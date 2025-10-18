#===- act/pipeline/torch2act.py - Torch to ACT Converter ---------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Spec-free, input-free Torch → ACT converter. Converts a wrapped 
#   nn.Sequential model (with embedded InputSpecLayer and OutputSpecLayer)
#   into an ACT Net/Layer graph consumable by downstream analyzers/solvers.
#
#===---------------------------------------------------------------------===#

# torch2act.py — Spec-free, input-free Torch → ACT converter
# Converts a *wrapped* nn.Sequential model (with embedded InputSpecLayer and OutputSpecLayer)
# into an ACT Net/Layer graph consumable by downstream analyzers/solvers.
#
# Contract:
#   - Exactly one InputLayer must be present (source of input shape).
#   - At least one InputSpecLayer exists (constraints embedded in post-adapter space).
#   - The last module is OutputSpecLayer (→ ASSERT as last ACT layer).
#
# Mapping:
#   InputLayer           → INPUT                   (allocates initial var block)
#   InputAdapterLayer    → PERMUTE/REORDER/SLICE/PAD/SCALE_SHIFT/LINEAR_PROJ
#   InputSpecLayer       → INPUT_SPEC              (constraint-only, no new vars)
#   nn.Flatten           → FLATTEN
#   nn.Linear            → DENSE
#   nn.ReLU              → RELU
#   OutputSpecLayer      → ASSERT                  (constraint-only, no new vars)
#
# Notes:
#   • No external input_shape or spec objects are accepted; everything is read from the wrapper.
#   • All numeric tensors (weights, bounds, etc.) go in Layer.params (torch.Tensor).
#   • Small flags/metadata go in Layer.meta (JSON-serializable).
#
# Optional helpers included:
#   - SolveResult enum + interpret_validation() (maps SAT/UNSAT to VIOLATED/VALID semantics).
#
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn

from act.front_end.model_inference import model_inference
from act.front_end.model_synthesis import model_synthesis
from act.back_end.core import Net, Layer
from act.back_end.layer_schema import LayerKind
from act.back_end.layer_util import create_layer
from act.back_end.solver.solver_torch import TorchLPSolver
from act.back_end.solver.solver_gurobi import GurobiSolver

# -----------------------------------------------------------------------------
# Public helper for solver interpretation (optional)
# -----------------------------------------------------------------------------

class SolveResult:
    SAT = "SAT"         # counterexample exists → property VIOLATED
    UNSAT = "UNSAT"     # no counterexample → property VALID
    UNKNOWN = "UNKNOWN"


def interpret_validation(net: Net, solve_status: str) -> Dict[str, Any]:
    last = net.last_validation()
    k = last.meta.get("kind") if last else None
    verdict = "VALID" if solve_status == SolveResult.UNSAT else (
              "VIOLATED" if solve_status == SolveResult.SAT else "UNKNOWN")
    return {
        "verdict": verdict,
        "solver_status": solve_status,
        "spec_kind": k,
        "assert_layer_id": last.id if last else None,
    }


# -----------------------------------------------------------------------------
# Torch → ACT converter
# -----------------------------------------------------------------------------

def _prod(shape_tail: Tuple[int, ...]) -> int:
    """Helper function to compute product of shape dimensions."""
    p = 1
    for s in shape_tail:
        p *= int(s)
    return int(p)


class TorchToACT:
    """
    Convert a *wrapped* nn.Sequential to ACT Net/Layers.
    Requirements (asserted in __init__):
      - Contains exactly one InputLayer (first-class source of input shape).
      - Contains at least one InputSpecLayer.
      - Ends with an OutputSpecLayer (producing ASSERT).
    No input_shape is accepted; InputLayer provides it.
    """
    # Type names are matched by isinstance; these references are not imported here to avoid circular deps.
    _InputLayerTypeName = "InputLayer"
    _InputAdapterLayerTypeName = "InputAdapterLayer"
    _InputSpecLayerTypeName = "InputSpecLayer"
    _OutputSpecLayerTypeName = "OutputSpecLayer"

    def __init__(self, wrapped: nn.Sequential):
        if not isinstance(wrapped, nn.Sequential):
            raise TypeError("TorchToACT expects an nn.Sequential wrapper model.")
        self.m = wrapped
        mods = list(self.m)

        # --- Assertions: InputSpecLayer and OutputSpecLayer existence ---
        has_input_spec = any(type(x).__name__ == self._InputSpecLayerTypeName for x in mods)
        has_output_spec = any(type(x).__name__ == self._OutputSpecLayerTypeName for x in mods)
        if not has_input_spec:
            raise AssertionError("Wrapper must include an InputSpecLayer (post-adapter) — none found.")
        if not has_output_spec:
            raise AssertionError("Wrapper must include an OutputSpecLayer as the final assertion — none found.")

        # Exactly one InputLayer
        input_layers = [x for x in mods if type(x).__name__ == self._InputLayerTypeName]
        if len(input_layers) != 1:
            raise AssertionError(f"Wrapper must contain exactly one InputLayer; found {len(input_layers)}.")
        self.input_layer = input_layers[0]

        # Must end with OutputSpecLayer
        if type(mods[-1]).__name__ != self._OutputSpecLayerTypeName:
            raise AssertionError("Wrapper should end with OutputSpecLayer so last ACT layer is ASSERT.")

        # Init state
        self.layers: List[Layer] = []
        self.next_var = 0
        self.prev_out: List[int] = []
        # Expect InputLayer to have a 'shape' attribute (tuple) that includes batch=1 first.
        shape = getattr(self.input_layer, "shape", None)
        if shape is None:
            raise AssertionError("InputLayer must expose a 'shape' attribute (e.g., (1, C, H, W) or (1, F)).")
        self.shape: Tuple[int, ...] = tuple(int(s) for s in shape)

    # --- var id management ---
    def _alloc_ids(self, n: int) -> List[int]:
        ids = list(range(self.next_var, self.next_var + n))
        self.next_var += n
        return ids

    def _add(self, kind: str, params: Dict[str, torch.Tensor], meta: Dict[str, Any],
             in_vars: List[int], out_vars: List[int]) -> int:
        layer = create_layer(
            id=len(self.layers),
            kind=kind,
            params=params,
            meta=meta,
            in_vars=in_vars,
            out_vars=out_vars,
        )
        self.layers.append(layer)
        return layer.id

    def _same_size_forward(self) -> List[int]:
        return self._alloc_ids(len(self.prev_out))

    # --- mapping helpers ---

    def _emit_input(self):
        """Emit INPUT layer using the to_act_layers() protocol."""
        new_layers, out_vars = self.input_layer.to_act_layers(len(self.layers), [])
        self.layers.extend(new_layers)
        self.prev_out = out_vars




    # --- main conversion ---

    def run(self) -> Net:
        # Emit INPUT from InputLayer
        self._emit_input()

        # Walk modules in order and emit ACT layers
        for mod in self.m:
            tname = type(mod).__name__

            if tname == self._InputLayerTypeName:
                # already emitted at start
                continue

            # Use standardized conversion with to_act_layers() protocol
            if hasattr(mod, 'to_act_layers'):
                new_layers, out_vars = mod.to_act_layers(len(self.layers), self.prev_out)
                self.layers.extend(new_layers)
                self.prev_out = out_vars
                continue

            # Handle standard PyTorch modules that don't have to_act_layers()
            if isinstance(mod, nn.Flatten):
                out_vars = self._same_size_forward()
                flattened_shape = (1, _prod(self.shape[1:]))
                self._add(LayerKind.FLATTEN.value, params={}, 
                          meta={"input_shape": self.shape, "output_shape": flattened_shape},
                          in_vars=self.prev_out, out_vars=out_vars)
                self.shape = flattened_shape
                self.prev_out = out_vars
                continue

            if isinstance(mod, nn.Linear):
                outF = int(mod.out_features)
                W = mod.weight.detach().clone()
                bvec = mod.bias.detach().clone() if mod.bias is not None else torch.zeros(outF, dtype=W.dtype, device=W.device)
                
                out_vars = self._alloc_ids(outF)
                self._add(LayerKind.DENSE.value, params={"W": W, "b": bvec},
                          meta={"input_shape": self.shape, "output_shape": (1, outF)},
                          in_vars=self.prev_out, out_vars=out_vars)
                self.shape = (1, outF)
                self.prev_out = out_vars
                continue

            if isinstance(mod, nn.ReLU):
                out_vars = self._same_size_forward()
                self._add(LayerKind.RELU.value, params={}, 
                          meta={"input_shape": self.shape, "output_shape": self.shape},
                          in_vars=self.prev_out, out_vars=out_vars)
                self.prev_out = out_vars
                continue

            if isinstance(mod, nn.Conv2d):
                # Handle Conv2d layers
                weight = mod.weight.detach().clone()
                bias = mod.bias.detach().clone() if mod.bias is not None else None
                
                # Infer input shape for conv
                if len(self.shape) == 2:  # (1, features) - need to reshape to spatial
                    n_features = self.shape[1]
                    if n_features == 3072:  # CIFAR-10
                        input_shape = (1, 3, 32, 32)
                    elif n_features == 784:  # MNIST
                        input_shape = (1, 1, 28, 28)
                    else:
                        channels = mod.in_channels
                        spatial_size = int((n_features / channels) ** 0.5)
                        input_shape = (1, channels, spatial_size, spatial_size)
                else:
                    input_shape = self.shape
                
                # Calculate output shape
                batch, in_c, in_h, in_w = input_shape
                out_c = mod.out_channels
                out_h = (in_h + 2 * mod.padding[0] - mod.dilation[0] * (mod.kernel_size[0] - 1) - 1) // mod.stride[0] + 1
                out_w = (in_w + 2 * mod.padding[1] - mod.dilation[1] * (mod.kernel_size[1] - 1) - 1) // mod.stride[1] + 1
                output_shape = (1, out_c, out_h, out_w)
                out_features = out_c * out_h * out_w
                
                params = {"weight": weight}
                if bias is not None:
                    params["bias"] = bias
                    
                meta = {
                    "input_shape": input_shape,
                    "output_shape": output_shape,
                    "kernel_size": mod.kernel_size,
                    "stride": mod.stride,
                    "padding": mod.padding,
                    "dilation": mod.dilation,
                    "groups": mod.groups,
                    "in_channels": in_c,
                    "out_channels": out_c
                }
                
                out_vars = self._alloc_ids(out_features)
                self._add(LayerKind.CONV2D.value, params=params, meta=meta,
                          in_vars=self.prev_out, out_vars=out_vars)
                self.shape = (1, out_features)  # Flatten for next layer
                self.prev_out = out_vars
                continue

            # Handle nested Sequential models (PyTorch models inside wrapper)
            if isinstance(mod, nn.Sequential):
                for sub_mod in mod:
                    # Recursively process sub-modules using the same logic
                    if hasattr(sub_mod, 'to_act_layers'):
                        new_layers, out_vars = sub_mod.to_act_layers(len(self.layers), self.prev_out)
                        self.layers.extend(new_layers)
                        self.prev_out = out_vars
                    elif isinstance(sub_mod, (nn.Flatten, nn.Linear, nn.ReLU, nn.Conv2d)):
                        # Reuse the same conversion logic as above
                        if isinstance(sub_mod, nn.Flatten):
                            out_vars = self._same_size_forward()
                            flattened_shape = (1, _prod(self.shape[1:]))
                            self._add(LayerKind.FLATTEN.value, params={}, 
                                      meta={"input_shape": self.shape, "output_shape": flattened_shape},
                                      in_vars=self.prev_out, out_vars=out_vars)
                            self.shape = flattened_shape
                            self.prev_out = out_vars
                        elif isinstance(sub_mod, nn.Linear):
                            outF = int(sub_mod.out_features)
                            W = sub_mod.weight.detach().clone()
                            bvec = sub_mod.bias.detach().clone() if sub_mod.bias is not None else torch.zeros(outF, dtype=W.dtype, device=W.device)
                            
                            out_vars = self._alloc_ids(outF)
                            self._add(LayerKind.DENSE.value, params={"W": W, "b": bvec},
                                      meta={"input_shape": self.shape, "output_shape": (1, outF)},
                                      in_vars=self.prev_out, out_vars=out_vars)
                            self.shape = (1, outF)
                            self.prev_out = out_vars
                        elif isinstance(sub_mod, nn.ReLU):
                            out_vars = self._same_size_forward()
                            self._add(LayerKind.RELU.value, params={}, 
                                      meta={"input_shape": self.shape, "output_shape": self.shape},
                                      in_vars=self.prev_out, out_vars=out_vars)
                            self.prev_out = out_vars
                        elif isinstance(sub_mod, nn.Conv2d):
                            # Same Conv2d logic as above
                            weight = sub_mod.weight.detach().clone()
                            bias = sub_mod.bias.detach().clone() if sub_mod.bias is not None else None
                            
                            if len(self.shape) == 2:
                                n_features = self.shape[1]
                                if n_features == 3072:
                                    input_shape = (1, 3, 32, 32)
                                elif n_features == 784:
                                    input_shape = (1, 1, 28, 28)
                                else:
                                    channels = sub_mod.in_channels
                                    spatial_size = int((n_features / channels) ** 0.5)
                                    input_shape = (1, channels, spatial_size, spatial_size)
                            else:
                                input_shape = self.shape
                            
                            batch, in_c, in_h, in_w = input_shape
                            out_c = sub_mod.out_channels
                            out_h = (in_h + 2 * sub_mod.padding[0] - sub_mod.dilation[0] * (sub_mod.kernel_size[0] - 1) - 1) // sub_mod.stride[0] + 1
                            out_w = (in_w + 2 * sub_mod.padding[1] - sub_mod.dilation[1] * (sub_mod.kernel_size[1] - 1) - 1) // sub_mod.stride[1] + 1
                            output_shape = (1, out_c, out_h, out_w)
                            out_features = out_c * out_h * out_w
                            
                            params = {"weight": weight}
                            if bias is not None:
                                params["bias"] = bias
                                
                            meta = {
                                "input_shape": input_shape,
                                "output_shape": output_shape,
                                "kernel_size": sub_mod.kernel_size,
                                "stride": sub_mod.stride,
                                "padding": sub_mod.padding,
                                "dilation": sub_mod.dilation,
                                "groups": sub_mod.groups,
                                "in_channels": in_c,
                                "out_channels": out_c
                            }
                            
                            out_vars = self._alloc_ids(out_features)
                            self._add(LayerKind.CONV2D.value, params=params, meta=meta,
                                      in_vars=self.prev_out, out_vars=out_vars)
                            self.shape = (1, out_features)
                            self.prev_out = out_vars
                    else:
                        raise NotImplementedError(f"Unsupported sub-module in Sequential: {type(sub_mod).__name__}")
                continue

            # Unsupported module
            raise NotImplementedError(f"Unsupported module in converter: {tname}")

        # Build linear preds/succs
        preds = {i: ([] if i == 0 else [i - 1]) for i in range(len(self.layers))}
        succs = {i: ([] if i == len(self.layers) - 1 else [i + 1]) for i in range(len(self.layers))}
        net = Net(layers=self.layers, preds=preds, succs=succs)

        # Validate the created network structure
        from act.back_end.layer_util import validate_graph
        validate_graph(self.layers)  # Pass layers list, not net

        # Final sanity check
        net.assert_last_is_validation()
        return net

    
if __name__ == "__main__":
    print("🚀 Starting Spec-Free, Input-Free Torch→ACT Verification Demo")
    
    # Step 1: Synthesize all wrapped models
    print("\n📦 Step 1: Synthesizing wrapped models...")
    wrapped_models, input_data = model_synthesis()
    print(f"  ✅ Generated {len(wrapped_models)} wrapped models")
    
    # Step 2: Test all models with inference
    print("\n🧪 Step 2: Testing model inference...")
    successful_models = model_inference(wrapped_models, input_data)
    print(f"  ✅ {len(successful_models)} models passed inference tests")
    
    if not successful_models:
        print("  ❌ No successful models to verify!")
        exit(1)
    
    # Step 3: Convert all successful models to ACT
    print(f"\n🎯 Step 3: Converting all {len(successful_models)} successful models to ACT...")
    
    conversion_results = {}
    successful_conversions = {}
    
    for model_id, wrapped_model in successful_models.items():
        print(f"\n  🔄 Converting '{model_id}'...")
        try:
            # Convert wrapped model to ACT Net (spec-free)
            net = TorchToACT(wrapped_model).run()
            
            # Verify the conversion produced a valid net
            assert net.layers, "Net should have layers"
            assert net.layers[0].kind == "INPUT", "First layer should be INPUT"
            assert net.layers[-1].kind == "ASSERT", "Last layer should be ASSERT"
            
            # Store successful conversion
            successful_conversions[model_id] = (wrapped_model, net)
            conversion_results[model_id] = "SUCCESS"
            
            # Get layer summary
            layer_types = " → ".join([layer.kind for layer in net.layers])
            print(f"    ✅ SUCCESS: {len(net.layers)} layers ({layer_types})")
            
        except Exception as e:
            conversion_results[model_id] = f"FAILED: {str(e)[:100]}..."
            print(f"    ❌ FAILED: {e}")
            continue
    
    # Summary of conversions
    success_count = len(successful_conversions)
    total_count = len(successful_models)
    print(f"\n📊 Conversion Summary:")
    print(f"  ✅ Successful conversions: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
    
    if not successful_conversions:
        print("  ❌ No successful conversions to verify!")
        exit(1)
    
    # Step 4: Initialize solvers
    print("\n🔧 Step 4: Initializing solvers...")
    gurobi_solver = None
    torch_solver = None
    
    try:
        gurobi_solver = GurobiSolver()
        gurobi_solver.begin("act_verification")
        print("  ✅ Gurobi solver available")
    except Exception as e:
        print(f"  ⚠️  Gurobi initialization failed: {e}")
    
    try:
        torch_solver = TorchLPSolver()
        # Use default device for TorchLP solver
        torch_solver.begin("act_verification", device="gpu")
        print("  ✅ TorchLP solver available (device: gpu)")
    except Exception as e:
        print(f"  ❌ TorchLP initialization failed: {e}")
    
    solvers_to_test = []
    if gurobi_solver:
        solvers_to_test.append(("Gurobi", gurobi_solver))
    if torch_solver:
        solvers_to_test.append(("TorchLP", torch_solver))
    
    if not solvers_to_test:
        print("  ❌ No solvers available!")
        exit(1)
    
    # Step 5: Run verification on just the first model for debugging
    print(f"\n🔍 Step 5: Running verification on first model for debugging...")
    
    # Import verification functions here to avoid early import issues
    from act.back_end.verifier import verify_once, verify_bab
    
    verification_results = {}
    
    # Just test the first model
    first_model_id = list(successful_conversions.keys())[0]
    wrapped_model, net = successful_conversions[first_model_id]
    
    print(f"\n🎯 Debugging model: '{first_model_id}'")
    print(f"  📐 Net structure: {' → '.join([layer.kind for layer in net.layers])}")
    
    model_results = {}
    
    for solver_name, solver in solvers_to_test:
        print(f"\n  --- Testing with {solver_name} solver ---")
        
        try:
            # Single-shot verification
            print("    🎯 Running single-shot verification...")
            res = verify_once(net, solver=solver, timelimit=30.0)
            print(f"      Status: {res.status}")
            if res.model_stats:
                print(f"      Stats: {res.model_stats}")
            
            model_results[solver_name] = {
                'single_shot': res.status
            }
            
        except Exception as e:
            print(f"    ❌ Verification failed with {solver_name}: {e}")
            print(f"    🔍 Full exception type: {type(e).__name__}")
            print(f"    🔍 Full exception message: {str(e)}")
            import traceback
            print(f"    🔍 Traceback:")
            traceback.print_exc()
            model_results[solver_name] = {'error': str(e)}
            continue
    
    verification_results[first_model_id] = model_results
    
    # Final verification summary
    print(f"\n📊 Debug Verification Summary:")
    print(f"  🔄 Models converted: {len(successful_conversions)}/{len(successful_models)}")
    print(f"  🔧 Solvers tested: {len(solvers_to_test)}")
    
    for model_id, results in verification_results.items():
        print(f"\n  📋 {model_id}:")
        for solver_name, solver_results in results.items():
            if 'error' in solver_results:
                print(f"    {solver_name}: ERROR - {solver_results['error'][:100]}...")
            else:
                single_status = solver_results.get('single_shot', 'N/A')
                print(f"    {solver_name}: Single={single_status}")
    
    print("\n🔍 Debug verification completed!")