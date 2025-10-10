#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#########################################################################
##   Abstract Constraint Transformer (ACT) - HybridZ Verifier          ##
##                                                                     ##
##   doctormeeee (https://github.com/doctormeeee) and contributors     ##
##   Copyright (C) 2024-2025                                           ##
##                                                                     ##
#########################################################################

import torch
import torch.nn as nn
import numpy as np
import time
from typing import Optional, List


from act.interval.base_verifier import BaseVerifier
from act.util.stats import ACTStats
from act.util.inference import perform_model_inference
from act.input_parser.spec import Spec
from act.input_parser.type import VerifyResult
from onnx2pytorch.operations.flatten import Flatten as OnnxFlatten
from onnx2pytorch.operations.base import OperatorWrapper
from act.hybridz.hybridz_transformers import HybridZonotopeGrid, HybridZonotopeElem
from act.hybridz.hybridz_operations import HybridZonotopeOps

try:
    from auto_LiRPA import BoundedModule, PerturbationLpNorm, BoundedTensor
    AUTOLIRPA_AVAILABLE = True
except ImportError:
    print("Warning: auto_LiRPA not available. HybridZonotopeVerifier will use standard bounds computation.")
    AUTOLIRPA_AVAILABLE = False

class HybridZonotopeVerifier(BaseVerifier):
    def __init__(self, method : str, spec: Spec, device: str = 'cpu',
                 relaxation_ratio: float = 1.0, enable_generator_merging: bool = False, cosine_threshold: float = 0.95,
                 ci_mode: bool = False):

        from act.hybridz.hybridz_transformers import HybridZonotopeElem
        self.HybridZonotopeElem = HybridZonotopeElem

        super().__init__(spec, device)

        self.method = method
        self.device = device
        self.relaxation_ratio = relaxation_ratio
        self.enable_generator_merging = enable_generator_merging
        self.cosine_threshold = cosine_threshold
        self.ci_mode = ci_mode

        self.hz_layer_bounds = {}
        self.autolirpa_layer_bounds = {}
        self.concrete_layer_values = {}
        self.layer_precision_comparison = {}
        self.soundness_check_results = {}

        self.use_auto_lirpa = False
        self.enable_layer_comparison = False
        self.enable_soundness_check = False

        if self.method == 'hybridz_relaxed':
            print(f"Relaxation Strategy: ratio={relaxation_ratio:.1f} ({'Full MILP (exact)' if relaxation_ratio == 0.0 else 'Full LP (relaxed)' if relaxation_ratio == 1.0 else f'{int(relaxation_ratio*100)}% Relaxed + {int((1-relaxation_ratio)*100)}% Exact'})")
        print(f"Generator Merging: {'Enabled' if enable_generator_merging else 'Disabled'}{f' (threshold={cosine_threshold})' if enable_generator_merging else ''}")
        if enable_generator_merging:
            print(f"Strategy: Automatically enabling parallel generator merging at the last fully-connected layer")

        self.late_stage_config = {
            'enabled': False,
            'start_layer': -3,
            'refinement_layers': ['ReLU', 'Linear'],
            'base_verifier': 'auto_lirpa',
            'bound_method': 'IBP+backward',
        }

        if self.late_stage_config['enabled']:
            print(f"Late-stage refinement enabled:")
            print(f"Base verifier: {self.late_stage_config['base_verifier']}")
            print(f"HybridZ starts from layer: {self.late_stage_config['start_layer']}")
            print(f"Refinement on: {self.late_stage_config['refinement_layers']}")
        else:
            print(" HybridZonotopeVerifier: auto_LiRPA pre-run disabled, using standard bound computation")


    def _setup_auto_lirpa(self, input_example):

        if not self.use_auto_lirpa:
            return False

        try:
            print("Setting up auto_LiRPA BoundedModule...")

            self.bounded_model = BoundedModule(
                self.spec.model.pytorch_model,
                input_example,
                device=self.device
            )
            print("BoundedModule setup complete.")
            return True
        except Exception as e:
            print(f"Warning: auto_LiRPA setup failed: {e}")
            self.use_auto_lirpa = False
            return False

    def _create_autolirpa_ordered_layer_bounds(self):

        self.autolirpa_ordered_layer_bounds = []

        node_layer_mapping = [
            ("/input-1", "input"),
            ("/input", "conv"),
            ("/input-4", "conv"),
            ("/input-8", "conv"),
            ("/24", "relu"),

            ("/25", "flatten"),
            ("/input-12", "linear"),
            ("/27", "relu"),
            ("/28", "linear"),
        ]

        print("🔄 Creating ordered layer bounds mapping...")
        for i, (node_name, layer_type) in enumerate(node_layer_mapping):
            if node_name in self.autolirpa_layer_bounds:
                bounds = self.autolirpa_layer_bounds[node_name]
                self.autolirpa_ordered_layer_bounds.append({
                    'node_name': node_name,
                    'layer_type': layer_type,
                    'layer_index': i,
                    'lb': bounds['lb'],
                    'ub': bounds['ub'],
                    'shape': bounds['shape']
                })
                print(f"✅ Mapped {node_name} -> Layer {i} ({layer_type}): {bounds['shape']}")
            else:
                print(f"⚠️  Missing bounds for {node_name} (Layer {i}, {layer_type})")

        print(f"✅ Created ordered mapping for {len(self.autolirpa_ordered_layer_bounds)} layers")

    def _compute_autolirpa_bounds(self, input_bounds, eps=None, method='CROWN'):

        if not self.use_auto_lirpa or not hasattr(self, 'bounded_model'):
            return False

        input_lb, input_ub = input_bounds

        print(f"[Auto_LiRPA] Computing all layer bounds using method: {method}")

        try:

            input_center = (input_lb + input_ub) / 2.0

            if eps is not None and eps > 0:

                print(f"Using center point + eps={eps} perturbation")
                ptb = PerturbationLpNorm(norm=np.inf, eps=eps)
                bounded_input = BoundedTensor(input_center.unsqueeze(0), ptb)
            else:

                print("Using explicit bounds (x_L, x_U)")
                ptb = PerturbationLpNorm(
                    norm=np.inf,
                    x_L=input_lb.unsqueeze(0),
                    x_U=input_ub.unsqueeze(0)
                )
                bounded_input = BoundedTensor(input_center.unsqueeze(0), ptb)

            lb, ub = self.bounded_model.compute_bounds(
                x=(bounded_input,),
                method=method,
                IBP=(method in ['IBP', 'IBP+backward', 'CROWN-IBP']),
                forward=(method in ['Forward', 'Forward+Backward']),
                bound_lower=True,
                bound_upper=True,
                return_A=False
            )

            print(f"Final bounds range: [{lb.min().item():.6f}, {ub.max().item():.6f}]")
            for idx, (lb_, ub_) in enumerate(zip(lb, ub)):
                if isinstance(lb_, torch.Tensor):
                    lb_vals = [f"{x:.6f}" for x in lb_.detach().cpu().numpy()]
                    ub_vals = [f"{x:.6f}" for x in ub_.detach().cpu().numpy()]
                    print(f"Output[{idx}]: [{', '.join(lb_vals)}] to [{', '.join(ub_vals)}]")
                else:
                    print(f"Output[{idx}]: [{lb_:.6f}] to [{ub_:.6f}]")

            self.autolirpa_layer_bounds = {}
            layer_count = 0

            try:
                intermediate_bounds = self.bounded_model.save_intermediate()
                if intermediate_bounds:
                    print("🔄 Collecting intermediate bounds from save_intermediate()...")
                    print(f"📊 Found {len(intermediate_bounds)} nodes in intermediate_bounds")

                    for node_name, bounds in intermediate_bounds.items():
                        print(f"Processing node: {node_name}")
                        print(f"Bounds type: {type(bounds)}")
                        print(f"Bounds keys: {bounds.keys() if isinstance(bounds, dict) else 'Not a dict'}")

                        lower, upper = None, None

                        if isinstance(bounds, tuple) and len(bounds) >= 2:
                            lower = bounds[0]
                            upper = bounds[1]
                            print(f"Found {type(bounds).__name__} bounds for {node_name}")

                        elif torch.is_tensor(bounds):
                            lower = bounds
                            upper = bounds
                            print(f"Found tensor bounds for {node_name}")

                        else:
                            print(f"⚠️  Unknown bounds structure for {node_name}: {type(bounds)}")
                            continue

                        if (lower is not None and upper is not None and
                            torch.is_tensor(lower) and torch.is_tensor(upper) and
                            lower.numel() > 0 and upper.numel() > 0):

                            self.autolirpa_layer_bounds[node_name] = {
                                'lb': lower.detach().clone(),
                                'ub': upper.detach().clone(),
                                'shape': lower.shape,
                                'method': 'CROWN',
                                'node_type': 'auto_lirpa_node'
                            }
                            layer_count += 1
                            print(f"✅ Saved bounds for {node_name}: {lower.shape}")
                        else:
                            print(f"❌ Invalid bounds for {node_name}: lower={type(lower)}, upper={type(upper)}")

                    print(f"Method 1 (save_intermediate): collected {layer_count} layers")
            except Exception as e:
                print(f"⚠️  save_intermediate() failed: {e}")

            if layer_count == 0:
                print("🔄 Fallback: saving final layer bounds only...")
                self.autolirpa_layer_bounds['final_output'] = {
                    'lb': lb.detach().clone(),
                    'ub': ub.detach().clone(),
                    'shape': lb.shape,
                    'method': 'CROWN',
                    'node_type': 'final_output'
                }
                layer_count = 1

            print(f"✅ Successfully computed bounds for {layer_count} layers using auto_LiRPA")

            self._create_autolirpa_ordered_layer_bounds()

            return True

        except Exception as e:
            print(f"❌ auto_LiRPA bounds computation failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _get_autolirpa_activation_bounds(self, layer_name):

        if layer_name in self.autolirpa_layer_bounds:
            bounds = self.autolirpa_layer_bounds[layer_name]
            return bounds['lb'], bounds['ub']
        return None, None

    def _get_autolirpa_stable_unstable_neurons(self, layer_name):

        lower, upper = self._get_autolirpa_activation_bounds(layer_name)
        if lower is None or upper is None:
            return None, None, None

        lower_flat = lower.view(-1)
        upper_flat = upper.view(-1)

        stable_positive = (lower_flat > 0).nonzero(as_tuple=True)[0]
        stable_negative = (upper_flat < 0).nonzero(as_tuple=True)[0]
        unstable = ((lower_flat <= 0) & (upper_flat >= 0)).nonzero(as_tuple=True)[0]

        print(f"Layer {layer_name}: {len(stable_positive)} stable+, {len(stable_negative)} stable-, {len(unstable)} unstable")

        return stable_positive, stable_negative, unstable

    def _compute_hz_layer_bounds(self, hz, layer_name, layer_type="unknown"):

        try:
            print(f"[HZ Bounds] Computing bounds for {layer_name} ({layer_type})")
            print(f"[HZ Bounds] Current time: {time.time()}")

            if hasattr(hz, 'PreActivationGetFlattenedTensor'):

                flat_center, flat_G_c, flat_G_b = hz.PreActivationGetFlattenedTensor()
                A_c, A_b, b = hz.A_c_tensor, hz.A_b_tensor, hz.b_tensor
            else:

                flat_center, flat_G_c, flat_G_b = hz.center, hz.G_c, hz.G_b
                A_c, A_b, b = hz.A_c, hz.A_b, hz.b

            print(f"[HZ Bounds] Data extracted, about to call GetLayerWiseBounds")

            method = hz.method
            print(f"[HZ Bounds] Using method: {method} for layer {layer_name}")

            print(f"[HZ Bounds] Calling GetLayerWiseBounds with method={method}, time_limit=500")
            start_time = time.time()

            lb, ub = HybridZonotopeOps.GetLayerWiseBounds(
                flat_center, flat_G_c, flat_G_b, A_c, A_b, b,
                method, time_limit=500, ci_mode=self.ci_mode
            )

            end_time = time.time()
            print(f"[HZ Bounds] GetLayerWiseBounds completed in {end_time - start_time:.2f}s")

            self.hz_layer_bounds[layer_name] = {
                'lb': lb.detach().clone(),
                'ub': ub.detach().clone(),
                'shape': lb.shape,
                'layer_type': layer_type,
                'method': method
            }

            print(f"✅ [HZ Bounds] {layer_name}: range=[{lb.min():.6f}, {ub.max():.6f}]")
            return lb, ub

        except Exception as e:
            print(f"❌ [HZ Bounds] Failed for {layer_name}: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def _compute_concrete_inference(self, input_center, model):

        try:
            print(f"[Concrete] Computing concrete network inference at input center")
            print(f"[Concrete] Input center shape: {input_center.shape}")

            if input_center.dim() == 1:
                x = input_center.unsqueeze(0)
            elif input_center.dim() == 3:
                x = input_center.unsqueeze(0)
            else:
                x = input_center

            x = x.to(self.device)
            layer_values = {}
            layer_count = 0

            print(f"[Concrete] Starting forward pass with input shape: {x.shape}")

            for layer in model.children():
                layer_name = f"layer_{layer_count}_{type(layer).__name__}"
                layer_type = type(layer).__name__.lower()

                print(f"[Concrete] Processing {layer_name} ({layer_type})")
                print(f"Input shape: {x.shape}")

                try:

                    x_prev = x.clone()
                    x = layer(x)

                    print(f"Raw output type: {type(x)}")
                    if hasattr(x, 'shape'):
                        print(f"Raw output shape: {x.shape}")

                    if isinstance(x, tuple):

                        print(f"⚠️  Layer {layer_name} returned tuple with {len(x)} elements, using first element")
                        print(f"Tuple elements types: {[type(elem) for elem in x]}")
                        if len(x) > 0:
                            x = x[0]
                        else:
                            print(f"❌ Empty tuple from {layer_name}, skipping layer")
                            continue
                    elif isinstance(x, list):

                        print(f"⚠️  Layer {layer_name} returned list with {len(x)} elements, using first element")
                        print(f"List elements types: {[type(elem) for elem in x]}")
                        if len(x) > 0:
                            x = x[0]
                        else:
                            print(f"❌ Empty list from {layer_name}, skipping layer")
                            continue

                    if not torch.is_tensor(x):
                        print(f"❌ Layer {layer_name} output is not a tensor: {type(x)}")
                        continue

                    if x.dim() > 1:
                        x_flat = x.squeeze(0).view(-1) if x.shape[0] == 1 else x.view(-1)
                    else:
                        x_flat = x

                    layer_values[layer_name] = {
                        'values': x_flat.detach().clone(),
                        'shape': x.shape,
                        'layer_type': layer_type,
                        'layer_index': layer_count
                    }

                    print(f"✅ [Concrete] {layer_name}: {x.shape} -> flattened: {x_flat.shape}")
                    print(f"Range: [{x_flat.min():.6f}, {x_flat.max():.6f}]")

                except Exception as layer_e:
                    print(f"❌ Error processing layer {layer_name}: {layer_e}")
                    print(f"Layer type: {type(layer)}")
                    print(f"Input shape: {x_prev.shape if 'x_prev' in locals() else 'unknown'}")
                    continue

                layer_count += 1

            self.concrete_layer_values = layer_values

            print(f"✅ [Concrete] Completed inference for {layer_count} layers")
            return layer_values

        except Exception as e:
            print(f"❌ [Concrete] Failed: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def _check_soundness(self, layer_name, hz_bounds=None, autolirpa_bounds=None, concrete_values=None):

        try:
            print(f"\n[Soundness Check] {layer_name}")
            print("="*80)

            if concrete_values is None and layer_name in self.concrete_layer_values:
                concrete_data = self.concrete_layer_values[layer_name]
                concrete_vals = concrete_data['values']
            elif concrete_values is not None:
                concrete_vals = concrete_values
            else:
                print(f"⚠️  [Soundness] No concrete values for {layer_name}")
                return

            if hz_bounds is None and layer_name in self.hz_layer_bounds:
                hz_data = self.hz_layer_bounds[layer_name]
                hz_lb, hz_ub = hz_data['lb'], hz_data['ub']
            elif hz_bounds is not None:
                hz_lb, hz_ub = hz_bounds
            else:
                hz_lb, hz_ub = None, None

            if autolirpa_bounds is None and layer_name in self.autolirpa_layer_bounds:
                crown_data = self.autolirpa_layer_bounds[layer_name]
                crown_lb, crown_ub = crown_data['lb'], crown_data['ub']
            elif autolirpa_bounds is not None:
                crown_lb, crown_ub = autolirpa_bounds
            else:
                crown_lb, crown_ub = None, None

            print(f"Data shapes: Concrete={concrete_vals.shape}")
            if hz_lb is not None:
                print(f"HZ={hz_lb.shape}")
            if crown_lb is not None:
                print(f"CROWN={crown_lb.shape}")

            soundness_results = {
                'layer_name': layer_name,
                'hz_sound': None,
                'crown_sound': None,
                'hz_violations': 0,
                'crown_violations': 0,
                'total_elements': concrete_vals.numel()
            }

            if hz_lb is not None and hz_ub is not None:

                if concrete_vals.shape != hz_lb.shape:
                    if hz_lb.dim() > 1:
                        hz_lb_flat = hz_lb.squeeze(0).view(-1) if hz_lb.shape[0] == 1 else hz_lb.view(-1)
                        hz_ub_flat = hz_ub.squeeze(0).view(-1) if hz_ub.shape[0] == 1 else hz_ub.view(-1)
                    else:
                        hz_lb_flat = hz_lb
                        hz_ub_flat = hz_ub

                    if concrete_vals.shape == hz_lb_flat.shape:
                        hz_lb, hz_ub = hz_lb_flat, hz_ub_flat
                    else:
                        print(f"❌ HZ shape still mismatched: Concrete={concrete_vals.shape}, HZ={hz_lb_flat.shape}")
                        hz_lb, hz_ub = None, None

                if hz_lb is not None:

                    eps_abs = 1e-5
                    eps_rel = 1e-5

                    hz_lb_tolerance = torch.max(torch.full_like(hz_lb, eps_abs), eps_rel * torch.abs(hz_lb))
                    hz_ub_tolerance = torch.max(torch.full_like(hz_ub, eps_abs), eps_rel * torch.abs(hz_ub))

                    lb_violations = (concrete_vals < (hz_lb - hz_lb_tolerance)).sum().item()
                    ub_violations = (concrete_vals > (hz_ub + hz_ub_tolerance)).sum().item()
                    total_violations = lb_violations + ub_violations

                    soundness_results['hz_sound'] = (total_violations == 0)
                    soundness_results['hz_violations'] = total_violations

                    print(f"🔍 HZ Soundness: {'✅ SOUND' if total_violations == 0 else f'❌ VIOLATIONS'}")
                    print(f"Lower bound violations: {lb_violations} (tolerance: {eps_abs:.0e})")
                    print(f"Upper bound violations: {ub_violations} (tolerance: {eps_abs:.0e})")
                    print(f"Total violations: {total_violations}/{concrete_vals.numel()}")

                    if total_violations > 0:

                        violation_indices = ((concrete_vals < (hz_lb - hz_lb_tolerance)) | (concrete_vals > (hz_ub + hz_ub_tolerance))).nonzero(as_tuple=True)[0]
                        print(f"Violation details (first 5):")
                        for i, idx in enumerate(violation_indices[:5]):
                            idx = idx.item()
                            concrete_val = concrete_vals[idx].item()
                            lb_val = hz_lb[idx].item()
                            ub_val = hz_ub[idx].item()
                            lb_tol = hz_lb_tolerance[idx].item()
                            ub_tol = hz_ub_tolerance[idx].item()
                            print(f"Index {idx}: concrete={concrete_val:.6f}, bounds=[{lb_val:.6f}, {ub_val:.6f}], tolerance=[{lb_tol:.0e}, {ub_tol:.0e}]")

            if crown_lb is not None and crown_ub is not None:

                if concrete_vals.shape != crown_lb.shape:
                    if crown_lb.dim() > 1:
                        crown_lb_flat = crown_lb.squeeze(0).view(-1) if crown_lb.shape[0] == 1 else crown_lb.view(-1)
                        crown_ub_flat = crown_ub.squeeze(0).view(-1) if crown_ub.shape[0] == 1 else crown_ub.view(-1)
                    else:
                        crown_lb_flat = crown_lb
                        crown_ub_flat = crown_ub

                    if concrete_vals.shape == crown_lb_flat.shape:
                        crown_lb, crown_ub = crown_lb_flat, crown_ub_flat
                    else:
                        print(f"❌ CROWN shape still mismatched: Concrete={concrete_vals.shape}, CROWN={crown_lb_flat.shape}")
                        crown_lb, crown_ub = None, None

                if crown_lb is not None:

                    eps_abs = 1e-5
                    eps_rel = 1e-5

                    crown_lb_tolerance = torch.max(torch.full_like(crown_lb, eps_abs), eps_rel * torch.abs(crown_lb))
                    crown_ub_tolerance = torch.max(torch.full_like(crown_ub, eps_abs), eps_rel * torch.abs(crown_ub))

                    lb_violations = (concrete_vals < (crown_lb - crown_lb_tolerance)).sum().item()
                    ub_violations = (concrete_vals > (crown_ub + crown_ub_tolerance)).sum().item()
                    total_violations = lb_violations + ub_violations

                    soundness_results['crown_sound'] = (total_violations == 0)
                    soundness_results['crown_violations'] = total_violations

                    print(f"🔍 CROWN Soundness: {'✅ SOUND' if total_violations == 0 else f'❌ VIOLATIONS'}")
                    print(f"Lower bound violations: {lb_violations} (tolerance: {eps_abs:.0e})")
                    print(f"Upper bound violations: {ub_violations} (tolerance: {eps_abs:.0e})")
                    print(f"Total violations: {total_violations}/{concrete_vals.numel()}")

                    if total_violations > 0:

                        violation_indices = ((concrete_vals < (crown_lb - crown_lb_tolerance)) | (concrete_vals > (crown_ub + crown_ub_tolerance))).nonzero(as_tuple=True)[0]
                        print(f"Violation details (first 5):")
                        for i, idx in enumerate(violation_indices[:5]):
                            idx = idx.item()
                            concrete_val = concrete_vals[idx].item()
                            lb_val = crown_lb[idx].item()
                            ub_val = crown_ub[idx].item()
                            lb_tol = crown_lb_tolerance[idx].item()
                            ub_tol = crown_ub_tolerance[idx].item()
                            print(f"Index {idx}: concrete={concrete_val:.6f}, bounds=[{lb_val:.6f}, {ub_val:.6f}], tolerance=[{lb_tol:.0e}, {ub_tol:.0e}]")

            self.soundness_check_results[layer_name] = soundness_results

            print("="*80)

        except Exception as e:
            print(f"❌ [Soundness Check] Failed for {layer_name}: {e}")
            import traceback
            traceback.print_exc()


    def _print_final_soundness_summary(self):
        if not self.soundness_check_results:
            print("[Final Soundness Summary] No soundness check data available")
            return

        print("\n" + "="*80)
        print("🔍 FINAL SOUNDNESS CHECK SUMMARY")
        print("="*80)

        hz_sound_count = 0
        crown_sound_count = 0
        total_layers = 0
        total_hz_violations = 0
        total_crown_violations = 0
        total_elements = 0

        for layer_name, result in self.soundness_check_results.items():
            total_layers += 1
            total_elements += result['total_elements']

            if result['hz_sound'] is not None:
                if result['hz_sound']:
                    hz_sound_count += 1
                total_hz_violations += result['hz_violations']
                hz_status = "✅ SOUND" if result['hz_sound'] else f"❌ {result['hz_violations']} violations"
            else:
                hz_status = "N/A"

            if result['crown_sound'] is not None:
                if result['crown_sound']:
                    crown_sound_count += 1
                total_crown_violations += result['crown_violations']
                crown_status = "✅ SOUND" if result['crown_sound'] else f"❌ {result['crown_violations']} violations"
            else:
                crown_status = "N/A"

            print(f"{layer_name:25s}: HZ={hz_status:15s} CROWN={crown_status:15s}")

        print("-" * 80)
        print(f"📊 Overall Soundness Results:")
        print(f"Total Layers: {total_layers}")
        print(f"Total Elements: {total_elements}")

        hz_valid_layers = sum(1 for r in self.soundness_check_results.values() if r['hz_sound'] is not None)
        crown_valid_layers = sum(1 for r in self.soundness_check_results.values() if r['crown_sound'] is not None)

        if hz_valid_layers > 0:
            print(f"HZ Soundness: {hz_sound_count}/{hz_valid_layers} layers ({hz_sound_count/hz_valid_layers:.1%})")
            print(f"HZ Violations: {total_hz_violations}/{total_elements} elements ({total_hz_violations/total_elements:.2%})")

        if crown_valid_layers > 0:
            print(f"CROWN Soundness: {crown_sound_count}/{crown_valid_layers} layers ({crown_sound_count/crown_valid_layers:.1%})")
            print(f"CROWN Violations: {total_crown_violations}/{total_elements} elements ({total_crown_violations/total_elements:.2%})")

        if hz_valid_layers > 0 and total_hz_violations == 0:
            print("🏆 HZ Overall: ✅ COMPLETELY SOUND")
        elif hz_valid_layers > 0:
            print("🏆 HZ Overall: ❌ SOUNDNESS VIOLATIONS DETECTED")

        if crown_valid_layers > 0 and total_crown_violations == 0:
            print("🏆 CROWN Overall: ✅ COMPLETELY SOUND")
        elif crown_valid_layers > 0:
            print("🏆 CROWN Overall: ❌ SOUNDNESS VIOLATIONS DETECTED")

        print("="*80)

    def _get_autolirpa_layer_bounds(self, layer_name, layer_index=None):

        try:

            if hasattr(self, 'autolirpa_ordered_layer_bounds') and layer_index is not None:
                if 0 <= layer_index < len(self.autolirpa_ordered_layer_bounds):
                    bounds_info = self.autolirpa_ordered_layer_bounds[layer_index]
                    lb, ub = bounds_info['lb'], bounds_info['ub']

                    if layer_name not in self.autolirpa_layer_bounds:
                        self.autolirpa_layer_bounds[layer_name] = {
                            'lb': lb.detach().clone(),
                            'ub': ub.detach().clone(),
                            'shape': lb.shape,
                            'method': 'CROWN',
                            'auto_lirpa_node': bounds_info['node_name']
                        }

                    return lb, ub

            if layer_name in self.autolirpa_layer_bounds:
                bounds = self.autolirpa_layer_bounds[layer_name]
                return bounds['lb'], bounds['ub']

            return None, None

        except Exception as e:
            print(f"❌ [CROWN Bounds] Failed for {layer_name}: {e}")
            return None, None

    def _compare_layer_precision(self, layer_name, hz_bounds=None, autolirpa_bounds=None):

        try:
            print(f"\n🔍 [Detailed Precision Comparison] {layer_name}")
            print("="*80)

            if self.enable_soundness_check:
                self._check_soundness(layer_name, hz_bounds, autolirpa_bounds)

            if hz_bounds is None and layer_name in self.hz_layer_bounds:
                hz_data = self.hz_layer_bounds[layer_name]
                hz_lb, hz_ub = hz_data['lb'], hz_data['ub']
            elif hz_bounds is not None:
                hz_lb, hz_ub = hz_bounds
            else:
                hz_lb, hz_ub = None, None

            if autolirpa_bounds is None and layer_name in self.autolirpa_layer_bounds:
                crown_data = self.autolirpa_layer_bounds[layer_name]
                crown_lb, crown_ub = crown_data['lb'], crown_data['ub']
            elif autolirpa_bounds is not None:
                crown_lb, crown_ub = autolirpa_bounds
            else:
                crown_lb, crown_ub = None, None

            if hz_lb is None or crown_lb is None:
                print(f"⚠️  [Precision] Incomplete bounds for {layer_name}")
                return

            print(f"[DEBUG] HZ bounds source: {type(hz_lb)}, shape: {hz_lb.shape}")
            print(f"[DEBUG] CROWN bounds source: {type(crown_lb)}, shape: {crown_lb.shape}")
            print(f"[DEBUG] HZ bounds range: [{hz_lb.min():.6f}, {hz_ub.max():.6f}]")
            print(f"[DEBUG] CROWN bounds range: [{crown_lb.min():.6f}, {crown_ub.max():.6f}]")
            print(f"[DEBUG] Are HZ and CROWN lb the same tensor? {torch.equal(hz_lb, crown_lb) if hz_lb.shape == crown_lb.shape else 'Different shapes'}")
            print(f"[DEBUG] Are HZ and CROWN ub the same tensor? {torch.equal(hz_ub, crown_ub) if hz_ub.shape == crown_ub.shape else 'Different shapes'}")

            concrete_vals = None
            if layer_name in self.concrete_layer_values:
                concrete_data = self.concrete_layer_values[layer_name]
                concrete_vals = concrete_data['values']
                print(f"Original shapes: HZ={hz_lb.shape}, CROWN={crown_lb.shape}, Concrete={concrete_vals.shape}")
            else:
                print(f"Original shapes: HZ={hz_lb.shape}, CROWN={crown_lb.shape}, Concrete=N/A")

            if hz_lb.shape != crown_lb.shape:
                print(f"Shape mismatch, attempting to flatten CROWN bounds...")

                if crown_lb.dim() > 1:

                    if crown_lb.shape[0] == 1:
                        crown_lb_flat = crown_lb.squeeze(0).view(-1)
                        crown_ub_flat = crown_ub.squeeze(0).view(-1)
                    else:
                        crown_lb_flat = crown_lb.view(-1)
                        crown_ub_flat = crown_ub.view(-1)

                    print(f"CROWN flattened shape: {crown_lb_flat.shape}")

                    if hz_lb.shape == crown_lb_flat.shape:
                        crown_lb, crown_ub = crown_lb_flat, crown_ub_flat
                        print(f"✅ Shapes matched successfully: {hz_lb.shape}")
                    else:
                        print(f"❌ Still mismatched after flatten: HZ={hz_lb.shape}, CROWN={crown_lb_flat.shape}")
                        return
                else:
                    print(f"❌ CROWN bounds are already 1D, cannot flatten further")
                    return

            hz_width = (hz_ub - hz_lb).abs()
            crown_width = (crown_ub - crown_lb).abs()

            if torch.isnan(hz_width).any() or torch.isinf(hz_width).any():
                print("⚠️  Warning: HZ bounds contain NaN or Inf values")
            if torch.isnan(crown_width).any() or torch.isinf(crown_width).any():
                print("⚠️  Warning: CROWN bounds contain NaN or Inf values")

            hz_mean_width = hz_width.mean().item()
            crown_mean_width = crown_width.mean().item()
            hz_max_width = hz_width.max().item()
            crown_max_width = crown_width.max().item()
            hz_min_lb = hz_lb.min().item()
            hz_max_ub = hz_ub.max().item()
            crown_min_lb = crown_lb.min().item()
            crown_max_ub = crown_ub.max().item()

            width_improvement = ((hz_width - crown_width) / (hz_width + 1e-8)).mean().item() * 100

            hz_range = hz_max_ub - hz_min_lb
            crown_range = crown_max_ub - crown_min_lb

            tighter_crown = (crown_width < hz_width).sum().item()
            tighter_hz = (hz_width < crown_width).sum().item()
            equal_bounds = (torch.abs(hz_width - crown_width) < 1e-6).sum().item()
            total_neurons = hz_lb.numel()

            print(f"📊 Statistics Comparison:")
            print(f"HybridZonotope: mean width={hz_mean_width:.6f}, max width={hz_max_width:.6f}")
            print(f"CROWN:          mean width={crown_mean_width:.6f}, max width={crown_max_width:.6f}")
            print(f"CROWN improvement: {width_improvement:.2f}% (positive = CROWN better)")
            print(f"Bound ranges:       HZ=[{hz_min_lb:.6f}, {hz_max_ub:.6f}], CROWN=[{crown_min_lb:.6f}, {crown_max_ub:.6f}]")
            print(f"Neuron comparison:  CROWN tighter={tighter_crown}/{total_neurons} ({tighter_crown/total_neurons*100:.1f}%)")
            print(f"HZ tighter={tighter_hz}/{total_neurons} ({tighter_hz/total_neurons*100:.1f}%)")
            print(f"Equal={equal_bounds}/{total_neurons} ({equal_bounds/total_neurons*100:.1f}%)")


            if total_neurons <= 50:
                print(f"\n📋 Element-wise detailed comparison (first {min(total_neurons, 20)} neurons):")

                indices_to_show = list(range(min(10, total_neurons))) + list(range(max(total_neurons-10, 10), total_neurons))
                indices_to_show = sorted(list(set(indices_to_show)))

                if concrete_vals is not None:
                    print(f"{'Index':<4} {'Concrete':<12} {'HZ_LB':<12} {'HZ_UB':<12} {'CROWN_LB':<12} {'CROWN_UB':<12} {'HZ_Width':<10} {'CROWN_Width':<10} {'Improve':<8}")
                    print("-" * 100)
                else:
                    print(f"{'Index':<4} {'HZ_LB':<12} {'HZ_UB':<12} {'CROWN_LB':<12} {'CROWN_UB':<12} {'HZ_Width':<10} {'CROWN_Width':<10} {'Improve':<8}")
                    print("-" * 80)

                print(f"Showing neuron indices: {indices_to_show} (total {len(indices_to_show)})")
                try:
                    for i in indices_to_show:

                        try:
                            hz_l = hz_lb[i].item()
                            hz_u = hz_ub[i].item()
                            crown_l = crown_lb[i].item()
                            crown_u = crown_ub[i].item()
                            hz_w = hz_width[i].item()
                            crown_w = crown_width[i].item()
                            improvement = ((hz_w - crown_w) / (hz_w + 1e-8)) * 100

                            if concrete_vals is not None and i < concrete_vals.numel():
                                concrete_val = concrete_vals[i].item()
                                print(f"{i:<4} {concrete_val:<12.6f} {hz_l:<12.6f} {hz_u:<12.6f} {crown_l:<12.6f} {crown_u:<12.6f} {hz_w:<10.6f} {crown_w:<10.6f} {improvement:<8.2f}%")
                            else:
                                print(f"{i:<4} {'N/A':<12} {hz_l:<12.6f} {hz_u:<12.6f} {crown_l:<12.6f} {crown_u:<12.6f} {hz_w:<10.6f} {crown_w:<10.6f} {improvement:<8.2f}%")
                        except Exception as detail_e:
                            print(f"{i:<4} Error accessing data: {detail_e}")
                            continue
                except Exception as loop_e:
                    print(f"⚠️  Error in detailed comparison loop: {loop_e}")
                    print("Skipping detailed element-wise comparison...")

            elif total_neurons <= 200:
                print(f"\n📋 Sampling comparison (show first 10 + last 10 neurons, consistent with ERAN):")

                indices_to_show = list(range(min(10, total_neurons))) + list(range(max(total_neurons-10, 10), total_neurons))
                indices_to_show = sorted(list(set(indices_to_show)))

                if concrete_vals is not None:
                    print(f"{'Index':<4} {'Concrete':<12} {'HZ_LB':<12} {'HZ_UB':<12} {'CROWN_LB':<12} {'CROWN_UB':<12} {'HZ_Width':<10} {'CROWN_Width':<10} {'Improve':<8}")
                    print("-" * 100)
                else:
                    print(f"{'Index':<4} {'HZ_LB':<12} {'HZ_UB':<12} {'CROWN_LB':<12} {'CROWN_UB':<12} {'HZ_Width':<10} {'CROWN_Width':<10} {'Improve':<8}")
                    print("-" * 80)

                print(f"Showing neuron indices: {indices_to_show} (total {len(indices_to_show)})")
                try:
                    for i in indices_to_show:
                        try:
                            hz_l = hz_lb[i].item()
                            hz_u = hz_ub[i].item()
                            crown_l = crown_lb[i].item()
                            crown_u = crown_ub[i].item()
                            hz_w = hz_width[i].item()
                            crown_w = crown_width[i].item()
                            improvement = ((hz_w - crown_w) / (hz_w + 1e-8)) * 100

                            if concrete_vals is not None and i < concrete_vals.numel():
                                concrete_val = concrete_vals[i].item()
                                print(f"{i:<4} {concrete_val:<12.6f} {hz_l:<12.6f} {hz_u:<12.6f} {crown_l:<12.6f} {crown_u:<12.6f} {hz_w:<10.6f} {crown_w:<10.6f} {improvement:<8.2f}%")
                            else:
                                print(f"{i:<4} {'N/A':<12} {hz_l:<12.6f} {hz_u:<12.6f} {crown_l:<12.6f} {crown_u:<12.6f} {hz_w:<10.6f} {crown_w:<10.6f} {improvement:<8.2f}%")
                        except Exception as detail_e:
                            print(f"{i:<4} Error accessing data: {detail_e}")
                            continue
                except Exception as loop_e:
                    print(f"⚠️  Error in detailed comparison loop: {loop_e}")
                    print("Skipping detailed element-wise comparison...")
                if concrete_vals is not None:
                    print(f"{'Index':<4} {'Concrete':<12} {'HZ LB':<12} {'HZ UB':<12} {'CROWN LB':<12} {'CROWN UB':<12} {'HZ Width':<10} {'CROWN Width':<10} {'Improve':<8}")
                    print("-" * 100)
                else:
                    print(f"{'Index':<4} {'HZ LB':<12} {'HZ UB':<12} {'CROWN LB':<12} {'CROWN UB':<12} {'HZ Width':<10} {'CROWN Width':<10} {'Improve':<8}")
                    print("-" * 80)

                try:
                    step = max(1, total_neurons // 10)
                    for i in range(0, total_neurons, step):
                        try:
                            hz_l = hz_lb[i].item()
                            hz_u = hz_ub[i].item()
                            crown_l = crown_lb[i].item()
                            crown_u = crown_ub[i].item()
                            hz_w = hz_width[i].item()
                            crown_w = crown_width[i].item()
                            improvement = ((hz_w - crown_w) / (hz_w + 1e-8)) * 100

                            if concrete_vals is not None and i < concrete_vals.numel():
                                concrete_val = concrete_vals[i].item()
                                print(f"{i:<4} {concrete_val:<12.6f} {hz_l:<12.6f} {hz_u:<12.6f} {crown_l:<12.6f} {crown_u:<12.6f} {hz_w:<10.6f} {crown_w:<10.6f} {improvement:<8.2f}%")
                            else:
                                print(f"{i:<4} {'N/A':<12} {hz_l:<12.6f} {hz_u:<12.6f} {crown_l:<12.6f} {crown_u:<12.6f} {hz_w:<10.6f} {crown_w:<10.6f} {improvement:<8.2f}%")
                        except Exception as detail_e:
                            print(f"{i:<4} Error accessing data: {detail_e}")
                            continue
                except Exception as sampling_e:
                    print(f"⚠️  Error in sampling comparison: {sampling_e}")

            else:

                print(f"\n📋 Large layer neuron comparison (showing first 10 + last 10 neurons, consistent with ERAN):")

                indices_to_show = list(range(min(10, total_neurons))) + list(range(max(total_neurons-10, 10), total_neurons))
                indices_to_show = sorted(list(set(indices_to_show)))

                if concrete_vals is not None:
                    print(f"{'Index':<4} {'Concrete':<12} {'HZ LB':<12} {'HZ UB':<12} {'CROWN LB':<12} {'CROWN UB':<12} {'HZ Width':<10} {'CROWN Width':<10} {'Improve':<8}")
                    print("-" * 100)
                else:
                    print(f"{'Index':<4} {'HZ LB':<12} {'HZ UB':<12} {'CROWN LB':<12} {'CROWN UB':<12} {'HZ Width':<10} {'CROWN Width':<10} {'Improve':<8}")
                    print("-" * 80)

                print(f"Showing neuron indices: {indices_to_show} (total {len(indices_to_show)})")
                try:
                    for i in indices_to_show:
                        try:
                            hz_l = hz_lb[i].item()
                            hz_u = hz_ub[i].item()
                            crown_l = crown_lb[i].item()
                            crown_u = crown_ub[i].item()
                            hz_w = hz_width[i].item()
                            crown_w = crown_width[i].item()
                            improvement = ((hz_w - crown_w) / (hz_w + 1e-8)) * 100

                            if concrete_vals is not None and i < concrete_vals.numel():
                                concrete_val = concrete_vals[i].item()
                                print(f"{i:<4} {concrete_val:<12.6f} {hz_l:<12.6f} {hz_u:<12.6f} {crown_l:<12.6f} {crown_u:<12.6f} {hz_w:<10.6f} {crown_w:<10.6f} {improvement:<8.2f}%")
                            else:
                                print(f"{i:<4} {'N/A':<12} {hz_l:<12.6f} {hz_u:<12.6f} {crown_l:<12.6f} {crown_u:<12.6f} {hz_w:<10.6f} {crown_w:<10.6f} {improvement:<8.2f}%")
                        except Exception as detail_e:
                            print(f"{i:<4} Error accessing data: {detail_e}")
                            continue
                except Exception as large_layer_e:
                    print(f"⚠️  Error in large layer comparison: {large_layer_e}")
                    print("Skipping large layer detailed comparison...")

            comparison = {
                'layer_name': layer_name,
                'hz_mean_width': hz_mean_width,
                'crown_mean_width': crown_mean_width,
                'hz_max_width': hz_max_width,
                'crown_max_width': crown_max_width,
                'width_improvement': width_improvement,
                'hz_range': hz_range,
                'crown_range': crown_range,
                'tighter_crown_ratio': tighter_crown / total_neurons,
                'tighter_hz_ratio': tighter_hz / total_neurons,
                'shape': hz_lb.shape,
                'total_neurons': total_neurons,
                'better_method': 'CROWN' if crown_mean_width < hz_mean_width else 'HybridZonotope'
            }

            self.layer_precision_comparison[layer_name] = comparison

            winner = "🥇 CROWN" if crown_mean_width < hz_mean_width else "🥇 HybridZonotope"
            print(f"\n🏆 Conclusion: {winner} (smaller average width)")
            print("="*80)

        except Exception as e:
            print(f"❌ [Precision Comparison] Failed for {layer_name}: {e}")
            import traceback
            traceback.print_exc()

    def _print_final_precision_summary(self):
        if not self.layer_precision_comparison:
            print("[Final Summary] No precision comparison data available")
            return

        print("\n" + "="*80)
        print("🏆 FINAL PRECISION COMPARISON SUMMARY")
        print("="*80)

        hz_wins = 0
        crown_wins = 0
        total_layers = len(self.layer_precision_comparison)

        for layer_name, comp in self.layer_precision_comparison.items():
            if comp['better_method'] == 'HybridZonotope':
                hz_wins += 1
            else:
                crown_wins += 1

            improvement = comp['width_improvement']
            winner_mark = "🥇" if comp['better_method'] == 'CROWN' else "🥈"

            print(f"{winner_mark} {layer_name:20s}: {improvement:+8.2%} improvement (CROWN vs HZ)")

        print("-" * 80)
        print(f"📊 Overall Results:")
        print(f"Total Layers:        {total_layers}")
        print(f"CROWN Wins:          {crown_wins} ({crown_wins/total_layers:.1%})")
        print(f"HybridZonotope Wins: {hz_wins} ({hz_wins/total_layers:.1%})")

        avg_improvement = np.mean([comp['width_improvement'] for comp in self.layer_precision_comparison.values()])
        print(f"Average Improvement: {avg_improvement:+.2%} (CROWN vs HZ)")

        if avg_improvement > 0:
            print("🏆 Overall Winner: CROWN (auto_LiRPA)")
        else:
            print("🏆 Overall Winner: HybridZonotope")

        print("="*80)

    def _abstract_constraint_solving_core(self, model, input_hz, method, sample_idx=0):

        model = model.pytorch_model
        hz = input_hz

        verification_core_start_time = time.time()
        layer_count = 0
        total_layer_time = 0.0

        all_layers = list(model.children())
        linear_layers = [i for i, layer in enumerate(all_layers) if isinstance(layer, nn.Linear)]
        last_linear_index = linear_layers[-1] if linear_layers else -1

        print(f"\n🕒 Starting layer-by-layer verification - method: {method}")
        print(f"Network structure analysis: total layers={len(all_layers)}, Linear layers={len(linear_layers)}")
        if self.enable_generator_merging and last_linear_index >= 0:
            print(f"Generator merging will be automatically enabled at layer {last_linear_index} (last Linear layer)")
        print("="*60)

        if self.enable_soundness_check:
            print(f"[Soundness] Computing concrete network inference...")

            if hasattr(hz, 'center_grid'):

                input_center = hz.center_grid.squeeze(-1)
            else:

                input_center = hz.center.squeeze(-1) if hz.center.dim() > 1 else hz.center

            self._compute_concrete_inference(input_center, model)

        if self.enable_layer_comparison or self.bab_config['enabled']:
            input_layer_name = f"input_layer"
            print(f"[Layer Bounds] Computing input layer bounds (layer_comparison={self.enable_layer_comparison}, bab_enabled={self.bab_config['enabled']})")

            original_input_center = self.input_center

            if original_input_center.dim() > 1:
                input_center_flat = original_input_center.view(-1)
            else:
                input_center_flat = original_input_center

            self.concrete_layer_values[input_layer_name] = {
                'values': input_center_flat.detach().clone(),
                'shape': original_input_center.shape,
                'layer_type': 'input',
                'layer_index': -1
            }
            print(f"✅ [Concrete] {input_layer_name}: {original_input_center.shape} -> flattened: {input_center_flat.shape}")
            print(f"Range: [{input_center_flat.min():.6f}, {input_center_flat.max():.6f}]")

            hz_input_bounds = self._compute_hz_layer_bounds(hz, input_layer_name, "input")

            autolirpa_input_bounds = self._get_autolirpa_layer_bounds(input_layer_name, layer_index=0)

            if hz_input_bounds[0] is not None and autolirpa_input_bounds[0] is not None:
                self._compare_layer_precision(input_layer_name, hz_input_bounds, autolirpa_input_bounds)

        for layer in model.children():
            layer_name = f"layer_{layer_count}_{type(layer).__name__}"
            layer_type = type(layer).__name__.lower()
            print(f"📋 Processing layer {layer_count}: {type(layer).__name__}")

            autolirpa_index = layer_count + 1

            if isinstance(layer, nn.Linear):
                layer_start_time = time.time()

                W = layer.weight
                b = layer.bias
                hz.set_method(method)

                is_last_linear = (layer_count == last_linear_index)
                enable_merging = self.enable_generator_merging and is_last_linear

                if enable_merging:
                    print(f"Last Linear layer (layer {layer_count}): enabling generator merging optimization")
                    hz = hz.linear(W, b, enable_generator_merging=True, cosine_threshold=self.cosine_threshold)
                else:
                    if self.enable_generator_merging and not is_last_linear:
                        print(f"⏭️  Intermediate Linear layer (layer {layer_count}): skipping generator merging optimization")
                    hz = hz.linear(W, b, enable_generator_merging=False)

                layer_end_time = time.time()
                layer_duration = layer_end_time - layer_start_time
                total_layer_time += layer_duration
                print(f"⏱️  Linear layer processing time: {layer_duration:.4f} seconds")

            elif isinstance(layer, nn.Conv2d):
                layer_start_time = time.time()

                hz.set_method(method)
                print(f"Processing Conv2d layer: verifier method {method}, hz.method {hz.method}")
                hz = hz.conv(layer.weight, layer.bias, stride=layer.stride, padding=layer.padding,
                             dilation=layer.dilation, groups=layer.groups)

                layer_end_time = time.time()
                layer_duration = layer_end_time - layer_start_time
                total_layer_time += layer_duration
                print(f"⏱️  Conv2d layer processing time: {layer_duration:.4f} seconds")

            elif isinstance(layer, nn.ReLU):
                layer_start_time = time.time()

                hz.set_method(method)

                relu_layer_name = layer_name
                debug_applied_constraints = []
                relu_constraints_to_apply = []

                if hasattr(self, 'current_relu_constraints') and self.current_relu_constraints:
                    for constraint in self.current_relu_constraints:
                        constraint_layer = constraint['layer']

                        if (constraint_layer == relu_layer_name or
                            constraint_layer == layer_name or
                            (constraint_layer.startswith('layer_') and constraint_layer.endswith('_ReLU') and layer_name == constraint_layer)):
                            relu_constraints_to_apply.append(constraint)
                            debug_applied_constraints.append(f"ReLU[{constraint['neuron_idx']}]={constraint['constraint_type']}")

                if debug_applied_constraints:
                    print(f"[{layer_name}] applyingReLUconstraint: {debug_applied_constraints}")

                hz = hz.relu(
                    auto_lirpa_info=None,
                    relu_constraints=relu_constraints_to_apply
                )

                layer_end_time = time.time()
                layer_duration = layer_end_time - layer_start_time
                total_layer_time += layer_duration
                print(f"⏱️  ReLU layer processing time: {layer_duration:.4f} seconds")

            elif isinstance(layer, nn.Sigmoid):
                layer_start_time = time.time()

                hz.set_method(method)
                hz = hz.sigmoid_or_tanh('sigmoid')

                layer_end_time = time.time()
                layer_duration = layer_end_time - layer_start_time
                total_layer_time += layer_duration
                print(f"⏱️  Sigmoid layer processing time: {layer_duration:.4f} seconds")

            elif isinstance(layer, nn.MaxPool2d):
                layer_start_time = time.time()

                hz.set_method(method)

                for name, bounds in self.autolirpa_layer_bounds.items():
                    print(f"Layer {name} bounds: lower={bounds['lb'].shape}, upper={bounds['ub'].shape}")

                hz = hz.maxpool(
                    kernel_size=layer.kernel_size,
                    stride=layer.stride,
                    padding=layer.padding,
                    auto_lirpa_info=None

                )

                layer_end_time = time.time()
                layer_duration = layer_end_time - layer_start_time
                total_layer_time += layer_duration
                print(f"⏱️  MaxPool2d layer processing time: {layer_duration:.4f} seconds")

            elif isinstance(layer, nn.Flatten) or isinstance(layer, OnnxFlatten):
                layer_start_time = time.time()

                hz = HybridZonotopeOps.FlattenHybridZonotopeGridIntersection(hz)

                layer_end_time = time.time()
                layer_duration = layer_end_time - layer_start_time
                total_layer_time += layer_duration
                print(f"⏱️  Flatten layer processing time: {layer_duration:.4f} seconds")

            elif isinstance(layer, OperatorWrapper):
                layer_start_time = time.time()

                layer_type_str = type(layer).__name__.lower()
                op_type = getattr(layer, 'op_type', None)

                if layer_type_str == 'tanh' or op_type == 'Tanh':

                    print(f"📋 Processing layer {layer_count}: tanh (via OperatorWrapper)")
                    hz.set_method(method)
                    hz = hz.sigmoid_or_tanh('tanh')

                    layer_end_time = time.time()
                    layer_duration = layer_end_time - layer_start_time
                    total_layer_time += layer_duration
                    print(f"⏱️  Tanh layer processing time: {layer_duration:.4f} seconds")

                elif hasattr(layer, 'op_type') and layer.op_type in ["Add", "Sub", "Mul", "Div"]:

                        other = getattr(layer, 'other', None)
                        if other is None:
                            raise ValueError(f"OperatorWrapper {layer.op_type} missing 'other' attribute for scalar operation.")

                        hz.set_method(method)

                        if layer.op_type == "Add":
                            hz = hz.add(other)
                            print(f"Applied Add scalar operation with value: {other}")
                        elif layer.op_type == "Sub":
                            hz = hz.subtract(other)
                            print(f"Applied Sub scalar operation with value: {other}")
                        elif layer.op_type == "Mul":
                            hz = hz.multiply(other)
                            print(f"Applied Mul scalar operation with value: {other}")
                        elif layer.op_type == "Div":
                            if other == 0:
                                raise ValueError("Division by zero encountered in OperatorWrapper Div layer.")
                            hz = hz.divide(other)
                            print(f"Applied Div scalar operation with value: {other}")

                        layer_end_time = time.time()
                        layer_duration = layer_end_time - layer_start_time
                        total_layer_time += layer_duration
                        print(f"⏱️  {layer.op_type} operation processing time: {layer_duration:.4f} seconds")
                else:

                    layer_type_str = type(layer).__name__
                    op_type = getattr(layer, 'op_type', 'not_found')
                    raise NotImplementedError(
                        f"OperatorWrapper layer not supported in HybridZonotopeVerifier.\n"
                        f"  Layer type: {layer_type_str}\n"
                        f"  Layer repr: {repr(layer)}\n"
                        f"  op_type: {op_type}\n"
                        f"  Supported op_types: ['Add', 'Sub', 'Mul', 'Div']\n"
                        f"  Supported layer names: ['tanh']"
                    )

            else:
                raise NotImplementedError(f"Layer {layer} not supported in HybridZonotopeVerifier.")

            print("-" * 60)
            layer_count += 1

        print(f"✅ All {layer_count} layers processed")
        print(f"📊 Total layer-by-layer processing time: {total_layer_time:.4f} seconds")
        print("="*60)

        if isinstance(hz, HybridZonotopeGrid):
            hz_elem = HybridZonotopeOps.FlattenHybridZonotopeGridIntersection(hz)
        else:
            hz_elem = hz

        print("Output: ", hz_elem.n)
        verification_core_end_time = time.time()
        verification_core_time = verification_core_end_time - verification_core_start_time

        print("\nVerification core time statistics:")
        print(f"📊 Total layer-by-layer processing time: {total_layer_time:.4f} seconds")
        print(f"⏱️  Other processing time: {verification_core_time - total_layer_time:.4f} seconds")
        print(f"🕒 Total verification core time: {verification_core_time:.4f} seconds")
        print("="*60)

        if self.enable_layer_comparison:
            self._print_final_precision_summary()

        if self.enable_soundness_check:
            self._print_final_soundness_summary()

        print(f"\nComputing output layer dimension-wise bounds (total {hz_elem.n} output neurons)")
        output_lbs, output_ubs = self._concretize_hz(hz_elem, method=method)

        print(f"Returning output layer HybridZonotope and bounds for two-stage verification")
        return hz_elem, output_lbs, output_ubs

    def _concretize_hz(self, hz_elem, method='hybridz', time_limit=500):

        print("="*60)
        spec_verification_start_time = time.time()

        print(f"Using GetLayerWiseBounds for {hz_elem.n} outputs, method={method}")

        lbs_tensor, ubs_tensor = HybridZonotopeOps.GetLayerWiseBounds(
            hz_elem.center, hz_elem.G_c, hz_elem.G_b,
            hz_elem.A_c, hz_elem.A_b, hz_elem.b,
            method=method, time_limit=time_limit, ci_mode=self.ci_mode
        )

        if isinstance(lbs_tensor, (int, float)):

            lbs_tensor = torch.tensor([lbs_tensor], dtype=hz_elem.dtype, device=hz_elem.device)
            ubs_tensor = torch.tensor([ubs_tensor], dtype=hz_elem.dtype, device=hz_elem.device)
            lbs = [lbs_tensor.item()]
            ubs = [ubs_tensor.item()]
        else:

            if not isinstance(lbs_tensor, torch.Tensor):
                lbs_tensor = torch.tensor(lbs_tensor, dtype=hz_elem.dtype, device=hz_elem.device)
                ubs_tensor = torch.tensor(ubs_tensor, dtype=hz_elem.dtype, device=hz_elem.device)

            lbs = lbs_tensor.cpu().numpy().flatten().tolist()
            ubs = ubs_tensor.cpu().numpy().flatten().tolist()

        for i in range(len(lbs)):
            print(f"✅ Output {i}: [{lbs[i]:.6f}, {ubs[i]:.6f}]")

        if not isinstance(lbs_tensor, torch.Tensor):
            lbs_tensor = torch.tensor(lbs, dtype=hz_elem.dtype, device=hz_elem.device)
            ubs_tensor = torch.tensor(ubs, dtype=hz_elem.dtype, device=hz_elem.device)

        spec_verification_end_time = time.time()
        print(f"🕒 Total Output Spec verification time: {spec_verification_end_time - spec_verification_start_time:.2f} seconds")
        print("="*60)

        return lbs_tensor, ubs_tensor

    def _single_result_verdict_hz(self, output_hz,
                                  output_lbs: torch.Tensor,
                                  output_ubs: torch.Tensor,
                                  output_constraints: Optional[List[List[float]]],
                                  true_label: Optional[int]) -> VerifyResult:

        if output_constraints is not None:
            print(f"HZ verification: processing linear constraints ({len(output_constraints)} constraints)")
            for row in output_constraints:
                a = torch.tensor(row[:-1], device=output_lbs.device)
                b = row[-1]
                worst = torch.sum(torch.where(a>=0, a*output_lbs, a*output_ubs)) + b
                if worst < 0:
                    return VerifyResult.UNSAT
            return VerifyResult.SAT

        if true_label is not None:
            print(f"HZ verification: using two-stage verification strategy (conservative judgment first, then precise difference)")
            return self._classify_with_two_stage_strategy_hz(output_hz, output_lbs, output_ubs, true_label)

        return VerifyResult.UNKNOWN

    def _classify_with_two_stage_strategy_hz(self, output_hz, output_lbs: torch.Tensor, output_ubs: torch.Tensor, true_label: int) -> VerifyResult:

        if len(output_lbs) <= true_label:
            print(f"❌ true_label {true_label} exceeds output dimension {len(output_lbs)}")
            return VerifyResult.UNKNOWN

        num_outputs = len(output_lbs)
        print(f"🔍 Starting two-stage HZ verification: true_label={true_label}, num_outputs={num_outputs}")

        print(f"Stage 1: Conservative bound comparison")
        true_label_lb = output_lbs[true_label].item()

        conservative_safe = True
        for j in range(num_outputs):
            if j == true_label:
                continue
            other_ub = output_ubs[j].item()
            if true_label_lb <= other_ub:
                print(f"⚠️  Conservative judgment failed: output[{true_label}]_lb={true_label_lb:.6f} <= output[{j}]_ub={other_ub:.6f}")
                conservative_safe = False
                break
            else:
                print(f"✅ output[{true_label}]_lb={true_label_lb:.6f} > output[{j}]_ub={other_ub:.6f}")

        if conservative_safe:
            print(f"✅ Stage 1 success: true_label lower bound greater than all other neuron upper bounds, directly judged robust")
            return VerifyResult.SAT

        print(f"🔍 Stage 2: Precise difference verification (ERAN style)")
        return self._classify_with_difference_bounds_hz(output_hz, true_label)

    def _classify_with_difference_bounds_hz(self, output_hz, true_label: int) -> VerifyResult:

        if output_hz.G_c.shape[0] <= true_label:
            print(f"❌ true_label {true_label} exceeds output dimension {output_hz.G_c.shape[0]}")
            return VerifyResult.UNKNOWN

        num_outputs = output_hz.G_c.shape[0]
        print(f"🔍 Starting HZ difference verification: true_label={true_label}, num_outputs={num_outputs}")

        for j in range(num_outputs):
            if j == true_label:
                continue

            print(f"Checking difference: output[{true_label}] - output[{j}]")

            try:

                diff_hz = HybridZonotopeOps.ConstructNeuronDifferenceHZ(output_hz, true_label, j)

                diff_lb, diff_ub = self._concretize_hz(diff_hz, method=self.method)

                if isinstance(diff_lb, torch.Tensor):
                    diff_lb_val = diff_lb.item() if diff_lb.numel() == 1 else diff_lb[0].item()
                    diff_ub_val = diff_ub.item() if diff_ub.numel() == 1 else diff_ub[0].item()
                else:
                    diff_lb_val = float(diff_lb)
                    diff_ub_val = float(diff_ub)

                print(f"Difference range: [{diff_lb_val:.6f}, {diff_ub_val:.6f}]")

                if diff_lb_val <= 0:
                    print(f"❌ Difference output[{true_label}] - output[{j}] lower bound <= 0: {diff_lb_val:.6f}")
                    return VerifyResult.UNSAT
                else:
                    print(f"✅ Difference output[{true_label}] - output[{j}] lower bound > 0: {diff_lb_val:.6f}")

            except Exception as e:
                print(f"⚠️  Error constructing difference zonotope: {e}")
                print(f"Cannot complete HZ difference verification, returning UNKNOWN")
                return VerifyResult.UNKNOWN

        print(f"✅ All HZ difference verifications passed")
        return VerifyResult.SAT

    def _abstract_constraint_solving(self, input_lb: torch.Tensor, input_ub: torch.Tensor, sample_idx: int) -> VerifyResult:

        print(f"Creating HybridZonotope abstract domain")
        self.input_hz = HybridZonotopeGrid(
            input_lb=input_lb,
            input_ub=input_ub,
            method=self.method,
            time_limit=500,
            relaxation_ratio=self.relaxation_ratio,
            device=self.device,
            ci_mode=self.ci_mode
        )

        print(f"Using method: {self.method}")
        verification_result = self._abstract_constraint_solving_core(model=self.model, input_hz=self.input_hz, method=self.method, sample_idx=sample_idx)

        if verification_result is not None and len(verification_result) == 3:
            output_hz, output_lbs, output_ubs = verification_result
        else:

            print(f"❌ Verification core returned abnormal value")
            return VerifyResult.UNKNOWN

        self.enforce_neuron_activation_constraints()

        if output_hz is not None:
            verdict = self._single_result_verdict_hz(
                output_hz,
                output_lbs,
                output_ubs,
                self.spec.output_spec.output_constraints if self.spec.output_spec.output_constraints is not None else None,
                self.spec.output_spec.labels[sample_idx].item() if self.spec.output_spec.labels is not None else None
            )
        else:

            verdict = VerifyResult.UNKNOWN

        print(f"📊 Verification result: {verdict.name}")
        return verdict

    def verify(self, proof=None, public_inputs=None):

        ACTStats.print_memory_usage("HybridZonotopeVerifier Start")
        print("Starting complete verification process - conforming to theoretical architecture design")

        if self.input_center is None:
            print("❌ Error: input_center is None. Cannot proceed with verification.")
            return {"verified": False, "error": "input_center is None"}

        num_samples = self.input_center.shape[0] if self.input_center.ndim > 1 else 1
        print(f"Total samples: {num_samples}")
        print(f"Input center shape: {self.input_center.shape}")
        print(f"Input boundary shapes: {self.spec.input_spec.input_lb.shape}, {self.spec.input_spec.input_ub.shape}")

        results = []
        for idx in range(num_samples):
            ACTStats.print_memory_usage(f"Sample {idx+1}")
            print(f"\n🔍 Processing sample {idx+1}/{num_samples}")
            print("="*80)

            center_input, true_label = self.get_sample_label_pair(idx)
            if not perform_model_inference(
                model=self.spec.model.pytorch_model,
                sample_tensor=center_input,
                ground_truth_label=true_label,
                input_adaptor=self.input_adaptor,
                prediction_stats=self.clean_prediction_stats,
                sample_index=idx,
                verbose=self.verbose
            ):

                print(f"⏭️  Skipping verification for sample {idx+1}")
                results.append(VerifyResult.CLEAN_FAILURE)
                continue

            self.clean_prediction_stats['verification_attempted'] += 1


            if self.spec.input_spec.input_lb.shape[0] == 1:
                lb_i = self.spec.input_spec.input_lb[0] if self.spec.input_spec.input_lb.ndim > 1 else self.spec.input_spec.input_lb
                ub_i = self.spec.input_spec.input_ub[0] if self.spec.input_spec.input_ub.ndim > 1 else self.spec.input_spec.input_ub
            elif self.spec.input_spec.input_lb.shape[0] > idx:
                lb_i = self.spec.input_spec.input_lb[idx]
                ub_i = self.spec.input_spec.input_ub[idx]
            else:
                lb_i = self.spec.input_spec.input_lb[0] if self.spec.input_spec.input_lb.ndim > 1 else self.spec.input_spec.input_lb
                ub_i = self.spec.input_spec.input_ub[0] if self.spec.input_spec.input_ub.ndim > 1 else self.spec.input_spec.input_ub

            if self.use_auto_lirpa:
                input_example = (lb_i + ub_i) / 2.0
                if self._setup_auto_lirpa(input_example.unsqueeze(0)):
                    eps = getattr(self.spec.input_spec, 'epsilon', None)

                    if eps is not None and hasattr(self.dataset, 'std') and self.dataset.std is not None:

                        std_val = self.dataset.std
                        if isinstance(std_val, list):
                            if len(std_val) == 1:

                                eps_normalized = eps / std_val[0]
                                print(f"[Auto_LiRPA] Original eps: {eps}, Normalized eps: {eps_normalized} (divided by std[0]: {std_val[0]})")
                            else:

                                eps_normalized = eps / std_val[0]
                                print(f"[Auto_LiRPA] Original eps: {eps}, Normalized eps: {eps_normalized} (divided by std[0]: {std_val[0]}, full std: {std_val})")
                        elif isinstance(std_val, (int, float)):

                            eps_normalized = eps / std_val
                            print(f"[Auto_LiRPA] Original eps: {eps}, Normalized eps: {eps_normalized} (divided by std: {std_val})")
                        else:

                            eps_normalized = eps
                            print(f"[Auto_LiRPA] Using original eps: {eps} (std format not recognized: {type(std_val)})")
                        eps = eps_normalized
                    self._compute_autolirpa_bounds((lb_i, ub_i), eps=eps)

            print("Step 1: HybridZonotope abstract constraint solving")
            initial_verdict = self._abstract_constraint_solving(lb_i, ub_i, idx)

            if initial_verdict == VerifyResult.SAT:
                self.clean_prediction_stats['verification_sat'] += 1

                print(f"✅ HybridZonotope verification successful - sample {idx+1} is safe")
                results.append(initial_verdict)
                continue
            elif initial_verdict == VerifyResult.UNSAT:
                self.clean_prediction_stats['verification_unsat'] += 1
            else:
                self.clean_prediction_stats['verification_unknown'] += 1

            if initial_verdict == VerifyResult.UNSAT:
                print(f"❌ HybridZonotope found potential violation - sample {idx+1}")
            else:
                print(f"❓ HybridZonotope result uncertain - sample {idx+1}")

            print("Automatically activating Specification Refinement BaB process")
            print("="*60)

            if self.bab_config['enabled']:

                refinement_verdict = self._spec_refinement(lb_i, ub_i, idx)
                results.append(refinement_verdict)
            else:
                print("⚠️  BaB not enabled, returning initial verdict")
                results.append(initial_verdict)

        return ACTStats.print_final_verification_summary(results)
