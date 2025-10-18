#===- act/main.py - ACT Entry Point ------------------------------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Entry point for ACT native torch2act verification pipeline.
#   Supports only the --verifier act option for spec-free, input-free
#   PyTorch model verification using the torch2act converter.
#
#===---------------------------------------------------------------------===#

import sys
import time
import os

# Import command line parser
from act.util.options import get_parser


def run_act_native_verifier(args_dict):
    """
    Run the ACT native torch2act verification pipeline.
    This implements the spec-free, input-free verification approach.
    """    
    # Import torch2act components
    from act.front_end.model_synthesis import model_synthesis
    from act.front_end.model_inference import model_inference
    from act.pipeline.torch2act import TorchToACT, interpret_validation
    from act.back_end.solver.solver_torch import TorchLPSolver
    from act.back_end.solver.solver_gurobi import GurobiSolver
    from act.back_end.verifier import verify_once, verify_bab
    
    # Get device and dtype from device manager (automatically reads command line)
    from act.util.device_manager import get_default_device, get_default_dtype
    device = get_default_device()
    dtype = get_default_dtype()
    print(f"🔧 Using device: {device}, dtype: {dtype}")
    
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
        return False
    
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
        return False
    
    # Step 4: Initialize solvers
    print("\n🔧 Step 4: Initializing solvers...")
    gurobi_solver = None
    torch_solver = None
    
    # Try TorchLP solver first (more reliable for large models)
    try:
        torch_solver = TorchLPSolver()
        # Use device from device manager
        device_str = str(device).split(':')[0]  # Convert torch.device to string (e.g., 'cuda:0' -> 'cuda')
        torch_solver.begin("act_verification", device=device_str)
        print(f"  ✅ TorchLP solver available (device: {device})")
    except Exception as e:
        print(f"  ❌ TorchLP initialization failed: {e}")
    
    # Try Gurobi solver with size limitation awareness
    try:
        gurobi_solver = GurobiSolver()
        gurobi_solver.begin("act_verification")
        print("  ✅ Gurobi solver available")
        print("  ⚠️  Note: Gurobi has license size limitations - will use TorchLP for large models")
    except Exception as e:
        print(f"  ⚠️  Gurobi initialization failed: {e}")
    
    # Prioritize TorchLP solver for large models
    solvers_to_test = []
    if torch_solver:
        solvers_to_test.append(("TorchLP", torch_solver))
    if gurobi_solver:
        solvers_to_test.append(("Gurobi", gurobi_solver))
    
    if not solvers_to_test:
        print("  ❌ No solvers available!")
        return False
    
    # Step 5: Run verification on just the first model for debugging
    print(f"\n🔍 Step 5: Running verification on first model for debugging...")
    
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
            error_msg = str(e)
            
            # Handle specific Gurobi license limitation
            if "Model too large for size-limited license" in error_msg:
                print(f"    ⚠️  {solver_name} license limitation: Model too large for size-limited license")
                print(f"    💡 Consider using TorchLP solver or upgrading Gurobi license")
                model_results[solver_name] = {'error': 'LICENSE_SIZE_LIMIT'}
            else:
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
                error = solver_results['error']
                if error == 'LICENSE_SIZE_LIMIT':
                    print(f"    {solver_name}: LICENSE LIMITATION - Model too large for size-limited license")
                else:
                    print(f"    {solver_name}: ERROR - {error[:100]}...")
            else:
                single_status = solver_results.get('single_shot', 'N/A')
                print(f"    {solver_name}: Single={single_status}")
    
    print("\n🔍 Debug verification completed!")
    return True


def main():
    """
    Main entry point for ACT native torch2act verification.
    Defaults to --verifier act if no verifier is specified.
    """
    parser = get_parser()
    parsed_args = parser.parse_args(sys.argv[1:])
    args_dict = vars(parsed_args)
    
    verifier_type = args_dict.get("verifier")
    
    # Default to 'act' verifier if none specified
    if verifier_type is None:
        verifier_type = 'act'
        print("🔧 No verifier specified, defaulting to ACT native verifier")
    
    if verifier_type == 'act':
        print("🎯 Starting ACT Native Verifier (torch2act pipeline)")
        success = run_act_native_verifier(args_dict)
        if not success:
            print("❌ ACT Native verification failed")
            sys.exit(1)
        else:
            print("✅ ACT Native verification completed successfully")
    
    else:
        print(f"❌ This main.py only supports --verifier act")
        print(f"❌ Unsupported verifier: {verifier_type}")
        print("📋 Supported verifier: act")
        print("\nUsage:")
        print("  python act/main.py                    # Uses ACT verifier by default")
        print("  python act/main.py --verifier act     # Explicitly specify ACT verifier")
        print("\nThis will run the spec-free, input-free torch2act verification pipeline.")
        sys.exit(1)


if __name__ == "__main__":
    main()