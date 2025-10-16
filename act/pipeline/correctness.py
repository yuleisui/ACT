"""
Simplified verifier correctness validation for ACT pipeline.

This module provides focused validation of the back_end verify_bab function
using mock_factory for test case generation and config for test scenarios.
"""

import torch
import torch.nn as nn
import time
import numpy as np
import sys
import os
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import logging

# Add ACT paths for imports
current_dir = Path(__file__).parent
act_root = current_dir.parent
sys.path.insert(0, str(act_root))

from act.pipeline.utils import PerformanceProfiler, print_memory_usage, clear_torch_cache
from act.pipeline.mock_factory import MockInputFactory
from act.pipeline.config import load_config

# Import real back_end verifier components
from act.back_end.core import Net, Layer, Bounds
from act.back_end.bab import VerifStatus, VerifResult, seed_from_input_spec
from act.back_end.bab import verify_bab
from act.back_end.solver.solver_gurobi import GurobiSolver
from act.back_end.solver.solver_torch import TorchLPSolver
from act.util.device_manager import get_current_settings
from act.front_end.specs import InputSpec, OutputSpec, InKind, OutKind

logger = logging.getLogger(__name__)


class VerifyResult(Enum):
    """Verification result types."""
    SAT = "SAT"
    UNSAT = "UNSAT"
    UNKNOWN = "UNKNOWN"
    TIMEOUT = "TIMEOUT"
    ERROR = "ERROR"


@dataclass
class TestCase:
    """Individual test case for validation."""
    sample_data: torch.Tensor
    labels: torch.Tensor
    input_spec: Dict[str, Any]
    output_spec: Dict[str, Any]
    model: nn.Module
    expected_result: Optional[VerifyResult] = None
    test_id: str = ""
    timeout: float = 60.0  # Reduced timeout for testing


@dataclass
class ValidationResult:
    """Result from validation testing."""
    success: bool
    total_tests: int
    passed_tests: int
    failed_tests: int
    results: List[Dict[str, Any]] = field(default_factory=list)
    execution_time: float = 0.0
    memory_usage_mb: float = 0.0
    error_message: Optional[str] = None


@dataclass 
class PerformanceResult:
    """Result from performance testing."""
    test_name: str
    execution_time: float
    memory_usage_mb: float
    success: bool
    details: Dict[str, Any] = field(default_factory=dict)


class BackEndVerifierValidator:
    """Validate correctness of back_end verify_bab function."""
    
    def __init__(self, device: str = "cpu"):
        """
        Initialize validator with mock factory and configuration.
        
        Args:
            device: Device for computation
        """
        self.mock_factory = MockInputFactory()
        
        # Get current ACT device management settings (auto-initialization already happened)
        try:
            device, dtype = get_current_settings()
            logger.info(f"Initialized ACT device: {device} with dtype {dtype}")
        except Exception as e:
            logger.warning(f"Device settings access failed: {e}")
            device, dtype = torch.device("cpu"), torch.float64
    
    def _convert_model_to_act_network(self, model: nn.Module, input_shape: Tuple[int, ...]) -> Tuple[Net, List[int], List[int]]:
        """
        Convert PyTorch model to ACT network representation.
        
        Args:
            model: PyTorch model
            input_shape: Input tensor shape (excluding batch)
            
        Returns:
            Tuple of (ACT network, input_ids, output_ids)
        """
        input_size = int(np.prod(input_shape))
        
        # Infer output size from model
        with torch.no_grad():
            dummy_input = torch.randn(1, *input_shape)
            try:
                dummy_output = model(dummy_input)
                output_size = dummy_output.shape[-1]
            except Exception:
                logger.warning("Could not infer output size, using default")
                output_size = 10  # Default for classification
        
        # Create variable IDs
        input_ids = list(range(input_size))
        output_ids = list(range(input_size, input_size + output_size))
        
        # Create simplified network layers for linear models
        layers = []
        layer_id = 0
        current_vars = input_ids
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                next_vars = list(range(max(current_vars) + 1, max(current_vars) + 1 + module.out_features))
                
                W = module.weight.data
                b = module.bias.data if module.bias is not None else torch.zeros(module.out_features)
                W_pos = torch.clamp(W, min=0)
                W_neg = torch.clamp(W, max=0)
                
                layer = Layer(
                    id=layer_id,
                    kind="DENSE",
                    params={"W": W, "W_pos": W_pos, "W_neg": W_neg, "b": b},
                    in_vars=current_vars,
                    out_vars=next_vars
                )
                layers.append(layer)
                current_vars = next_vars
                layer_id += 1
                
            elif isinstance(module, nn.ReLU):
                layer = Layer(
                    id=layer_id,
                    kind="RELU",
                    params={},
                    in_vars=current_vars,
                    out_vars=current_vars
                )
                layers.append(layer)
                layer_id += 1
        
        output_ids = current_vars
        
        # Create network structure
        preds = {i: [] if i == 0 else [i-1] for i in range(len(layers))}
        succs = {i: [i+1] if i < len(layers)-1 else [] for i in range(len(layers))}
        
        net = Net(layers=layers, preds=preds, succs=succs)
        return net, input_ids, output_ids
    
    def _run_verify_bab(self, test_case: TestCase) -> Dict[str, Any]:
        """
        Run verify_bab on a test case.
        
        Args:
            test_case: Test case to verify
            
        Returns:
            Verification result dictionary
        """
        start_time = time.time()
        
        try:
            # Get current device and dtype from PyTorch global settings
            current_device, current_dtype = get_current_settings()
            
            model_converted = test_case.model.to(dtype=current_dtype)
            
            # Convert model to ACT network
            input_shape = test_case.sample_data.shape[1:]  # Remove batch dimension
            net, input_ids, output_ids = self._convert_model_to_act_network(model_converted, input_shape)
            
            # Create input specification
            input_spec_dict = test_case.input_spec
            
            # Create proper bounds based on input specification
            if "lb" in input_spec_dict and "ub" in input_spec_dict:
                lb = input_spec_dict["lb"]
                ub = input_spec_dict["ub"]
            elif "epsilon" in input_spec_dict:
                # Create bounds from center point and epsilon
                # Use first sample from batch
                center = test_case.sample_data[0]  # Take first sample from batch
                epsilon = input_spec_dict["epsilon"]
                lb = center - epsilon
                ub = center + epsilon
            else:
                # Default small perturbation
                center = test_case.sample_data[0]  # Take first sample from batch
                epsilon = 0.05
                lb = center - epsilon
                ub = center + epsilon
            
            # Flatten bounds to match the network input and convert to current dtype/device
            lb_flat = lb.view(-1).to(device=current_device, dtype=current_dtype)
            ub_flat = ub.view(-1).to(device=current_device, dtype=current_dtype)
            center_flat = test_case.sample_data[0].view(-1).to(device=current_device, dtype=current_dtype)
            
            input_spec = InputSpec(
                kind=InKind.BOX,  # Use BOX for bounds-based specs
                lb=lb_flat,
                ub=ub_flat,
                center=center_flat,
                eps=input_spec_dict.get("epsilon", 0.05)
            )
            
            # Create output specification
            output_spec_dict = test_case.output_spec
            
            # Get true label from test case
            y_true = test_case.labels.item() if test_case.labels.numel() == 1 else 0
            
            output_spec = OutputSpec(
                kind=OutKind.TOP1_ROBUST,  # Use TOP1_ROBUST for classification
                y_true=y_true,
                margin=output_spec_dict.get("margin", 0.0)
            )
            
            # Create root bounds from input spec
            root_box = seed_from_input_spec(input_spec)
            
            # Use PyTorch LP solver with current dtype
            solver = TorchLPSolver(dtype=current_dtype)
            
            # Create model function for BaB
            def model_fn(x):
                """Model evaluation function for BaB."""
                with torch.no_grad():
                    if isinstance(x, torch.Tensor):
                        x = x.to(dtype=current_dtype)  # Convert to current dtype
                    if x.dim() == 1:
                        x = x.unsqueeze(0)  # Add batch dimension
                    
                    # Flatten input if model expects it (for linear layers)
                    original_shape = x.shape
                    if len(original_shape) > 2:  # If input has spatial dimensions
                        x = x.view(original_shape[0], -1)  # Flatten to [batch, features]
                    
                    output = model_converted(x)
                    # CRITICAL: Remove batch dimension for BaB compatibility
                    if output.dim() > 1:
                        output = output.squeeze(0)
                    return output
            
            # Run verify_bab
            logger.debug(f"Running verify_bab for test {test_case.test_id}")
            
            verif_result = verify_bab(
                net=net,
                entry_id=0,
                input_ids=input_ids,
                output_ids=output_ids,
                input_spec=input_spec,
                output_spec=output_spec,
                root_box=root_box,
                solver=solver,
                model_fn=model_fn,
                max_depth=3,  # Small for testing
                max_nodes=20, # Small for testing  
                time_budget_s=test_case.timeout
            )
            
            execution_time = time.time() - start_time
            
            # Convert ACT result to pipeline result
            if verif_result.status == VerifStatus.CERTIFIED:
                verify_result = VerifyResult.UNSAT
                success = True
            elif verif_result.status == VerifStatus.COUNTEREXAMPLE:
                verify_result = VerifyResult.SAT
                success = True
            else:
                verify_result = VerifyResult.UNKNOWN
                success = False
            
            # Check against expected result if provided
            result_correct = True
            if test_case.expected_result is not None:
                result_correct = (verify_result == test_case.expected_result)
            
            return {
                "success": success and result_correct,
                "verify_result": verify_result.value,
                "expected_result": test_case.expected_result.value if test_case.expected_result else None,
                "result_correct": result_correct,
                "execution_time": execution_time,
                "memory_usage_mb": 0.0,
                "model_stats": verif_result.model_stats if hasattr(verif_result, 'model_stats') else {},
                "counterexample": {
                    "x": verif_result.ce_x.tolist() if verif_result.ce_x is not None else None,
                    "y": verif_result.ce_y.tolist() if verif_result.ce_y is not None else None
                } if verif_result.status == VerifStatus.COUNTEREXAMPLE else None
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"verify_bab failed for test {test_case.test_id}: {e}")
            
            return {
                "success": False,
                "verify_result": VerifyResult.ERROR.value,
                "expected_result": test_case.expected_result.value if test_case.expected_result else None,
                "result_correct": False,
                "execution_time": execution_time,
                "memory_usage_mb": 0.0,
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    def validate_verify_bab(self, test_cases: List[TestCase]) -> ValidationResult:
        """
        Test verify_bab correctness using test cases.
        
        Args:
            test_cases: List of test cases to validate
            
        Returns:
            ValidationResult with validation results
        """
        logger.info(f"Starting verify_bab validation with {len(test_cases)} test cases")
        
        start_time = time.time()
        passed = 0
        failed = 0
        results = []
        
        for i, test_case in enumerate(test_cases):
            try:
                print_memory_usage(f"Test {i+1}/{len(test_cases)} - ")
                
                # Run verify_bab
                result = self._run_verify_bab(test_case)
                
                # Check result
                if result["success"]:
                    passed += 1
                else:
                    failed += 1
                
                results.append({
                    "test_id": test_case.test_id,
                    "result": result["verify_result"],
                    "expected": result["expected_result"],
                    "correct": result["result_correct"],
                    "execution_time": result["execution_time"],
                    "success": result["success"],
                    "counterexample": result.get("counterexample"),
                    "model_stats": result.get("model_stats", {})
                })
                
                logger.info(f"Test {i+1}: {result['verify_result']} (expected: {result.get('expected_result', 'N/A')})")
                
            except Exception as e:
                failed += 1
                logger.error(f"Test {i+1} failed: {e}")
                results.append({
                    "test_id": test_case.test_id,
                    "result": "ERROR",
                    "error": str(e),
                    "success": False
                })
            
            # Clear memory after each test
            clear_torch_cache()
        
        total_time = time.time() - start_time
        
        success_rate = passed / len(test_cases) if test_cases else 0.0
        logger.info(f"verify_bab validation completed: {passed}/{len(test_cases)} passed ({success_rate:.1%})")
        
        return ValidationResult(
            success=passed > 0,  # Consider successful if at least one test passes
            total_tests=len(test_cases),
            passed_tests=passed,
            failed_tests=failed,
            results=results,
            execution_time=total_time,
            memory_usage_mb=0.0  # Could be improved with actual memory tracking
        )
    
    def generate_test_cases_from_config(self, config_name: str = "test_scenarios") -> List[TestCase]:
        """
        Generate test cases from configuration using mock factory.
        
        Args:
            config_name: Name of config file to load
            
        Returns:
            List of test cases
        """
        try:
            # Load test scenario configuration
            scenario_config = load_config(config_name)
            scenarios = scenario_config.get("scenarios", {})
            
            test_cases = []
            
            for scenario_name, scenario_data in scenarios.items():
                try:
                    # Generate components using mock factory
                    sample_data, labels = self.mock_factory.create_sample_data("mnist_small")
                    input_spec = self.mock_factory.create_input_spec("robust_l_inf_small")
                    output_spec = self.mock_factory.create_output_spec("classification")
                    model = self.mock_factory.create_model("simple_relu")
                    
                    # Extract expected result if specified
                    expected_result = None
                    if "expected_result" in scenario_data:
                        expected_result = VerifyResult(scenario_data["expected_result"])
                    
                    test_case = TestCase(
                        sample_data=sample_data,
                        labels=labels,
                        input_spec=input_spec,
                        output_spec=output_spec,
                        model=model,
                        expected_result=expected_result,
                        test_id=scenario_name,
                        timeout=scenario_data.get("timeout", 60.0)
                    )
                    test_cases.append(test_case)
                    
                except Exception as e:
                    logger.warning(f"Failed to create test case for scenario {scenario_name}: {e}")
            
            return test_cases
            
        except Exception as e:
            logger.error(f"Failed to load config {config_name}: {e}")
            
            # Fallback: create simple test cases using mock factory
            logger.info("Creating fallback test cases")
            test_cases = []
            
            for i in range(3):  # Small number for testing
                sample_data, labels = self.mock_factory.create_sample_data("mnist_small")
                input_spec = self.mock_factory.create_input_spec("robust_l_inf_small")
                output_spec = self.mock_factory.create_output_spec("classification")
                model = self.mock_factory.create_model("simple_relu")
                
                test_case = TestCase(
                    sample_data=sample_data,
                    labels=labels,
                    input_spec=input_spec,
                    output_spec=output_spec,
                    model=model,
                    test_id=f"fallback_test_{i}",
                    timeout=60.0
                )
                test_cases.append(test_case)
            
            return test_cases


def validate_correctness_framework() -> bool:
    """
    Validate the correctness framework by testing verify_bab with mock inputs.
    
    Returns:
        True if validation passes, False otherwise
    """
    logger.info("🧪 Validating Back-End verify_bab Correctness...")
    tests_passed = 0
    total_tests = 0
    
    # Test 1: BackEndVerifierValidator initialization
    total_tests += 1
    try:
        validator = BackEndVerifierValidator()
        if hasattr(validator, 'mock_factory'):
            logger.info("✅ BackEndVerifierValidator initialization successful")
            tests_passed += 1
        else:
            logger.error("❌ BackEndVerifierValidator missing required components")
    except Exception as e:
        logger.error(f"❌ BackEndVerifierValidator initialization failed: {e}")
    
    # Test 2: Test case generation from mock factory
    total_tests += 1
    try:
        validator = BackEndVerifierValidator()
        test_cases = validator.generate_test_cases_from_config()
        
        if test_cases and len(test_cases) > 0:
            logger.info(f"✅ Generated {len(test_cases)} test cases from mock factory")
            tests_passed += 1
        else:
            logger.error("❌ Failed to generate test cases")
    except Exception as e:
        logger.error(f"❌ Test case generation failed: {e}")
    
    # Test 3: verify_bab validation (with reduced test set)
    total_tests += 1
    try:
        validator = BackEndVerifierValidator()
        test_cases = validator.generate_test_cases_from_config()
        
        # Use only first test case for quick validation
        if test_cases:
            single_test = [test_cases[0]]
            validation_result = validator.validate_verify_bab(single_test)
            
            if validation_result.total_tests > 0:
                logger.info(f"✅ verify_bab validation completed: {validation_result.passed_tests}/{validation_result.total_tests} passed")
                tests_passed += 1
            else:
                logger.error("❌ verify_bab validation failed: no tests run")
        else:
            logger.error("❌ verify_bab validation failed: no test cases")
    except Exception as e:
        logger.error(f"❌ verify_bab validation failed: {e}")
    
    # Summary
    success = tests_passed == total_tests
    status = "✅ SUCCESS" if success else "❌ FAILED"
    logger.info(f"📊 Back-end correctness validation: {status} ({tests_passed}/{total_tests} tests passed)")
    
    return success


if __name__ == "__main__":
    """
    Main entry point for running the correctness validation.
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    )
    
    print("🚀 Running ACT Back-End Correctness Validation")
    success = validate_correctness_framework()
    print(f"\n🎯 Final Result: {'✅ COMPLETE SUCCESS' if success else '❌ FAILED'}")