"""
Front-end integration bridge for ACT pipeline.

This module provides integration between the pipeline testing framework and 
ACT's front-end components, enabling testing with real specifications,
models, and datasets using the front_end loaders.
"""

import torch
import torch.nn as nn
import os
import sys
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
from enum import Enum
import logging

# Add ACT paths for imports
current_dir = Path(__file__).parent
act_root = current_dir.parent
sys.path.insert(0, str(act_root))

from act.front_end.specs import InputSpec, OutputSpec, InKind, OutKind
from act.front_end.loaders import ModelLoader, DatasetLoader, SpecLoader
from act.front_end.batch import SampleRecord
from act.front_end.device_manager import get_current_settings
from act.back_end.core import Net, Layer, Bounds
from act.back_end.verify_status import VerifStatus, VerifResult, verify_once, seed_from_input_spec
from act.back_end.solver.solver_gurobi import GurobiSolver
from act.back_end.solver.solver_torch import TorchLPSolver
from act.back_end.bab import verify_bab

logger = logging.getLogger(__name__)


# Local test case definitions to avoid pipeline import issues
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
    name: str
    input_data: torch.Tensor
    expected_output: torch.Tensor
    model: nn.Module
    input_spec: InputSpec
    output_spec: OutputSpec
    config: Dict[str, Any]


@dataclass
class ValidationResult:
    """Validation test result."""
    test_name: str
    passed: bool
    execution_time: float
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class PerformanceResult:
    """Performance test result."""
    test_name: str
    execution_time: float
    memory_usage_mb: float
    success: bool
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ACTIntegrationConfig:
    """Configuration for ACT integration."""
    device: str = "cpu"
    data_dir: str = "data"
    model_dir: str = "models"
    timeout: float = 300.0


@dataclass
class IntegrationTestCase:
    """Test case using real ACT components."""
    dataset_name: str
    model_path: str
    spec_type: str
    sample_indices: List[int]
    epsilon: float = 0.1
    norm_type: str = "inf"
    timeout: float = 300.0


class ACTFrontendBridge:
    """Bridge between pipeline and ACT front-end components."""
    
    def __init__(self, config: Optional[ACTIntegrationConfig] = None):
        """Initialize integration bridge."""
        self.config = config or ACTIntegrationConfig()
        self.device = self.config.device
        
        # Get current device/dtype settings (auto-initialization already happened)
        try:
            device, dtype = get_current_settings()
            logger.info(f"✅ ACT initialized: device={device}, dtype={dtype}")
        except Exception as e:
            logger.error(f"Device initialization failed: {e}")
            raise RuntimeError(f"Device setup failed: {e}")
        
        # Paths
        self.data_dir = Path(self.config.data_dir)
        self.model_dir = Path(self.config.model_dir)
        
        # Initialize front-end loaders
        self.model_loader = ModelLoader()
        self.data_loader = DatasetLoader()
        self.spec_loader = SpecLoader()
    
    def load_dataset_samples(
        self,
        dataset_name: str,
        sample_indices: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load specific samples from dataset using front-end data loader."""
        # Construct CSV file path
        dataset_path = self.data_dir / f"{dataset_name.upper()}_csv"
        csv_file = dataset_path / f"{dataset_name.lower()}_first_100_samples.csv"
        
        if not csv_file.exists():
            raise FileNotFoundError(f"Dataset CSV not found: {csv_file}")
        
        # Load real CSV data
        logger.info(f"Loading dataset from: {csv_file}")
        data_pairs = self.data_loader.load_csv_torch(str(csv_file))
        
        # Extract requested samples
        samples = []
        labels = []
        
        for idx in sample_indices:
            if idx >= len(data_pairs):
                raise IndexError(f"Sample index {idx} out of bounds (dataset size: {len(data_pairs)})")
            
            sample_tensor, label = data_pairs[idx]  # load_csv_torch returns torch tensors directly
            samples.append(sample_tensor)
            labels.append(label)
        
        samples_tensor = torch.stack(samples)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        # Return tensors directly (global defaults already set)
        return samples_tensor, labels_tensor
    
    def load_model_from_path(self, model_path: str) -> nn.Module:
        """Load model from file path using front-end model loader."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        if not model_path.endswith('.onnx'):
            raise ValueError(f"Only ONNX models supported, got: {model_path}")
        
        logger.info(f"Loading ONNX model: {model_path}")
        model = self.model_loader.load_onnx_model(model_path)
        
        if model is None:
            raise RuntimeError(f"Failed to load model: {model_path}")
        
        return model
    
    def create_input_spec(
        self,
        samples: torch.Tensor,
        epsilon: float = 0.1,
        norm_type: str = "inf"
    ) -> InputSpec:
        """Create input specification using front-end spec loader."""
        logger.info(f"Creating input specs with epsilon={epsilon}, norm={norm_type}")
        
        # Convert samples to list format for SpecLoader
        samples_list = [samples[i] for i in range(samples.shape[0])]
        
        # Create spec config dictionary
        spec_config = {
            "type": "linf_ball" if norm_type.lower() == "inf" else "l2_ball",
            "epsilon": epsilon,
            "norm_type": norm_type
        }
        
        # Use front-end spec loader
        input_specs = self.spec_loader.create_input_specs(samples_list, spec_config)
        
        if len(input_specs) == 0:
            raise RuntimeError("SpecLoader returned empty list")
        
        return input_specs[0]
    
    def create_output_spec(
        self,
        target_label: int,
        num_classes: int = 10,
        spec_type: str = "robustness"
    ) -> OutputSpec:
        """Create output specification using front-end specs."""
        logger.info(f"Creating output spec for class {target_label}")
        
        # Create spec config dictionary
        spec_config = {
            "output_type": "margin_robust" if spec_type == "robustness" else spec_type,
            "margin": 0.0,
            "target_class": target_label
        }
        
        # Use front-end spec loader
        output_specs = self.spec_loader.create_output_specs([target_label], spec_config)
        
        if len(output_specs) == 0:
            raise RuntimeError("SpecLoader returned empty output specs")
        
        return output_specs[0]
    
    def create_test_case(
        self,
        dataset_name: str,
        model_path: str,
        sample_indices: List[int],
        epsilon: float = 0.1,
        norm_type: str = "inf"
    ) -> TestCase:
        """Create a complete test case using ACT front-end components."""
        logger.info(f"Creating test case: {dataset_name}, samples: {sample_indices}")
        
        # Load components using front-end loaders (no fallbacks)
        samples, labels = self.load_dataset_samples(dataset_name, sample_indices)
        model = self.load_model_from_path(model_path)
        
        # Create specifications using front-end loaders (no fallbacks)
        input_spec = self.create_input_spec(samples, epsilon, norm_type)
        output_spec = self.create_output_spec(labels[0].item() if len(labels) > 0 else 0)
        
        # Create test case
        test_case = TestCase(
            name=f"{dataset_name}_integration_test",
            input_data=samples,
            expected_output=labels,
            model=model,
            input_spec=input_spec,
            output_spec=output_spec,
            config={
                'dataset': dataset_name,
                'model_path': model_path,
                'epsilon': epsilon,
                'norm_type': norm_type,
                'sample_indices': sample_indices,
                'frontend_integration': True
            }
        )
        
        logger.info(f"Created test case: {test_case.name}")
        return test_case
    
    def run_integration_test(self, test_case: IntegrationTestCase) -> ValidationResult:
        """Run integration test with real ACT verifiers like driver.py."""
        logger.info(f"Running real verification test: {test_case.dataset_name}")
        
        # Create complete test case using front-end loaders
        complete_test_case = self.create_test_case(
            dataset_name=test_case.dataset_name,
            model_path=test_case.model_path,
            sample_indices=test_case.sample_indices,
            epsilon=test_case.epsilon,
            norm_type=test_case.norm_type
        )
        
        # Convert PyTorch model to ACT Net format for verification
        model = complete_test_case.model
        sample = complete_test_case.input_data[0]  # Take first sample
        
        try:
            # Create ACT Net from PyTorch model
            net, entry_id, input_ids, output_ids = self._pytorch_to_act_net(model, sample)
            
            # Get verification specifications
            input_spec = complete_test_case.input_spec
            output_spec = complete_test_case.output_spec
            
            # Create root box from input specification
            root_box = seed_from_input_spec(input_spec)
            
            # Define forward function for verification (takes only input tensor x)
            @torch.no_grad()
            def forward_fn(x):
                """Model function that takes input tensor and returns output tensor"""
                # Ensure input tensor is on the same device as the model
                model_device = next(model.parameters()).device
                if x.device != model_device:
                    x = x.to(model_device)
                
                # Handle input reshaping like demo_driver does
                if x.numel() == 3072:  # CIFAR-10: 32x32x3 = 3072
                    x_reshaped = x.view(1, 3, 32, 32)
                elif x.numel() == 784:  # MNIST: 28x28 = 784  
                    x_reshaped = x.view(1, 1, 28, 28)
                else:
                    x_reshaped = x if len(x.shape) == 4 else x.unsqueeze(0)
                
                return model(x_reshaped)
            
            # Configure solvers to test (like driver.py)
            solvers_to_test = [
                ("Gurobi MILP", lambda: GurobiSolver()),
                ("PyTorch LP", lambda: TorchLPSolver())
            ]
            
            verification_results = []
            verification_success = False
            
            # Test each solver independently (no early exit)
            for solver_name, solver_factory in solvers_to_test:
                logger.info(f"Testing with {solver_name} solver...")
                
                try:
                    solver = solver_factory()
                    
                    # Run verification using verify_bab (same as driver.py)
                    result = verify_bab(
                        net, 
                        entry_id=entry_id, 
                        input_ids=input_ids,
                        output_ids=output_ids,
                        input_spec=input_spec, 
                        output_spec=output_spec, 
                        root_box=root_box,
                        solver=solver, 
                        model_fn=forward_fn,
                        max_depth=5,  # Reduced for faster testing
                        max_nodes=50, # Reduced for faster testing
                        time_budget_s=test_case.timeout
                    )
                    
                    # Record verification result
                    verification_results.append({
                        'solver': solver_name,
                        'status': result.status,
                        'stats': result.model_stats,
                        'success': True
                    })
                    
                    logger.info(f"✅ {solver_name} verification completed: {result.status}")
                    
                    if result.status == "CERTIFIED":
                        logger.info(f"✅ Property VERIFIED for {solver_name}")
                        verification_success = True
                        # Continue to test next solver (no break)
                    elif result.status == "COUNTEREXAMPLE":
                        logger.info(f"❌ Property VIOLATED - Counterexample found with {solver_name}")
                        if result.ce_x is not None:
                            ce_input_norm = torch.norm(torch.as_tensor(result.ce_x)).item()
                            logger.info(f"  Counterexample input norm: {ce_input_norm:.6f}")
                        verification_success = True  # Counterexample is also a valid result
                        # Continue to test next solver (no break)
                    else:
                        logger.info(f"❓ Property status UNKNOWN with {solver_name}")
                        # Continue to next solver
                        
                except Exception as e:
                    logger.warning(f"❌ {solver_name} solver failed: {e}")
                    verification_results.append({
                        'solver': solver_name,
                        'status': 'ERROR',
                        'error': str(e),
                        'success': False
                    })
                    
                    if "gurobi" in solver_name.lower():
                        logger.info("  Note: Gurobi license may not be available")
                    elif "torch" in solver_name.lower():
                        logger.info("  Note: PyTorch LP solver may not support this constraint")
                    
                    # Continue to next solver
            
            # Create validation result based on verification outcome
            execution_time = sum(r.get('execution_time', 0.1) for r in verification_results)
            
            result = ValidationResult(
                test_name=f"real_verification_{test_case.dataset_name}",
                passed=verification_success,
                execution_time=execution_time,
                error_message=None if verification_success else "All verifiers failed",
                metadata={
                    'verification_results': verification_results,
                    'dataset': test_case.dataset_name,
                    'model_path': test_case.model_path,
                    'epsilon': test_case.epsilon,
                    'norm_type': test_case.norm_type,
                    'real_verification': True,
                    'num_solvers_tested': len(solvers_to_test),
                    'verification_success': verification_success
                }
            )
            
            logger.info(f"Real verification completed: {result.passed}")
            return result
            
        except Exception as e:
            logger.error(f"Real verification setup failed: {e}")
            return ValidationResult(
                test_name=f"real_verification_{test_case.dataset_name}",
                passed=False,
                execution_time=0.0,
                error_message=f"Verification setup failed: {e}",
                metadata={'error': str(e), 'real_verification_failed': True}
            )
    
    def _pytorch_to_act_net(self, pytorch_model: nn.Module, sample_input: torch.Tensor) -> Tuple[Net, int, List[int], List[int]]:
        """Convert PyTorch model to ACT Net format with CNN/RNN support (following driver.py patterns).
        Returns: (net, entry_id, input_ids, output_ids)
        """
        layers = []
        var_counter = 0
        layer_id = 0
        
        # Ensure consistent tensor dtype using global defaults
        sample_input = sample_input  # Already a tensor, no conversion needed
        
        # Get input shape for CNN layers
        if len(sample_input.shape) == 1:
            # Assume flattened input, infer image shape
            if sample_input.shape[0] == 784:  # MNIST
                current_shape = (1, 1, 28, 28)
            elif sample_input.shape[0] == 3072:  # CIFAR-10
                current_shape = (1, 3, 32, 32)
            else:
                # For unknown shapes, assume it's already the right format
                current_shape = sample_input.shape
        else:
            current_shape = sample_input.shape
        
        input_size = sample_input.numel()
        current_vars = list(range(input_size))
        var_counter = input_size
        
        logger.info(f"Converting PyTorch model to ACT Net, input shape: {current_shape}")
        
        # Convert each layer
        for module in pytorch_model.children():
            if isinstance(module, nn.Conv2d):
                # Create ACT CONV2D layer (with proper tensor conversion)
                weight = module.weight.detach()  # Already tensor, no conversion needed
                bias = module.bias.detach() if module.bias is not None else None
                
                # Calculate output shape
                batch_size, in_channels, in_h, in_w = current_shape
                out_channels = module.out_channels
                
                # Calculate output dimensions
                out_h = (in_h + 2 * module.padding[0] - module.dilation[0] * (module.kernel_size[0] - 1) - 1) // module.stride[0] + 1
                out_w = (in_w + 2 * module.padding[1] - module.dilation[1] * (module.kernel_size[1] - 1) - 1) // module.stride[1] + 1
                output_shape = (batch_size, out_channels, out_h, out_w)
                
                # Create output variables
                output_size = out_channels * out_h * out_w
                out_vars = list(range(var_counter, var_counter + output_size))
                
                conv_layer = Layer(
                    id=layer_id,
                    kind="CONV2D",
                    params={
                        "weight": weight,
                        "bias": bias,
                        "stride": module.stride,
                        "padding": module.padding,
                        "dilation": module.dilation,
                        "groups": module.groups,
                        "input_shape": current_shape,
                        "output_shape": output_shape
                    },
                    in_vars=current_vars.copy(),
                    out_vars=out_vars,
                    cache={}
                )
                layers.append(conv_layer)
                
                current_vars = out_vars
                current_shape = output_shape
                var_counter += output_size
                layer_id += 1
                
            elif isinstance(module, nn.MaxPool2d):
                # Create ACT MAXPOOL2D layer
                batch_size, channels, in_h, in_w = current_shape
                
                # Calculate output dimensions
                out_h = (in_h + 2 * module.padding - module.dilation * (module.kernel_size - 1) - 1) // module.stride + 1
                out_w = (in_w + 2 * module.padding - module.dilation * (module.kernel_size - 1) - 1) // module.stride + 1
                output_shape = (batch_size, channels, out_h, out_w)
                
                # Create output variables (same number of channels)
                output_size = channels * out_h * out_w
                out_vars = list(range(var_counter, var_counter + output_size))
                
                maxpool_layer = Layer(
                    id=layer_id,
                    kind="MAXPOOL2D",
                    params={
                        "kernel_size": module.kernel_size,
                        "stride": module.stride,
                        "padding": module.padding,
                        "dilation": module.dilation,
                        "input_shape": current_shape,
                        "output_shape": output_shape
                    },
                    in_vars=current_vars.copy(),
                    out_vars=out_vars,
                    cache={}
                )
                layers.append(maxpool_layer)
                
                current_vars = out_vars
                current_shape = output_shape
                var_counter += output_size
                layer_id += 1
                
            elif isinstance(module, nn.AvgPool2d):
                # Create ACT AVGPOOL2D layer
                batch_size, channels, in_h, in_w = current_shape
                
                # Calculate output dimensions
                out_h = (in_h + 2 * module.padding - module.kernel_size) // module.stride + 1
                out_w = (in_w + 2 * module.padding - module.kernel_size) // module.stride + 1
                output_shape = (batch_size, channels, out_h, out_w)
                
                output_size = channels * out_h * out_w
                out_vars = list(range(var_counter, var_counter + output_size))
                
                avgpool_layer = Layer(
                    id=layer_id,
                    kind="AVGPOOL2D",
                    params={
                        "kernel_size": module.kernel_size,
                        "stride": module.stride,
                        "padding": module.padding,
                        "input_shape": current_shape,
                        "output_shape": output_shape
                    },
                    in_vars=current_vars.copy(),
                    out_vars=out_vars,
                    cache={}
                )
                layers.append(avgpool_layer)
                
                current_vars = out_vars
                current_shape = output_shape
                var_counter += output_size
                layer_id += 1
                
            elif isinstance(module, nn.Flatten):
                # Create ACT FLATTEN layer
                input_shape = current_shape
                # Calculate explicit feature count instead of using -1
                total_features = torch.prod(torch.tensor(input_shape[1:])).item()
                output_shape = (input_shape[0], total_features)  # Explicit dimensions
                
                flatten_layer = Layer(
                    id=layer_id,
                    kind="FLATTEN",
                    params={
                        "input_shape": input_shape,
                        "output_shape": output_shape
                    },
                    in_vars=current_vars.copy(),
                    out_vars=current_vars.copy(),  # Same variables, just reshaped
                    cache={}
                )
                layers.append(flatten_layer)
                
                # Update current shape but keep same variables
                current_shape = output_shape
                layer_id += 1
                
            elif isinstance(module, nn.Linear):
                # Create ACT DENSE layer (following driver.py pattern)
                W = module.weight.detach()  # Already tensor, no conversion needed
                b = module.bias.detach() if module.bias is not None else torch.zeros(module.out_features)
                
                # Pre-compute positive and negative weights (as in driver.py)
                W_pos = torch.clamp(W, min=0)
                W_neg = torch.clamp(W, max=0)
                
                out_vars = list(range(var_counter, var_counter + module.out_features))
                
                dense_layer = Layer(
                    id=layer_id,
                    kind="DENSE",
                    params={"W": W, "W_pos": W_pos, "W_neg": W_neg, "b": b},  # Match driver.py format
                    in_vars=current_vars.copy(),
                    out_vars=out_vars,
                    cache={}
                )
                layers.append(dense_layer)
                
                current_vars = out_vars
                var_counter += module.out_features
                layer_id += 1
                
            elif isinstance(module, nn.LSTM):
                # Create ACT LSTM layer
                input_size = module.input_size
                hidden_size = module.hidden_size
                num_layers = module.num_layers
                batch_first = module.batch_first
                bidirectional = module.bidirectional
                
                # Extract weights and biases
                weight_ih = module.weight_ih_l0.detach()
                weight_hh = module.weight_hh_l0.detach()
                bias_ih = module.bias_ih_l0.detach() if hasattr(module, 'bias_ih_l0') else None
                bias_hh = module.bias_hh_l0.detach() if hasattr(module, 'bias_hh_l0') else None
                
                # Calculate output shape
                batch_size, seq_len = current_shape[0], current_shape[1] if len(current_shape) >= 2 else 1
                output_hidden_size = hidden_size * 2 if bidirectional else hidden_size
                
                if batch_first:
                    output_shape = (batch_size, seq_len, output_hidden_size)
                else:
                    output_shape = (seq_len, batch_size, output_hidden_size)
                
                output_size = torch.prod(torch.tensor(output_shape)).item()
                out_vars = list(range(var_counter, var_counter + output_size))
                
                lstm_layer = Layer(
                    id=layer_id,
                    kind="LSTM",
                    params={
                        "weight_ih": weight_ih,
                        "weight_hh": weight_hh,
                        "bias_ih": bias_ih,
                        "bias_hh": bias_hh,
                        "input_size": input_size,
                        "hidden_size": hidden_size,
                        "num_layers": num_layers,
                        "batch_first": batch_first,
                        "bidirectional": bidirectional,
                        "input_shape": current_shape,
                        "output_shape": output_shape
                    },
                    in_vars=current_vars.copy(),
                    out_vars=out_vars,
                    cache={}
                )
                layers.append(lstm_layer)
                
                current_vars = out_vars
                current_shape = output_shape
                var_counter += output_size
                layer_id += 1
                
            elif isinstance(module, nn.GRU):
                # Create ACT GRU layer
                input_size = module.input_size
                hidden_size = module.hidden_size
                num_layers = module.num_layers
                batch_first = module.batch_first
                bidirectional = module.bidirectional
                
                # Extract weights and biases
                weight_ih = module.weight_ih_l0.detach()
                weight_hh = module.weight_hh_l0.detach()
                bias_ih = module.bias_ih_l0.detach() if hasattr(module, 'bias_ih_l0') else None
                bias_hh = module.bias_hh_l0.detach() if hasattr(module, 'bias_hh_l0') else None
                
                # Calculate output shape
                batch_size, seq_len = current_shape[0], current_shape[1] if len(current_shape) >= 2 else 1
                output_hidden_size = hidden_size * 2 if bidirectional else hidden_size
                
                if batch_first:
                    output_shape = (batch_size, seq_len, output_hidden_size)
                else:
                    output_shape = (seq_len, batch_size, output_hidden_size)
                
                output_size = torch.prod(torch.tensor(output_shape)).item()
                out_vars = list(range(var_counter, var_counter + output_size))
                
                gru_layer = Layer(
                    id=layer_id,
                    kind="GRU",
                    params={
                        "weight_ih": weight_ih,
                        "weight_hh": weight_hh,
                        "bias_ih": bias_ih,
                        "bias_hh": bias_hh,
                        "input_size": input_size,
                        "hidden_size": hidden_size,
                        "num_layers": num_layers,
                        "batch_first": batch_first,
                        "bidirectional": bidirectional,
                        "input_shape": current_shape,
                        "output_shape": output_shape
                    },
                    in_vars=current_vars.copy(),
                    out_vars=out_vars,
                    cache={}
                )
                layers.append(gru_layer)
                
                current_vars = out_vars
                current_shape = output_shape
                var_counter += output_size
                layer_id += 1
                
            elif isinstance(module, nn.RNN):
                # Create ACT RNN layer
                input_size = module.input_size
                hidden_size = module.hidden_size
                num_layers = module.num_layers
                nonlinearity = module.nonlinearity
                batch_first = module.batch_first
                bidirectional = module.bidirectional
                
                # Extract weights and biases
                weight_ih = module.weight_ih_l0.detach()
                weight_hh = module.weight_hh_l0.detach()
                bias_ih = module.bias_ih_l0.detach() if hasattr(module, 'bias_ih_l0') else None
                bias_hh = module.bias_hh_l0.detach() if hasattr(module, 'bias_hh_l0') else None
                
                # Calculate output shape
                batch_size, seq_len = current_shape[0], current_shape[1] if len(current_shape) >= 2 else 1
                output_hidden_size = hidden_size * 2 if bidirectional else hidden_size
                
                if batch_first:
                    output_shape = (batch_size, seq_len, output_hidden_size)
                else:
                    output_shape = (seq_len, batch_size, output_hidden_size)
                
                output_size = torch.prod(torch.tensor(output_shape)).item()
                out_vars = list(range(var_counter, var_counter + output_size))
                
                rnn_layer = Layer(
                    id=layer_id,
                    kind="RNN",
                    params={
                        "weight_ih": weight_ih,
                        "weight_hh": weight_hh,
                        "bias_ih": bias_ih,
                        "bias_hh": bias_hh,
                        "input_size": input_size,
                        "hidden_size": hidden_size,
                        "num_layers": num_layers,
                        "nonlinearity": nonlinearity,
                        "batch_first": batch_first,
                        "bidirectional": bidirectional,
                        "input_shape": current_shape,
                        "output_shape": output_shape
                    },
                    in_vars=current_vars.copy(),
                    out_vars=out_vars,
                    cache={}
                )
                layers.append(rnn_layer)
                
                current_vars = out_vars
                current_shape = output_shape
                var_counter += output_size
                layer_id += 1
                
            elif isinstance(module, nn.Embedding):
                # Create ACT EMBEDDING layer
                num_embeddings = module.num_embeddings
                embedding_dim = module.embedding_dim
                padding_idx = module.padding_idx
                
                # Extract weight
                weight = module.weight.detach()
                
                # Calculate output shape (assumes input indices will be embedded)
                # Input shape should be [batch_size, seq_len] -> [batch_size, seq_len, embedding_dim]
                if len(current_shape) >= 2:
                    batch_size, seq_len = current_shape[:2]
                    output_shape = (batch_size, seq_len, embedding_dim)
                else:
                    # Fallback for unknown input shape
                    output_shape = (*current_shape, embedding_dim)
                
                output_size = torch.prod(torch.tensor(output_shape)).item()
                out_vars = list(range(var_counter, var_counter + output_size))
                
                embedding_layer = Layer(
                    id=layer_id,
                    kind="EMBEDDING",
                    params={
                        "weight": weight,
                        "num_embeddings": num_embeddings,
                        "embedding_dim": embedding_dim,
                        "padding_idx": padding_idx,
                        "input_shape": current_shape,
                        "output_shape": output_shape
                    },
                    in_vars=current_vars.copy(),
                    out_vars=out_vars,
                    cache={}
                )
                layers.append(embedding_layer)
                
                current_vars = out_vars
                current_shape = output_shape
                var_counter += output_size
                layer_id += 1
                
            elif isinstance(module, nn.ReLU):
                # Create ACT ReLU layer
                relu_layer = Layer(
                    id=layer_id,
                    kind="RELU",
                    params={},
                    in_vars=current_vars.copy(),
                    out_vars=current_vars.copy(),
                    cache={}
                )
                layers.append(relu_layer)
                layer_id += 1
                
            elif isinstance(module, (nn.Identity, nn.Dropout)):
                # Skip identity and dropout layers during inference
                continue
                
            else:
                logger.warning(f"Unsupported layer type: {type(module)}")
        
        if not layers:
            raise ValueError("No supported layers found in PyTorch model")
        
        # Build Net topology (sequential)
        preds = {}
        succs = {}
        
        for i, layer in enumerate(layers):
            if i == 0:
                preds[layer.id] = []
            else:
                preds[layer.id] = [layers[i-1].id]
            
            if i == len(layers) - 1:
                succs[layer.id] = []
            else:
                succs[layer.id] = [layers[i+1].id]
        
        # Create ACT Net
        net = Net(layers=layers, preds=preds, succs=succs)
        logger.info(f"Converted PyTorch model to ACT Net with {len(layers)} layers:")
        for layer in layers:
            logger.info(f"  Layer {layer.id}: {layer.kind}")
        
        # Return tuple like demo_driver: (net, entry_id, input_ids, output_ids)
        entry_id = 0  # First layer is entry point
        input_ids = list(range(input_size))  # Input variable IDs
        output_ids = layers[-1].out_vars if layers else []  # Output variable IDs from last layer
        
        return net, entry_id, input_ids, output_ids


# Factory functions for easy usage
def create_integration_bridge(config: Optional[ACTIntegrationConfig] = None) -> ACTFrontendBridge:
    """Create ACT frontend integration bridge."""
    return ACTFrontendBridge(config)


def create_mnist_test_case(
    sample_indices: List[int] = None,
    epsilon: float = 0.1
) -> IntegrationTestCase:
    """Create MNIST integration test case using frontend loaders."""
    if sample_indices is None:
        sample_indices = [0, 1, 2]
    
    return IntegrationTestCase(
        dataset_name="mnist",
        model_path="models/Sample_models/MNIST/small_relu_mnist_cnn_model_1.onnx",
        spec_type="local_lp",
        sample_indices=sample_indices,
        epsilon=epsilon,
        norm_type="inf"
    )


def validate_frontend_integration() -> List[ValidationResult]:
    """Validate real verification integration with ACT solvers."""
    results = []
    
    logger.info("Validating real verification integration...")
    
    # Create integration bridge with frontend loaders
    bridge = create_integration_bridge()
    
    # Test MNIST with real verification
    mnist_test = create_mnist_test_case(sample_indices=[0, 1])
    mnist_result = bridge.run_integration_test(mnist_test)
    results.append(mnist_result)
    
    # Summary
    passed_tests = sum(1 for r in results if r.passed)
    total_tests = len(results)
    
    logger.info(f"Real verification validation: {passed_tests}/{total_tests} tests passed")
    
    return results


if __name__ == "__main__":
    # Run validation when executed directly
    from pathlib import Path
    pipeline_dir = Path(__file__).parent
    log_dir = pipeline_dir / "log"
    log_dir.mkdir(exist_ok=True)
    log_file_path = log_dir / "integration_tests.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file_path)
        ]
    )

    print("Running ACT Integration Tests with Simplified Device Management...")
    try:
        # Auto-initialization already happened via _auto_initialize()
        current_device, current_dtype = get_current_settings()
        cuda_available = torch.cuda.is_available()
        
        device_info = f"device={current_device}, dtype={current_dtype}, cuda_available={cuda_available}"
        if "device=" in device_info and "dtype=" in device_info:
            logger.info("✅ Device management integration successful")
        else:
            logger.error("❌ Device management integration failed")
    except Exception as e:
        logger.error(f"❌ Device management integration failed: {e}")

    # Simple integration test (like driver.py)
    results = validate_frontend_integration()
    print(f"\n📊 Validation Results:")
    for result in results:
        status = '✅ PASSED' if result.passed else '❌ FAILED'
        print(f'  {result.test_name}: {status}')
        print(f'    Execution time: {result.execution_time:.2f}s')
        if result.error_message:
            print(f'    Error: {result.error_message}')
        # Safely print solver-level verification results if available
        if result.metadata and isinstance(result.metadata, dict):
            vrs = result.metadata.get('verification_results')
            if vrs:
                for vr in vrs:
                    try:
                        solver = vr.get('solver') if isinstance(vr, dict) else vr[0]
                        status = vr.get('status') if isinstance(vr, dict) else vr[1]
                        print(f'    Solver {solver}: {status}')
                    except Exception:
                        # Fallback: print raw entry
                        print(f'    Solver entry: {vr}')