#===- verifier.interval.interval_verifier.py interval arithmetic verifier --#
#
#                 ACT: Abstract Constraints Transformer
#
# Copyright (C) <2025->  ACT Team
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#
# Purpose:
# Interval arithmetic-based neural network verifier for robustness analysis.
# Implements abstract constraint solving using interval bound propagation through
# neural network layers with support for BaB refinement constraints.
#
#===----------------------------------------------------------------------===#

import torch
import torch.nn as nn
from typing import Tuple, Optional, List

from abstract_constraint_solver.base_verifier import BaseVerifier
from input_parser.dataset import Dataset
from input_parser.spec import Spec
from input_parser.type import VerifyResult
from util.stats import ACTStats, ACTLog
from onnx2pytorch.operations.flatten import Flatten as OnnxFlatten
from onnx2pytorch.operations.add import Add as OnnxAdd
from onnx2pytorch.operations.div import Div as OnnxDiv
from onnx2pytorch.operations.clip import Clip as OnnxClip
from onnx2pytorch.operations.reshape import Reshape as OnnxReshape
from onnx2pytorch.operations.squeeze import Squeeze as OnnxSqueeze
from onnx2pytorch.operations.unsqueeze import Unsqueeze as OnnxUnsqueeze
from onnx2pytorch.operations.transpose import Transpose as OnnxTranspose
from onnx2pytorch.operations.base import OperatorWrapper

class IntervalVerifier(BaseVerifier):
    """
    Interval arithmetic-based neural network verifier for robustness analysis.
    
    Implements abstract constraint solving using interval bound propagation through
    neural network layers with support for BaB refinement constraints.
    """
    
    # =============================================================================
    # PUBLIC INTERFACE METHODS
    # =============================================================================
    
    def __init__(self, dataset: Dataset, method: str, spec: Spec, device: str = 'cpu') -> None:
        """Initialize the interval verifier with validation.
        
        Args:
            dataset: Input dataset for verification
            method: Verification method (must be 'interval')
            spec: Verification specification including model and constraints
            device: Compute device ('cpu' or 'cuda')
            
        Raises:
            ValueError: If method is not 'interval'
        """
        super().__init__(dataset, spec, device)
        if method != 'interval':
            raise ValueError(f"IntervalVerifier only supports 'interval' method, got {method}.")

    def verify(self) -> VerifyResult:
        """
        Execute the complete interval verification pipeline for all samples.
        
        Processes each sample through interval bound propagation, applies BaB refinement
        when enabled, and aggregates results.
        
        Returns:
            Final verification status aggregated across all samples
        """
        ACTLog.log_verification_info("Starting Interval verification pipeline")

        num_samples = self.input_center.shape[0] if self.input_center.ndim > 1 else 1
        ACTLog.log_verification_info(f"Total samples: {num_samples}")
        results = []
        
        for idx in range(num_samples):
            ACTLog.log_verification_info(f"\n🔍 Processing sample {idx+1}/{num_samples}")
            ACTLog.log_verification_info("="*80)

            center_input, true_label = self.extract_sample_input_and_label(idx)
            if not self.validate_unperturbed_prediction(center_input, true_label, idx):
                ACTLog.log_verification_info(f"⏭️  Skipping verification for sample {idx+1}")
                results.append(VerifyResult.CLEAN_FAILURE)
                continue

            # Extract bounds for current sample
            sample_lb = self.input_lb if self.input_lb.ndim == 1 else self.input_lb[idx]
            sample_ub = self.input_ub if self.input_ub.ndim == 1 else self.input_ub[idx]

            self.clean_prediction_stats['verification_attempted'] += 1

            ACTLog.log_verification_info("Step 1: Interval abstract constraint solving")
            initial_verdict = self._solve_constraints(sample_lb, sample_ub, idx)

            if initial_verdict == VerifyResult.SAT:
                self.clean_prediction_stats['verification_sat'] += 1
                ACTLog.log_verification_info(f"✅ Interval verification success - Sample {idx+1} safe")
                results.append(initial_verdict)
                continue
            elif initial_verdict == VerifyResult.UNSAT:
                self.clean_prediction_stats['verification_unsat'] += 1
            else:
                self.clean_prediction_stats['verification_unknown'] += 1

            if initial_verdict == VerifyResult.UNSAT:
                ACTLog.log_verification_info(f"❌ Interval potential violation detected - Sample {idx+1}")
            else:
                ACTLog.log_verification_info(f"❓ Interval inconclusive - Sample {idx+1}")

            ACTLog.log_verification_info("Launching Specification Refinement BaB process")
            ACTLog.log_verification_info("="*60)

            if self.bab_config['enabled']:
                refinement_verdict = self._spec_refinement(sample_lb, sample_ub, idx)
                if refinement_verdict == VerifyResult.SAT:
                    self.clean_prediction_stats['verification_sat'] += 1
                elif refinement_verdict == VerifyResult.UNSAT:
                    self.clean_prediction_stats['verification_unsat'] += 1
                else:
                    self.clean_prediction_stats['verification_unknown'] += 1
                results.append(refinement_verdict)
            else:
                ACTLog.log_verification_info("⚠️  BaB disabled, returning initial verdict")
                results.append(initial_verdict)

        ACTStats.print_verification_stats(self.clean_prediction_stats)
        return ACTStats.print_final_verification_summary(results)

    # =============================================================================
    # CORE VERIFICATION ALGORITHMS
    # =============================================================================

    def _solve_constraints(self, input_lb: torch.Tensor, input_ub: torch.Tensor, sample_idx: int) -> VerifyResult:
        """
        Perform interval constraint solving for a single sample.
        
        Applies interval bound propagation through the network and evaluates results
        against verification constraints to determine if the sample is safe.
        
        Args:
            input_lb: Input lower bounds tensor for the sample
            input_ub: Input upper bounds tensor for the sample  
            sample_idx: Index of the current sample being processed
            
        Returns:
            VerifyResult indicating the verification result (SAT/UNSAT/UNKNOWN)
        """
        ACTLog.log_verification_info(f"Performing Interval propagation")

        output_lb, output_ub, _ = self._propagate_bounds(
            self.spec.model.pytorch_model, input_lb, input_ub
        )

        verdict = self._evaluate_output_bounds(
            output_lb, output_ub,
            self.spec.output_spec.output_constraints if self.spec.output_spec.output_constraints is not None else None,
            self.spec.output_spec.labels[sample_idx].item() if self.spec.output_spec.labels is not None else None
        )

        ACTLog.log_verification_info(f"📊 Verification verdict: {verdict.name}")
        return verdict

    def _propagate_bounds(self, model: nn.Module, input_lb: torch.Tensor, input_ub: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, None]:
        """
        Propagate interval bounds through network layers using interval arithmetic.
        
        Sequentially processes each layer applying interval arithmetic rules to maintain
        tight bounds while supporting ReLU constraints from BaB refinement.
        
        Args:
            model: PyTorch neural network model to analyze
            input_lb: Input lower bounds tensor
            input_ub: Input upper bounds tensor
            
        Returns:
            Tuple containing (output_lb, output_ub, None) for BaseVerifier compatibility
            
        Raises:
            NotImplementedError: If unsupported layer type encountered
            ValueError: If numerical instability or division by zero detected
            AttributeError: If required layer attributes missing
        """
        lb = input_lb.clone()
        ub = input_ub.clone()
        
        ACTLog.log_verification_info("Starting interval bound propagation through network layers")
        
        # Track ReLU layers separately for constraint application
        relu_count = 0
        
        for idx, layer in enumerate(model.children()):
            ACTLog.log_verification_info(f"Processing layer {idx}: {type(layer).__name__}")
            
            # Validate numerical stability before processing each layer
            if torch.any(torch.isnan(lb)) or torch.any(torch.isnan(ub)):
                raise ValueError(f"NaN values detected at layer {idx} - numerical instability")
            
            if isinstance(layer, nn.Linear):
                lb, ub = self._handle_linear(layer, lb, ub, idx)
            elif isinstance(layer, nn.Conv2d):
                lb, ub = self._handle_conv2d(layer, lb, ub, idx)
            elif isinstance(layer, nn.ReLU):
                lb, ub = self._handle_relu(layer, lb, ub, relu_count)
                relu_count += 1
            elif isinstance(layer, (nn.Sigmoid, nn.MaxPool2d)):
                lb, ub = self._handle_activation(layer, lb, ub)
            elif isinstance(layer, (nn.Flatten, OnnxFlatten, OnnxReshape, OnnxSqueeze, OnnxUnsqueeze, OnnxTranspose)):
                lb, ub = self._handle_structural(layer, lb, ub)
            elif isinstance(layer, nn.BatchNorm2d):
                lb, ub = self._handle_batchnorm(layer, lb, ub)
            elif isinstance(layer, (OnnxAdd, OnnxDiv, OnnxClip, OperatorWrapper)):
                lb, ub = self._handle_onnx_op(layer, lb, ub)
            else:
                raise NotImplementedError(f"Layer type {type(layer)} not supported in interval propagation")
            
            # Log progress periodically
            if idx % 10 == 0 or idx < 5:
                ACTLog.log_verification_info(f"Layer {idx} completed: bounds shape {lb.shape}")
        
        ACTLog.log_verification_info("Interval bound propagation completed successfully")
        return lb, ub, None

    # =============================================================================
    # LAYER HANDLING METHODS - NEURAL NETWORK LAYERS
    # =============================================================================

    def _handle_linear(self, layer: nn.Linear, lb: torch.Tensor, ub: torch.Tensor, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Handle linear layer using weight decomposition for tight interval arithmetic."""
        weight = layer.weight
        bias = layer.bias
        
        # Decompose weights: W = W+ + W- for optimal bound computation
        w_pos = torch.clamp(weight, min=0)
        w_neg = torch.clamp(weight, max=0)
        
        # Apply interval arithmetic: W+ uses same-sign bounds, W- uses opposite-sign bounds
        next_lb = torch.matmul(w_pos, lb) + torch.matmul(w_neg, ub)
        next_ub = torch.matmul(w_pos, ub) + torch.matmul(w_neg, lb)
        
        if bias is not None:
            next_lb += bias
            next_ub += bias
        
        ACTLog.log_verification_info(f"Linear layer processed: output shape {next_lb.shape}")
        return next_lb, next_ub

    def _handle_conv2d(self, layer: nn.Conv2d, lb: torch.Tensor, ub: torch.Tensor, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Handle 2D convolution layer with interval arithmetic using weight decomposition."""
        weight = layer.weight
        bias = layer.bias
        stride = layer.stride
        padding = layer.padding
        
        w_pos = torch.clamp(weight, min=0)
        w_neg = torch.clamp(weight, max=0)
        
        # Apply convolution with interval bounds (add batch dimension)
        next_lb = (
            nn.functional.conv2d(lb.unsqueeze(0), w_pos, None, stride, padding) +
            nn.functional.conv2d(ub.unsqueeze(0), w_neg, None, stride, padding)
        ).squeeze(0)
        
        next_ub = (
            nn.functional.conv2d(ub.unsqueeze(0), w_pos, None, stride, padding) +
            nn.functional.conv2d(lb.unsqueeze(0), w_neg, None, stride, padding)
        ).squeeze(0)
        
        if bias is not None:
            next_lb += bias.view(-1, 1, 1)
            next_ub += bias.view(-1, 1, 1)
        
        ACTLog.log_verification_info(f"Conv2d layer processed: output shape {next_lb.shape}")
        return next_lb, next_ub

    def _handle_relu(self, layer: nn.ReLU, lb: torch.Tensor, ub: torch.Tensor, relu_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Handle ReLU layer with max(0, x) transformation and optional BaB constraints."""
        layer_name = f"relu_{relu_idx}"
        constraints = []
        
        # Apply neuron-specific constraints from BaB refinement if available
        if hasattr(self, 'current_relu_constraints'):
            for constraint in self.current_relu_constraints:
                if constraint['layer'] == layer_name:
                    neuron_idx = constraint['neuron_idx']
                    constraint_type = constraint['constraint_type']
                    
                    if neuron_idx < lb.numel():
                        flat_lb = lb.view(-1)
                        flat_ub = ub.view(-1)
                        
                        if constraint_type == 'inactive':
                            # Force neuron output to 0: upper bound ≤ 0
                            flat_ub[neuron_idx] = min(flat_ub[neuron_idx], 0.0)
                            constraints.append(f"ReLU[{neuron_idx}]=inactive")
                        elif constraint_type == 'active':
                            # Force neuron to pass-through: lower bound ≥ 0
                            flat_lb[neuron_idx] = max(flat_lb[neuron_idx], 0.0)
                            constraints.append(f"ReLU[{neuron_idx}]=active")
                        
                        lb = flat_lb.view(lb.shape)
                        ub = flat_ub.view(ub.shape)
        
        if constraints:
            ACTLog.log_verification_info(f"Applied {layer_name} constraints: {constraints}")
        
        # Apply standard ReLU transformation: max(0, x)
        return torch.clamp(lb, min=0), torch.clamp(ub, min=0)

    def _handle_batchnorm(self, layer: nn.BatchNorm2d, lb: torch.Tensor, ub: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Handle batch normalization layer with running statistics."""
        mean = layer.running_mean
        var = layer.running_var
        eps = layer.eps
        weight = layer.weight
        bias = layer.bias
        
        # Apply batch norm transformation: (x - mean) / sqrt(var + eps) * weight + bias
        norm_factor = weight[None, :, None, None] / torch.sqrt(var[None, :, None, None] + eps)
        offset = bias[None, :, None, None] - mean[None, :, None, None] * norm_factor
        
        return lb * norm_factor + offset, ub * norm_factor + offset

    # =============================================================================
    # LAYER HANDLING METHODS - ACTIVATION FUNCTIONS
    # =============================================================================

    def _handle_activation(self, layer: nn.Module, lb: torch.Tensor, ub: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Handle activation layers with element-wise monotonic functions."""
        if isinstance(layer, nn.Sigmoid):
            return torch.sigmoid(lb), torch.sigmoid(ub)
        elif isinstance(layer, nn.MaxPool2d):
            return (
                nn.functional.max_pool2d(
                    lb, kernel_size=layer.kernel_size, 
                    stride=layer.stride, padding=layer.padding
                ),
                nn.functional.max_pool2d(
                    ub, kernel_size=layer.kernel_size,
                    stride=layer.stride, padding=layer.padding
                )
            )
        else:
            raise NotImplementedError(f"Activation layer {type(layer)} not supported")

    # =============================================================================
    # LAYER HANDLING METHODS - STRUCTURAL OPERATIONS
    # =============================================================================

    def _handle_structural(self, layer: nn.Module, lb: torch.Tensor, ub: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Handle structural layers that manipulate tensor shape without computation."""
        if isinstance(layer, (nn.Flatten, OnnxFlatten)):
            ACTLog.log_verification_info("Processing flatten layer")
            return torch.flatten(lb, start_dim=0), torch.flatten(ub, start_dim=0)
        
        elif isinstance(layer, OnnxReshape):
            # Extract shape dimensions from ONNX reshape layer
            shape = None
            for attr in ["shape", "target_shape", "_shape"]:
                if hasattr(layer, attr):
                    shape = getattr(layer, attr)
                    break
            
            if shape is None:
                raise AttributeError(f"Cannot find shape attribute in Reshape layer. Available: {dir(layer)}")
            
            return lb.reshape(list(shape)), ub.reshape(list(shape))
        
        elif isinstance(layer, OnnxSqueeze):
            dim = layer.dim
            return lb.squeeze(dim), ub.squeeze(dim)
        
        elif isinstance(layer, OnnxUnsqueeze):
            dim = layer.dim
            return lb.unsqueeze(dim), ub.unsqueeze(dim)
        
        elif isinstance(layer, OnnxTranspose):
            # Extract permutation dimensions from ONNX transpose layer
            perm = None
            for attr in ["perm", "dims"]:
                if hasattr(layer, attr):
                    perm = getattr(layer, attr)
                    break
            
            if perm is None:
                raise AttributeError(f"Cannot find permutation attribute in Transpose layer. Available: {dir(layer)}")
            
            return lb.permute(*perm), ub.permute(*perm)
        
        else:
            raise NotImplementedError(f"Structural layer {type(layer)} not supported")

    # =============================================================================
    # LAYER HANDLING METHODS - ONNX OPERATIONS
    # =============================================================================

    def _handle_onnx_op(self, layer: nn.Module, lb: torch.Tensor, ub: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Handle ONNX operation layers with proper interval arithmetic."""
        if isinstance(layer, OnnxAdd):
            return layer(lb), layer(ub)
        
        elif isinstance(layer, OnnxDiv):
            new_lb = layer(lb)
            new_ub = layer(ub)
            # Division can flip bound ordering depending on divisor sign
            return torch.min(new_lb, new_ub), torch.max(new_lb, new_ub)
        
        elif isinstance(layer, OnnxClip):
            min_val = layer.min
            max_val = layer.max
            return (
                torch.clamp(lb, min=min_val, max=max_val),
                torch.clamp(ub, min=min_val, max=max_val)
            )
        
        elif isinstance(layer, OperatorWrapper):
            return self._handle_operator(layer, lb, ub)
        
        else:
            raise NotImplementedError(f"ONNX operation layer {type(layer)} not supported")

    def _handle_operator(self, layer: OperatorWrapper, lb: torch.Tensor, ub: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Handle generic operator wrapper with interval arithmetic for constant operations."""
        if not (hasattr(layer, 'op_type') and layer.op_type in ["Add", "Sub", "Mul", "Div"]):
            # Generic operator without specific interval handling
            return layer(lb), layer(ub)
        
        other = getattr(layer, 'other', None)
        if other is None:
            # No constant operand - apply layer directly
            return layer(lb), layer(ub)
        
        # Apply interval arithmetic with constant operand
        if layer.op_type == "Add":
            return lb + other, ub + other
        elif layer.op_type == "Sub":
            return lb - other, ub - other
        elif layer.op_type == "Mul":
            # Multiplication can flip bounds depending on sign of constant
            new_lb = torch.min(lb * other, ub * other)
            new_ub = torch.max(lb * other, ub * other)
            return new_lb, new_ub
        elif layer.op_type == "Div":
            # Validate no division by zero
            if torch.any(other == 0):
                raise ValueError("Division by zero encountered in OperatorWrapper")
            # Division can flip bounds depending on sign of divisor
            new_lb = torch.min(lb / other, ub / other)
            new_ub = torch.max(lb / other, ub / other)
            return new_lb, new_ub

    # =============================================================================
    # OUTPUT EVALUATION METHODS
    # =============================================================================

    def _evaluate_output_bounds(self, lb: torch.Tensor, ub: torch.Tensor,
                             output_constraints: Optional[List[List[float]]],
                             true_label: Optional[int]) -> VerifyResult:
        """
        Evaluate output bounds to determine verification result.
        
        Analyzes computed interval bounds against either linear output constraints
        or classification robustness requirements to determine verification status.
        
        Args:
            lb: Lower bounds tensor for network outputs
            ub: Upper bounds tensor for network outputs
            output_constraints: Linear constraint matrix (optional)
            true_label: Ground truth class index for classification (optional)
            
        Returns:
            VerifyResult indicating SAT (safe), UNSAT (unsafe), or UNKNOWN
        """
        if output_constraints is not None:
            return self._eval_linear_constraints(lb, ub, output_constraints)
        
        if true_label is not None:
            return self._eval_classification(lb, ub, true_label)
        
        return VerifyResult.UNKNOWN

    def _eval_linear_constraints(self, lb: torch.Tensor, ub: torch.Tensor, 
                               constraints: List[List[float]]) -> VerifyResult:
        """Evaluate linear output constraints using interval arithmetic."""
        # Check each linear constraint: coefficients·outputs + bias ≥ 0
        for row in constraints:
            # Split constraint: [coeff1, coeff2, ..., coeffN, bias]
            coeffs = torch.tensor(row[:-1], device=lb.device)  # All except last
            bias = row[-1]  # Last element
            
            # Interval arithmetic: find worst-case (minimum) constraint value
            # Positive coefficients use lower bounds, negative coefficients use upper bounds
            terms = torch.where(coeffs >= 0, coeffs * lb, coeffs * ub)
            worst_case = torch.sum(terms) + bias
            
            # If worst case is negative, constraint is definitely violated
            if worst_case < 0:
                return VerifyResult.UNSAT
        
        # All constraints passed worst-case check - potentially satisfiable
        return VerifyResult.SAT

    def _eval_classification(self, lb: torch.Tensor, ub: torch.Tensor, 
                           true_label: int) -> VerifyResult:
        """Evaluate classification robustness using interval bounds comparison."""
        ACTLog.log_verification_info(f"Checking classification robustness for true_label: {true_label}")
        
        # Validate true_label is within valid output class range
        num_classes = lb.shape[0]
        if true_label < 0 or true_label >= num_classes:
            ACTLog.log_verification_warning(f"true_label {true_label} out of bounds (valid range: 0-{num_classes-1})")
            return VerifyResult.UNKNOWN

        # Check for numerical instability in output bounds
        if torch.any(torch.isnan(lb)) or torch.any(torch.isnan(ub)):
            ACTLog.log_verification_warning("Found NaN values in output bounds - numerical instability detected")
            return VerifyResult.UNKNOWN

        # Core adversarial robustness check: verify no other class can exceed true class
        ACTLog.log_verification_info("Performing adversarial robustness verification via interval comparison")
        correct_min = lb[true_label]  # Minimum possible output for correct class
        
        for other_idx in range(num_classes):
            if other_idx != true_label:
                other_max = ub[other_idx]  # Maximum possible output for other class
                
                # If any other class can achieve higher output than true class minimum,
                # then adversarial examples exist within the input perturbation region
                if other_max >= correct_min:
                    ACTLog.log_verification_info(f"Adversarial vulnerability detected: class {other_idx} "
                                               f"(max: {other_max:.4f}) can exceed true class {true_label} "
                                               f"(min: {correct_min:.4f})")
                    return VerifyResult.UNSAT
        
        # All other classes have maximum outputs below true class minimum - robustness proven
        ACTLog.log_verification_info("Classification robustness verified: no adversarial examples possible")
        return VerifyResult.SAT
