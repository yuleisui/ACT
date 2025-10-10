#===- act.interval.bounds_propagation.py interval bounds propagation --#
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
# Interval bounds propagation through neural network layers with comprehensive
# layer support. Implements tight interval arithmetic for linear layers,
# convolutions, activations, and ONNX operations with BaB constraint integration.
#
#===----------------------------------------------------------------------===#

import torch
import torch.nn as nn
from typing import Tuple, Optional

from util.stats import ACTLog
from util.device import DeviceManager
from interval.bounds_prop_helper import (
    BoundsPropMetadata, 
    BoundsPropagationMetadata, 
    IntervalPropagationError,
    NumericalInstabilityError,
    InvalidBoundsError,
    UnsupportedLayerError
)
from onnx2pytorch.operations.flatten import Flatten as OnnxFlatten
from onnx2pytorch.operations.add import Add as OnnxAdd
from onnx2pytorch.operations.div import Div as OnnxDiv
from onnx2pytorch.operations.clip import Clip as OnnxClip
from onnx2pytorch.operations.reshape import Reshape as OnnxReshape
from onnx2pytorch.operations.squeeze import Squeeze as OnnxSqueeze
from onnx2pytorch.operations.unsqueeze import Unsqueeze as OnnxUnsqueeze
from onnx2pytorch.operations.transpose import Transpose as OnnxTranspose
from onnx2pytorch.operations.base import OperatorWrapper


class BoundsPropagate:
    """
    Interval bound propagation through neural network layers using interval arithmetic.
    
    Supports comprehensive layer types including linear, convolutional, activation,
    structural, and ONNX operations with tight interval arithmetic computations.
    """
    
    def __init__(self, relu_constraints: Optional[list] = None, enable_metadata_tracking: bool = True):
        """
        Initialize the interval bound propagator.
        
        Args:
            relu_constraints: Optional list of ReLU constraints from BaB refinement
            enable_metadata_tracking: Whether to enable metadata tracking (default: True)
        """
        # Always create a defensive copy to prevent external modification
        self.current_relu_constraints = (relu_constraints or []).copy()
        
        # Initialize the metadata tracker for bounds propagation tracking
        self.metadata_tracker = BoundsPropMetadata(enable_metadata_tracking)
        
        # Always keep device manager available for essential functionality
        self.device_manager = DeviceManager()
    
    def propagate_bounds(self, model: nn.Module, input_lb: torch.Tensor, input_ub: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, BoundsPropagationMetadata]:
        """
        Propagate interval bounds through the neural network model.
        
        This method performs forward propagation of interval bounds through each layer
        of the model, handling different layer types with appropriate interval arithmetic.
        
        Args:
            model: PyTorch neural network model
            input_lb: Lower bounds for input (same shape as model input)
            input_ub: Upper bounds for input (same shape as model input)
            
        Returns:
            Tuple of (output_lb, output_ub, metadata):
                - output_lb: Lower bounds of model output
                - output_ub: Upper bounds of model output  
                - metadata: Detailed propagation metadata from BoundsPropMetadata
                
        Raises:
            IntervalPropagationError: Base exception for propagation errors
            NumericalInstabilityError: When numerical instability detected
            InvalidBoundsError: When bounds become invalid or inconsistent
            UnsupportedLayerError: When encountering unsupported layer type
        """
        # Comprehensive input validation
        if input_lb.shape != input_ub.shape:
            raise InvalidBoundsError(f"Input bounds shape mismatch: lb={input_lb.shape}, ub={input_ub.shape}")
        
        if torch.any(input_lb > input_ub):
            raise InvalidBoundsError("Input lower bounds exceed upper bounds")
        
        if len(input_lb.shape) == 0:
            raise InvalidBoundsError("Input bounds must have at least one dimension")
        
        if torch.any(torch.isnan(input_lb)) or torch.any(torch.isnan(input_ub)):
            raise InvalidBoundsError("Input bounds contain NaN values")
        
        if torch.any(torch.isinf(input_lb)) or torch.any(torch.isinf(input_ub)):
            raise InvalidBoundsError("Input bounds contain infinite values")
        
        # Initialize tracking and validation
        self.metadata_tracker.start_propagation()
        
        # Ensure device consistency and create working copies (always needed)
        lb, ub = self.device_manager.ensure_device_consistency(model, input_lb.clone(), input_ub.clone())
        
        ACTLog.log_verification_info("Starting interval bound propagation through network layers")
        
        # Track ReLU layers separately for constraint application
        relu_count = 0
        
        for idx, layer in enumerate(model.children()):
            ACTLog.log_verification_info(f"Processing layer {idx}: {type(layer).__name__}")
            
            try:
                # Perform essential bounds validation and optional tracking
                self.metadata_tracker.validate_bounds_essential(lb, ub, idx)
                self.metadata_tracker.process_layer(layer, idx, lb, ub)
                
                if isinstance(layer, nn.Linear):
                    lb, ub = self._handle_linear(layer, lb, ub, idx)
                    self.metadata_tracker.validate_layer_output(lb, ub, idx, "Linear")
                elif isinstance(layer, nn.Conv2d):
                    lb, ub = self._handle_conv2d(layer, lb, ub, idx)
                    self.metadata_tracker.validate_layer_output(lb, ub, idx, "Conv2d")
                elif isinstance(layer, nn.ReLU):
                    lb, ub = self._handle_relu(layer, lb, ub, relu_count)
                    self.metadata_tracker.validate_layer_output(lb, ub, idx, "ReLU")
                    relu_count += 1
                    # Track ReLU constraints application
                    if relu_count <= len(self.current_relu_constraints):
                        self.metadata_tracker.track_constraint_application()
                elif isinstance(layer, (nn.Sigmoid, nn.MaxPool2d)):
                    lb, ub = self._handle_activation(layer, lb, ub)
                    self.metadata_tracker.validate_layer_output(lb, ub, idx, f"Activation({type(layer).__name__})")
                elif isinstance(layer, (nn.Flatten, OnnxFlatten, OnnxReshape, OnnxSqueeze, OnnxUnsqueeze, OnnxTranspose)):
                    lb, ub = self._handle_structural(layer, lb, ub)
                    self.metadata_tracker.validate_layer_output(lb, ub, idx, f"Structural({type(layer).__name__})")
                elif isinstance(layer, nn.BatchNorm2d):
                    lb, ub = self._handle_batchnorm(layer, lb, ub)
                    self.metadata_tracker.validate_layer_output(lb, ub, idx, "BatchNorm2d")
                elif isinstance(layer, (OnnxAdd, OnnxDiv, OnnxClip, OperatorWrapper)):
                    lb, ub = self._handle_onnx_op(layer, lb, ub)
                    self.metadata_tracker.validate_layer_output(lb, ub, idx, f"ONNX({type(layer).__name__})")
                else:
                    raise UnsupportedLayerError(f"Layer type {type(layer)} not supported in interval propagation")
                    
            except (NumericalInstabilityError, InvalidBoundsError) as e:
                self.metadata_tracker.track_numerical_warning(f"Layer {idx} error: {e}")
                raise
            except Exception as e:
                raise UnsupportedLayerError(f"Failed to process layer {idx} ({type(layer).__name__}): {e}") from e
            
            # Log progress periodically
            if idx % 10 == 0 or idx < 5:
                ACTLog.log_verification_info(f"Layer {idx} completed: bounds shape {lb.shape}")
        
        ACTLog.log_verification_info("Interval bound propagation completed successfully")
        
        # Finalize propagation and collect comprehensive metadata
        metadata = self.metadata_tracker.finalize_propagation(lb)
        return lb, ub, metadata

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
        if hasattr(self, 'current_relu_constraints') and self.current_relu_constraints:
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
        
        # Apply batch norm transformation with numerical stability check
        sqrt_var_eps = torch.sqrt(var[None, :, None, None] + eps)
        
        # Validate numerical stability
        if torch.any(sqrt_var_eps < 1e-10):
            raise NumericalInstabilityError("BatchNorm variance too small, potential division by zero")
        
        norm_factor = weight[None, :, None, None] / sqrt_var_eps
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