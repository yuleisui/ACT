#===- verifier.interval.bounds_propagation.py interval bounds propagation --#
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
    
    def __init__(self, relu_constraints: Optional[list] = None):
        """
        Initialize the interval bound propagator.
        
        Args:
            relu_constraints: Optional list of ReLU constraints from BaB refinement
        """
        self.current_relu_constraints = relu_constraints if relu_constraints else []
    
    def propagate_bounds(self, model: nn.Module, input_lb: torch.Tensor, input_ub: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, None]:
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