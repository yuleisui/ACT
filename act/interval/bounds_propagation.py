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

from act.util.stats import ACTLog
from act.util.device import DeviceManager
from act.util.bounds import Bounds, WeightDecomposer
from act.interval.bounds_prop_helper import (
    BoundsPropMetadata, 
    BoundsPropagationMetadata, 
    TrackingMode,
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
    """
    
    def __init__(self, relu_constraints: Optional[list] = None, mode: TrackingMode = TrackingMode.DEBUG):
        """
        Initialize the interval bound propagator.
        
        Args:
            relu_constraints: Optional list of ReLU constraints from BaB refinement
            mode: Tracking mode controlling logging, validation, and metadata collection
                - TrackingMode.PRODUCTION: Minimal overhead for production use
                - TrackingMode.PERFORMANCE: Fast with metadata tracking  
                - TrackingMode.DEBUG: Full logging and validation (default)
        """
        # Always create a defensive copy to prevent external modification
        self.current_relu_constraints = (relu_constraints or []).copy()
        
        # Initialize the metadata tracker with clean single-mode design
        self.metadata_tracker = BoundsPropMetadata(mode)
        
        # Always keep device manager available for essential functionality
        self.device_manager = DeviceManager()
        
        # Track ReLU layers for constraint application  
        self.relu_layer_index = 0
        
        # Weight decomposition cache for performance optimization
        self.weight_decomposer = WeightDecomposer()
    
    def propagate_bounds(self, model: nn.Module, input_bounds: Bounds) -> Bounds:
        """
        Propagate interval bounds through the neural network model.
        
        Args:
            model: Neural network model to propagate bounds through
            input_bounds: Input bounds object containing lower and upper bound tensors
            
        Returns:
            Bounds object containing output lower and upper bounds
        """

        # Input validation handled by metadata tracker
        self.metadata_tracker.validate_input_bounds(input_bounds.lb, input_bounds.ub)
        
        # Initialize tracking and validation
        self.metadata_tracker.start_propagation()
        
        # Device consistency - metadata tracker handles performance optimizations
        input_lb_clean, input_ub_clean = self.device_manager.ensure_device_consistency(model, input_bounds.lb, input_bounds.ub)
        
        # Create initial bounds object - use _internal=True since input validation already done above
        bounds = Bounds(input_lb_clean, input_ub_clean, _internal=True)
        
        # Logging handled by metadata tracker performance settings
        self.metadata_tracker.log_if_enabled("Starting interval bound propagation through network layers")
        
        # Reset ReLU layer index for this propagation
        self.relu_layer_index = 0
        
        for idx, layer in enumerate(model.children()):
            # Logging handled by metadata tracker
            self.metadata_tracker.log_if_enabled(f"Processing layer {idx}: {type(layer).__name__}")
            
            try:
                # Validation and processing handled by metadata tracker
                self.metadata_tracker.validate_bounds_essential(bounds.lb, bounds.ub, idx)
                self.metadata_tracker.process_layer(layer, idx, bounds.lb, bounds.ub)
                
                # Fast path layer processing with optimized dispatching
                if isinstance(layer, nn.Linear):
                    bounds = self._handle_linear(layer, bounds, idx)
                elif isinstance(layer, nn.Conv2d):
                    bounds = self._handle_conv2d(layer, bounds, idx)
                elif isinstance(layer, (nn.ReLU, nn.Sigmoid, nn.MaxPool2d)):
                    bounds = self._handle_activation(layer, bounds)
                elif isinstance(layer, (nn.Flatten, OnnxFlatten, OnnxReshape, OnnxSqueeze, OnnxUnsqueeze, OnnxTranspose)):
                    bounds = self._handle_structural(layer, bounds)
                elif isinstance(layer, nn.BatchNorm2d):
                    bounds = self._handle_batchnorm(layer, bounds)
                elif isinstance(layer, (OnnxAdd, OnnxDiv, OnnxClip, OperatorWrapper)):
                    bounds = self._handle_onnx_op(layer, bounds)
                else:
                    raise UnsupportedLayerError(f"Layer type {type(layer)} not supported in interval propagation")
                
                # Validation and metadata tracking handled by metadata tracker
                # Validate layer output once for all layer types (determines category internally)
                self.metadata_tracker.validate_layer_output(bounds.lb, bounds.ub, idx, layer)
                # Finalize layer processing for timing and memory tracking
                self.metadata_tracker.finalize_layer_processing(idx)
                    
            except (NumericalInstabilityError, InvalidBoundsError) as e:
                self.metadata_tracker.track_numerical_warning(f"Layer {idx} error: {e}")
                raise
            except Exception as e:
                raise UnsupportedLayerError(f"Failed to process layer {idx} ({type(layer).__name__}): {e}") from e
            
            # Progress logging handled by metadata tracker
            if idx % 10 == 0 or idx < 5:
                self.metadata_tracker.log_if_enabled(f"Layer {idx} completed: bounds shape {bounds.shape}")
            
            # Track layer count for metadata
            layer_count = idx + 1
        
        ACTLog.log_verification_info("Interval bound propagation completed successfully")
        
        # Finalize propagation and collect comprehensive metadata
        metadata = self.metadata_tracker.finalize_propagation(bounds.lb)
        return bounds

    # =============================================================================
    # LAYER HANDLING METHODS - NEURAL NETWORK LAYERS
    # =============================================================================

    def _handle_linear(self, layer: nn.Linear, bounds: Bounds, idx: int) -> Bounds:
        """Handle linear layer with performance optimizations."""
        if self.metadata_tracker.mode.enable_logging:
            ACTLog.log_verification_info(f"Linear layer {idx}: input {layer.in_features} -> {layer.out_features}")
        
        # Handle bounds flattening if needed
        if hasattr(bounds, 'flatten') and len(bounds.shape) > 2:
            bounds = bounds.flatten(_internal=True)  # Essential operation
        
        # Tracking handled by metadata tracker if available
        if hasattr(self.metadata_tracker, 'track_linear_layer'):
            self.metadata_tracker.track_linear_layer(layer, bounds.lb, bounds.ub, idx)
        
        # Apply linear transformation using interval arithmetic
        weight = layer.weight
        bias = layer.bias
        
        # Use cached weight decomposition for performance
        w_pos, w_neg = self.weight_decomposer.decompose(weight)
        
        # Cache attribute access for performance
        lb = bounds.lb
        ub = bounds.ub
        
        # Apply interval arithmetic: W+ uses same-sign bounds, W- uses opposite-sign bounds
        new_lb = torch.matmul(w_pos, lb) + torch.matmul(w_neg, ub)
        new_ub = torch.matmul(w_pos, ub) + torch.matmul(w_neg, lb)
        
        if bias is not None:
            new_lb += bias
            new_ub += bias
        
        result = Bounds(new_lb, new_ub, _internal=True)
        
        if self.metadata_tracker.mode.enable_logging:
            ACTLog.log_verification_info(f"Linear layer {idx} output shape: {result.shape}")
        
        return result

    def _handle_conv2d(self, layer: nn.Conv2d, bounds: Bounds, idx: int) -> Bounds:
        """Handle Conv2d layer with performance optimizations."""
        if self.metadata_tracker.mode.enable_logging:
            ACTLog.log_verification_info(f"Conv2d layer {idx}: channels {layer.in_channels} -> {layer.out_channels}")
        
        # Tracking handled by metadata tracker if available
        if hasattr(self.metadata_tracker, 'track_conv_layer'):
            self.metadata_tracker.track_conv_layer(layer, bounds.lb, bounds.ub, idx)
        
        # Apply convolution transformation using interval arithmetic
        # Use cached weight decomposition for performance
        w_pos, w_neg = self.weight_decomposer.decompose(layer.weight)
        
        # Inline convolution with interval arithmetic
        weight, bias = layer.weight, layer.bias
        stride, padding = layer.stride, layer.padding
        
        # Ensure input has batch dimension for conv2d
        lb_input = bounds.lb if bounds.lb.dim() == 4 else bounds.lb.unsqueeze(0)
        ub_input = bounds.ub if bounds.ub.dim() == 4 else bounds.ub.unsqueeze(0)
        
        # Apply convolution with interval bounds
        new_lb = (
            nn.functional.conv2d(lb_input, w_pos, None, stride, padding) +
            nn.functional.conv2d(ub_input, w_neg, None, stride, padding)
        )
        
        new_ub = (
            nn.functional.conv2d(ub_input, w_pos, None, stride, padding) +
            nn.functional.conv2d(lb_input, w_neg, None, stride, padding)
        )
        
        # Remove batch dimension if it was added
        if bounds.lb.dim() == 3:
            new_lb = new_lb.squeeze(0)
            new_ub = new_ub.squeeze(0)
        
        if bias is not None:
            if new_lb.dim() == 4:  # Batch dimension present
                bias_shape = [1, -1] + [1] * (new_lb.dim() - 2)
            else:  # No batch dimension
                bias_shape = [-1] + [1] * (new_lb.dim() - 1)
            new_lb += bias.view(*bias_shape)
            new_ub += bias.view(*bias_shape)
            
        result = Bounds(new_lb, new_ub, _internal=True)
        
        if self.metadata_tracker.mode.enable_logging:
            ACTLog.log_verification_info(f"Conv2d layer {idx} output shape: {result.shape}")
        
        return result

    def _handle_batchnorm(self, layer: nn.BatchNorm2d, bounds: Bounds) -> Bounds:
        """Handle batch normalization layer with running statistics."""
        # Inline batch normalization with interval arithmetic
        running_mean, running_var = layer.running_mean, layer.running_var
        weight, bias, eps = layer.weight, layer.bias, layer.eps
        
        # Apply batch norm transformation with numerical stability check
        sqrt_var_eps = torch.sqrt(running_var[None, :, None, None] + eps)
        
        # Validate numerical stability
        if torch.any(sqrt_var_eps < 1e-10):
            raise ValueError("BatchNorm variance too small, potential division by zero")
        
        norm_factor = weight[None, :, None, None] / sqrt_var_eps
        offset = bias[None, :, None, None] - running_mean[None, :, None, None] * norm_factor
        
        return Bounds(
            bounds.lb * norm_factor + offset,
            bounds.ub * norm_factor + offset,
            _internal=True
        )

    # =============================================================================
    # LAYER HANDLING METHODS - ACTIVATION FUNCTIONS
    # =============================================================================

    def _handle_activation(self, layer: nn.Module, bounds: Bounds) -> Bounds:
        """Handle activation layers with element-wise monotonic functions and BaB constraints."""
        if isinstance(layer, nn.ReLU):
            # Handle ReLU layer with max(0, x) transformation and optional BaB constraints
            layer_name = f"relu_{self.relu_layer_index}"
            
            # Apply neuron-specific constraints from BaB refinement if available
            if hasattr(self, 'current_relu_constraints') and self.current_relu_constraints:
                constrained_bounds = bounds.apply_relu_constraints(self.current_relu_constraints, layer_name)
                
                # Constraint logging handled by metadata tracker
                if self.metadata_tracker.mode.enable_logging and hasattr(constrained_bounds, '_applied_constraints') and constrained_bounds._applied_constraints:
                    ACTLog.log_verification_info(f"Applied {layer_name} constraints: {constrained_bounds._applied_constraints}")
                
                # Apply ReLU transformation to constrained bounds
                result_bounds = constrained_bounds.clamp_relu(_internal=True)
            else:
                # No constraints - apply standard ReLU
                result_bounds = bounds.clamp_relu(_internal=True)
            
            # Increment ReLU layer index and track constraints application
            self.relu_layer_index += 1
            if hasattr(self.metadata_tracker, 'track_constraint_application') and self.relu_layer_index <= len(self.current_relu_constraints):
                self.metadata_tracker.track_constraint_application()
            
            return result_bounds
            
        elif isinstance(layer, nn.Sigmoid):
            return bounds.apply_sigmoid(_internal=True)
        elif isinstance(layer, nn.MaxPool2d):
            # Inline max pooling operation
            kernel_size, stride, padding = layer.kernel_size, layer.stride, layer.padding
            return Bounds(
                nn.functional.max_pool2d(bounds.lb, kernel_size, stride, padding),
                nn.functional.max_pool2d(bounds.ub, kernel_size, stride, padding),
                _internal=True
            )
        else:
            raise NotImplementedError(f"Activation layer {type(layer)} not supported")

    # =============================================================================
    # LAYER HANDLING METHODS - STRUCTURAL OPERATIONS
    # =============================================================================

    def _handle_structural(self, layer: nn.Module, bounds: Bounds) -> Bounds:
        """Handle structural layers that manipulate tensor shape without computation."""
        if isinstance(layer, (nn.Flatten, OnnxFlatten)):
            self.metadata_tracker.log_if_enabled("Processing flatten layer")
            return bounds.flatten(start_dim=0, _internal=True)
        
        elif isinstance(layer, OnnxReshape):
            # Extract shape dimensions from ONNX reshape layer
            shape = None
            for attr in ["shape", "target_shape", "_shape"]:
                if hasattr(layer, attr):
                    shape = getattr(layer, attr)
                    break
            
            if shape is None:
                raise AttributeError(f"Cannot find shape attribute in Reshape layer. Available: {dir(layer)}")
            
            return bounds.reshape(list(shape), _internal=True)
        
        elif isinstance(layer, OnnxSqueeze):
            return bounds.squeeze(layer.dim, _internal=True)
        
        elif isinstance(layer, OnnxUnsqueeze):
            return bounds.unsqueeze(layer.dim, _internal=True)
        
        elif isinstance(layer, OnnxTranspose):
            # Extract permutation dimensions from ONNX transpose layer
            perm = None
            for attr in ["perm", "dims"]:
                if hasattr(layer, attr):
                    perm = getattr(layer, attr)
                    break
            
            if perm is None:
                raise AttributeError(f"Cannot find permutation attribute in Transpose layer. Available: {dir(layer)}")
            
            return bounds.permute(*perm, _internal=True)
        
        else:
            raise NotImplementedError(f"Structural layer {type(layer)} not supported")

    # =============================================================================
    # LAYER HANDLING METHODS - ONNX OPERATIONS
    # =============================================================================

    def _handle_onnx_op(self, layer: nn.Module, bounds: Bounds) -> Bounds:
        """Handle ONNX operation layers with proper interval arithmetic."""
        if isinstance(layer, OnnxAdd):
            return Bounds(layer(bounds.lb), layer(bounds.ub), _internal=True)
        
        elif isinstance(layer, OnnxDiv):
            new_lb = layer(bounds.lb)
            new_ub = layer(bounds.ub)
            # Division can flip bound ordering depending on divisor sign
            return Bounds(torch.min(new_lb, new_ub), torch.max(new_lb, new_ub), _internal=True)
        
        elif isinstance(layer, OnnxClip):
            min_val = layer.min
            max_val = layer.max
            return Bounds(
                torch.clamp(bounds.lb, min=min_val, max=max_val),
                torch.clamp(bounds.ub, min=min_val, max=max_val),
                _internal=True
            )
        
        elif isinstance(layer, OperatorWrapper):
            return self._handle_operator(layer, bounds)
        
        else:
            raise NotImplementedError(f"ONNX operation layer {type(layer)} not supported")

    def _handle_operator(self, layer: OperatorWrapper, bounds: Bounds) -> Bounds:
        """Handle generic operator wrapper with interval arithmetic for constant operations."""
        if not (hasattr(layer, 'op_type') and layer.op_type in ["Add", "Sub", "Mul", "Div"]):
            # Generic operator without specific interval handling
            return Bounds(layer(bounds.lb), layer(bounds.ub), _internal=True)
        
        other = getattr(layer, 'other', None)
        if other is None:
            # No constant operand - apply layer directly
            return Bounds(layer(bounds.lb), layer(bounds.ub), _internal=True)
        
        # Use the new Bounds class methods for operator arithmetic
        return bounds.apply_operator(layer.op_type, other)