#===- util.bounds.py interval bounds data structure ----------------------#
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
# Bounds data structure for interval arithmetic in neural network verification.
# Encapsulates lower and upper bounds with validation and common operations.
#
#===----------------------------------------------------------------------===#

import torch
import torch.nn as nn
from typing import Tuple, Optional, Union
from torch import Tensor


class WeightDecomposer:
    """
    Utility class for caching weight decompositions to improve performance.
    
    Caches positive and negative weight components to avoid recomputation
    in repeated interval arithmetic operations. Includes automatic memory management.
    """
    
    def __init__(self, max_cache_size: int = 100, max_memory_mb: float = 50.0):
        """
        Initialize weight decomposer with cache limits.
        
        Args:
            max_cache_size: Maximum number of cached decompositions
            max_memory_mb: Maximum memory usage in MB before cache cleanup
        """
        self._cache = {}
        self.max_cache_size = max_cache_size
        self.max_memory_mb = max_memory_mb
    
    def decompose(self, weight: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Get or compute weight decomposition W = W+ + W-.
        
        Args:
            weight: Weight tensor to decompose
            
        Returns:
            Tuple of (w_pos, w_neg) where w_pos = max(W, 0) and w_neg = min(W, 0)
        """
        # Use tensor id as cache key (assumes weights don't change)
        cache_key = id(weight)
        
        if cache_key not in self._cache:
            # Check if we need to clean cache before adding new entry
            self._check_cache_limits()
            
            w_pos = torch.clamp(weight, min=0)
            w_neg = torch.clamp(weight, max=0)
            self._cache[cache_key] = (w_pos, w_neg)
        
        return self._cache[cache_key]
    
    def _check_cache_limits(self) -> None:
        """Check cache limits and cleanup if necessary."""
        # Check cache size limit
        if len(self._cache) >= self.max_cache_size:
            self.clear_cache()
            return
        
        # Check memory limit
        if self.get_memory_usage() > self.max_memory_mb:
            self.clear_cache()
    
    def clear_cache(self):
        """Clear the decomposition cache."""
        self._cache.clear()
    
    def get_cache_size(self) -> int:
        """Get the number of cached weight decompositions."""
        return len(self._cache)
    
    def get_memory_usage(self) -> float:
        """Estimate memory usage of cached decompositions in MB."""
        total_elements = 0
        for w_pos, w_neg in self._cache.values():
            total_elements += w_pos.numel() + w_neg.numel()
        # Estimate: 4 bytes per float32 element
        return (total_elements * 4) / (1024 * 1024)


class InvalidBoundsError(Exception):
    """Exception raised when bounds are invalid (lb > ub)."""
    pass


class Bounds:
    """
    Interval bounds container for neural network verification.
    
    Encapsulates lower and upper bound tensors with validation and common
    interval arithmetic operations for improved code readability and maintainability.
    """
    
    __slots__ = ['lb', 'ub']  # Memory optimization
    
    def __init__(self, lb: Tensor, ub: Tensor, _internal: bool = False):
        """
        Initialize bounds with lower and upper bound tensors.
        
        Args:
            lb: Lower bound tensor
            ub: Upper bound tensor
            _internal: Internal flag to skip validation for trusted operations
            
        Raises:
            InvalidBoundsError: If lb > ub (only when _internal=False)
        """
        self.lb = lb
        self.ub = ub
        
        # Only validate for external/public API calls, skip for internal operations
        if not _internal:
            self._validate()
    
    def _validate(self) -> None:
        """Validate that lower bounds are <= upper bounds."""
        if torch.any(self.lb > self.ub):
            raise InvalidBoundsError("Lower bounds must be <= upper bounds")
    
    @property
    def shape(self) -> torch.Size:
        """Return the shape of the bounds tensors."""
        return self.lb.shape
    
    @property
    def device(self) -> torch.device:
        """Return the device of the bounds tensors."""
        return self.lb.device
    
    @property
    def dtype(self) -> torch.dtype:
        """Return the dtype of the bounds tensors."""
        return self.lb.dtype
    
    # =============================================================================
    # INTERVAL ARITHMETIC OPERATIONS
    # =============================================================================
    
    def clamp_relu(self, _internal: bool = False) -> 'Bounds':
        """Apply ReLU activation: max(0, x)."""
        return Bounds(
            torch.clamp(self.lb, min=0),
            torch.clamp(self.ub, min=0),
            _internal=True
        )
    
    def flatten(self, start_dim: int = 0, end_dim: int = -1, _internal: bool = False) -> 'Bounds':
        """
        Flatten bounds tensors.
        
        Args:
            start_dim: First dim to flatten (default: 0)
            end_dim: Last dim to flatten (default: -1, which means last dim)
            
        Returns:
            New Bounds object with flattened tensors
        """
        return Bounds(
            torch.flatten(self.lb, start_dim=start_dim, end_dim=end_dim),
            torch.flatten(self.ub, start_dim=start_dim, end_dim=end_dim),
            _internal=True
        )
    
    def apply_sigmoid(self, _internal: bool = False) -> 'Bounds':
        """Apply sigmoid activation: 1 / (1 + exp(-x))."""
        return Bounds(
            torch.sigmoid(self.lb),
            torch.sigmoid(self.ub),
            _internal=True
        )
    
    def apply_shape_transform(self, operation: str, **kwargs) -> 'Bounds':
        """
        Apply shape transformation operations to bounds tensors.
        
        Args:
            operation: Transform type ('reshape', 'squeeze', 'unsqueeze', 'permute', 'flatten')
            **kwargs: Operation-specific arguments
                - reshape: shape (list)
                - squeeze/unsqueeze: dim (int)
                - permute: dims (tuple/list)
                - flatten: start_dim (int, default=1), end_dim (int, default=-1)
        
        Returns:
            New Bounds object with transformed tensors
        """
        if operation == 'reshape':
            shape = kwargs['shape']
            return Bounds(
                self.lb.reshape(shape),
                self.ub.reshape(shape),
                _internal=True
            )
        elif operation == 'squeeze':
            dim = kwargs['dim']
            return Bounds(
                self.lb.squeeze(dim),
                self.ub.squeeze(dim),
                _internal=True
            )
        elif operation == 'unsqueeze':
            dim = kwargs['dim']
            return Bounds(
                self.lb.unsqueeze(dim),
                self.ub.unsqueeze(dim),
                _internal=True
            )
        elif operation == 'permute':
            dims = kwargs['dims']
            return Bounds(
                self.lb.permute(*dims),
                self.ub.permute(*dims),
                _internal=True
            )
        elif operation == 'flatten':
            start_dim = kwargs.get('start_dim', 1)
            end_dim = kwargs.get('end_dim', -1)
            return Bounds(
                torch.flatten(self.lb, start_dim=start_dim, end_dim=end_dim),
                torch.flatten(self.ub, start_dim=start_dim, end_dim=end_dim),
                _internal=True
            )
        else:
            raise ValueError(f"Unsupported shape transformation: {operation}")
    
    def reshape(self, shape: list, _internal: bool = False) -> 'Bounds':
        """Reshape bounds tensors to specified shape."""
        return self.apply_shape_transform('reshape', shape=shape)
    
    def squeeze(self, dim: int, _internal: bool = False) -> 'Bounds':
        """Squeeze bounds tensors along specified dimension."""
        return self.apply_shape_transform('squeeze', dim=dim)
    
    def unsqueeze(self, dim: int, _internal: bool = False) -> 'Bounds':
        """Unsqueeze bounds tensors along specified dimension."""
        return self.apply_shape_transform('unsqueeze', dim=dim)
    
    def permute(self, *dims, _internal: bool = False) -> 'Bounds':
        """Permute bounds tensors dimensions."""
        return self.apply_shape_transform('permute', dims=dims)
    
    def apply_relu_constraints(self, relu_constraints: list, layer_name: str) -> 'Bounds':
        """Apply ReLU constraints from BaB refinement (placeholder implementation)."""
        # For now, return self - this should be implemented based on the constraint format
        # This is a placeholder to maintain API compatibility
        return self
    
    # =============================================================================
    # OPERATOR WRAPPER ARITHMETIC OPERATIONS
    # =============================================================================
    
    def apply_operator(self, op_type: str, other: Union[float, torch.Tensor]) -> 'Bounds':
        """Apply arithmetic operator with constant operand using interval arithmetic."""
        # Handle division by zero validation
        if op_type == "Div":
            if isinstance(other, torch.Tensor):
                if torch.any(other == 0):
                    raise ValueError("Division by zero encountered in bounds division")
            elif other == 0:
                raise ValueError("Division by zero encountered in bounds division")
        
        # Apply operation to both bounds
        if op_type == "Add":
            new_lb, new_ub = self.lb + other, self.ub + other
        elif op_type == "Sub":
            new_lb, new_ub = self.lb - other, self.ub - other
        elif op_type in ["Mul", "Div"]:
            # Multiplication/division can flip bounds depending on operand sign
            # Use in-place operations for better performance
            op_func = torch.mul if op_type == "Mul" else torch.div
            lb_result = op_func(self.lb, other)
            ub_result = op_func(self.ub, other)
            new_lb = torch.minimum(lb_result, ub_result)
            new_ub = torch.maximum(lb_result, ub_result)
        else:
            raise ValueError(f"Unsupported operator type: {op_type}")
        
        return Bounds(new_lb, new_ub, _internal=True)

    def __repr__(self) -> str:
        """String representation of bounds."""
        return f"Bounds(shape={self.shape}, device={self.device})"
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        lb_range = f"[{self.lb.min().item():.4f}, {self.lb.max().item():.4f}]"
        ub_range = f"[{self.ub.min().item():.4f}, {self.ub.max().item():.4f}]"
        return f"Bounds(lb_range={lb_range}, ub_range={ub_range}, shape={self.shape})"