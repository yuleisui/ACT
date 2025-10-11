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
    in repeated interval arithmetic operations.
    """
    
    def __init__(self):
        self._cache = {}
    
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
            w_pos = torch.clamp(weight, min=0)
            w_neg = torch.clamp(weight, max=0)
            self._cache[cache_key] = (w_pos, w_neg)
        
        return self._cache[cache_key]
    
    def clear_cache(self):
        """Clear the decomposition cache."""
        self._cache.clear()


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
    
    def __init__(self, lb: Tensor, ub: Tensor, validate: bool = True):
        """
        Initialize bounds with lower and upper bound tensors.
        
        Args:
            lb: Lower bound tensor
            ub: Upper bound tensor  
            validate: Whether to validate that lb <= ub (default: True)
            
        Raises:
            InvalidBoundsError: If lb > ub when validation is enabled
        """
        self.lb = lb
        self.ub = ub
        
        if validate:
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
    
    def to(self, device: torch.device) -> 'Bounds':
        """Move bounds to specified device."""
        return Bounds(self.lb.to(device), self.ub.to(device), validate=False)
    
    def clone(self) -> 'Bounds':
        """Create a deep copy of the bounds."""
        return Bounds(self.lb.clone(), self.ub.clone(), validate=False)
    
    def detach(self) -> 'Bounds':
        """Detach bounds from computation graph."""
        return Bounds(self.lb.detach(), self.ub.detach(), validate=False)
    
    # =============================================================================
    # INTERVAL ARITHMETIC OPERATIONS
    # =============================================================================
    
    def clamp_relu(self) -> 'Bounds':
        """Apply ReLU activation: max(0, x)."""
        return Bounds(
            torch.clamp(self.lb, min=0),
            torch.clamp(self.ub, min=0),
            validate=False
        )
    
    def flatten(self, start_dim: int = 0, end_dim: int = -1) -> 'Bounds':
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
            validate=False
        )
    
    def __repr__(self) -> str:
        """String representation of bounds."""
        return f"Bounds(shape={self.shape}, device={self.device})"
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        lb_range = f"[{self.lb.min().item():.4f}, {self.lb.max().item():.4f}]"
        ub_range = f"[{self.ub.min().item():.4f}, {self.ub.max().item():.4f}]"
        return f"Bounds(lb_range={lb_range}, ub_range={ub_range}, shape={self.shape})"