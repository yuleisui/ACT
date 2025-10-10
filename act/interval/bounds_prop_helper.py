#===- act.interval.bounds_prop_helper.py helper utilities --#
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
# Helper utilities for bounds propagation including metadata tracking,
# error handling, memory management, and device consistency.
#
#===----------------------------------------------------------------------===#

import time
import torch
import gc
from typing import List, Tuple
from dataclasses import dataclass

from util.stats import ACTLog
from util.device import DeviceManager, DeviceConsistencyError


# =============================================================================
# CUSTOM EXCEPTION CLASSES
# =============================================================================

class IntervalPropagationError(Exception):
    """Base exception for interval propagation errors."""
    pass


class NumericalInstabilityError(IntervalPropagationError):
    """Raised when numerical instability is detected during propagation."""
    pass


class InvalidBoundsError(IntervalPropagationError):
    """Raised when bounds become invalid (NaN, infinite, or inconsistent)."""
    pass


class UnsupportedLayerError(IntervalPropagationError):
    """Raised when an unsupported layer type is encountered."""
    pass


# =============================================================================
# METADATA DATACLASS
# =============================================================================

@dataclass
class BoundsPropagationMetadata:
    """Enhanced metadata from bounds propagation for BaseVerifier integration."""
    layers_processed: int
    constraints_applied: int
    memory_cleanups: int
    numerical_warnings: List[str]
    layer_types_processed: List[str]
    final_bounds_shape: Tuple[int, ...]
    processing_time_ms: float


# =============================================================================
# METADATA TRACKING CLASS
# =============================================================================

class BoundsPropMetadata:
    """
    Metadata tracking class for bounds propagation operations.
    
    This class provides a unified interface for metadata tracking, memory management,
    device consistency, and bounds validation with optional tracking control.
    """
    
    def __init__(self, enable_tracking: bool = True):
        """
        Initialize the metadata tracker with optional tracking control.
        
        Args:
            enable_tracking: Whether to enable metadata tracking (default: True)
        """
        self.enable_tracking = enable_tracking
        self._reset_metadata()
        self.device_manager = DeviceManager()
    
    def _reset_metadata(self):
        """Reset metadata tracking for a new propagation."""
        self._layers_processed = 0
        self._constraints_applied = 0
        self._memory_cleanups = 0
        self._numerical_warnings = []
        self._layer_types_processed = []
        self._start_time = None
    
    def start_propagation(self):
        """Initialize for a new propagation run."""
        if self.enable_tracking:
            self._reset_metadata()
            self._start_time = time.time()
            ACTLog.log_verification_info("Starting bounds propagation with metadata tracking")
        else:
            ACTLog.log_verification_info("Starting bounds propagation (metadata tracking disabled)")
    
    def process_layer(self, layer: torch.nn.Module, layer_idx: int, 
                     lb: torch.Tensor, ub: torch.Tensor) -> Tuple[str, List[str]]:
        """
        Process a layer with optional validation and tracking.
        
        Args:
            layer: PyTorch layer being processed
            layer_idx: Index of the layer
            lb: Lower bounds tensor
            ub: Upper bounds tensor
            
        Returns:
            Tuple of (layer_type, warnings) - warnings will be empty if tracking disabled
        """
        layer_type = type(layer).__name__
        
        # Always validate bounds for correctness (performed by caller)
        # self._validate_bounds(lb, ub, layer_idx)  # Moved to caller for efficiency
        
        stability_warnings = []
        if self.enable_tracking:
            # Check for stability issues only when tracking is enabled
            stability_warnings = self._check_bounds_stability(lb, ub)
            for warning in stability_warnings:
                self.track_numerical_warning(f"Layer {layer_idx}: {warning}")
            
            # Track layer processing
            self._track_layer_processing(layer_type)
            
            # Periodic memory cleanup
            if self._cleanup_memory(layer_idx):
                self._memory_cleanups += 1
        
        return layer_type, stability_warnings
    
    def _validate_bounds(self, lb: torch.Tensor, ub: torch.Tensor, layer_idx: int):
        """Validate interval bounds for numerical stability and consistency."""
        # Check for NaN values
        if torch.any(torch.isnan(lb)) or torch.any(torch.isnan(ub)):
            raise NumericalInstabilityError(
                f"NaN values detected in bounds at layer {layer_idx}"
            )
        
        # Check for infinite values
        if torch.any(torch.isinf(lb)) or torch.any(torch.isinf(ub)):
            raise NumericalInstabilityError(
                f"Infinite values detected in bounds at layer {layer_idx}"
            )
        
        # Check bounds consistency (lb <= ub)
        if torch.any(lb > ub):
            raise InvalidBoundsError(
                f"Invalid bounds at layer {layer_idx}: lower bounds exceed upper bounds"
            )
        
        # Check for extremely large values that might cause overflow
        max_val = max(torch.max(torch.abs(lb)).item(), torch.max(torch.abs(ub)).item())
        if max_val > 1e10:
            ACTLog.log_verification_info(
                f"Warning: Very large bounds detected at layer {layer_idx}: max={max_val:.2e}"
            )
    
    def _check_bounds_stability(self, lb: torch.Tensor, ub: torch.Tensor, 
                              threshold: float = 1e-8) -> List[str]:
        """Check bounds for potential numerical stability issues."""
        warnings = []
        
        # Check for very small intervals
        intervals = ub - lb
        small_intervals = torch.sum(intervals < threshold).item()
        if small_intervals > 0:
            warnings.append(f"{small_intervals} intervals smaller than {threshold}")
        
        # Check for zero intervals (degenerate cases)
        zero_intervals = torch.sum(intervals == 0).item()
        if zero_intervals > 0:
            warnings.append(f"{zero_intervals} zero-width intervals detected")
        
        return warnings
    
    def _cleanup_memory(self, idx: int, force: bool = False) -> bool:
        """Periodic memory cleanup during bounds propagation."""
        if force or (idx % 5 == 0 and torch.cuda.is_available()):
            torch.cuda.empty_cache()
            gc.collect()
            ACTLog.log_verification_info(f"GPU memory cleaned at layer {idx}")
            return True
        return False
    
    def _track_layer_processing(self, layer_type: str):
        """Track a layer being processed (only if tracking enabled)."""
        if self.enable_tracking:
            self._layer_types_processed.append(layer_type)
            self._layers_processed += 1
    
    def track_constraint_application(self):
        """Public method to track application of a constraint (only if tracking enabled)."""
        if self.enable_tracking:
            self._constraints_applied += 1
    
    def track_numerical_warning(self, warning_msg: str):
        """Public method to track a numerical warning (only if tracking enabled)."""
        if self.enable_tracking:
            self._numerical_warnings.append(warning_msg)
        ACTLog.log_verification_info(f"Numerical warning: {warning_msg}")
    
    def _track_constraint_application(self):
        """Private method for internal use - kept for backward compatibility."""
        self.track_constraint_application()
    
    def _track_numerical_warning(self, warning_msg: str):
        """Private method for internal use - kept for backward compatibility."""
        self.track_numerical_warning(warning_msg)
    
    def finalize_propagation(self, final_bounds: torch.Tensor) -> BoundsPropagationMetadata:
        """
        Finalize propagation and collect metadata if tracking is enabled.
        
        Args:
            final_bounds: Final bounds tensor
            
        Returns:
            Complete propagation metadata (empty if tracking disabled)
        """
        if not self.enable_tracking:
            return BoundsPropagationMetadata(
                layers_processed=0,
                constraints_applied=0,
                memory_cleanups=0,
                numerical_warnings=[],
                layer_types_processed=[],
                final_bounds_shape=tuple(final_bounds.shape),
                processing_time_ms=0.0
            )
        
        processing_time_ms = (time.time() - self._start_time) * 1000 if self._start_time else 0.0
        
        metadata = BoundsPropagationMetadata(
            layers_processed=self._layers_processed,
            constraints_applied=self._constraints_applied,
            memory_cleanups=self._memory_cleanups,
            numerical_warnings=self._numerical_warnings.copy(),
            layer_types_processed=self._layer_types_processed.copy(),
            final_bounds_shape=tuple(final_bounds.shape),
            processing_time_ms=processing_time_ms
        )
        
        ACTLog.log_verification_info(
            f"Propagation complete: {metadata.layers_processed} layers, "
            f"{metadata.processing_time_ms:.1f}ms, {metadata.memory_cleanups} cleanups"
        )
        return metadata
    
    def validate_layer_output(self, lb: torch.Tensor, ub: torch.Tensor, layer_idx: int, layer_name: str):
        """
        Validate bounds after layer processing to catch issues immediately.
        
        Args:
            lb: Lower bounds tensor
            ub: Upper bounds tensor
            layer_idx: Index of the processed layer
            layer_name: Name/type of the processed layer
            
        Raises:
            NumericalInstabilityError: If NaN or infinite values detected
            InvalidBoundsError: If bounds are inconsistent
        """
        if torch.any(torch.isnan(lb)) or torch.any(torch.isnan(ub)):
            raise NumericalInstabilityError(f"NaN detected after {layer_name} layer {layer_idx}")
        
        if torch.any(torch.isinf(lb)) or torch.any(torch.isinf(ub)):
            raise NumericalInstabilityError(f"Infinite values detected after {layer_name} layer {layer_idx}")
        
        if torch.any(lb > ub):
            raise InvalidBoundsError(f"Invalid bounds after {layer_name} layer {layer_idx}: lower > upper")

    def validate_input_bounds(self, input_lb: torch.Tensor, input_ub: torch.Tensor):
        """
        Comprehensive input bounds validation before starting propagation.
        
        Args:
            input_lb: Lower bounds tensor for input
            input_ub: Upper bounds tensor for input
            
        Raises:
            InvalidBoundsError: If input bounds are invalid, inconsistent, or malformed
        """
        # Check shape consistency
        if input_lb.shape != input_ub.shape:
            raise InvalidBoundsError(f"Input bounds shape mismatch: lb={input_lb.shape}, ub={input_ub.shape}")
        
        # Check bounds ordering
        if torch.any(input_lb > input_ub):
            raise InvalidBoundsError("Input lower bounds exceed upper bounds")
        
        # Check dimensionality
        if len(input_lb.shape) == 0:
            raise InvalidBoundsError("Input bounds must have at least one dimension")
        
        # Check for NaN values
        if torch.any(torch.isnan(input_lb)) or torch.any(torch.isnan(input_ub)):
            raise InvalidBoundsError("Input bounds contain NaN values")
        
        # Check for infinite values
        if torch.any(torch.isinf(input_lb)) or torch.any(torch.isinf(input_ub)):
            raise InvalidBoundsError("Input bounds contain infinite values")

    def validate_bounds_essential(self, lb: torch.Tensor, ub: torch.Tensor, layer_idx: int):
        """
        Essential bounds validation (always performed for correctness).
        
        Args:
            lb: Lower bounds tensor
            ub: Upper bounds tensor  
            layer_idx: Current layer index for error reporting
            
        Raises:
            NumericalInstabilityError: If NaN or infinite values detected
            InvalidBoundsError: If bounds are inconsistent
        """
        # Check for NaN values
        if torch.any(torch.isnan(lb)) or torch.any(torch.isnan(ub)):
            raise NumericalInstabilityError(f"NaN values detected in bounds at layer {layer_idx}")
        
        # Check for infinite values  
        if torch.any(torch.isinf(lb)) or torch.any(torch.isinf(ub)):
            raise NumericalInstabilityError(f"Infinite values detected in bounds at layer {layer_idx}")
        
        # Check bounds consistency (lb <= ub)
        if torch.any(lb > ub):
            raise InvalidBoundsError(f"Invalid bounds at layer {layer_idx}: lower bounds exceed upper bounds")