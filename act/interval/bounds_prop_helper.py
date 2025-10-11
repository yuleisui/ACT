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
import torch.nn as nn
import gc
import os
from typing import List, Tuple, Union
from enum import Enum
from onnx2pytorch.operations.flatten import Flatten as OnnxFlatten

from onnx2pytorch.operations.add import Add as OnnxAdd
from onnx2pytorch.operations.div import Div as OnnxDiv
from onnx2pytorch.operations.clip import Clip as OnnxClip
from onnx2pytorch.operations.reshape import Reshape as OnnxReshape
from onnx2pytorch.operations.squeeze import Squeeze as OnnxSqueeze
from onnx2pytorch.operations.unsqueeze import Unsqueeze as OnnxUnsqueeze
from onnx2pytorch.operations.transpose import Transpose as OnnxTranspose
from onnx2pytorch.operations.base import OperatorWrapper
from dataclasses import dataclass

from act.util.stats import ACTLog, ACTStats
from act.util.device import DeviceManager, DeviceConsistencyError


# =============================================================================
# TRACKING MODE ENUM
# =============================================================================

class TrackingMode(Enum):
    """
    Unified tracking and performance mode for bounds propagation.
    
    This replaces the confusing enable_tracking + performance_mode combination.
    """
    PRODUCTION = "production"    # Minimal overhead: no logging, no validation, no metadata
    PERFORMANCE = "performance"  # Fast with metadata: metadata tracking but minimal logging
    DEBUG = "debug"             # Full tracking: comprehensive logging, validation, and metadata
    
    @property
    def collect_metadata(self) -> bool:
        """Whether to collect timing and memory metadata."""
        return self in (TrackingMode.PERFORMANCE, TrackingMode.DEBUG)
    
    @property
    def enable_logging(self) -> bool:
        """Whether to enable detailed logging."""
        return self == TrackingMode.DEBUG
    
    @property
    def enable_validation(self) -> bool:
        """Whether to enable comprehensive validation."""
        return self == TrackingMode.DEBUG
    
    @property
    def cleanup_frequency(self) -> int:
        """Memory cleanup frequency based on mode."""
        return {
            TrackingMode.PRODUCTION: 20,   # Very infrequent
            TrackingMode.PERFORMANCE: 10, # Moderate
            TrackingMode.DEBUG: 5         # Frequent for debugging
        }[self]


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
    
    # Enhanced time and memory statistics for BoundsPropagate
    start_time_ms: float
    end_time_ms: float
    layer_processing_times_ms: List[float]
    peak_memory_usage_mb: float
    initial_memory_usage_mb: float
    final_memory_usage_mb: float
    memory_usage_per_layer_mb: List[float]
    total_memory_allocated_mb: float
    gpu_memory_available_mb: float
    cpu_memory_usage_mb: float


# =============================================================================
# METADATA TRACKING CLASS
# =============================================================================

class BoundsPropMetadata:
    """
    Metadata tracking class for bounds propagation operations.
    
    This class provides a unified interface for metadata tracking, memory management,
    device consistency, and bounds validation with a clean single-mode design.
    """
    
    def __init__(self, mode: TrackingMode = TrackingMode.DEBUG):
        """
        Initialize the metadata tracker with a unified tracking mode.
        
        Args:
            mode: Tracking mode controlling logging, validation, and metadata collection
                - PRODUCTION: Minimal overhead, no logging/validation/metadata
                - PERFORMANCE: Fast with metadata tracking but minimal logging  
                - DEBUG: Full logging, validation, and metadata (default)
        """
        self.mode = mode
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
        
        # Enhanced time and memory tracking
        self._start_time_ms = None
        self._end_time_ms = None
        self._layer_processing_times_ms = []
        self._layer_start_time = None
        self._peak_memory_usage_mb = 0.0
        self._initial_memory_usage_mb = 0.0
        self._final_memory_usage_mb = 0.0
        self._memory_usage_per_layer_mb = []
        self._total_memory_allocated_mb = 0.0
    
    def start_propagation(self):
        """Initialize for a new propagation run."""
        if self.mode.collect_metadata:
            self._reset_metadata()
            current_time = time.time()
            self._start_time = current_time
            self._start_time_ms = current_time * 1000
            
            # Capture initial memory usage
            self._initial_memory_usage_mb = ACTStats.get_memory_usage_mb()
            self._peak_memory_usage_mb = self._initial_memory_usage_mb
            
            # Log based on mode
            if self.mode.enable_logging:
                ACTLog.log_verification_info(
                    f"Starting bounds propagation with metadata tracking. "
                    f"Initial memory: {self._initial_memory_usage_mb:.1f}MB"
                )
        else:
            if self.mode.enable_logging:
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
        
        # Start timing for this layer if metadata collection enabled
        if self.mode.collect_metadata:
            self._layer_start_time = time.time()
        
        # Always validate bounds for correctness (performed by caller)
        # self._validate_bounds(lb, ub, layer_idx)  # Moved to caller for efficiency
        
        stability_warnings = []
        if self.mode == TrackingMode.DEBUG:
            # Check for stability issues only in debug mode
            stability_warnings = self._check_bounds_stability(lb, ub)
            for warning in stability_warnings:
                self.track_numerical_warning(f"Layer {layer_idx}: {warning}")
        
        if self.mode.collect_metadata:
            # Track layer processing
            self._track_layer_processing(layer_type)
            
            # Memory cleanup based on mode frequency
            if self._cleanup_memory(layer_idx, force=False, frequency=self.mode.cleanup_frequency):
                self._memory_cleanups += 1
        
        return layer_type, stability_warnings
    
    def finalize_layer_processing(self, layer_idx: int):
        """Finalize processing for a layer (call after layer processing is complete)."""
        if self.mode.collect_metadata and self._layer_start_time is not None:
            # Record processing time for this layer
            layer_time_ms = (time.time() - self._layer_start_time) * 1000
            self._layer_processing_times_ms.append(layer_time_ms)
            
            # Record memory usage after processing this layer
            current_memory = ACTStats.get_memory_usage_mb()
            self._memory_usage_per_layer_mb.append(current_memory)
            
            # Update peak memory if needed
            if current_memory > self._peak_memory_usage_mb:
                self._peak_memory_usage_mb = current_memory
            
            self._layer_start_time = None
    
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
    
    def _cleanup_memory(self, idx: int, force: bool = False, frequency: int = 5) -> bool:
        """Periodic memory cleanup during bounds propagation."""
        if force or (idx % frequency == 0 and torch.cuda.is_available()):
            torch.cuda.empty_cache()
            gc.collect()
            if self.mode.enable_logging:
                ACTLog.log_verification_info(f"GPU memory cleaned at layer {idx}")
            return True
        return False
    
    def _track_layer_processing(self, layer_type: str):
        """Track a layer being processed (only if metadata collection enabled)."""
        if self.mode.collect_metadata:
            self._layer_types_processed.append(layer_type)
            self._layers_processed += 1
    
    def track_constraint_application(self):
        """Public method to track application of a constraint (only if metadata collection enabled)."""
        if self.mode.collect_metadata:
            self._constraints_applied += 1
    
    def track_numerical_warning(self, warning_msg: str):
        """Public method to track a numerical warning (only if metadata collection enabled)."""
        if self.mode.collect_metadata:
            self._numerical_warnings.append(warning_msg)
        # Log based on mode
        if self.mode.enable_logging:
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
        if not self.mode.collect_metadata:
            return BoundsPropagationMetadata(
                layers_processed=0,
                constraints_applied=0,
                memory_cleanups=0,
                numerical_warnings=[],
                layer_types_processed=[],
                final_bounds_shape=tuple(final_bounds.shape),
                processing_time_ms=0.0,
                start_time_ms=0.0,
                end_time_ms=0.0,
                layer_processing_times_ms=[],
                peak_memory_usage_mb=0.0,
                initial_memory_usage_mb=0.0,
                final_memory_usage_mb=0.0,
                memory_usage_per_layer_mb=[],
                total_memory_allocated_mb=0.0,
                gpu_memory_available_mb=0.0,
                cpu_memory_usage_mb=0.0
            )
        
        # Capture final state
        current_time = time.time()
        self._end_time_ms = current_time * 1000
        processing_time_ms = (current_time - self._start_time) * 1000 if self._start_time else 0.0
        self._final_memory_usage_mb = ACTStats.get_memory_usage_mb()
        
        # Get additional memory info
        gpu_available_mb, gpu_total_mb = ACTStats.get_gpu_memory_info()
        cpu_memory_mb = ACTStats.get_cpu_memory_usage()
        
        # Calculate total memory allocated during propagation
        if torch.cuda.is_available():
            self._total_memory_allocated_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        else:
            self._total_memory_allocated_mb = self._peak_memory_usage_mb
        
        metadata = BoundsPropagationMetadata(
            layers_processed=self._layers_processed,
            constraints_applied=self._constraints_applied,
            memory_cleanups=self._memory_cleanups,
            numerical_warnings=self._numerical_warnings.copy(),
            layer_types_processed=self._layer_types_processed.copy(),
            final_bounds_shape=tuple(final_bounds.shape),
            processing_time_ms=processing_time_ms,
            start_time_ms=self._start_time_ms or 0.0,
            end_time_ms=self._end_time_ms,
            layer_processing_times_ms=self._layer_processing_times_ms.copy(),
            peak_memory_usage_mb=self._peak_memory_usage_mb,
            initial_memory_usage_mb=self._initial_memory_usage_mb,
            final_memory_usage_mb=self._final_memory_usage_mb,
            memory_usage_per_layer_mb=self._memory_usage_per_layer_mb.copy(),
            total_memory_allocated_mb=self._total_memory_allocated_mb,
            gpu_memory_available_mb=gpu_available_mb,
            cpu_memory_usage_mb=cpu_memory_mb
        )
        
        # Log summary based on mode
        if self.mode.enable_logging:
            ACTLog.log_verification_info(
                f"Propagation complete: {metadata.layers_processed} layers, "
                f"{metadata.processing_time_ms:.1f}ms, {metadata.memory_cleanups} cleanups, "
                f"peak memory: {metadata.peak_memory_usage_mb:.1f}MB"
            )
        return metadata
    
    def validate_layer_output(self, lb: torch.Tensor, ub: torch.Tensor, layer_idx: int, layer_or_name: Union[nn.Module, str]):
        """
        Validate bounds after layer processing to catch issues immediately.
        Skips validation in production and performance modes.
        
        Args:
            lb: Lower bounds tensor
            ub: Upper bounds tensor
            layer_idx: Index of the processed layer
            layer_or_name: Layer object or name/type string of the processed layer
            
        Raises:
            NumericalInstabilityError: If NaN or infinite values detected
            InvalidBoundsError: If bounds are inconsistent
        """
        # Skip validation unless in debug mode
        if not self.mode.enable_validation:
            return
        
        # Determine layer name/category from layer object or use provided string
        if isinstance(layer_or_name, str):
            layer_name = layer_or_name
        else:
            layer = layer_or_name
            if isinstance(layer, nn.Linear):
                layer_name = "Linear"
            elif isinstance(layer, nn.Conv2d):
                layer_name = "Conv2d"
            elif isinstance(layer, (nn.ReLU, nn.Sigmoid, nn.MaxPool2d)):
                layer_name = f"Activation({type(layer).__name__})"
            elif isinstance(layer, (nn.Flatten, OnnxFlatten, OnnxReshape, OnnxSqueeze, OnnxUnsqueeze, OnnxTranspose)):
                layer_name = f"Structural({type(layer).__name__})"
            elif isinstance(layer, nn.BatchNorm2d):
                layer_name = "BatchNorm2d"
            elif isinstance(layer, (OnnxAdd, OnnxDiv, OnnxClip, OperatorWrapper)):
                layer_name = f"ONNX({type(layer).__name__})"
            else:
                layer_name = f"Unknown({type(layer).__name__})"
        
        if torch.any(torch.isnan(lb)) or torch.any(torch.isnan(ub)):
            raise NumericalInstabilityError(f"NaN detected after {layer_name} layer {layer_idx}")
        
        if torch.any(torch.isinf(lb)) or torch.any(torch.isinf(ub)):
            raise NumericalInstabilityError(f"Infinite values detected after {layer_name} layer {layer_idx}")
        
        if torch.any(lb > ub):
            raise InvalidBoundsError(f"Invalid bounds after {layer_name} layer {layer_idx}: lower > upper")

    def validate_input_bounds(self, input_lb: torch.Tensor, input_ub: torch.Tensor):
        """
        Comprehensive input bounds validation before starting propagation.
        Skips expensive checks in performance mode.
        
        Args:
            input_lb: Lower bounds tensor for input
            input_ub: Upper bounds tensor for input
            
        Raises:
            InvalidBoundsError: If input bounds are invalid, inconsistent, or malformed
        """
        # Always check critical issues that could cause crashes
        if input_lb.shape != input_ub.shape:
            raise InvalidBoundsError(f"Input bounds shape mismatch: lb={input_lb.shape}, ub={input_ub.shape}")
        
        if torch.any(input_lb > input_ub):
            raise InvalidBoundsError("Input lower bounds exceed upper bounds")
        
        # Skip expensive validation in production/performance modes
        if not self.mode.enable_validation:
            return
        
        # Comprehensive validation in normal mode
        if len(input_lb.shape) == 0:
            raise InvalidBoundsError("Input bounds must have at least one dimension")
        
        if torch.any(torch.isnan(input_lb)) or torch.any(torch.isnan(input_ub)):
            raise InvalidBoundsError("Input bounds contain NaN values")
        
        if torch.any(torch.isinf(input_lb)) or torch.any(torch.isinf(input_ub)):
            raise InvalidBoundsError("Input bounds contain infinite values")

    def validate_bounds_essential(self, lb: torch.Tensor, ub: torch.Tensor, layer_idx: int):
        """
        Essential bounds validation (always performed for correctness).
        Skips some checks in performance mode for speed.
        
        Args:
            lb: Lower bounds tensor
            ub: Upper bounds tensor  
            layer_idx: Current layer index for error reporting
            
        Raises:
            NumericalInstabilityError: If NaN or infinite values detected
            InvalidBoundsError: If bounds are inconsistent
        """
        # Always check bounds consistency (critical for correctness)
        if torch.any(lb > ub):
            raise InvalidBoundsError(f"Invalid bounds at layer {layer_idx}: lower bounds exceed upper bounds")
        
        # Skip expensive NaN/inf checks in production/performance modes  
        if not self.mode.enable_validation:
            return
        
        # Comprehensive validation in normal mode
        if torch.any(torch.isnan(lb)) or torch.any(torch.isnan(ub)):
            raise NumericalInstabilityError(f"NaN values detected in bounds at layer {layer_idx}")
        
        if torch.any(torch.isinf(lb)) or torch.any(torch.isinf(ub)):
            raise NumericalInstabilityError(f"Infinite values detected in bounds at layer {layer_idx}")