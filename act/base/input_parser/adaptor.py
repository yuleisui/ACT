#===- verifier.input_adaptor.py Input adaptation utilities ---------------#
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
#   Input adaptation utilities for neural network verification in the Abstract
#   Constraints Transformer (ACT). Handles specification bounds adaptation,
#   sample tensor normalization, and pre-computation of model-ready sample-label
#   pairs for efficient verification workflows.
#
#===----------------------------------------------------------------------===#

from typing import Tuple, List
import torch
import numpy as np

from act.base.input_parser.spec import Spec
from act.base.util.stats import ACTLog


class InputAdaptor:
    """
    Handles input adaptation for neural network verification.
    
    Provides utilities for adapting verification specification bounds to model shape,
    pre-computing normalized sample-label pairs, and validating tensor formats for
    model inference. Centralizes all input preprocessing logic for verification workflows.
    """
    
    def __init__(self, spec: Spec, input_center: torch.Tensor, dataset, verbose: bool = True):
        """
        Initialize InputAdaptor with specification and input configuration.
        
        Args:
            spec: Verification specification with input/output constraints
            input_center: Center point tensor for verification region
            dataset: Dataset containing samples and labels
            verbose: Enable verbose logging (default: True)
        """
        self.spec = spec
        self.input_center = input_center
        self.dataset = dataset
        self.verbose = verbose
        self._sample_label_pairs = []
        
    def adapt_all_inputs_to_model_shape(self) -> None:
        """
        Adapt both specification bounds and sample-label pairs to model requirements.
        
        Performs complete input adaptation workflow:
        1. Adapts verification specification bounds to model input shape
        2. Pre-computes all sample-label pairs adapted to model format
        
        This is typically called during verifier initialization.
        """
        # Adapt specification bounds to model requirements
        expected_shape = self.spec.model.get_expected_input_shape()
        self.adapt_spec_bounds_to_model_shape(expected_shape)      
        # Pre-compute adapted sample-label pairs for efficient access
        self.adapt_sample_label_pairs_to_model_shape()
        
    def adapt_spec_bounds_to_model_shape(self, expected_shape: Tuple[int, ...]) -> None:
        """
        Adapt verification specification bounds to match expected model input shape.
        
        Reshapes the verification constraint tensors (input_lb, input_ub, input_center) to be
        compatible with the model's expected input format. This is a one-time initialization
        operation that ensures the verification problem is properly defined.
        
        Handles: flat → multidimensional reshaping, missing channel dimension insertion.
        Modifies specification bounds in-place.
        
        Args:
            expected_shape: Target shape from model.get_expected_input_shape()
            
        Raises:
            ValueError: If specification bounds have mismatched shapes
            RuntimeError: If shape adaptation is not supported
        """
        spec_input_lb = self.spec.input_spec.input_lb
        spec_input_ub = self.spec.input_spec.input_ub
        
        if spec_input_lb.shape != spec_input_ub.shape:
            raise ValueError(f"Specification bounds shape mismatch: {spec_input_lb.shape} vs {spec_input_ub.shape}")

        current_shape = spec_input_lb.shape
        if current_shape[1:] == expected_shape[1:]:
            return

        # Flat specification bounds reshaping (e.g., [1, 784] -> [1, 1, 28, 28])
        if (len(current_shape) == 2 and 
            int(np.prod(current_shape[1:])) == int(np.prod(expected_shape[1:]))):
            target_shape = (current_shape[0], *expected_shape[1:])
            self.spec.input_spec.input_lb = spec_input_lb.view(target_shape)
            self.spec.input_spec.input_ub = spec_input_ub.view(target_shape)
            return

        # Add channel dimension to specification bounds (e.g., [1, 28, 28] -> [1, 1, 28, 28])
        if current_shape[1:] == expected_shape[2:] and expected_shape[1] == 1:
            self.spec.input_spec.input_lb = spec_input_lb.unsqueeze(1)
            self.spec.input_spec.input_ub = spec_input_ub.unsqueeze(1)
            if self.input_center.shape[1:] == expected_shape[2:]:
                self.input_center = self.input_center.unsqueeze(1)
            return

        raise RuntimeError(f"Cannot adapt specification bounds {current_shape} to model shape {expected_shape}")

    def adapt_sample_label_pairs_to_model_shape(self) -> None:
        """
        Adapt sample-label pairs to model shape during initialization for efficient access.
        
        Determines the number of samples based on dataset size and pre-computes all
        sample tensors adapted to model input format, paired with their corresponding truth labels. 
        This performs shape adaptation, batch dimension handling, and tensor reshaping to ensure
        samples match the model's expected input shape. Avoids repeated adaptation computation 
        during verification and ensures samples and labels are always retrieved together as a coherent unit.
        """
        # Determine number of samples to pre-compute
        num_samples = 1  # Default single sample
        
        if hasattr(self.dataset, 'labels') and self.dataset.labels is not None:
            num_samples = len(self.dataset.labels)
        elif hasattr(self.dataset, 'true_labels') and self.dataset.true_labels is not None:
            num_samples = len(self.dataset.true_labels)
        elif (hasattr(self.spec.output_spec, 'true_labels') and 
              self.spec.output_spec.true_labels is not None):
            num_samples = len(self.spec.output_spec.true_labels)
        elif (self.input_center.ndim > 1 and self.input_center.shape[0] > 1):
            num_samples = self.input_center.shape[0]
        
        # Pre-compute all adapted sample-label pairs
        self._sample_label_pairs = []
        
        for idx in range(num_samples):
            adapted_sample_tensor, truth_label = self.extract_normalized_sample_and_label(idx)
            self._sample_label_pairs.append((adapted_sample_tensor, truth_label))
            
        ACTLog.log_verification_info(f"Pre-computed {num_samples} model-adapted samples for efficient access")

    def get_sample_label_pairs(self) -> List[Tuple[torch.Tensor, int]]:
        """
        Get all pre-computed sample-label pairs.
        
        Returns:
            List of (model_adapted_sample_tensor, truth_label) tuples
        """
        return self._sample_label_pairs

    def extract_normalized_sample_and_label(self, sample_index: int = 0) -> Tuple[torch.Tensor, int]:
        """
        Extract and adapt a sample tensor with ground truth label for model-ready verification testing.
        
        Takes the center point from the verification specification and adapts it to create
        a concrete sample tensor ready for model inference. Performs shape normalization,
        batch dimension handling, and tensor reshaping to match model's expected input format.
        This is used to validate that the model correctly classifies the center point before 
        attempting verification of the entire region.
        
        Label retrieval fallback strategy: dataset.labels → dataset.true_labels → spec.output_spec.true_labels → default 0
        
        Args:
            sample_index: Index of sample to extract and adapt (default: 0)
            
        Returns:
            Tuple of (model_ready_sample_tensor, ground_truth_label)
            The sample tensor is guaranteed to have proper batch dimension and be model-ready.
        """
        # Extract center sample from specification - ensure proper batch dimension
        if self.input_center.ndim > 1 and self.input_center.shape[0] > 1:
            center_sample = self.input_center[sample_index:sample_index+1]
        elif self.input_center.ndim == 1:
            center_sample = self.input_center.unsqueeze(0)
        elif self.input_center.ndim == 3:
            center_sample = self.input_center.unsqueeze(0)
        else:
            center_sample = self.input_center

        # Apply full sample tensor adaptation to ensure model compatibility
        # Handles shape adaptation: 1D (flat), 2D (batched), 3D (no batch), 4D (batched images).
        # Always returns sample tensor with batch dimension = 1 and proper model shape.
        model_shape = self.spec.model.get_expected_input_shape()
        sample_dims = center_sample.ndim
        
        if sample_dims == 1:
            adapted_sample = (center_sample.view(1, *model_shape[1:]) if len(model_shape) == 4 
                   else center_sample.unsqueeze(0))
        elif sample_dims == 2:
            if center_sample.shape[0] == 1:
                adapted_sample = (center_sample.view(1, *model_shape[1:]) 
                       if len(model_shape) == 4 and center_sample.shape[1] == int(np.prod(model_shape[1:]))
                       else center_sample)
            else:
                adapted_sample = center_sample[0:1]
        elif sample_dims == 3:
            adapted_sample = center_sample.unsqueeze(0)
        elif sample_dims == 4:
            adapted_sample = center_sample[0:1] if center_sample.shape[0] != 1 else center_sample
        else:
            raise RuntimeError(f"Unsupported sample tensor dimensions: {sample_dims}")

        # Retrieve ground truth label with fallback strategy
        # Try dataset.labels first
        if hasattr(self.dataset, 'labels') and self.dataset.labels is not None:
            idx = min(sample_index, len(self.dataset.labels) - 1)
            ground_truth_label = self.dataset.labels[idx].item()
        # Try dataset.true_labels
        elif hasattr(self.dataset, 'true_labels') and self.dataset.true_labels is not None:
            idx = min(sample_index, len(self.dataset.true_labels) - 1)
            ground_truth_label = self.dataset.true_labels[idx].item()
        # Try spec.output_spec.true_labels
        elif (hasattr(self.spec.output_spec, 'true_labels') and 
              self.spec.output_spec.true_labels is not None):
            idx = min(sample_index, len(self.spec.output_spec.true_labels) - 1)
            ground_truth_label = self.spec.output_spec.true_labels[idx].item()
        else:
            # Fallback with warning
            ACTLog.log_label_warning(sample_index)
            ground_truth_label = 0

        return adapted_sample, ground_truth_label

    def is_sample_tensor_properly_normalized(self, sample_tensor: torch.Tensor) -> bool:
        """
        Check if sample tensor is properly normalized for model inference.
        
        Validates batch size = 1 and tensor shape matches expected model input shape.
        
        Args:
            sample_tensor: Sample tensor to validate
            
        Returns:
            True if sample tensor is properly normalized, False otherwise
        """
        model_shape = self.spec.model.get_expected_input_shape()
        
        # Check batch size is 1
        if sample_tensor.shape[0] != 1:
            return False
            
        # Check sample tensor dimensions match model dimensions
        if len(sample_tensor.shape) != len(model_shape):
            return False
            
        # Check sample tensor shape matches model shape (excluding batch dimension)
        if sample_tensor.shape[1:] != model_shape[1:]:
            return False
            
        return True