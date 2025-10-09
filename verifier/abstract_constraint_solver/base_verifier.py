#===- verifier.base_verifier.py the basic verification class --------------#
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
#   Abstract base class for neural network verification implementations in the
#   Abstract Constraints Transformer (ACT), providing common interfaces for
#   input handling, model inference, and verification result processing.
#
#===----------------------------------------------------------------------===#

from typing import Dict, List, Optional, Any, Tuple
import torch
import numpy as np

from input_parser.spec import Spec
from input_parser.type import VerifyResult
from bab_refinement.bab_spec_refinement import create_bab_refinement
from util.stats import ACTLog

class BaseVerifier:
    def __init__(self, spec: Spec, device: str = 'cpu'):
        """
        Initialize BaseVerifier with specification and device configuration.
        
        Args:
            spec: Verification specification with input/output constraints and dataset
            device: Computing device ('cpu' or 'cuda')
        """
        self.spec = spec
        self.dataset = spec.dataset
        self.model = spec.model
        self.device = device
        self.dtype = torch.float32
        self.verbose = True
        # Initialize paired sample-label data structure for pre-computation
        self._sample_label_pairs = []
        
        # Initialize input center from specification
        spec_input = self.spec.input_spec
        self.input_center = (spec_input.input_center 
                           if hasattr(spec_input, 'input_center') and spec_input.input_center is not None
                           else (spec_input.input_lb + spec_input.input_ub) / 2.0)

        # Initialize branch-and-bound config, constraints, and prediction statistics
        self.bab_config = {
            'enabled': False,
            'max_depth': 8,
            'max_subproblems': 500,
            'time_limit': 1500.0,
            'split_tolerance': 1e-6,
            'verbose': True
        }
        
        self.current_relu_constraints = []
        
        self.clean_prediction_stats = {
            'total_samples': 0,
            'clean_correct': 0,
            'clean_incorrect': 0,
            'verification_attempted': 0,
            'verification_sat': 0,
            'verification_unsat': 0,
            'verification_unknown': 0
        }
        
        # Adapt specification bounds to model requirements
        expected_shape = self.model.get_expected_input_shape()
        self._adapt_spec_bounds_to_model_shape(expected_shape)      
        # Pre-compute adapted sample-label pairs for efficient access
        self._adapt_sample_label_pairs_to_model_shape()

    def verify(self, proof, public_inputs):
        """Abstract method for verification - must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement verify method")

    def perform_model_inference(
        self, 
        sample_tensor: torch.Tensor, 
        ground_truth_label: int, 
        sample_index: int = 0
    ) -> bool:
        """
        Perform model inference on a sample tensor and validate prediction accuracy.
        
        Executes forward pass on a concrete sample tensor and compares prediction with ground truth. 
        Essential validation before verification attempts. Expects sample tensor to be already normalized.
        
        Args:
            sample_tensor: Sample tensor (must be normalized with proper batch dimension)
            ground_truth_label: Expected correct classification label
            sample_index: Sample index for logging (default: 0)
            
        Returns:
            True if model prediction matches ground truth, False otherwise
        """
        self.clean_prediction_stats['total_samples'] += 1
        
        try:
            # Assert that sample tensor is properly normalized
            assert self._is_sample_tensor_properly_normalized(sample_tensor), \
                "Sample tensor must be properly normalized for model inference"
                
            try:
                with torch.no_grad():
                    self.spec.model.pytorch_model.eval()
                    outputs = self.spec.model.pytorch_model(sample_tensor)
                    predicted = torch.argmax(outputs, dim=1).item()
            except Exception as e:
                raise ValueError(f"Model inference failed: {e}") from e
            
            # Evaluate prediction accuracy (inlined from _evaluate_prediction_accuracy)
            is_correct = (predicted == ground_truth_label)
            
            if is_correct:
                self.clean_prediction_stats['clean_correct'] += 1
                ACTLog.log_correct_prediction(predicted, ground_truth_label, sample_index, self.verbose)
            else:
                self.clean_prediction_stats['clean_incorrect'] += 1
                ACTLog.log_incorrect_prediction(predicted, ground_truth_label, sample_index, self.verbose)
                
            return is_correct
            
        except (RuntimeError, ValueError) as inference_error:
            ACTLog.log_prediction_failure(inference_error)
            self.clean_prediction_stats['clean_incorrect'] += 1
            return False

    def get_sample_label_pair(self, sample_idx: int = 0) -> Tuple[torch.Tensor, int]:
        """
        Get pre-computed model-adapted sample tensor and label for efficient access.
        
        Args:
            sample_idx: Index of the sample to retrieve (0-based)
            
        Returns:
            Tuple of (model_adapted_sample_tensor, truth_label) where sample_tensor is properly
            shaped, normalized, and ready for model inference
            
        Raises:
            IndexError: If sample_idx is out of bounds for pre-computed sample-label pairs
        """
        if sample_idx < 0 or sample_idx >= len(self._sample_label_pairs):
            raise IndexError(f"Sample index {sample_idx} is out of bounds. "
                           f"Available samples: 0 to {len(self._sample_label_pairs) - 1}")
        
        return self._sample_label_pairs[sample_idx]

    def _extract_normalized_sample_and_label(self, sample_index: int = 0) -> Tuple[torch.Tensor, int]:
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

    def set_relu_constraints(self, relu_constraints: List[Dict[str, Any]]) -> None:
        """
        Set ReLU activation constraints for verification.
        
        Args:
            relu_constraints: List of constraint dictionaries with keys:
                'layer', 'neuron_idx', 'constraint_type' ('active'/'inactive')
        """
        self.current_relu_constraints = relu_constraints.copy() if relu_constraints else []
        if self.verbose:
            ACTLog.log_relu_constraints_set(self.current_relu_constraints, self.verbose)

    def get_relu_constraints(self) -> List[Dict[str, Any]]:
        """Return a copy of current ReLU constraints."""
        return self.current_relu_constraints.copy()

    def enforce_neuron_activation_constraints(self) -> None:
        """
        Apply ReLU constraints to cached layer bounds.
        
        Modifies bounds in-place: 'inactive' constraints clamp upper bounds ≤ 0,
        'active' constraints clamp lower bounds ≥ 0. Supports HybridZonotope 
        and AutoLiRPA bounds caches. No effect if no constraints or cache available.
        """
        if not self.current_relu_constraints:
            return
            
        bounds_cache_info = self._identify_available_bounds_cache()
        if bounds_cache_info is None:
            return
            
        bounds_cache, cache_type_name = bounds_cache_info
        ACTLog.log_constraint_application_start(cache_type_name, len(self.current_relu_constraints), self.verbose)
        
        layer_names = list(bounds_cache.keys())
        for constraint_config in self.current_relu_constraints:
            self._apply_single_neuron_constraint(constraint_config, bounds_cache, layer_names)
            
        ACTLog.log_constraint_application_complete(self.verbose)

    # ========================= PRIVATE METHODS =========================
    
    def _adapt_spec_bounds_to_model_shape(self, expected_shape: Tuple[int, ...]) -> None:
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
            assert hasattr(self, 'input_center') and self.input_center is not None, \
                "input_center should be initialized before calling _adapt_spec_bounds_to_model_shape"
            if self.input_center.shape[1:] == expected_shape[2:]:
                self.input_center = self.input_center.unsqueeze(1)
            return

        raise RuntimeError(f"Cannot adapt specification bounds {current_shape} to model shape {expected_shape}")

    def _adapt_sample_label_pairs_to_model_shape(self) -> None:
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
            adapted_sample_tensor, truth_label = self._extract_normalized_sample_and_label(idx)
            self._sample_label_pairs.append((adapted_sample_tensor, truth_label))
            
        ACTLog.log_verification_info(f"Pre-computed {num_samples} model-adapted samples for efficient access")

    def _is_sample_tensor_properly_normalized(self, sample_tensor: torch.Tensor) -> bool:
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

    def _identify_available_bounds_cache(self) -> Optional[Tuple[Dict[str, Any], str]]:
        """
        Find available bounds cache and return with type identifier.
        
        Returns:
            (bounds_cache, cache_type_name) or None if no cache available
        """
        if hasattr(self, 'hz_layer_bounds') and self.hz_layer_bounds:
            return self.hz_layer_bounds, "HybridZonotope"
        elif hasattr(self, 'autolirpa_layer_bounds') and self.autolirpa_layer_bounds:
            return self.autolirpa_layer_bounds, "AutoLiRPA"
        else:
            ACTLog.log_no_bounds_cache_warning(self.verbose)
            return None

    def _apply_single_neuron_constraint(self, constraint_config: Dict[str, Any], 
                                      bounds_cache: Dict[str, Any], 
                                      layer_names: List[str]) -> None:
        """
        Apply single ReLU constraint to modify bounds in previous layer.
        
        Validates constraint config, finds target layer, and modifies bounds:
        - 'inactive': clamp upper bound ≤ 0  
        - 'active': clamp lower bound ≥ 0
        
        Args:
            constraint_config: Dict with 'layer', 'neuron_idx', 'constraint_type'
            bounds_cache: Layer bounds data
            layer_names: Ordered layer names
        """
        # Extract and validate constraint details
        required_keys = ['layer', 'neuron_idx', 'constraint_type']
        if not all(key in constraint_config for key in required_keys):
            ACTLog.log_constraint_error("Invalid constraint: missing required keys", self.verbose)
            return
        
        layer_name = constraint_config['layer']
        neuron_idx = constraint_config['neuron_idx']
        constraint_type = constraint_config['constraint_type']
        
        ACTLog.log_constraint_processing(layer_name, neuron_idx, constraint_type, self.verbose)
        
        # Find layer position
        try:
            layer_pos = layer_names.index(layer_name)
            if layer_pos == 0:
                ACTLog.log_constraint_error(f"ReLU layer {layer_name} is first layer", self.verbose)
                return
        except ValueError:
            ACTLog.log_constraint_error(f"ReLU layer not found: {layer_name}", self.verbose)
            return
        
        # Get previous layer bounds
        prev_layer_name = layer_names[layer_pos - 1]
        prev_layer_data = bounds_cache[prev_layer_name]
        
        if 'lb' not in prev_layer_data or 'ub' not in prev_layer_data:
            ACTLog.log_constraint_error(f"Previous layer {prev_layer_name} missing bounds", self.verbose)
            return
        
        lb, ub = prev_layer_data['lb'], prev_layer_data['ub']
        
        # Validate neuron index
        if neuron_idx >= lb.numel():
            ACTLog.log_constraint_error(f"Neuron index {neuron_idx} out of bounds", self.verbose)
            return
        
        # Apply constraint
        flat_lb, flat_ub = lb.view(-1), ub.view(-1)
        
        if constraint_type == 'inactive':
            orig = flat_ub[neuron_idx].item()
            flat_ub[neuron_idx] = min(orig, 0.0)
            ACTLog.log_constraint_bound_update('inactive', neuron_idx, orig, flat_ub[neuron_idx].item(), self.verbose)
        elif constraint_type == 'active':
            orig = flat_lb[neuron_idx].item()
            flat_lb[neuron_idx] = max(orig, 0.0)
            ACTLog.log_constraint_bound_update('active', neuron_idx, orig, flat_lb[neuron_idx].item(), self.verbose)
        else:
            ACTLog.log_constraint_error(f"Unknown constraint type: {constraint_type}", self.verbose)

    def _spec_refinement(self, input_lb: torch.Tensor, input_ub: torch.Tensor, 
                        sample_idx: int = 0) -> VerifyResult:
        """
        Perform branch-and-bound verification using specification refinement.
        
        Args:
            input_lb: Input lower bounds
            input_ub: Input upper bounds
            sample_idx: Sample index for logging (default: 0)
            
        Returns:
            VerifyResult from branch-and-bound search
        """
        ACTLog.log_bab_start(sample_idx)

        spec_refinement = create_bab_refinement(
            max_depth=self.bab_config['max_depth'],
            max_subproblems=self.bab_config['max_subproblems'],
            time_limit=self.bab_config['time_limit'],
            spurious_check_enabled=self.bab_config.get('spurious_check', True),
            verbose=self.bab_config['verbose']
        )

        spec_refinement._current_verifier = self

        try:
            result = spec_refinement.search(input_lb, input_ub, self, self.spec.model.pytorch_model)
            ACTLog.log_bab_results(result.status.name, result.total_subproblems, 
                                 len(result.spurious_counterexamples), result.real_counterexample,
                                 result.max_depth, result.total_time)
            return result.status
        except Exception as e:
            ACTLog.log_bab_error(e)
            return VerifyResult.UNKNOWN


