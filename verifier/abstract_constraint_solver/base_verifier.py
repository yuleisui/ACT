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

from input_parser.dataset import Dataset
from input_parser.spec import Spec
from input_parser.type import VerificationStatus
from bab_refinement.bab_spec_refinement import create_spec_refinement_core
from util.stats import ACTStats, ACTLog

class BaseVerifier:
    def __init__(self, dataset: Dataset, spec: Spec, device: str = 'cpu'):
        """
        Initialize BaseVerifier with dataset, specification, and device configuration.
        
        Args:
            dataset: Input dataset containing samples and labels
            spec: Verification specification with input/output constraints
            device: Computing device ('cpu' or 'cuda')
        """
        self.dataset = dataset
        self.spec = spec
        self.device = device
        self.dtype = torch.float32
        self.verbose = True
        
        # Initialize input tensors and model
        self._initialize_input_tensors()
        self._initialize_configurations()
        
        # Adapt input tensors to model requirements
        expected_shape = self.model.get_expected_input_shape()
        self._adapt_input_tensors_to_model_shape(expected_shape)

    def verify(self, proof, public_inputs):
        """Abstract method for verification - must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement verify method")

    def validate_unperturbed_prediction(
        self, 
        input_tensor: torch.Tensor, 
        ground_truth_label: int, 
        sample_index: int = 0
    ) -> bool:
        """
        Validate that the model correctly classifies an unperturbed input sample.
        
        Normalizes input tensor shapes, performs inference, and compares prediction
        with ground truth. Essential validation before verification attempts.
        
        Args:
            input_tensor: Input sample tensor (various shapes supported)
            ground_truth_label: Expected correct classification label
            sample_index: Sample index for logging (default: 0)
            
        Returns:
            True if model prediction matches ground truth, False otherwise
        """
        self._increment_sample_count()
        
        try:
            normalized_input = self._normalize_input_tensor_shape(input_tensor)
            prediction_result = self._perform_model_inference(normalized_input)
            return self._evaluate_prediction_accuracy(
                prediction_result, ground_truth_label, sample_index
            )
        except (RuntimeError, ValueError) as inference_error:
            ACTLog.log_prediction_failure(inference_error)
            self.clean_prediction_stats['clean_incorrect'] += 1
            return False

    def extract_sample_input_and_label(self, sample_index: int = 0) -> Tuple[torch.Tensor, int]:
        """
        Extract input center tensor and ground truth label for a specific sample.
        
        Searches for labels across multiple sources with fallback strategy:
        dataset.labels → dataset.true_labels → spec.output_spec.true_labels → default 0
        
        Args:
            sample_index: Index of sample to extract (default: 0)
            
        Returns:
            Tuple of (normalized_input_center, ground_truth_label)
        """
        normalized_input_center = self._normalize_input_center_tensor(sample_index)
        ground_truth_label = self._retrieve_ground_truth_label(sample_index)
        return normalized_input_center, ground_truth_label

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
    
    def _initialize_input_tensors(self) -> None:
        """Initialize input bounds, center, and model from specification."""
        # Get input center, compute if not available
        spec_input = self.spec.input_spec
        self.input_center = (spec_input.input_center 
                           if hasattr(spec_input, 'input_center') and spec_input.input_center is not None
                           else (spec_input.input_lb + spec_input.input_ub) / 2.0)
        
        self.input_lb = spec_input.input_lb
        self.input_ub = spec_input.input_ub
        self.model = self.spec.model

    def _initialize_configurations(self) -> None:
        """Initialize branch-and-bound config, constraints, and prediction statistics."""
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

    def _adapt_input_tensors_to_model_shape(self, expected_shape: Tuple[int, ...]) -> None:
        """
        Adapt input bounds to match expected model input shape.
        
        Handles: flat → multidimensional reshaping, missing channel dimension insertion.
        Modifies input_lb, input_ub, and input_center in-place.
        
        Args:
            expected_shape: Target shape from model.get_expected_input_shape()
            
        Raises:
            ValueError: If input bounds have mismatched shapes
            RuntimeError: If shape adaptation is not supported
        """
        if self.input_lb.shape != self.input_ub.shape:
            raise ValueError(f"Input bounds shape mismatch: {self.input_lb.shape} vs {self.input_ub.shape}")

        current_shape = self.input_lb.shape
        if current_shape[1:] == expected_shape[1:]:
            return

        # Flat input reshaping (e.g., [1, 784] -> [1, 1, 28, 28])
        if (len(current_shape) == 2 and 
            int(np.prod(current_shape[1:])) == int(np.prod(expected_shape[1:]))):
            target_shape = (current_shape[0], *expected_shape[1:])
            self.input_lb = self.input_lb.view(target_shape)
            self.input_ub = self.input_ub.view(target_shape)
            return

        # Add channel dimension (e.g., [1, 28, 28] -> [1, 1, 28, 28])
        if current_shape[1:] == expected_shape[2:] and expected_shape[1] == 1:
            self.input_lb = self.input_lb.unsqueeze(1)
            self.input_ub = self.input_ub.unsqueeze(1)
            if (hasattr(self, 'input_center') and self.input_center is not None and 
                self.input_center.shape[1:] == expected_shape[2:]):
                self.input_center = self.input_center.unsqueeze(1)
            return

        raise RuntimeError(f"Cannot adapt {current_shape} to model shape {expected_shape}")

    def _increment_sample_count(self) -> None:
        """Increment the total samples counter for statistics tracking."""
        self.clean_prediction_stats['total_samples'] += 1

    def _normalize_input_tensor_shape(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Normalize input tensor to single-batch format compatible with model.
        
        Handles: 1D (flat), 2D (batched), 3D (no batch), 4D (batched images).
        Always returns tensor with batch dimension = 1.
        
        Args:
            input_tensor: Input tensor in any supported format
            
        Returns:
            Normalized tensor with batch dimension [1, ...]
        """
        model_shape = self.model.get_expected_input_shape()
        dims = input_tensor.ndim
        
        if dims == 1:
            return (input_tensor.view(1, *model_shape[1:]) if len(model_shape) == 4 
                   else input_tensor.unsqueeze(0))
        elif dims == 2:
            if input_tensor.shape[0] == 1:
                return (input_tensor.view(1, *model_shape[1:]) 
                       if len(model_shape) == 4 and input_tensor.shape[1] == int(np.prod(model_shape[1:]))
                       else input_tensor)
            return input_tensor[0:1]
        elif dims == 3:
            return input_tensor.unsqueeze(0)
        elif dims == 4:
            return input_tensor[0:1] if input_tensor.shape[0] != 1 else input_tensor
        
        raise RuntimeError(f"Unsupported input tensor dimensions: {dims}")

    def _perform_model_inference(self, normalized_input: torch.Tensor) -> int:
        """
        Perform forward pass and return predicted class label.
        
        Args:
            normalized_input: Preprocessed tensor ready for model
            
        Returns:
            Predicted class index (int)
        """
        try:
            with torch.no_grad():
                self.model.pytorch_model.eval()
                outputs = self.model.pytorch_model(normalized_input)
                return torch.argmax(outputs, dim=1).item()
        except Exception as e:
            raise ValueError(f"Model inference failed: {e}") from e

    def _evaluate_prediction_accuracy(self, predicted: int, ground_truth: int, sample_idx: int) -> bool:
        """
        Compare prediction with ground truth and update statistics.
        
        Args:
            predicted: Model's predicted class
            ground_truth: Correct class label  
            sample_idx: Sample index for logging
            
        Returns:
            True if prediction matches ground truth
        """
        is_correct = (predicted == ground_truth)
        
        if is_correct:
            self.clean_prediction_stats['clean_correct'] += 1
            ACTLog.log_correct_prediction(predicted, ground_truth, sample_idx, self.verbose)
        else:
            self.clean_prediction_stats['clean_incorrect'] += 1
            ACTLog.log_incorrect_prediction(predicted, ground_truth, sample_idx, self.verbose)
            
        return is_correct

    def _normalize_input_center_tensor(self, sample_index: int) -> torch.Tensor:
        """
        Ensure input center tensor has proper batch dimension.
        
        Extracts specific sample from multi-sample tensors or adds batch 
        dimension to single-sample tensors.
        
        Args:
            sample_index: Index of sample to extract
            
        Returns:
            Input center with batch dimension [1, ...]
        """
        if self.input_center.ndim > 1 and self.input_center.shape[0] > 1:
            return self.input_center[sample_index:sample_index+1]
        
        if self.input_center.ndim == 1:
            return self.input_center.unsqueeze(0)
        elif self.input_center.ndim == 3:
            return self.input_center.unsqueeze(0)
        return self.input_center

    def _retrieve_ground_truth_label(self, sample_index: int) -> int:
        """
        Retrieve ground truth label with fallback strategy.
        
        Search order: dataset.labels → dataset.true_labels → spec.output_spec.true_labels → 0
        Uses closest available index if sample_index exceeds array bounds.
        
        Args:
            sample_index: Index of sample
            
        Returns:
            Ground truth label (int)
        """
        # Try dataset.labels first
        if hasattr(self.dataset, 'labels') and self.dataset.labels is not None:
            idx = min(sample_index, len(self.dataset.labels) - 1)
            return self.dataset.labels[idx].item()
        
        # Try dataset.true_labels
        if hasattr(self.dataset, 'true_labels') and self.dataset.true_labels is not None:
            idx = min(sample_index, len(self.dataset.true_labels) - 1)
            return self.dataset.true_labels[idx].item()
        
        # Try spec.output_spec.true_labels
        if (hasattr(self.spec.output_spec, 'true_labels') and 
            self.spec.output_spec.true_labels is not None):
            idx = min(sample_index, len(self.spec.output_spec.true_labels) - 1)
            return self.spec.output_spec.true_labels[idx].item()
        
        # Fallback with warning
        ACTLog.log_label_warning(sample_index)
        return 0

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

    def _single_result_verdict(self, lb: torch.Tensor, ub: torch.Tensor,
                             output_constraints: Optional[List[List[float]]],
                             true_label: Optional[int]) -> VerificationStatus:
        """
        Determine verification result from output bounds.
        
        For output constraints: checks if any constraint is violated (UNSAT/SAT).
        For classification: checks if any other class can exceed true label (UNSAT/SAT).
        
        Args:
            lb: Lower bounds tensor
            ub: Upper bounds tensor  
            output_constraints: Linear constraint matrix (optional)
            true_label: Ground truth class index (optional)
            
        Returns:
            VerificationStatus (SAT/UNSAT/UNKNOWN)
        """
        if output_constraints is not None:
            for row in output_constraints:
                a = torch.tensor(row[:-1], device=lb.device)
                b = row[-1]
                worst = torch.sum(torch.where(a >= 0, a * lb, a * ub)) + b
                if worst < 0:
                    return VerificationStatus.UNSAT
            return VerificationStatus.SAT

        if true_label is not None:
            ACTLog.log_verification_info(f"Checking classification for true_label: {true_label}")
            
            if true_label < 0 or true_label >= lb.shape[0]:
                ACTLog.log_verification_warning(f"true_label {true_label} out of bounds")
                return VerificationStatus.UNKNOWN

            if torch.any(torch.isnan(lb)) or torch.any(torch.isnan(ub)):
                ACTLog.log_verification_warning("Found NaN values in output bounds")
                return VerificationStatus.UNKNOWN

            ACTLog.log_verification_info("Using traditional bounds comparison")
            for j in range(lb.shape[0]):
                if j != true_label and ub[j] >= lb[true_label]:
                    return VerificationStatus.UNSAT
            return VerificationStatus.SAT

        return VerificationStatus.UNKNOWN

    def _spec_refinement_verification(self, input_lb: torch.Tensor, input_ub: torch.Tensor, 
                                    sample_idx: int = 0) -> VerificationStatus:
        """
        Perform branch-and-bound verification using specification refinement.
        
        Args:
            input_lb: Input lower bounds
            input_ub: Input upper bounds
            sample_idx: Sample index for logging (default: 0)
            
        Returns:
            VerificationStatus from branch-and-bound search
        """
        ACTLog.log_bab_start(sample_idx)

        spec_refinement = create_spec_refinement_core(
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
            return VerificationStatus.UNKNOWN


