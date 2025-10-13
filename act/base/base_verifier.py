#===- act.base.base_verifier.py - Base Verifier ----#
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

from act.base.input_parser.spec import Spec
from act.base.input_parser.type import VerifyResult
from act.base.input_parser.adaptor import InputAdaptor
from act.base.refinement.bab_spec_refinement import create_bab_refinement
from act.base.util.stats import ACTLog, ACTStats
from act.base.util.inference import perform_model_inference
from act.base.util.bounds import Bounds
from act.base.bounds_propagation import BoundsPropagate
from act.base.bounds_prop_helper import TrackingMode
from act.base.outputs_evaluation import OutputsEvaluate

class BaseVerifier:
    def __init__(self, spec: Spec, device: str = 'cpu', enable_metadata_tracking: bool = True):
        """
        Initialize BaseVerifier with specification and device configuration.
        
        Args:
            spec: Verification specification with input/output constraints and dataset
            device: Computing device ('cpu' or 'cuda')
            enable_metadata_tracking: Whether to enable metadata tracking in bounds propagation
        """
        self.spec = spec
        self.dataset = spec.dataset
        self.model = spec.model
        self.device = device
        self.enable_metadata_tracking = enable_metadata_tracking
        self.dtype = torch.float32
        self.verbose = True
        
        # Initialize input center from specification
        spec_input = self.spec.input_spec
        self.input_center = (spec_input.input_center 
                           if hasattr(spec_input, 'input_center') and spec_input.input_center is not None
                           else (spec_input.input_lb + spec_input.input_ub) / 2.0)

        # Initialize input adaptor for shape adaptation and sample preprocessing
        self.input_adaptor = InputAdaptor(spec, self.input_center, self.dataset, self.verbose)

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
        
        # Adapt specification bounds and pre-compute sample-label pairs
        self.input_adaptor.adapt_all_inputs_to_model_shape()

    def verify(self) -> VerifyResult:
        """
        Execute the complete interval verification pipeline for all samples.
        
        Processes each sample through base bound propagation, applies BaB refinement
        when enabled, and aggregates results.
        
        Returns:
            Final verification status aggregated across all samples
        """
        ACTLog.log_verification_info("Starting Interval verification pipeline")

        num_samples = self.input_center.shape[0] if self.input_center.ndim > 1 else 1
        ACTLog.log_verification_info(f"Total samples: {num_samples}")
        results = []
        
        for idx in range(num_samples):
            ACTLog.log_verification_info(f"\n🔍 Processing sample {idx+1}/{num_samples}")
            ACTLog.log_verification_info("="*80)

            center_input, true_label = self.get_sample_label_pair(idx)
            if not perform_model_inference(
                model=self.spec.model.pytorch_model,
                sample_tensor=center_input,
                ground_truth_label=true_label,
                input_adaptor=self.input_adaptor,
                prediction_stats=self.clean_prediction_stats,
                sample_index=idx,
                verbose=self.verbose
            ):
                ACTLog.log_verification_info(f"⏭️  Skipping verification for sample {idx+1}")
                results.append(VerifyResult.CLEAN_FAILURE)
                continue

            # Extract bounds for current sample and create Bounds object
            sample_lb = self.spec.input_spec.input_lb if self.spec.input_spec.input_lb.ndim == 1 else self.spec.input_spec.input_lb[idx]
            sample_ub = self.spec.input_spec.input_ub if self.spec.input_spec.input_ub.ndim == 1 else self.spec.input_spec.input_ub[idx]
            input_bounds = Bounds(sample_lb, sample_ub, _internal=True)

            self.clean_prediction_stats['verification_attempted'] += 1

            ACTLog.log_verification_info("Step 1: Interval abstract constraint solving")
            initial_verdict = self._solve_constraints(input_bounds, idx)

            if initial_verdict == VerifyResult.SAT:
                self.clean_prediction_stats['verification_sat'] += 1
                ACTLog.log_verification_info(f"✅ Interval verification success - Sample {idx+1} safe")
                results.append(initial_verdict)
                continue
            elif initial_verdict == VerifyResult.UNSAT:
                self.clean_prediction_stats['verification_unsat'] += 1
            else:
                self.clean_prediction_stats['verification_unknown'] += 1

            if initial_verdict == VerifyResult.UNSAT:
                ACTLog.log_verification_info(f"❌ Interval potential violation detected - Sample {idx+1}")
            else:
                ACTLog.log_verification_info(f"❓ Interval inconclusive - Sample {idx+1}")

            ACTLog.log_verification_info("Launching Specification Refinement BaB process")
            ACTLog.log_verification_info("="*60)

            if self.bab_config['enabled']:
                refinement_verdict = self._spec_refinement(input_bounds, idx)
                if refinement_verdict == VerifyResult.SAT:
                    self.clean_prediction_stats['verification_sat'] += 1
                elif refinement_verdict == VerifyResult.UNSAT:
                    self.clean_prediction_stats['verification_unsat'] += 1
                else:
                    self.clean_prediction_stats['verification_unknown'] += 1
                results.append(refinement_verdict)
            else:
                ACTLog.log_verification_info("⚠️  BaB disabled, returning initial verdict")
                results.append(initial_verdict)

        ACTStats.print_verification_stats(self.clean_prediction_stats)
        return ACTStats.print_final_verification_summary(results)



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
        sample_label_pairs = self.input_adaptor.get_sample_label_pairs()
        if sample_idx < 0 or sample_idx >= len(sample_label_pairs):
            raise IndexError(f"Sample index {sample_idx} is out of bounds. "
                           f"Available samples: 0 to {len(sample_label_pairs) - 1}")
        
        return sample_label_pairs[sample_idx]

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

    def _solve_constraints(self, input_bounds: Bounds, sample_idx: int) -> VerifyResult:
        """
        Perform interval constraint solving for a single sample.
        
        Applies base bound propagation through the network and evaluates results
        against verification constraints to determine if the sample is safe.
        
        Args:
            input_bounds: Input bounds object containing lb and ub tensors
            sample_idx: Index of the current sample being processed
            
        Returns:
            VerifyResult indicating the verification result (SAT/UNSAT/UNKNOWN)
        """
        ACTLog.log_verification_info(f"Performing Base propagation")

        # Get ReLU constraints from BaB refinement if available
        relu_constraints = getattr(self, 'current_relu_constraints', None)
        
        # Create propagator and run bound propagation
        # Use dedicated interval bound propagator with BaB constraints
        # Convert boolean metadata tracking flag to TrackingMode enum
        tracking_mode = TrackingMode.DEBUG if self.enable_metadata_tracking else TrackingMode.PRODUCTION
        propagator = BoundsPropagate(relu_constraints, tracking_mode)
        output_bounds = propagator.propagate_bounds(
            self.spec.model.pytorch_model, input_bounds
        )

        verdict = OutputsEvaluate.evaluate_output_bounds(
            output_bounds,
            self.spec.output_spec.output_constraints if self.spec.output_spec.output_constraints is not None else None,
            self.spec.output_spec.labels[sample_idx].item() if self.spec.output_spec.labels is not None else None
        )

        ACTLog.log_verification_info(f"📊 Verification verdict: {verdict.name}")
        return verdict

    # ========================= PRIVATE METHODS =========================
    
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

    def _spec_refinement(self, input_bounds: Bounds, 
                        sample_idx: int = 0) -> VerifyResult:
        """
        Perform branch-and-bound verification using specification refinement.
        
        Args:
            input_bounds: Input bounds object containing lb and ub tensors
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
            result = spec_refinement.search(input_bounds.lb, input_bounds.ub, self, self.spec.model.pytorch_model)
            ACTLog.log_bab_results(result.status.name, result.total_subproblems, 
                                 len(result.spurious_counterexamples), result.real_counterexample,
                                 result.max_depth, result.total_time)
            return result.status
        except Exception as e:
            ACTLog.log_bab_error(e)
            return VerifyResult.UNKNOWN


