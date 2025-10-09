#===- verifier.interval.interval_verifier.py interval arithmetic verifier --#
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
# Interval arithmetic-based neural network verifier for robustness analysis.
# Implements abstract constraint solving using interval bound propagation through
# neural network layers with support for BaB refinement constraints.
#
#===----------------------------------------------------------------------===#

import torch
import torch.nn as nn
from typing import Tuple, Optional, List

from abstract_constraint_solver.base_verifier import BaseVerifier
from abstract_constraint_solver.interval.bounds_propagation import BoundsPropagate
from abstract_constraint_solver.interval.outputs_evaluation import OutputsEvaluate
from input_parser.spec import Spec
from input_parser.type import VerifyResult
from util.stats import ACTStats, ACTLog

class IntervalVerifier(BaseVerifier):
    """
    Interval arithmetic-based neural network verifier for robustness analysis.
    
    Implements abstract constraint solving using interval bound propagation through
    neural network layers with support for BaB refinement constraints.
    """
    
    # =============================================================================
    # PUBLIC INTERFACE METHODS
    # =============================================================================
    
    def __init__(self, method: str, spec: Spec, device: str = 'cpu') -> None:
        """Initialize the interval verifier with validation.
        
        Args:
            method: Verification method (must be 'interval')
            spec: Verification specification including model and constraints
            device: Compute device ('cpu' or 'cuda')
            
        Raises:
            ValueError: If method is not 'interval'
        """
        super().__init__(spec, device)
        if method != 'interval':
            raise ValueError(f"IntervalVerifier only supports 'interval' method, got {method}.")

    def verify(self) -> VerifyResult:
        """
        Execute the complete interval verification pipeline for all samples.
        
        Processes each sample through interval bound propagation, applies BaB refinement
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
            if not self.perform_model_inference(center_input, true_label, idx):
                ACTLog.log_verification_info(f"⏭️  Skipping verification for sample {idx+1}")
                results.append(VerifyResult.CLEAN_FAILURE)
                continue

            # Extract bounds for current sample
            sample_lb = self.spec.input_spec.input_lb if self.spec.input_spec.input_lb.ndim == 1 else self.spec.input_spec.input_lb[idx]
            sample_ub = self.spec.input_spec.input_ub if self.spec.input_spec.input_ub.ndim == 1 else self.spec.input_spec.input_ub[idx]

            self.clean_prediction_stats['verification_attempted'] += 1

            ACTLog.log_verification_info("Step 1: Interval abstract constraint solving")
            initial_verdict = self._solve_constraints(sample_lb, sample_ub, idx)

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
                refinement_verdict = self._spec_refinement(sample_lb, sample_ub, idx)
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

    # =============================================================================
    # CORE VERIFICATION ALGORITHMS
    # =============================================================================

    def _solve_constraints(self, input_lb: torch.Tensor, input_ub: torch.Tensor, sample_idx: int) -> VerifyResult:
        """
        Perform interval constraint solving for a single sample.
        
        Applies interval bound propagation through the network and evaluates results
        against verification constraints to determine if the sample is safe.
        
        Args:
            input_lb: Input lower bounds tensor for the sample
            input_ub: Input upper bounds tensor for the sample  
            sample_idx: Index of the current sample being processed
            
        Returns:
            VerifyResult indicating the verification result (SAT/UNSAT/UNKNOWN)
        """
        ACTLog.log_verification_info(f"Performing Interval propagation")

        # Get ReLU constraints from BaB refinement if available
        relu_constraints = getattr(self, 'current_relu_constraints', None)
        
        # Create propagator and run bound propagation
        # Use dedicated interval bound propagator with BaB constraints
        propagator = BoundsPropagate(relu_constraints)
        output_lb, output_ub, _ = propagator.propagate_bounds(
            self.spec.model.pytorch_model, input_lb, input_ub
        )

        verdict = OutputsEvaluate.evaluate_output_bounds(
            output_lb, output_ub,
            self.spec.output_spec.output_constraints if self.spec.output_spec.output_constraints is not None else None,
            self.spec.output_spec.labels[sample_idx].item() if self.spec.output_spec.labels is not None else None
        )

        ACTLog.log_verification_info(f"📊 Verification verdict: {verdict.name}")
        return verdict
