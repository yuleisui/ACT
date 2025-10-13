#===- verifier.interval.outputs_evaluation.py output bounds evaluation --#
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
# Output bounds evaluation for neural network verification results analysis.
# Implements constraint checking for linear output constraints and classification
# robustness verification using interval arithmetic bounds comparison.
#
#===----------------------------------------------------------------------===#

import torch
from typing import List, Optional

from act.interval.input_parser.type import VerifyResult
from act.interval.util.stats import ACTLog
from act.interval.util.bounds import Bounds


class OutputsEvaluate:
    """
    Output bounds evaluation for neural network verification results analysis.
    
    Provides methods for evaluating computed interval bounds against verification
    constraints including linear output constraints and classification robustness.
    """
    
    @staticmethod
    def evaluate_output_bounds(output_bounds: Bounds,
                             output_constraints: Optional[List[List[float]]],
                             true_label: Optional[int]) -> VerifyResult:
        """
        Evaluate output bounds to determine verification result.
        
        Analyzes computed interval bounds against either linear output constraints
        or classification robustness requirements to determine verification status.
        
        Args:
            output_bounds: Bounds object containing lower and upper bounds for network outputs
            output_constraints: Linear constraint matrix (optional)
            true_label: Ground truth class index for classification (optional)
            
        Returns:
            VerifyResult indicating SAT (safe), UNSAT (unsafe), or UNKNOWN
        """
        if output_constraints is not None:
            return OutputsEvaluate._eval_linear_constraints(output_bounds, output_constraints)
        
        if true_label is not None:
            return OutputsEvaluate._eval_classification(output_bounds, true_label)
        
        return VerifyResult.UNKNOWN

    @staticmethod
    def _eval_linear_constraints(output_bounds: Bounds, 
                               constraints: List[List[float]]) -> VerifyResult:
        """Evaluate linear output constraints using interval arithmetic."""
        lb, ub = output_bounds.lb, output_bounds.ub
        
        # Check each linear constraint: coefficients·outputs + bias ≥ 0
        for row in constraints:
            # Split constraint: [coeff1, coeff2, ..., coeffN, bias]
            coeffs = torch.tensor(row[:-1], device=lb.device)  # All except last
            bias = row[-1]  # Last element
            
            # Interval arithmetic: find worst-case (minimum) constraint value
            # Positive coefficients use lower bounds, negative coefficients use upper bounds
            terms = torch.where(coeffs >= 0, coeffs * lb, coeffs * ub)
            worst_case = torch.sum(terms) + bias
            
            # If worst case is negative, constraint is definitely violated
            if worst_case < 0:
                return VerifyResult.UNSAT
        
        # All constraints passed worst-case check - potentially satisfiable
        return VerifyResult.SAT

    @staticmethod
    def _eval_classification(output_bounds: Bounds, 
                           true_label: int) -> VerifyResult:
        """Evaluate classification robustness using interval bounds comparison."""
        lb, ub = output_bounds.lb, output_bounds.ub
        
        ACTLog.log_verification_info(f"Checking classification robustness for true_label: {true_label}")
        
        # Validate true_label is within valid output class range
        num_classes = lb.shape[0]
        if true_label < 0 or true_label >= num_classes:
            ACTLog.log_verification_warning(f"true_label {true_label} out of bounds (valid range: 0-{num_classes-1})")
            return VerifyResult.UNKNOWN

        # Check for numerical instability in output bounds
        if torch.any(torch.isnan(lb)) or torch.any(torch.isnan(ub)):
            ACTLog.log_verification_warning("Found NaN values in output bounds - numerical instability detected")
            return VerifyResult.UNKNOWN

        # Core adversarial robustness check: verify no other class can exceed true class
        ACTLog.log_verification_info("Performing adversarial robustness verification via interval comparison")
        correct_min = lb[true_label]  # Minimum possible output for correct class
        
        for other_idx in range(num_classes):
            if other_idx != true_label:
                other_max = ub[other_idx]  # Maximum possible output for other class
                
                # If any other class can achieve higher output than true class minimum,
                # then adversarial examples exist within the input perturbation region
                if other_max >= correct_min:
                    ACTLog.log_verification_info(f"Adversarial vulnerability detected: class {other_idx} "
                                               f"(max: {other_max:.4f}) can exceed true class {true_label} "
                                               f"(min: {correct_min:.4f})")
                    return VerifyResult.UNSAT
        
        # All other classes have maximum outputs below true class minimum - robustness proven
        ACTLog.log_verification_info("Classification robustness verified: no adversarial examples possible")
        return VerifyResult.SAT