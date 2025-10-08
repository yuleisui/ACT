#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#########################################################################
##   Abstract Constraint Transformer (ACT) - Interval Verifier         ##
##                                                                     ##
##   doctormeeee (https://github.com/doctormeeee) and contributors     ##
##   Copyright (C) 2024-2025                                           ##
##                                                                     ##
#########################################################################

import torch
import torch.nn as nn
from typing import Tuple, Optional, List

from abstract_constraint_solver.base_verifier import BaseVerifier
from input_parser.dataset import Dataset
from input_parser.spec import Spec
from input_parser.type import VerificationStatus
from util.stats import ACTStats, ACTLog
from onnx2pytorch.operations.flatten import Flatten as OnnxFlatten
from onnx2pytorch.operations.add import Add as OnnxAdd
from onnx2pytorch.operations.div import Div as OnnxDiv
from onnx2pytorch.operations.clip import Clip as OnnxClip
from onnx2pytorch.operations.reshape import Reshape as OnnxReshape
from onnx2pytorch.operations.squeeze import Squeeze as OnnxSqueeze
from onnx2pytorch.operations.unsqueeze import Unsqueeze as OnnxUnsqueeze
from onnx2pytorch.operations.transpose import Transpose as OnnxTranspose
from onnx2pytorch.operations.base import OperatorWrapper

class IntervalVerifier(BaseVerifier):
    def __init__(self, dataset : Dataset, method, spec : Spec, device: str = 'cpu'):
        super().__init__(dataset, spec, device)
        if method != 'interval':
            raise ValueError(f"IntervalVerifier only supports 'interval' method, got {method}.")

    def _abstract_constraint_solving_core(self, model: nn.Module, input_lb: torch.Tensor, input_ub: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        lb = input_lb.clone()
        ub = input_ub.clone()

        layer_index = 0

        for layer in model.children():
            print("Layer type:", type(layer))
        for layer in model.children():

            if isinstance(layer, nn.Linear):
                W = layer.weight
                b = layer.bias

                W_pos = torch.clamp(W, min=0)
                W_neg = torch.clamp(W, max=0)
                next_lb = torch.matmul(W_pos, lb) + torch.matmul(W_neg, ub)
                next_ub = torch.matmul(W_pos, ub) + torch.matmul(W_neg, lb)
                if b is not None:
                    next_lb = next_lb + b
                    next_ub = next_ub + b
                lb, ub = next_lb, next_ub
                layer_index += 1

            elif isinstance(layer, nn.Conv2d):

                W = layer.weight
                b = layer.bias
                stride = layer.stride
                padding = layer.padding
                W_pos = torch.clamp(W, min=0)
                W_neg = torch.clamp(W, max=0)

                next_lb = (
                    nn.functional.conv2d(lb.unsqueeze(0), W_pos, None, stride, padding) +
                    nn.functional.conv2d(ub.unsqueeze(0), W_neg, None, stride, padding)
                ).squeeze(0)
                next_ub = (
                    nn.functional.conv2d(ub.unsqueeze(0), W_pos, None, stride, padding) +
                    nn.functional.conv2d(lb.unsqueeze(0), W_neg, None, stride, padding)
                ).squeeze(0)

                if b is not None:

                    next_lb += b.view(-1, 1, 1)
                    next_ub += b.view(-1, 1, 1)
                lb, ub = next_lb, next_ub

            elif isinstance(layer, nn.ReLU):

                layer_name = f"relu_{layer_index}"
                applied_constraints = []

                if hasattr(self, 'current_relu_constraints'):
                    for constraint in self.current_relu_constraints:
                        if constraint['layer'] == layer_name:
                            neuron_idx = constraint['neuron_idx']
                            constraint_type = constraint['constraint_type']

                            if constraint_type == 'inactive':

                                if neuron_idx < lb.numel():
                                    flat_lb = lb.view(-1)
                                    flat_ub = ub.view(-1)

                                    flat_ub[neuron_idx] = min(flat_ub[neuron_idx], 0.0)
                                    lb = flat_lb.view(lb.shape)
                                    ub = flat_ub.view(ub.shape)
                                    applied_constraints.append(f"ReLU[{neuron_idx}]=inactive")

                            elif constraint_type == 'active':

                                if neuron_idx < lb.numel():
                                    flat_lb = lb.view(-1)
                                    flat_ub = ub.view(-1)

                                    flat_lb[neuron_idx] = max(flat_lb[neuron_idx], 0.0)
                                    lb = flat_lb.view(lb.shape)
                                    ub = flat_ub.view(ub.shape)
                                    applied_constraints.append(f"ReLU[{neuron_idx}]=active")

                if applied_constraints and hasattr(self, 'verbose') and getattr(self, 'verbose', False):
                    print(f"applying{layer_name}constraint: {applied_constraints}")

                lb = torch.clamp(lb, min=0)
                ub = torch.clamp(ub, min=0)
                layer_index += 1

            elif isinstance(layer, nn.Sigmoid):
                lb = torch.sigmoid(lb)
                ub = torch.sigmoid(ub)

            elif isinstance(layer, nn.MaxPool2d):
                lb = nn.functional.max_pool2d(lb, kernel_size=layer.kernel_size, stride=layer.stride, padding=layer.padding)
                ub = nn.functional.max_pool2d(ub, kernel_size=layer.kernel_size, stride=layer.stride, padding=layer.padding)

            elif isinstance(layer, nn.Flatten) or isinstance(layer, OnnxFlatten):
                print("Flattening layer detected.")
                lb = torch.flatten(lb, start_dim=0)
                ub = torch.flatten(ub, start_dim=0)

            elif isinstance(layer, nn.BatchNorm2d):
                mean = layer.running_mean
                var = layer.running_var
                eps = layer.eps
                weight = layer.weight
                bias = layer.bias
                lb = (lb - mean[None, :, None, None]) / torch.sqrt(var[None, :, None, None] + eps) * weight[None, :, None, None] + bias[None, :, None, None]
                ub = (ub - mean[None, :, None, None]) / torch.sqrt(var[None, :, None, None] + eps) * weight[None, :, None, None] + bias[None, :, None, None]

            elif isinstance(layer, OnnxAdd):
                print("OnnxAdd layer attributes:", dir(layer))
                print("OnnxAdd layer __dict__:", vars(layer))

                lb = layer(lb)
                ub = layer(ub)

            elif isinstance(layer, OnnxDiv):

                lb = layer(lb)
                ub = layer(ub)
                lb_new = torch.min(lb, ub)
                ub_new = torch.max(lb, ub)
                lb, ub = lb_new, ub_new

            elif isinstance(layer, OnnxClip):
                min_val = layer.min
                max_val = layer.max
                lb = torch.clamp(lb, min=min_val, max=max_val)
                ub = torch.clamp(ub, min=min_val, max=max_val)

            elif isinstance(layer, OnnxReshape):

                if hasattr(layer, "shape"):
                    new_shape = layer.shape
                elif hasattr(layer, "target_shape"):
                    new_shape = layer.target_shape
                elif hasattr(layer, "_shape"):
                    new_shape = layer._shape
                else:
                    raise AttributeError(f"Cannot find shape attribute in Reshape layer: {dir(layer)}")
                lb = lb.reshape(list(new_shape))
                ub = ub.reshape(list(new_shape))

            elif isinstance(layer, OnnxSqueeze):
                dim = layer.dim
                lb = lb.squeeze(dim)
                ub = ub.squeeze(dim)

            elif isinstance(layer, OnnxUnsqueeze):
                dim = layer.dim
                lb = lb.unsqueeze(dim)
                ub = ub.unsqueeze(dim)

            elif isinstance(layer, OnnxTranspose):
                if hasattr(layer, "perm"):
                    perm = layer.perm
                elif hasattr(layer, "dims"):
                    perm = layer.dims
                else:
                    raise AttributeError(f"Cannot find perm attribute in Transpose layer: {dir(layer)}")
                lb = lb.permute(*perm)
                ub = ub.permute(*perm)

            elif isinstance(layer, OperatorWrapper):

                if hasattr(layer, 'op_type') and layer.op_type in ["Add", "Sub", "Mul", "Div"]:
                    other = getattr(layer, 'other', None)
                    if other is None:

                        lb = layer(lb)
                        ub = layer(ub)
                    else:

                        if layer.op_type == "Add":
                            lb = lb + other
                            ub = ub + other
                        elif layer.op_type == "Sub":
                            lb = lb - other
                            ub = ub - other
                        elif layer.op_type == "Mul":

                            lb_new = torch.min(lb * other, ub * other)
                            ub_new = torch.max(lb * other, ub * other)
                            lb, ub = lb_new, ub_new
                        elif layer.op_type == "Div":

                            if torch.any(other == 0):
                                raise ValueError("Division by zero encountered in OperatorWrapper Div layer.")
                            lb_new = torch.min(lb / other, ub / other)
                            ub_new = torch.max(lb / other, ub / other)
                            lb, ub = lb_new, ub_new
                else:

                    lb = layer(lb)
                    ub = layer(ub)

            else:
                raise NotImplementedError(f"Layer {layer} not supported in interval propagation.")

        return lb, ub, None

    def _abstract_constraint_solving(self, input_lb: torch.Tensor, input_ub: torch.Tensor, sample_idx: int) -> VerificationStatus:
        print(f"Performing Interval propagation")

        output_lb, output_ub, _ = self._abstract_constraint_solving_core(
            self.spec.model.pytorch_model, input_lb, input_ub
        )

        verdict = self._evaluate_output_bounds(
            output_lb, output_ub,
            self.spec.output_spec.output_constraints if self.spec.output_spec.output_constraints is not None else None,
            self.spec.output_spec.labels[sample_idx].item() if self.spec.output_spec.labels is not None else None
        )

        print(f"📊 Verification verdict: {verdict.name}")
        return verdict

    def verify(self) -> VerificationStatus:
        print("Starting Interval verification pipeline")

        num_samples = self.input_center.shape[0] if self.input_center.ndim > 1 else 1
        print(f"Total samples: {num_samples}")
        results = []
        for idx in range(num_samples):
            print(f"\n🔍 Processing sample {idx+1}/{num_samples}")
            print("="*80)

            center_input, true_label = self.extract_sample_input_and_label(idx)
            if not self.validate_unperturbed_prediction(center_input, true_label, idx):
                print(f"⏭️  Skipping verification for sample {idx+1}")
                results.append(VerificationStatus.CLEAN_FAILURE)
                continue

            if self.input_lb.ndim == 1:
                lb_i = self.input_lb
                ub_i = self.input_ub
            else:
                lb_i = self.input_lb[idx]
                ub_i = self.input_ub[idx]

            self.clean_prediction_stats['verification_attempted'] += 1

            print("Step 1: Interval abstract constraint solving")
            initial_verdict = self._abstract_constraint_solving(lb_i, ub_i, idx)

            if initial_verdict == VerificationStatus.SAT:
                self.clean_prediction_stats['verification_sat'] += 1
                print(f"✅ Interval verification success - Sample {idx+1} safe")
                results.append(initial_verdict)
                continue
            elif initial_verdict == VerificationStatus.UNSAT:
                self.clean_prediction_stats['verification_unsat'] += 1
            else:
                self.clean_prediction_stats['verification_unknown'] += 1

            if initial_verdict == VerificationStatus.UNSAT:
                print(f"❌ Interval potential violation detected - Sample {idx+1}")
            else:
                print(f"❓ Interval inconclusive - Sample {idx+1}")

            print("Launching Specification Refinement BaB process")
            print("="*60)

            if self.bab_config['enabled']:
                refinement_verdict = self._spec_refinement_verification(lb_i, ub_i, idx)
                if refinement_verdict == VerificationStatus.SAT:
                    self.clean_prediction_stats['verification_sat'] += 1
                elif refinement_verdict == VerificationStatus.UNSAT:
                    self.clean_prediction_stats['verification_unsat'] += 1
                else:
                    self.clean_prediction_stats['verification_unknown'] += 1
                results.append(refinement_verdict)
            else:
                print("⚠️  BaB disabled, returning initial verdict")
                results.append(initial_verdict)

        ACTStats.print_verification_stats(self.clean_prediction_stats)
        return ACTStats.print_final_verification_summary(results)

    def _evaluate_output_bounds(self, lb: torch.Tensor, ub: torch.Tensor,
                             output_constraints: Optional[List[List[float]]],
                             true_label: Optional[int]) -> VerificationStatus:
        """
        Evaluate output bounds to determine verification result.
        
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
            # Check each linear constraint: coefficients·outputs + bias ≥ 0
            for row in output_constraints:
                # Split constraint: [coeff1, coeff2, ..., coeffN, bias]
                coeffs = torch.tensor(row[:-1], device=lb.device)  # All except last
                bias = row[-1]  # Last element
                
                # Interval arithmetic: find worst-case (minimum) constraint value
                # Positive coefficients use lower bounds, negative coefficients use upper bounds
                terms = torch.where(coeffs >= 0, coeffs * lb, coeffs * ub)
                worst_case = torch.sum(terms) + bias
                
                # If worst case is negative, constraint is definitely violated
                if worst_case < 0:
                    return VerificationStatus.UNSAT
            
            # All constraints passed worst-case check - potentially satisfiable
            return VerificationStatus.SAT

        if true_label is not None:
            ACTLog.log_verification_info(f"Checking classification robustness for true_label: {true_label}")
            
            # Validate true_label is within valid output class range
            num_classes = lb.shape[0]
            if true_label < 0 or true_label >= num_classes:
                ACTLog.log_verification_warning(f"true_label {true_label} out of bounds (valid range: 0-{num_classes-1})")
                return VerificationStatus.UNKNOWN

            # Check for numerical instability in output bounds
            if torch.any(torch.isnan(lb)) or torch.any(torch.isnan(ub)):
                ACTLog.log_verification_warning("Found NaN values in output bounds - numerical instability detected")
                return VerificationStatus.UNKNOWN

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
                        return VerificationStatus.UNSAT
            
            # All other classes have maximum outputs below true class minimum - robustness proven
            ACTLog.log_verification_info("Classification robustness verified: no adversarial examples possible")
            return VerificationStatus.SAT

        return VerificationStatus.UNKNOWN
