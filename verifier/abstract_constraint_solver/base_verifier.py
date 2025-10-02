#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#########################################################################
##   Abstract Constraint Transformer (ACT) - Base Verifier             ##
##                                                                     ##
##   doctormeeee (https://github.com/doctormeeee) and contributors     ##
##   Copyright (C) 2024-2025                                           ##
##                                                                     ##
#########################################################################

from typing import Dict, List, Optional, Any, Tuple
import torch
import torch.nn as nn
import numpy as np
import psutil
import os
import sys


import path_config

from input_parser.model import Model
from input_parser.dataset import Dataset
from input_parser.spec import Spec, InputSpec, OutputSpec
from input_parser.type import SpecType, VerificationStatus
from bab_refinement.bab_spec_refinement import create_spec_refinement_core

def print_memory_usage(stage_name=""):
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024

        system_memory = psutil.virtual_memory()
        total_mb = system_memory.total / 1024 / 1024
        available_mb = system_memory.available / 1024 / 1024
        used_percent = (memory_mb / total_mb) * 100

        print(f"[{stage_name}] Memory Usage:")
        print(f"Process: {memory_mb:.1f} MB ({used_percent:.1f}% of total)")
        print(f"System: {total_mb:.1f} MB total, {available_mb:.1f} MB available")

        if torch.cuda.is_available():
            gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
            gpu_cached_mb = torch.cuda.memory_reserved() / 1024 / 1024
            print(f"GPU: {gpu_memory_mb:.1f} MB allocated, {gpu_cached_mb:.1f} MB cached")

        return memory_mb
    except ImportError:
        print(f"⚠️ [{stage_name}] psutil not available, cannot monitor memory")
        return 0

class BaseVerifier:
    def __init__(self, dataset : Dataset, spec : Spec, device: str = 'cpu'):
        self.dataset = dataset
        self.spec = spec
        self.device = device
        
        # Get input center, compute if not available
        if hasattr(self.spec.input_spec, 'input_center') and self.spec.input_spec.input_center is not None:
            self.input_center = self.spec.input_spec.input_center
        else:
            # Calculate input_center from bounds for VNNLIB formats
            self.input_center = (self.spec.input_spec.input_lb + self.spec.input_spec.input_ub) / 2.0
            
        self.input_lb = self.spec.input_spec.input_lb
        self.input_ub = self.spec.input_spec.input_ub
        self.model = self.spec.model

        self.dtype = torch.float32

        self.bab_config = {
            'enabled': False,
            'max_depth': 8,
            'max_subproblems': 500,
            'time_limit': 1500.0,
            'split_tolerance': 1e-6,
            'verbose': True
        }

        self.current_relu_constraints = []

        self.verbose = True

        self.clean_prediction_stats = {
            'total_samples': 0,
            'clean_correct': 0,
            'clean_incorrect': 0,
            'verification_attempted': 0,
            'verification_sat': 0,
            'verification_unsat': 0,
            'verification_unknown': 0
        }

        model_shape = self.model.get_expected_input_shape()
        self._adapt_input_shape(model_shape)

    def _adapt_input_shape(self, model_shape):
        
        if self.input_lb.shape != self.input_ub.shape:
            raise ValueError("Input lower and upper bounds must have the same size.")

        # Case 1: Already in correct shape
        if self.input_lb.shape[1:] == model_shape[1:]:
            return
        
        # Case 2: Flat input needs reshaping to multi-dimensional (e.g., [1, 784] -> [1, 1, 28, 28])
        elif len(self.input_lb.shape) == 2 and self.input_lb.shape[1] == int(np.prod(model_shape[1:])):
            self.input_lb = self.input_lb.view(self.input_lb.shape[0], *model_shape[1:])
            self.input_ub = self.input_ub.view(self.input_ub.shape[0], *model_shape[1:])
            
        # Case 3: Missing channel dimension (e.g., [1, 28, 28] -> [1, 1, 28, 28])
        elif self.input_lb.shape[1:] == model_shape[2:] and model_shape[1] == 1:
            self.input_lb = self.input_lb.unsqueeze(1)
            self.input_ub = self.input_ub.unsqueeze(1)
            # Also adapt input_center
            if hasattr(self, 'input_center') and self.input_center is not None and self.input_center.shape[1:] == model_shape[2:]:
                self.input_center = self.input_center.unsqueeze(1)
            
        else:
            raise RuntimeError(f"Input shape mismatch: cannot adapt {self.input_lb.shape} to model shape {model_shape}. "
                             f"Expected total elements: {int(np.prod(model_shape[1:]))}, "
                             f"got: {int(np.prod(self.input_lb.shape[1:]))}")

    def verify(self, proof, public_inputs):
        raise NotImplementedError("Subclasses must implement verify method")

    def check_clean_prediction(self, sample_input, true_label, sample_idx=0):
        
        self.clean_prediction_stats['total_samples'] += 1

        try:
            # Handle different input shapes
            if sample_input.ndim == 1:
                # Flat input (e.g., [784] from CSV) - need to add batch dimension and reshape
                expected_shape = self.model.get_expected_input_shape()
                
                if len(expected_shape) == 4:  # CNN model expecting [batch, channels, height, width]
                    sample_input = sample_input.view(1, *expected_shape[1:])
                else:
                    sample_input = sample_input.unsqueeze(0)
                    
            elif sample_input.ndim == 2:
                # [batch, features] format - might need reshaping for CNN
                if sample_input.shape[0] == 1:  # Already has batch dimension
                    expected_shape = self.model.get_expected_input_shape()
                    if len(expected_shape) == 4 and sample_input.shape[1] == int(np.prod(expected_shape[1:])):
                        # Reshape [1, 784] to [1, 1, 28, 28] for MNIST or [1, 3072] to [1, 3, 32, 32] for CIFAR10
                        sample_input = sample_input.view(1, *expected_shape[1:])
                else:
                    # Multiple samples - take first one
                    sample_input = sample_input[0:1]
                    
            elif sample_input.ndim == 3:
                # [channels, height, width] format - add batch dimension
                sample_input = sample_input.unsqueeze(0)
                
            elif sample_input.ndim == 4:
                # Already in [batch, channels, height, width] format
                if sample_input.shape[0] != 1:
                    sample_input = sample_input[0:1]  # Take first sample

            with torch.no_grad():
                self.model.pytorch_model.eval()
                outputs = self.model.pytorch_model(sample_input)
                predicted_label = torch.argmax(outputs, dim=1).item()

                final_layer_values = outputs.squeeze(0).detach().cpu().numpy()

            is_correct = (predicted_label == true_label)

            if is_correct:
                self.clean_prediction_stats['clean_correct'] += 1
                if self.verbose:
                    print(f"✅ Sample {sample_idx+1} Clean Prediction: Correct (pred: {predicted_label}, true: {true_label})")
            else:
                self.clean_prediction_stats['clean_incorrect'] += 1
                if self.verbose:
                    print(f"❌ Sample {sample_idx+1} Clean Prediction: Incorrect (pred: {predicted_label}, true: {true_label})")
                    print(f"⚠️  Skipping verification - clean prediction already incorrect")

            return is_correct

        except Exception as e:
            print(f"❌ Clean prediction check failed: {e}")
            self.clean_prediction_stats['clean_incorrect'] += 1
            return False

    def get_sample_center_and_label(self, sample_idx=0):

        if self.input_center.ndim > 1 and self.input_center.shape[0] > 1:

            center_input = self.input_center[sample_idx:sample_idx+1]
        else:

            if self.input_center.ndim == 1:

                center_input = self.input_center.unsqueeze(0)
            elif self.input_center.ndim == 3:

                center_input = self.input_center.unsqueeze(0)
            else:

                center_input = self.input_center

        if hasattr(self.dataset, 'labels') and self.dataset.labels is not None:
            if sample_idx < len(self.dataset.labels):
                true_label = self.dataset.labels[sample_idx].item()
            else:
                true_label = self.dataset.labels[0].item()
        elif hasattr(self.dataset, 'true_labels') and self.dataset.true_labels is not None:
            if sample_idx < len(self.dataset.true_labels):
                true_label = self.dataset.true_labels[sample_idx].item()
            else:
                true_label = self.dataset.true_labels[0].item()
        else:

            if hasattr(self.spec.output_spec, 'true_labels') and self.spec.output_spec.true_labels is not None:
                if sample_idx < len(self.spec.output_spec.true_labels):
                    true_label = self.spec.output_spec.true_labels[sample_idx].item()
                else:
                    true_label = self.spec.output_spec.true_labels[0].item()
            else:

                print(f"⚠️  Could not get true label for sample {sample_idx+1}, using default label 0")
                true_label = 0

        return center_input, true_label

    def print_verification_stats(self):
        stats = self.clean_prediction_stats
        total = stats['total_samples']

        if total == 0:
            print("📊 Verification stats: no sample data")
            return
        print(f"\n📊 Verification summary:")
        print("="*60)
        print(f"Total samples: {total}")
        print(f"Clean Prediction correct: {stats['clean_correct']} ({stats['clean_correct']/total*100:.1f}%)")
        print(f"Clean Prediction incorrect: {stats['clean_incorrect']} ({stats['clean_incorrect']/total*100:.1f}%)")
        print(f"Samples attempted for verification: {stats['verification_attempted']} ({stats['verification_attempted']/total*100:.1f}%)")

        if stats['verification_attempted'] > 0:
            attempted = stats['verification_attempted']
            print(f"Verification result distribution:")
            print(f"SAT (safe): {stats['verification_sat']} ({stats['verification_sat']/attempted*100:.1f}%)")
            print(f"UNSAT (unsafe): {stats['verification_unsat']} ({stats['verification_unsat']/attempted*100:.1f}%)")
            print(f"UNKNOWN: {stats['verification_unknown']} ({stats['verification_unknown']/attempted*100:.1f}%)")

        print("="*60)

    def set_relu_constraints(self, relu_constraints: List[Dict[str, Any]]):

        self.current_relu_constraints = relu_constraints.copy() if relu_constraints else []

        if hasattr(self, 'verbose') and getattr(self, 'verbose', False):
            if self.current_relu_constraints:
                print(f"Set ReLU constraints: {len(self.current_relu_constraints)} constraints")
                for constraint in self.current_relu_constraints:
                    print(f"{constraint['layer']}[{constraint['neuron_idx']}] = {constraint['constraint_type']}")
            else:
                print(f"Cleared ReLU constraints")

    def get_relu_constraints(self) -> List[Dict[str, Any]]:
        return self.current_relu_constraints.copy()

    def apply_relu_constraints_to_bounds(self):

        if not self.current_relu_constraints:
            return

        if self.verbose:
            print(f"[Bounds Fix] Applying {len(self.current_relu_constraints)} ReLU constraints to layer bounds cache")

        bounds_cache = None
        if hasattr(self, 'hz_layer_bounds') and self.hz_layer_bounds:
            bounds_cache = self.hz_layer_bounds
            cache_type = "HybridZonotope"
        elif hasattr(self, 'autolirpa_layer_bounds') and self.autolirpa_layer_bounds:
            bounds_cache = self.autolirpa_layer_bounds
            cache_type = "AutoLiRPA"
        else:
            if self.verbose:
                print("⚠️  No layer bounds cache found, skipping constraint application")
            return

        if self.verbose:
            print(f"[Bounds Fix] Using {cache_type} bounds cache")

        layer_names = list(bounds_cache.keys())

        for constraint in self.current_relu_constraints:
            relu_layer = constraint['layer']
            neuron_idx = constraint['neuron_idx']
            constraint_type = constraint['constraint_type']

            if self.verbose:
                print(f"[Bounds Fix] Processing constraint: {relu_layer}[{neuron_idx}] = {constraint_type}")

            relu_layer_idx = None
            for i, layer_name in enumerate(layer_names):
                if layer_name == relu_layer:
                    relu_layer_idx = i
                    break

            if relu_layer_idx is None:
                if self.verbose:
                    print(f"⚠️  ReLU layer not found: {relu_layer}")
                continue

            if relu_layer_idx == 0:
                if self.verbose:
                    print(f"⚠️  ReLU layer is first layer, no previous layer")
                continue

            prev_layer_name = layer_names[relu_layer_idx - 1]
            prev_layer_data = bounds_cache[prev_layer_name]

            if 'lb' not in prev_layer_data or 'ub' not in prev_layer_data:
                if self.verbose:
                    print(f"⚠️  Previous layer {prev_layer_name} missing bounds data")
                continue

            lb = prev_layer_data['lb']
            ub = prev_layer_data['ub']

            if neuron_idx >= lb.numel():
                if self.verbose:
                    print(f"⚠️  Neuron index {neuron_idx} out of bounds, layer shape: {lb.shape}")
                continue

            if constraint_type == 'inactive':

                original_ub = ub.view(-1)[neuron_idx].item()
                ub.view(-1)[neuron_idx] = min(original_ub, 0.0)
                if self.verbose:
                    print(f"inactive constraint: neuron {neuron_idx} ub: {original_ub:.6f} → {ub.view(-1)[neuron_idx].item():.6f}")

            elif constraint_type == 'active':

                original_lb = lb.view(-1)[neuron_idx].item()
                lb.view(-1)[neuron_idx] = max(original_lb, 0.0)
                if self.verbose:
                    print(f"active constraint: neuron {neuron_idx} lb: {original_lb:.6f} → {lb.view(-1)[neuron_idx].item():.6f}")

            else:
                if self.verbose:
                    print(f"⚠️  Unknown constraint type: {constraint_type}")

        if self.verbose:
            print(f"✅ [Bounds Fix] ReLU constraint application done")

    def _single_result_verdict(self, lb: torch.Tensor, ub: torch.Tensor,
                       output_constraints: Optional[List[List[float]]],
                       true_label: Optional[int]) -> VerificationStatus:

        if output_constraints is not None:
            for row in output_constraints:
                a = torch.tensor(row[:-1], device=lb.device)
                b = row[-1]

                worst = torch.sum(torch.where(a>=0, a*lb, a*ub)) + b
                if worst < 0:
                    return VerificationStatus.UNSAT
            return VerificationStatus.SAT

        if true_label is not None:
            print(f"Checking classification for true_label: {true_label}")

            if true_label < 0 or true_label >= lb.shape[0]:
                print(f"Warning: true_label {true_label} is out of bounds [0, {lb.shape[0]-1}]. Skipping classification check.")
                return VerificationStatus.UNKNOWN

            if torch.any(torch.isnan(lb)) or torch.any(torch.isnan(ub)):
                print(f"Warning: Found NaN values in output bounds. lb contains NaN: {torch.any(torch.isnan(lb))}, ub contains NaN: {torch.any(torch.isnan(ub))}")
                return VerificationStatus.UNKNOWN

            print(f"Using traditional bounds comparison for classification check")
            for j in range(lb.shape[0]):
                if j == true_label: continue
                if ub[j] >= lb[true_label]:
                    return VerificationStatus.UNSAT
            return VerificationStatus.SAT

        return VerificationStatus.UNKNOWN

    def _spec_refinement_verification(self, input_lb: torch.Tensor, input_ub: torch.Tensor, sample_idx: int = 0) -> VerificationStatus:
        print(f"Starting generic BaB specification refinement verification (sample {sample_idx})")
        print(f"Framework: theoretically-aligned specification refinement")

        spec_refinement = create_spec_refinement_core(
            max_depth=self.bab_config['max_depth'],
            max_subproblems=self.bab_config['max_subproblems'],
            time_limit=self.bab_config['time_limit'],
            spurious_check_enabled=self.bab_config.get('spurious_check', True),
            verbose=self.bab_config['verbose']
        )

        spec_refinement._current_verifier = self

        concrete_network = self.spec.model.pytorch_model

        try:
            result = spec_refinement.search(
                input_lb, input_ub,
                self,
                concrete_network
            )

            print(f"Specification refinement verification finished: {result.status.name}")
            print(f"Total subproblems: {result.total_subproblems}")
            print(f"Spurious counterexamples: {len(result.spurious_counterexamples)}")
            print(f"Real counterexample: {'Yes' if result.real_counterexample else 'No'}")
            print(f"Max depth: {result.max_depth}")
            print(f"Total time: {result.total_time:.2f}s")

            return result.status

        except Exception as e:
            print(f"⚠️ Specification refinement verification error: {e}")
            return VerificationStatus.UNKNOWN

    def _all_results_verdict(self, results: List[VerificationStatus]) -> VerificationStatus:
        print("\n" + "🏆" + "="*70 + "🏆")
        print("📊 Final verification results summary")
        print("🏆" + "="*70 + "🏆")

        for idx, result in enumerate(results):
            print(f"Sample {idx+1}: {result.name}")

        print("-" * 60)

        sat_count = sum(1 for r in results if r == VerificationStatus.SAT)
        unsat_count = sum(1 for r in results if r == VerificationStatus.UNSAT)
        clean_failure_count = sum(1 for r in results if r == VerificationStatus.CLEAN_FAILURE)
        unknown_count = sum(1 for r in results if r == VerificationStatus.UNKNOWN)
        total_count = len(results)

        valid_count = total_count - clean_failure_count

        print("Verification statistics:")
        print(f"Total samples: {total_count}")
        print(f"✅ SAT (safe): {sat_count} ")
        print(f"❌ UNSAT (unsafe): {unsat_count} ")
        print(f"⚠️  CLEAN_FAILURE (clean prediction failed): {clean_failure_count} ")
        print(f"❓ UNKNOWN: {unknown_count} ")
        print(f"🔍 Valid verification samples: {valid_count} ")

        if valid_count > 0:
            sat_percentage = (sat_count / valid_count) * 100
            unsat_percentage = (unsat_count / valid_count) * 100
            print(f"📊 SAT over valid samples: {sat_percentage:.2f}% ({sat_count}/{valid_count})")
            print(f"📊 UNSAT over valid samples: {unsat_percentage:.2f}% ({unsat_count}/{valid_count})")
        else:
            print("  ⚠️  No valid verification samples")

        if total_count > 0:
            sat_total_percentage = (sat_count / total_count) * 100
            unsat_total_percentage = (unsat_count / total_count) * 100
            clean_failure_percentage = (clean_failure_count / total_count) * 100
            print(f"📊 SAT over total: {sat_total_percentage:.2f}% ({sat_count}/{total_count})")
            print(f"📊 UNSAT over total: {unsat_total_percentage:.2f}% ({unsat_count}/{total_count})")
            print(f"📊 CLEAN_FAILURE over total: {clean_failure_percentage:.2f}% ({clean_failure_count}/{total_count})")

        print("-" * 60)

        if all(r == VerificationStatus.SAT for r in results):
            final_result = VerificationStatus.SAT
            print("✅ Final Result: SAT - all samples verified safe")
        elif any(r == VerificationStatus.UNSAT for r in results):
            final_result = VerificationStatus.UNSAT
            print("❌ Final Result: UNSAT - at least one sample violates the property")
        else:
            final_result = VerificationStatus.UNKNOWN
            print("❓ Final Result: UNKNOWN - inconclusive")

        print("🏆" + "="*70 + "🏆")
        return final_result