#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#########################################################################
##   Abstract Constraint Transformer (ACT) - Unified Verification     ##
##                                                                     ##
##   Copyright (C) 2024-2025 ACT Development Team                      ##
##                                                                     ##
##   This program integrates multiple neural network verification      ##
##   tools and provides novel hybrid zonotope verification methods.    ##
##                                                                     ##
##   External tool compatibility parameters are adapted from:          ##
##   - α,β-CROWN: https://github.com/Verified-Intelligence/alpha-beta-CROWN ##
##     Copyright (C) 2021-2025 The α,β-CROWN Team                      ##
##     Licensed under BSD 3-Clause License                             ##
##   - ERAN: https://github.com/eth-sri/eran                           ##
##     Copyright ETH Zurich, Licensed under Apache 2.0 License         ##
##                                                                     ##
##   This integration enables unified command-line interface across    ##
##   different verification backends while maintaining ACT's native    ##
##   capabilities and novel contributions.                             ##
##                                                                     ##
#########################################################################

from enum import Enum
from typing import Dict, List, Optional, Union, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import sys
import os
import psutil
import gc
import subprocess
import time
import traceback
from model import Model
from dataset import Dataset
from spec import Spec, InputSpec, OutputSpec
from type import SpecType, VerificationStatus, LPNormType
from hybridz_tensorised import HybridZonotopeGrid, HybridZonotopeOps

from bab_spec_refinement import (SpecRefinement, create_spec_refinement_core)

torch.set_printoptions(
    linewidth=500,
    threshold=10000,
    sci_mode=False,
    precision=4
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'modules', 'abcrown', 'auto_LiRPA', 'auto_LiRPA')))
try:
    from auto_LiRPA import BoundedModule, PerturbationLpNorm, BoundedTensor
    AUTOLIRPA_AVAILABLE = True
except ImportError:
    print("Warning: auto_LiRPA not available. HybridZonotopeVerifier will use standard bounds computation.")
    AUTOLIRPA_AVAILABLE = False

from onnx2pytorch.operations.flatten import Flatten as OnnxFlatten

from onnx2pytorch.operations.add import Add as OnnxAdd

from onnx2pytorch.operations.div import Div as OnnxDiv

from onnx2pytorch.operations.clip import Clip as OnnxClip

from onnx2pytorch.operations.reshape import Reshape as OnnxReshape
from onnx2pytorch.operations.squeeze import Squeeze as OnnxSqueeze
from onnx2pytorch.operations.unsqueeze import Unsqueeze as OnnxUnsqueeze
from onnx2pytorch.operations.transpose import Transpose as OnnxTranspose

from onnx2pytorch.operations.base import OperatorWrapper

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'modules', 'abcrown', 'auto_LiRPA', 'auto_LiRPA')))
try:
    from auto_LiRPA import BoundedModule, PerturbationLpNorm, BoundedTensor
    AUTOLIRPA_AVAILABLE = True
except ImportError:
    print("Warning: auto_LiRPA not available. HybridZonotopeVerifier will use standard bounds computation.")
    AUTOLIRPA_AVAILABLE = False

def print_memory_usage(stage_name=""):
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024

        system_memory = psutil.virtual_memory()
        total_mb = system_memory.total / 1024 / 1024
        available_mb = system_memory.available / 1024 / 1024
        used_percent = (memory_mb / total_mb) * 100

        print(f"🧠 [{stage_name}] Memory Usage:")
        print(f"   Process: {memory_mb:.1f} MB ({used_percent:.1f}% of total)")
        print(f"   System: {total_mb:.1f} MB total, {available_mb:.1f} MB available")

        if torch.cuda.is_available():
            gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
            gpu_cached_mb = torch.cuda.memory_reserved() / 1024 / 1024
            print(f"   GPU: {gpu_memory_mb:.1f} MB allocated, {gpu_cached_mb:.1f} MB cached")

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
                    print(f"   ⚠️  Skipping verification - clean prediction already incorrect")

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
            print(f"  SAT (safe): {stats['verification_sat']} ({stats['verification_sat']/attempted*100:.1f}%)")
            print(f"  UNSAT (unsafe): {stats['verification_unsat']} ({stats['verification_unsat']/attempted*100:.1f}%)")
            print(f"  UNKNOWN: {stats['verification_unknown']} ({stats['verification_unknown']/attempted*100:.1f}%)")

        print("="*60)

    def set_relu_constraints(self, relu_constraints: List[Dict[str, Any]]):

        self.current_relu_constraints = relu_constraints.copy() if relu_constraints else []

        if hasattr(self, 'verbose') and getattr(self, 'verbose', False):
            if self.current_relu_constraints:
                print(f"🔒 Set ReLU constraints: {len(self.current_relu_constraints)} constraints")
                for constraint in self.current_relu_constraints:
                    print(f"   {constraint['layer']}[{constraint['neuron_idx']}] = {constraint['constraint_type']}")
            else:
                print(f"🔓 Cleared ReLU constraints")

    def get_relu_constraints(self) -> List[Dict[str, Any]]:
        return self.current_relu_constraints.copy()

    def apply_relu_constraints_to_bounds(self):

        if not self.current_relu_constraints:
            return

        if self.verbose:
            print(f"🔧 [Bounds Fix] Applying {len(self.current_relu_constraints)} ReLU constraints to layer bounds cache")

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
            print(f"🔧 [Bounds Fix] Using {cache_type} bounds cache")

        layer_names = list(bounds_cache.keys())

        for constraint in self.current_relu_constraints:
            relu_layer = constraint['layer']
            neuron_idx = constraint['neuron_idx']
            constraint_type = constraint['constraint_type']

            if self.verbose:
                print(f"🔧 [Bounds Fix] Processing constraint: {relu_layer}[{neuron_idx}] = {constraint_type}")

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
                    print(f"   🔒 inactive constraint: neuron {neuron_idx} ub: {original_ub:.6f} → {ub.view(-1)[neuron_idx].item():.6f}")

            elif constraint_type == 'active':

                original_lb = lb.view(-1)[neuron_idx].item()
                lb.view(-1)[neuron_idx] = max(original_lb, 0.0)
                if self.verbose:
                    print(f"   🔒 active constraint: neuron {neuron_idx} lb: {original_lb:.6f} → {lb.view(-1)[neuron_idx].item():.6f}")

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

            print(f"🔧 Using traditional bounds comparison for classification check")
            for j in range(lb.shape[0]):
                if j == true_label: continue
                if ub[j] >= lb[true_label]:
                    return VerificationStatus.UNSAT
            return VerificationStatus.SAT

        return VerificationStatus.UNKNOWN

    def _spec_refinement_verification(self, input_lb: torch.Tensor, input_ub: torch.Tensor, sample_idx: int = 0) -> VerificationStatus:
        print(f"🌳 Starting generic BaB specification refinement verification (sample {sample_idx})")
        print(f"   Framework: theoretically-aligned specification refinement")

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

            print(f"🌳 Specification refinement verification finished: {result.status.name}")
            print(f"   Total subproblems: {result.total_subproblems}")
            print(f"   Spurious counterexamples: {len(result.spurious_counterexamples)}")
            print(f"   Real counterexample: {'Yes' if result.real_counterexample else 'No'}")
            print(f"   Max depth: {result.max_depth}")
            print(f"   Total time: {result.total_time:.2f}s")

            return result.status

        except Exception as e:
            print(f"⚠️ Specification refinement verification error: {e}")
            return VerificationStatus.UNKNOWN

    def _all_results_verdict(self, results: List[VerificationStatus]) -> VerificationStatus:
        print("\n" + "🏆" + "="*70 + "🏆")
        print("📊 Final verification results summary")
        print("🏆" + "="*70 + "🏆")

        for idx, result in enumerate(results):
            print(f"   Sample {idx+1}: {result.name}")

        print("-" * 60)

        sat_count = sum(1 for r in results if r == VerificationStatus.SAT)
        unsat_count = sum(1 for r in results if r == VerificationStatus.UNSAT)
        clean_failure_count = sum(1 for r in results if r == VerificationStatus.CLEAN_FAILURE)
        unknown_count = sum(1 for r in results if r == VerificationStatus.UNKNOWN)
        total_count = len(results)

        valid_count = total_count - clean_failure_count

        print("📈 Verification statistics:")
        print(f"   🎯 Total samples: {total_count}")
        print(f"   ✅ SAT (safe): {sat_count} ")
        print(f"   ❌ UNSAT (unsafe): {unsat_count} ")
        print(f"   ⚠️  CLEAN_FAILURE (clean prediction failed): {clean_failure_count} ")
        print(f"   ❓ UNKNOWN: {unknown_count} ")
        print(f"   🔍 Valid verification samples: {valid_count} ")

        if valid_count > 0:
            sat_percentage = (sat_count / valid_count) * 100
            unsat_percentage = (unsat_count / valid_count) * 100
            print(f"   📊 SAT over valid samples: {sat_percentage:.2f}% ({sat_count}/{valid_count})")
            print(f"   📊 UNSAT over valid samples: {unsat_percentage:.2f}% ({unsat_count}/{valid_count})")
        else:
            print("   ⚠️  No valid verification samples")

        if total_count > 0:
            sat_total_percentage = (sat_count / total_count) * 100
            unsat_total_percentage = (unsat_count / total_count) * 100
            clean_failure_percentage = (clean_failure_count / total_count) * 100
            print(f"   📊 SAT over total: {sat_total_percentage:.2f}% ({sat_count}/{total_count})")
            print(f"   📊 UNSAT over total: {unsat_total_percentage:.2f}% ({unsat_count}/{total_count})")
            print(f"   📊 CLEAN_FAILURE over total: {clean_failure_percentage:.2f}% ({clean_failure_count}/{total_count})")

        print("-" * 60)

        if all(r == VerificationStatus.SAT for r in results):
            final_result = VerificationStatus.SAT
            print("🎉 Final Result: SAT - all samples verified safe")
        elif any(r == VerificationStatus.UNSAT for r in results):
            final_result = VerificationStatus.UNSAT
            print("❌ Final Result: UNSAT - at least one sample violates the property")
        else:
            final_result = VerificationStatus.UNKNOWN
            print("❓ Final Result: UNKNOWN - inconclusive")

        print("🏆" + "="*70 + "🏆")
        return final_result


class ERANVerifier(BaseVerifier):
    def __init__(self, dataset : Dataset, method, spec : Spec, device: str = 'cpu'):
        super().__init__(dataset, spec, device)

        self.method = method

        print(f"🔍 [ERAN DEBUG] Input bounds info:")
        print(f"    input_lb shape: {self.input_lb.shape}")
        print(f"    input_ub shape: {self.input_ub.shape}")
        print(f"    input_lb unique values: {len(torch.unique(self.input_lb.view(-1)))}")
        print(f"    input_ub unique values: {len(torch.unique(self.input_ub.view(-1)))}")
        print(f"    input_lb range: [{self.input_lb.min():.6f}, {self.input_lb.max():.6f}]")
        print(f"    input_ub range: [{self.input_ub.min():.6f}, {self.input_ub.max():.6f}]")
        print(f"    input_lb first 10 values: {self.input_lb.view(-1)[:10].tolist()}")
        print(f"    input_ub first 10 values: {self.input_ub.view(-1)[:10].tolist()}")
        print(f"    Dataset input center shape: {self.dataset.input_center.shape if hasattr(self.dataset, 'input_center') and self.dataset.input_center is not None else 'None'}")
        if hasattr(self.dataset, 'input_center') and self.dataset.input_center is not None:
            print(f"    Dataset input center unique values: {len(torch.unique(self.dataset.input_center.view(-1)))}")
            print(f"    Dataset input center first 10 values: {self.dataset.input_center.view(-1)[:10].tolist()}")
        print(f"🔍 [ERAN DEBUG] End of input bounds info")
        print("="*80)

    def verify(self, proof, public_inputs):

        netname = self.spec.model.model_path
        if netname is not None and not os.path.isabs(netname):
            netname = os.path.abspath(netname)

        vnnlib_path = self.spec.input_spec.vnnlib_path
        if vnnlib_path is not None and not os.path.isabs(vnnlib_path):
            vnnlib_path = os.path.abspath(vnnlib_path)
        args_dict = {

            "netname" : netname,
            "epsilon" : self.spec.input_spec.epsilon,
            "domain" : self.method,
            "vnn_lib_spec" : vnnlib_path,
            "dataset": self.dataset.dataset_path,
            "from_test" : self.dataset.start,
            "num_tests" : self.dataset.end - self.dataset.start,
            "mean" : self.dataset.mean,
            "std"  : self.dataset.std,
            "t-norm" : self.spec.input_spec.norm.value,
        }

        args_list = []
        for k, v in args_dict.items():
            if v is not None:
                args_list.append(f"--{k}")

                if isinstance(v, list) and k in ['mean', 'std']:

                    for val in v:
                        args_list.append(str(val))
                else:
                    args_list.append(str(v))

        conda_env_name = "act-eran"
        
        # Dynamic path detection for ERAN runner
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.join(current_dir, '..')
        eran_tf_verify_path = os.path.join(project_root, 'modules', 'eran', 'tf_verify')
        eran_tf_verify_path = os.path.abspath(eran_tf_verify_path)

        cmd = ["conda", "run", "--no-capture-output", "-n", conda_env_name, "python3", "-u", "__main__.py"] + args_list

        try:
            print("[ERANVerifier] ERAN verifier running now, please wait for result generation.")
            print(f"[ERANVerifier] Command: {' '.join(cmd)}")
            print(f"[ERANVerifier] Working directory: {eran_tf_verify_path}")
            
            # Use real-time output instead of waiting for completion
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                cwd=eran_tf_verify_path
            )
            
            # Print output in real-time
            print("[ERANVerifier] Real-time output:")
            print("-" * 60)
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(output.strip())
            
            return_code = process.poll()
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, cmd)
            print("[ERANVerifier] ERAN verification completed successfully")
            return return_code
        except subprocess.CalledProcessError as e:
            print("[ERANVerifier] ERAN execution failed:")
            print(f"Return code: {e.returncode}")
            raise RuntimeError("ERAN verification failed.") from e
        except Exception as e:
            print(f"[ERANVerifier] Unexpected error: {e}")
            if process.poll() is None:
                process.terminate()
            raise RuntimeError("ERAN verification failed.") from e


class ABCROWNVerifier(BaseVerifier):
    def __init__(self, dataset : Dataset, method, spec : Spec, device: str = 'cpu'):
        super().__init__(dataset, spec, device)
        self.method = method

    def verify(self, proof, public_inputs):
        print(self.spec.input_spec.norm)
        if self.spec.spec_type == SpecType.LOCAL_LP:
            spec_type = 'lp'
        elif self.spec.spec_type == SpecType.SET_BOX:
            spec_type = 'box'

        elif self.spec.spec_type == SpecType.SET_VNNLIB or self.spec.spec_type == SpecType.LOCAL_VNNLIB:
            spec_type = 'bound'
        else:
            raise ValueError(f"Unsupported specification type for ABCROWN: {self.spec.spec_type}. Supported types are 'local_lp', 'set_box', 'set_vnnlib'.")

        netname = self.spec.model.model_path
        if netname is not None and not os.path.isabs(netname):
            netname = os.path.abspath(netname)

        norm = float(self.spec.input_spec.norm.value)

        vnnlib_path = self.spec.input_spec.vnnlib_path
        if vnnlib_path is not None and not os.path.isabs(vnnlib_path):
            vnnlib_path = os.path.abspath(vnnlib_path)

        args_dict = {
            "config": "empty_config.yaml",

            "device": self.device,
            "dataset": self.dataset.dataset_path.upper(),
            "start" : self.dataset.start,
            "end" : self.dataset.end,
            "num_outputs"  : self.dataset.num_outputs,
            "mean" : self.dataset.mean,
            "std"  : self.dataset.std,
            "spec_type" : spec_type,
            "norm" : norm,
            "epsilon" : self.spec.input_spec.epsilon,
            "vnnlib_path" : vnnlib_path
        }

        if netname.endswith(".onnx"):
            args_dict["onnx_path"] = netname
        elif netname.endswith(".pth"):
            args_dict["model"] = netname
        else:
            raise ValueError(f"Unsupported model file type: {netname}")


        args_list = []
        for k, v in args_dict.items():
            if v is not None:
                args_list.append(f"--{k}")
                if isinstance(v, list) and k in ['mean', 'std']:
                    for val in v:
                        args_list.append(str(val))
                else:
                    args_list.append(str(v))

        print("aruguments checking for abcrown")
        print(args_dict)

        conda_env_name = "act-abcrown"
        
        # Dynamic path detection for ABCROWN runner
        current_dir = os.path.dirname(os.path.abspath(__file__))
        verifier_path = os.path.abspath(current_dir) 
        
        cmd = ["conda", "run", "--no-capture-output", "-n", conda_env_name, "python3", "-u", "abcrown_runner.py", self.method] + args_list

        try:
            print("[ABCROWNVerifier] ABCROWN verifier running now, please wait for result generation.")
            print(f"[ABCROWNVerifier] Command: {' '.join(cmd)}")
            print(f"[ABCROWNVerifier] Working directory: {verifier_path}")
            
            # Use real-time output instead of waiting for completion
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                cwd=verifier_path
            )
            
            # Print output in real-time
            print("[ABCROWNVerifier] Real-time output:")
            print("-" * 60)
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(output.strip())
            
            return_code = process.poll()
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, cmd)
            print("[ABCROWNVerifier] ABCROWN verification completed successfully")
            return return_code
        except subprocess.CalledProcessError as e:
            print("[ABCROWNVerifier] ABCROWN execution failed:")
            print(f"Return code: {e.returncode}")
            raise RuntimeError("ABCROWN verification failed.") from e
        except Exception as e:
            print(f"[ABCROWNVerifier] Unexpected error: {e}")
            if process.poll() is None:
                process.terminate()
            raise RuntimeError("ABCROWN verification failed.") from e

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
                    print(f"🔒 applying{layer_name}constraint: {applied_constraints}")

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
        print(f"   🔧 Performing Interval propagation")

        output_lb, output_ub, _ = self._abstract_constraint_solving_core(
            self.spec.model.pytorch_model, input_lb, input_ub
        )

        verdict = self._single_result_verdict(
            output_lb, output_ub,
            self.spec.output_spec.output_constraints if self.spec.output_spec.output_constraints is not None else None,
            self.spec.output_spec.labels[sample_idx].item() if self.spec.output_spec.labels is not None else None
        )

        print(f"   📊 Verification verdict: {verdict.name}")
        return verdict

    def verify(self) -> VerificationStatus:
        print("🚀 Starting Interval verification pipeline")

        num_samples = self.input_center.shape[0] if self.input_center.ndim > 1 else 1
        print(f"Total samples: {num_samples}")
        results = []
        for idx in range(num_samples):
            print(f"\n🔍 Processing sample {idx+1}/{num_samples}")
            print("="*80)

            center_input, true_label = self.get_sample_center_and_label(idx)
            if not self.check_clean_prediction(center_input, true_label, idx):
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

            print("🌟 Step 1: Interval abstract constraint solving")
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

            print("🌳 Launching Specification Refinement BaB process")
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

        self.print_verification_stats()
        return self._all_results_verdict(results)

class HybridZonotopeVerifier(BaseVerifier):
    def __init__(self, dataset: Dataset, method : str, spec: Spec, device: str = 'cpu',
                 relaxation_ratio: float = 1.0, enable_generator_merging: bool = False, cosine_threshold: float = 0.95):

        from hybridz_tensorised import HybridZonotopeElem
        self.HybridZonotopeElem = HybridZonotopeElem

        super().__init__(dataset, spec, device)

        self.method = method
        self.device = device
        self.relaxation_ratio = relaxation_ratio
        self.enable_generator_merging = enable_generator_merging
        self.cosine_threshold = cosine_threshold

        self.hz_layer_bounds = {}
        self.autolirpa_layer_bounds = {}
        self.concrete_layer_values = {}
        self.layer_precision_comparison = {}
        self.soundness_check_results = {}

        self.use_auto_lirpa = False
        self.enable_layer_comparison = False
        self.enable_soundness_check = False

        if self.method == 'hybridz_relaxed':
            print(f"🎭 Relaxation Strategy: ratio={relaxation_ratio:.1f} ({'Full MILP (exact)' if relaxation_ratio == 0.0 else 'Full LP (relaxed)' if relaxation_ratio == 1.0 else f'{int(relaxation_ratio*100)}% Relaxed + {int((1-relaxation_ratio)*100)}% Exact'})")
        print(f"🔧 Generator Merging: {'Enabled' if enable_generator_merging else 'Disabled'}{f' (threshold={cosine_threshold})' if enable_generator_merging else ''}")
        if enable_generator_merging:
            print(f"   Strategy: Automatically enabling parallel generator merging at the last fully-connected layer")

        self.late_stage_config = {
            'enabled': False,
            'start_layer': -3,
            'refinement_layers': ['ReLU', 'Linear'],
            'base_verifier': 'auto_lirpa',
            'bound_method': 'IBP+backward',
        }

        if self.late_stage_config['enabled']:
            print(f"🚀 Late-stage refinement enabled:")
            print(f"   Base verifier: {self.late_stage_config['base_verifier']}")
            print(f"   HybridZ starts from layer: {self.late_stage_config['start_layer']}")
            print(f"   Refinement on: {self.late_stage_config['refinement_layers']}")
        else:
            print("⚙️  HybridZonotopeVerifier: auto_LiRPA pre-run disabled, using standard bound computation")


    def _setup_auto_lirpa(self, input_example):

        if not self.use_auto_lirpa:
            return False

        try:
            print("Setting up auto_LiRPA BoundedModule...")

            self.bounded_model = BoundedModule(
                self.model.pytorch_model,
                input_example,
                device=self.device
            )
            print("BoundedModule setup complete.")
            return True
        except Exception as e:
            print(f"Warning: auto_LiRPA setup failed: {e}")
            self.use_auto_lirpa = False
            return False

    def _create_autolirpa_ordered_layer_bounds(self):

        self.autolirpa_ordered_layer_bounds = []

        node_layer_mapping = [
            ("/input-1", "input"),
            ("/input", "conv"),
            ("/input-4", "conv"),
            ("/input-8", "conv"),
            ("/24", "relu"),

            ("/25", "flatten"),
            ("/input-12", "linear"),
            ("/27", "relu"),
            ("/28", "linear"),
        ]

        print("🔄 Creating ordered layer bounds mapping...")
        for i, (node_name, layer_type) in enumerate(node_layer_mapping):
            if node_name in self.autolirpa_layer_bounds:
                bounds = self.autolirpa_layer_bounds[node_name]
                self.autolirpa_ordered_layer_bounds.append({
                    'node_name': node_name,
                    'layer_type': layer_type,
                    'layer_index': i,
                    'lb': bounds['lb'],
                    'ub': bounds['ub'],
                    'shape': bounds['shape']
                })
                print(f"  ✅ Mapped {node_name} -> Layer {i} ({layer_type}): {bounds['shape']}")
            else:
                print(f"  ⚠️  Missing bounds for {node_name} (Layer {i}, {layer_type})")

        print(f"✅ Created ordered mapping for {len(self.autolirpa_ordered_layer_bounds)} layers")

    def _compute_autolirpa_bounds(self, input_bounds, eps=None, method='CROWN'):

        if not self.use_auto_lirpa or not hasattr(self, 'bounded_model'):
            return False

        input_lb, input_ub = input_bounds

        print(f"🔍 [Auto_LiRPA] Computing all layer bounds using method: {method}")

        try:

            input_center = (input_lb + input_ub) / 2.0

            if eps is not None and eps > 0:

                print(f"Using center point + eps={eps} perturbation")
                ptb = PerturbationLpNorm(norm=np.inf, eps=eps)
                bounded_input = BoundedTensor(input_center.unsqueeze(0), ptb)
            else:

                print("Using explicit bounds (x_L, x_U)")
                ptb = PerturbationLpNorm(
                    norm=np.inf,
                    x_L=input_lb.unsqueeze(0),
                    x_U=input_ub.unsqueeze(0)
                )
                bounded_input = BoundedTensor(input_center.unsqueeze(0), ptb)

            lb, ub = self.bounded_model.compute_bounds(
                x=(bounded_input,),
                method=method,
                IBP=(method in ['IBP', 'IBP+backward', 'CROWN-IBP']),
                forward=(method in ['Forward', 'Forward+Backward']),
                bound_lower=True,
                bound_upper=True,
                return_A=False
            )

            print(f"Final bounds range: [{lb.min().item():.6f}, {ub.max().item():.6f}]")
            for idx, (lb_, ub_) in enumerate(zip(lb, ub)):
                if isinstance(lb_, torch.Tensor):
                    lb_vals = [f"{x:.6f}" for x in lb_.detach().cpu().numpy()]
                    ub_vals = [f"{x:.6f}" for x in ub_.detach().cpu().numpy()]
                    print(f"Output[{idx}]: [{', '.join(lb_vals)}] to [{', '.join(ub_vals)}]")
                else:
                    print(f"Output[{idx}]: [{lb_:.6f}] to [{ub_:.6f}]")

            self.autolirpa_layer_bounds = {}
            layer_count = 0

            try:
                intermediate_bounds = self.bounded_model.save_intermediate()
                if intermediate_bounds:
                    print("🔄 Collecting intermediate bounds from save_intermediate()...")
                    print(f"📊 Found {len(intermediate_bounds)} nodes in intermediate_bounds")

                    for node_name, bounds in intermediate_bounds.items():
                        print(f"Processing node: {node_name}")
                        print(f"  Bounds type: {type(bounds)}")
                        print(f"  Bounds keys: {bounds.keys() if isinstance(bounds, dict) else 'Not a dict'}")

                        lower, upper = None, None

                        if isinstance(bounds, tuple) and len(bounds) >= 2:
                            lower = bounds[0]
                            upper = bounds[1]
                            print(f"  Found {type(bounds).__name__} bounds for {node_name}")

                        elif torch.is_tensor(bounds):
                            lower = bounds
                            upper = bounds
                            print(f"  Found tensor bounds for {node_name}")

                        else:
                            print(f"  ⚠️  Unknown bounds structure for {node_name}: {type(bounds)}")
                            continue

                        if (lower is not None and upper is not None and
                            torch.is_tensor(lower) and torch.is_tensor(upper) and
                            lower.numel() > 0 and upper.numel() > 0):

                            self.autolirpa_layer_bounds[node_name] = {
                                'lb': lower.detach().clone(),
                                'ub': upper.detach().clone(),
                                'shape': lower.shape,
                                'method': 'CROWN',
                                'node_type': 'auto_lirpa_node'
                            }
                            layer_count += 1
                            print(f"✅ Saved bounds for {node_name}: {lower.shape}")
                        else:
                            print(f"  ❌ Invalid bounds for {node_name}: lower={type(lower)}, upper={type(upper)}")

                    print(f"Method 1 (save_intermediate): collected {layer_count} layers")
            except Exception as e:
                print(f"⚠️  save_intermediate() failed: {e}")

            if layer_count == 0:
                print("🔄 Fallback: saving final layer bounds only...")
                self.autolirpa_layer_bounds['final_output'] = {
                    'lb': lb.detach().clone(),
                    'ub': ub.detach().clone(),
                    'shape': lb.shape,
                    'method': 'CROWN',
                    'node_type': 'final_output'
                }
                layer_count = 1

            print(f"✅ Successfully computed bounds for {layer_count} layers using auto_LiRPA")

            self._create_autolirpa_ordered_layer_bounds()

            return True

        except Exception as e:
            print(f"❌ auto_LiRPA bounds computation failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _get_autolirpa_activation_bounds(self, layer_name):

        if layer_name in self.autolirpa_layer_bounds:
            bounds = self.autolirpa_layer_bounds[layer_name]
            return bounds['lb'], bounds['ub']
        return None, None

    def _get_autolirpa_stable_unstable_neurons(self, layer_name):

        lower, upper = self._get_autolirpa_activation_bounds(layer_name)
        if lower is None or upper is None:
            return None, None, None

        lower_flat = lower.view(-1)
        upper_flat = upper.view(-1)

        stable_positive = (lower_flat > 0).nonzero(as_tuple=True)[0]
        stable_negative = (upper_flat < 0).nonzero(as_tuple=True)[0]
        unstable = ((lower_flat <= 0) & (upper_flat >= 0)).nonzero(as_tuple=True)[0]

        print(f"Layer {layer_name}: {len(stable_positive)} stable+, {len(stable_negative)} stable-, {len(unstable)} unstable")

        return stable_positive, stable_negative, unstable

    def _compute_hz_layer_bounds(self, hz, layer_name, layer_type="unknown"):

        try:
            print(f"🔍 [HZ Bounds] Computing bounds for {layer_name} ({layer_type})")
            print(f"🔍 [HZ Bounds] Current time: {time.time()}")

            if hasattr(hz, 'PreActivationGetFlattenedTensor'):

                flat_center, flat_G_c, flat_G_b = hz.PreActivationGetFlattenedTensor()
                A_c, A_b, b = hz.A_c_tensor, hz.A_b_tensor, hz.b_tensor
            else:

                flat_center, flat_G_c, flat_G_b = hz.center, hz.G_c, hz.G_b
                A_c, A_b, b = hz.A_c, hz.A_b, hz.b

            print(f"🔍 [HZ Bounds] Data extracted, about to call GetLayerWiseBounds")

            method = hz.method
            print(f"🔍 [HZ Bounds] Using method: {method} for layer {layer_name}")

            print(f"🔍 [HZ Bounds] Calling GetLayerWiseBounds with method={method}, time_limit=500")
            start_time = time.time()

            lb, ub = HybridZonotopeOps.GetLayerWiseBounds(
                flat_center, flat_G_c, flat_G_b, A_c, A_b, b,
                method, time_limit=500
            )

            end_time = time.time()
            print(f"🔍 [HZ Bounds] GetLayerWiseBounds completed in {end_time - start_time:.2f}s")

            self.hz_layer_bounds[layer_name] = {
                'lb': lb.detach().clone(),
                'ub': ub.detach().clone(),
                'shape': lb.shape,
                'layer_type': layer_type,
                'method': method
            }

            print(f"✅ [HZ Bounds] {layer_name}: range=[{lb.min():.6f}, {ub.max():.6f}]")
            return lb, ub

        except Exception as e:
            print(f"❌ [HZ Bounds] Failed for {layer_name}: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def _compute_concrete_inference(self, input_center, model):

        try:
            print(f"🔍 [Concrete] Computing concrete network inference at input center")
            print(f"🔍 [Concrete] Input center shape: {input_center.shape}")

            if input_center.dim() == 1:
                x = input_center.unsqueeze(0)
            elif input_center.dim() == 3:
                x = input_center.unsqueeze(0)
            else:
                x = input_center

            x = x.to(self.device)
            layer_values = {}
            layer_count = 0

            print(f"🔍 [Concrete] Starting forward pass with input shape: {x.shape}")

            for layer in model.children():
                layer_name = f"layer_{layer_count}_{type(layer).__name__}"
                layer_type = type(layer).__name__.lower()

                print(f"🔍 [Concrete] Processing {layer_name} ({layer_type})")
                print(f"    Input shape: {x.shape}")

                try:

                    x_prev = x.clone()
                    x = layer(x)

                    print(f"    Raw output type: {type(x)}")
                    if hasattr(x, 'shape'):
                        print(f"    Raw output shape: {x.shape}")

                    if isinstance(x, tuple):

                        print(f"⚠️  Layer {layer_name} returned tuple with {len(x)} elements, using first element")
                        print(f"    Tuple elements types: {[type(elem) for elem in x]}")
                        if len(x) > 0:
                            x = x[0]
                        else:
                            print(f"❌ Empty tuple from {layer_name}, skipping layer")
                            continue
                    elif isinstance(x, list):

                        print(f"⚠️  Layer {layer_name} returned list with {len(x)} elements, using first element")
                        print(f"    List elements types: {[type(elem) for elem in x]}")
                        if len(x) > 0:
                            x = x[0]
                        else:
                            print(f"❌ Empty list from {layer_name}, skipping layer")
                            continue

                    if not torch.is_tensor(x):
                        print(f"❌ Layer {layer_name} output is not a tensor: {type(x)}")
                        continue

                    if x.dim() > 1:
                        x_flat = x.squeeze(0).view(-1) if x.shape[0] == 1 else x.view(-1)
                    else:
                        x_flat = x

                    layer_values[layer_name] = {
                        'values': x_flat.detach().clone(),
                        'shape': x.shape,
                        'layer_type': layer_type,
                        'layer_index': layer_count
                    }

                    print(f"✅ [Concrete] {layer_name}: {x.shape} -> flattened: {x_flat.shape}")
                    print(f"    Range: [{x_flat.min():.6f}, {x_flat.max():.6f}]")

                except Exception as layer_e:
                    print(f"❌ Error processing layer {layer_name}: {layer_e}")
                    print(f"    Layer type: {type(layer)}")
                    print(f"    Input shape: {x_prev.shape if 'x_prev' in locals() else 'unknown'}")
                    continue

                layer_count += 1

            self.concrete_layer_values = layer_values

            print(f"✅ [Concrete] Completed inference for {layer_count} layers")
            return layer_values

        except Exception as e:
            print(f"❌ [Concrete] Failed: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def _check_soundness(self, layer_name, hz_bounds=None, autolirpa_bounds=None, concrete_values=None):

        try:
            print(f"\n🔍 [Soundness Check] {layer_name}")
            print("="*80)

            if concrete_values is None and layer_name in self.concrete_layer_values:
                concrete_data = self.concrete_layer_values[layer_name]
                concrete_vals = concrete_data['values']
            elif concrete_values is not None:
                concrete_vals = concrete_values
            else:
                print(f"⚠️  [Soundness] No concrete values for {layer_name}")
                return

            if hz_bounds is None and layer_name in self.hz_layer_bounds:
                hz_data = self.hz_layer_bounds[layer_name]
                hz_lb, hz_ub = hz_data['lb'], hz_data['ub']
            elif hz_bounds is not None:
                hz_lb, hz_ub = hz_bounds
            else:
                hz_lb, hz_ub = None, None

            if autolirpa_bounds is None and layer_name in self.autolirpa_layer_bounds:
                crown_data = self.autolirpa_layer_bounds[layer_name]
                crown_lb, crown_ub = crown_data['lb'], crown_data['ub']
            elif autolirpa_bounds is not None:
                crown_lb, crown_ub = autolirpa_bounds
            else:
                crown_lb, crown_ub = None, None

            print(f"🔍 Data shapes: Concrete={concrete_vals.shape}")
            if hz_lb is not None:
                print(f"             HZ={hz_lb.shape}")
            if crown_lb is not None:
                print(f"             CROWN={crown_lb.shape}")

            soundness_results = {
                'layer_name': layer_name,
                'hz_sound': None,
                'crown_sound': None,
                'hz_violations': 0,
                'crown_violations': 0,
                'total_elements': concrete_vals.numel()
            }

            if hz_lb is not None and hz_ub is not None:

                if concrete_vals.shape != hz_lb.shape:
                    if hz_lb.dim() > 1:
                        hz_lb_flat = hz_lb.squeeze(0).view(-1) if hz_lb.shape[0] == 1 else hz_lb.view(-1)
                        hz_ub_flat = hz_ub.squeeze(0).view(-1) if hz_ub.shape[0] == 1 else hz_ub.view(-1)
                    else:
                        hz_lb_flat = hz_lb
                        hz_ub_flat = hz_ub

                    if concrete_vals.shape == hz_lb_flat.shape:
                        hz_lb, hz_ub = hz_lb_flat, hz_ub_flat
                    else:
                        print(f"❌ HZ shape still mismatched: Concrete={concrete_vals.shape}, HZ={hz_lb_flat.shape}")
                        hz_lb, hz_ub = None, None

                if hz_lb is not None:

                    eps_abs = 1e-5
                    eps_rel = 1e-5

                    hz_lb_tolerance = torch.max(torch.full_like(hz_lb, eps_abs), eps_rel * torch.abs(hz_lb))
                    hz_ub_tolerance = torch.max(torch.full_like(hz_ub, eps_abs), eps_rel * torch.abs(hz_ub))

                    lb_violations = (concrete_vals < (hz_lb - hz_lb_tolerance)).sum().item()
                    ub_violations = (concrete_vals > (hz_ub + hz_ub_tolerance)).sum().item()
                    total_violations = lb_violations + ub_violations

                    soundness_results['hz_sound'] = (total_violations == 0)
                    soundness_results['hz_violations'] = total_violations

                    print(f"🔍 HZ Soundness: {'✅ SOUND' if total_violations == 0 else f'❌ VIOLATIONS'}")
                    print(f"   Lower bound violations: {lb_violations} (tolerance: {eps_abs:.0e})")
                    print(f"   Upper bound violations: {ub_violations} (tolerance: {eps_abs:.0e})")
                    print(f"   Total violations: {total_violations}/{concrete_vals.numel()}")

                    if total_violations > 0:

                        violation_indices = ((concrete_vals < (hz_lb - hz_lb_tolerance)) | (concrete_vals > (hz_ub + hz_ub_tolerance))).nonzero(as_tuple=True)[0]
                        print(f"   Violation details (first 5):")
                        for i, idx in enumerate(violation_indices[:5]):
                            idx = idx.item()
                            concrete_val = concrete_vals[idx].item()
                            lb_val = hz_lb[idx].item()
                            ub_val = hz_ub[idx].item()
                            lb_tol = hz_lb_tolerance[idx].item()
                            ub_tol = hz_ub_tolerance[idx].item()
                            print(f"     Index {idx}: concrete={concrete_val:.6f}, bounds=[{lb_val:.6f}, {ub_val:.6f}], tolerance=[{lb_tol:.0e}, {ub_tol:.0e}]")

            if crown_lb is not None and crown_ub is not None:

                if concrete_vals.shape != crown_lb.shape:
                    if crown_lb.dim() > 1:
                        crown_lb_flat = crown_lb.squeeze(0).view(-1) if crown_lb.shape[0] == 1 else crown_lb.view(-1)
                        crown_ub_flat = crown_ub.squeeze(0).view(-1) if crown_ub.shape[0] == 1 else crown_ub.view(-1)
                    else:
                        crown_lb_flat = crown_lb
                        crown_ub_flat = crown_ub

                    if concrete_vals.shape == crown_lb_flat.shape:
                        crown_lb, crown_ub = crown_lb_flat, crown_ub_flat
                    else:
                        print(f"❌ CROWN shape still mismatched: Concrete={concrete_vals.shape}, CROWN={crown_lb_flat.shape}")
                        crown_lb, crown_ub = None, None

                if crown_lb is not None:

                    eps_abs = 1e-5
                    eps_rel = 1e-5

                    crown_lb_tolerance = torch.max(torch.full_like(crown_lb, eps_abs), eps_rel * torch.abs(crown_lb))
                    crown_ub_tolerance = torch.max(torch.full_like(crown_ub, eps_abs), eps_rel * torch.abs(crown_ub))

                    lb_violations = (concrete_vals < (crown_lb - crown_lb_tolerance)).sum().item()
                    ub_violations = (concrete_vals > (crown_ub + crown_ub_tolerance)).sum().item()
                    total_violations = lb_violations + ub_violations

                    soundness_results['crown_sound'] = (total_violations == 0)
                    soundness_results['crown_violations'] = total_violations

                    print(f"🔍 CROWN Soundness: {'✅ SOUND' if total_violations == 0 else f'❌ VIOLATIONS'}")
                    print(f"   Lower bound violations: {lb_violations} (tolerance: {eps_abs:.0e})")
                    print(f"   Upper bound violations: {ub_violations} (tolerance: {eps_abs:.0e})")
                    print(f"   Total violations: {total_violations}/{concrete_vals.numel()}")

                    if total_violations > 0:

                        violation_indices = ((concrete_vals < (crown_lb - crown_lb_tolerance)) | (concrete_vals > (crown_ub + crown_ub_tolerance))).nonzero(as_tuple=True)[0]
                        print(f"   Violation details (first 5):")
                        for i, idx in enumerate(violation_indices[:5]):
                            idx = idx.item()
                            concrete_val = concrete_vals[idx].item()
                            lb_val = crown_lb[idx].item()
                            ub_val = crown_ub[idx].item()
                            lb_tol = crown_lb_tolerance[idx].item()
                            ub_tol = crown_ub_tolerance[idx].item()
                            print(f"     Index {idx}: concrete={concrete_val:.6f}, bounds=[{lb_val:.6f}, {ub_val:.6f}], tolerance=[{lb_tol:.0e}, {ub_tol:.0e}]")

            self.soundness_check_results[layer_name] = soundness_results

            print("="*80)

        except Exception as e:
            print(f"❌ [Soundness Check] Failed for {layer_name}: {e}")
            import traceback
            traceback.print_exc()


    def _print_final_soundness_summary(self):
        if not self.soundness_check_results:
            print("🔍 [Final Soundness Summary] No soundness check data available")
            return

        print("\n" + "="*80)
        print("🔍 FINAL SOUNDNESS CHECK SUMMARY")
        print("="*80)

        hz_sound_count = 0
        crown_sound_count = 0
        total_layers = 0
        total_hz_violations = 0
        total_crown_violations = 0
        total_elements = 0

        for layer_name, result in self.soundness_check_results.items():
            total_layers += 1
            total_elements += result['total_elements']

            if result['hz_sound'] is not None:
                if result['hz_sound']:
                    hz_sound_count += 1
                total_hz_violations += result['hz_violations']
                hz_status = "✅ SOUND" if result['hz_sound'] else f"❌ {result['hz_violations']} violations"
            else:
                hz_status = "N/A"

            if result['crown_sound'] is not None:
                if result['crown_sound']:
                    crown_sound_count += 1
                total_crown_violations += result['crown_violations']
                crown_status = "✅ SOUND" if result['crown_sound'] else f"❌ {result['crown_violations']} violations"
            else:
                crown_status = "N/A"

            print(f"{layer_name:25s}: HZ={hz_status:15s} CROWN={crown_status:15s}")

        print("-" * 80)
        print(f"📊 Overall Soundness Results:")
        print(f"   Total Layers: {total_layers}")
        print(f"   Total Elements: {total_elements}")

        hz_valid_layers = sum(1 for r in self.soundness_check_results.values() if r['hz_sound'] is not None)
        crown_valid_layers = sum(1 for r in self.soundness_check_results.values() if r['crown_sound'] is not None)

        if hz_valid_layers > 0:
            print(f"   HZ Soundness: {hz_sound_count}/{hz_valid_layers} layers ({hz_sound_count/hz_valid_layers:.1%})")
            print(f"   HZ Violations: {total_hz_violations}/{total_elements} elements ({total_hz_violations/total_elements:.2%})")

        if crown_valid_layers > 0:
            print(f"   CROWN Soundness: {crown_sound_count}/{crown_valid_layers} layers ({crown_sound_count/crown_valid_layers:.1%})")
            print(f"   CROWN Violations: {total_crown_violations}/{total_elements} elements ({total_crown_violations/total_elements:.2%})")

        if hz_valid_layers > 0 and total_hz_violations == 0:
            print("🏆 HZ Overall: ✅ COMPLETELY SOUND")
        elif hz_valid_layers > 0:
            print("🏆 HZ Overall: ❌ SOUNDNESS VIOLATIONS DETECTED")

        if crown_valid_layers > 0 and total_crown_violations == 0:
            print("🏆 CROWN Overall: ✅ COMPLETELY SOUND")
        elif crown_valid_layers > 0:
            print("🏆 CROWN Overall: ❌ SOUNDNESS VIOLATIONS DETECTED")

        print("="*80)

    def _get_autolirpa_layer_bounds(self, layer_name, layer_index=None):

        try:

            if hasattr(self, 'autolirpa_ordered_layer_bounds') and layer_index is not None:
                if 0 <= layer_index < len(self.autolirpa_ordered_layer_bounds):
                    bounds_info = self.autolirpa_ordered_layer_bounds[layer_index]
                    lb, ub = bounds_info['lb'], bounds_info['ub']

                    if layer_name not in self.autolirpa_layer_bounds:
                        self.autolirpa_layer_bounds[layer_name] = {
                            'lb': lb.detach().clone(),
                            'ub': ub.detach().clone(),
                            'shape': lb.shape,
                            'method': 'CROWN',
                            'auto_lirpa_node': bounds_info['node_name']
                        }

                    return lb, ub

            if layer_name in self.autolirpa_layer_bounds:
                bounds = self.autolirpa_layer_bounds[layer_name]
                return bounds['lb'], bounds['ub']

            return None, None

        except Exception as e:
            print(f"❌ [CROWN Bounds] Failed for {layer_name}: {e}")
            return None, None

    def _compare_layer_precision(self, layer_name, hz_bounds=None, autolirpa_bounds=None):

        try:
            print(f"\n🔍 [Detailed Precision Comparison] {layer_name}")
            print("="*80)

            if self.enable_soundness_check:
                self._check_soundness(layer_name, hz_bounds, autolirpa_bounds)

            if hz_bounds is None and layer_name in self.hz_layer_bounds:
                hz_data = self.hz_layer_bounds[layer_name]
                hz_lb, hz_ub = hz_data['lb'], hz_data['ub']
            elif hz_bounds is not None:
                hz_lb, hz_ub = hz_bounds
            else:
                hz_lb, hz_ub = None, None

            if autolirpa_bounds is None and layer_name in self.autolirpa_layer_bounds:
                crown_data = self.autolirpa_layer_bounds[layer_name]
                crown_lb, crown_ub = crown_data['lb'], crown_data['ub']
            elif autolirpa_bounds is not None:
                crown_lb, crown_ub = autolirpa_bounds
            else:
                crown_lb, crown_ub = None, None

            if hz_lb is None or crown_lb is None:
                print(f"⚠️  [Precision] Incomplete bounds for {layer_name}")
                return

            print(f"🔍 [DEBUG] HZ bounds source: {type(hz_lb)}, shape: {hz_lb.shape}")
            print(f"🔍 [DEBUG] CROWN bounds source: {type(crown_lb)}, shape: {crown_lb.shape}")
            print(f"🔍 [DEBUG] HZ bounds range: [{hz_lb.min():.6f}, {hz_ub.max():.6f}]")
            print(f"🔍 [DEBUG] CROWN bounds range: [{crown_lb.min():.6f}, {crown_ub.max():.6f}]")
            print(f"🔍 [DEBUG] Are HZ and CROWN lb the same tensor? {torch.equal(hz_lb, crown_lb) if hz_lb.shape == crown_lb.shape else 'Different shapes'}")
            print(f"🔍 [DEBUG] Are HZ and CROWN ub the same tensor? {torch.equal(hz_ub, crown_ub) if hz_ub.shape == crown_ub.shape else 'Different shapes'}")

            concrete_vals = None
            if layer_name in self.concrete_layer_values:
                concrete_data = self.concrete_layer_values[layer_name]
                concrete_vals = concrete_data['values']
                print(f"🔍 Original shapes: HZ={hz_lb.shape}, CROWN={crown_lb.shape}, Concrete={concrete_vals.shape}")
            else:
                print(f"🔍 Original shapes: HZ={hz_lb.shape}, CROWN={crown_lb.shape}, Concrete=N/A")

            if hz_lb.shape != crown_lb.shape:
                print(f"🔧 Shape mismatch, attempting to flatten CROWN bounds...")

                if crown_lb.dim() > 1:

                    if crown_lb.shape[0] == 1:
                        crown_lb_flat = crown_lb.squeeze(0).view(-1)
                        crown_ub_flat = crown_ub.squeeze(0).view(-1)
                    else:
                        crown_lb_flat = crown_lb.view(-1)
                        crown_ub_flat = crown_ub.view(-1)

                    print(f"🔧 CROWN flattened shape: {crown_lb_flat.shape}")

                    if hz_lb.shape == crown_lb_flat.shape:
                        crown_lb, crown_ub = crown_lb_flat, crown_ub_flat
                        print(f"✅ Shapes matched successfully: {hz_lb.shape}")
                    else:
                        print(f"❌ Still mismatched after flatten: HZ={hz_lb.shape}, CROWN={crown_lb_flat.shape}")
                        return
                else:
                    print(f"❌ CROWN bounds are already 1D, cannot flatten further")
                    return

            hz_width = (hz_ub - hz_lb).abs()
            crown_width = (crown_ub - crown_lb).abs()

            if torch.isnan(hz_width).any() or torch.isinf(hz_width).any():
                print("⚠️  Warning: HZ bounds contain NaN or Inf values")
            if torch.isnan(crown_width).any() or torch.isinf(crown_width).any():
                print("⚠️  Warning: CROWN bounds contain NaN or Inf values")

            hz_mean_width = hz_width.mean().item()
            crown_mean_width = crown_width.mean().item()
            hz_max_width = hz_width.max().item()
            crown_max_width = crown_width.max().item()
            hz_min_lb = hz_lb.min().item()
            hz_max_ub = hz_ub.max().item()
            crown_min_lb = crown_lb.min().item()
            crown_max_ub = crown_ub.max().item()

            width_improvement = ((hz_width - crown_width) / (hz_width + 1e-8)).mean().item() * 100

            hz_range = hz_max_ub - hz_min_lb
            crown_range = crown_max_ub - crown_min_lb

            tighter_crown = (crown_width < hz_width).sum().item()
            tighter_hz = (hz_width < crown_width).sum().item()
            equal_bounds = (torch.abs(hz_width - crown_width) < 1e-6).sum().item()
            total_neurons = hz_lb.numel()

            print(f"📊 Statistics Comparison:")
            print(f"   HybridZonotope: mean width={hz_mean_width:.6f}, max width={hz_max_width:.6f}")
            print(f"   CROWN:          mean width={crown_mean_width:.6f}, max width={crown_max_width:.6f}")
            print(f"   CROWN improvement: {width_improvement:.2f}% (positive = CROWN better)")
            print(f"   Bound ranges:       HZ=[{hz_min_lb:.6f}, {hz_max_ub:.6f}], CROWN=[{crown_min_lb:.6f}, {crown_max_ub:.6f}]")
            print(f"   Neuron comparison:  CROWN tighter={tighter_crown}/{total_neurons} ({tighter_crown/total_neurons*100:.1f}%)")
            print(f"                       HZ tighter={tighter_hz}/{total_neurons} ({tighter_hz/total_neurons*100:.1f}%)")
            print(f"                       Equal={equal_bounds}/{total_neurons} ({equal_bounds/total_neurons*100:.1f}%)")


            if total_neurons <= 50:
                print(f"\n📋 Element-wise detailed comparison (first {min(total_neurons, 20)} neurons):")

                indices_to_show = list(range(min(10, total_neurons))) + list(range(max(total_neurons-10, 10), total_neurons))
                indices_to_show = sorted(list(set(indices_to_show)))

                if concrete_vals is not None:
                    print(f"{'Index':<4} {'Concrete':<12} {'HZ_LB':<12} {'HZ_UB':<12} {'CROWN_LB':<12} {'CROWN_UB':<12} {'HZ_Width':<10} {'CROWN_Width':<10} {'Improve':<8}")
                    print("-" * 100)
                else:
                    print(f"{'Index':<4} {'HZ_LB':<12} {'HZ_UB':<12} {'CROWN_LB':<12} {'CROWN_UB':<12} {'HZ_Width':<10} {'CROWN_Width':<10} {'Improve':<8}")
                    print("-" * 80)

                print(f"   Showing neuron indices: {indices_to_show} (total {len(indices_to_show)})")
                try:
                    for i in indices_to_show:

                        try:
                            hz_l = hz_lb[i].item()
                            hz_u = hz_ub[i].item()
                            crown_l = crown_lb[i].item()
                            crown_u = crown_ub[i].item()
                            hz_w = hz_width[i].item()
                            crown_w = crown_width[i].item()
                            improvement = ((hz_w - crown_w) / (hz_w + 1e-8)) * 100

                            if concrete_vals is not None and i < concrete_vals.numel():
                                concrete_val = concrete_vals[i].item()
                                print(f"{i:<4} {concrete_val:<12.6f} {hz_l:<12.6f} {hz_u:<12.6f} {crown_l:<12.6f} {crown_u:<12.6f} {hz_w:<10.6f} {crown_w:<10.6f} {improvement:<8.2f}%")
                            else:
                                print(f"{i:<4} {'N/A':<12} {hz_l:<12.6f} {hz_u:<12.6f} {crown_l:<12.6f} {crown_u:<12.6f} {hz_w:<10.6f} {crown_w:<10.6f} {improvement:<8.2f}%")
                        except Exception as detail_e:
                            print(f"{i:<4} Error accessing data: {detail_e}")
                            continue
                except Exception as loop_e:
                    print(f"⚠️  Error in detailed comparison loop: {loop_e}")
                    print("Skipping detailed element-wise comparison...")

            elif total_neurons <= 200:
                print(f"\n📋 Sampling comparison (show first 10 + last 10 neurons, consistent with ERAN):")

                indices_to_show = list(range(min(10, total_neurons))) + list(range(max(total_neurons-10, 10), total_neurons))
                indices_to_show = sorted(list(set(indices_to_show)))

                if concrete_vals is not None:
                    print(f"{'Index':<4} {'Concrete':<12} {'HZ_LB':<12} {'HZ_UB':<12} {'CROWN_LB':<12} {'CROWN_UB':<12} {'HZ_Width':<10} {'CROWN_Width':<10} {'Improve':<8}")
                    print("-" * 100)
                else:
                    print(f"{'Index':<4} {'HZ_LB':<12} {'HZ_UB':<12} {'CROWN_LB':<12} {'CROWN_UB':<12} {'HZ_Width':<10} {'CROWN_Width':<10} {'Improve':<8}")
                    print("-" * 80)

                print(f"   Showing neuron indices: {indices_to_show} (total {len(indices_to_show)})")
                try:
                    for i in indices_to_show:
                        try:
                            hz_l = hz_lb[i].item()
                            hz_u = hz_ub[i].item()
                            crown_l = crown_lb[i].item()
                            crown_u = crown_ub[i].item()
                            hz_w = hz_width[i].item()
                            crown_w = crown_width[i].item()
                            improvement = ((hz_w - crown_w) / (hz_w + 1e-8)) * 100

                            if concrete_vals is not None and i < concrete_vals.numel():
                                concrete_val = concrete_vals[i].item()
                                print(f"{i:<4} {concrete_val:<12.6f} {hz_l:<12.6f} {hz_u:<12.6f} {crown_l:<12.6f} {crown_u:<12.6f} {hz_w:<10.6f} {crown_w:<10.6f} {improvement:<8.2f}%")
                            else:
                                print(f"{i:<4} {'N/A':<12} {hz_l:<12.6f} {hz_u:<12.6f} {crown_l:<12.6f} {crown_u:<12.6f} {hz_w:<10.6f} {crown_w:<10.6f} {improvement:<8.2f}%")
                        except Exception as detail_e:
                            print(f"{i:<4} Error accessing data: {detail_e}")
                            continue
                except Exception as loop_e:
                    print(f"⚠️  Error in detailed comparison loop: {loop_e}")
                    print("Skipping detailed element-wise comparison...")
                if concrete_vals is not None:
                    print(f"{'Index':<4} {'Concrete':<12} {'HZ LB':<12} {'HZ UB':<12} {'CROWN LB':<12} {'CROWN UB':<12} {'HZ Width':<10} {'CROWN Width':<10} {'Improve':<8}")
                    print("-" * 100)
                else:
                    print(f"{'Index':<4} {'HZ LB':<12} {'HZ UB':<12} {'CROWN LB':<12} {'CROWN UB':<12} {'HZ Width':<10} {'CROWN Width':<10} {'Improve':<8}")
                    print("-" * 80)

                try:
                    step = max(1, total_neurons // 10)
                    for i in range(0, total_neurons, step):
                        try:
                            hz_l = hz_lb[i].item()
                            hz_u = hz_ub[i].item()
                            crown_l = crown_lb[i].item()
                            crown_u = crown_ub[i].item()
                            hz_w = hz_width[i].item()
                            crown_w = crown_width[i].item()
                            improvement = ((hz_w - crown_w) / (hz_w + 1e-8)) * 100

                            if concrete_vals is not None and i < concrete_vals.numel():
                                concrete_val = concrete_vals[i].item()
                                print(f"{i:<4} {concrete_val:<12.6f} {hz_l:<12.6f} {hz_u:<12.6f} {crown_l:<12.6f} {crown_u:<12.6f} {hz_w:<10.6f} {crown_w:<10.6f} {improvement:<8.2f}%")
                            else:
                                print(f"{i:<4} {'N/A':<12} {hz_l:<12.6f} {hz_u:<12.6f} {crown_l:<12.6f} {crown_u:<12.6f} {hz_w:<10.6f} {crown_w:<10.6f} {improvement:<8.2f}%")
                        except Exception as detail_e:
                            print(f"{i:<4} Error accessing data: {detail_e}")
                            continue
                except Exception as sampling_e:
                    print(f"⚠️  Error in sampling comparison: {sampling_e}")

            else:

                print(f"\n📋 Large layer neuron comparison (showing first 10 + last 10 neurons, consistent with ERAN):")

                indices_to_show = list(range(min(10, total_neurons))) + list(range(max(total_neurons-10, 10), total_neurons))
                indices_to_show = sorted(list(set(indices_to_show)))

                if concrete_vals is not None:
                    print(f"{'Index':<4} {'Concrete':<12} {'HZ LB':<12} {'HZ UB':<12} {'CROWN LB':<12} {'CROWN UB':<12} {'HZ Width':<10} {'CROWN Width':<10} {'Improve':<8}")
                    print("-" * 100)
                else:
                    print(f"{'Index':<4} {'HZ LB':<12} {'HZ UB':<12} {'CROWN LB':<12} {'CROWN UB':<12} {'HZ Width':<10} {'CROWN Width':<10} {'Improve':<8}")
                    print("-" * 80)

                print(f"   Showing neuron indices: {indices_to_show} (total {len(indices_to_show)})")
                try:
                    for i in indices_to_show:
                        try:
                            hz_l = hz_lb[i].item()
                            hz_u = hz_ub[i].item()
                            crown_l = crown_lb[i].item()
                            crown_u = crown_ub[i].item()
                            hz_w = hz_width[i].item()
                            crown_w = crown_width[i].item()
                            improvement = ((hz_w - crown_w) / (hz_w + 1e-8)) * 100

                            if concrete_vals is not None and i < concrete_vals.numel():
                                concrete_val = concrete_vals[i].item()
                                print(f"{i:<4} {concrete_val:<12.6f} {hz_l:<12.6f} {hz_u:<12.6f} {crown_l:<12.6f} {crown_u:<12.6f} {hz_w:<10.6f} {crown_w:<10.6f} {improvement:<8.2f}%")
                            else:
                                print(f"{i:<4} {'N/A':<12} {hz_l:<12.6f} {hz_u:<12.6f} {crown_l:<12.6f} {crown_u:<12.6f} {hz_w:<10.6f} {crown_w:<10.6f} {improvement:<8.2f}%")
                        except Exception as detail_e:
                            print(f"{i:<4} Error accessing data: {detail_e}")
                            continue
                except Exception as large_layer_e:
                    print(f"⚠️  Error in large layer comparison: {large_layer_e}")
                    print("Skipping large layer detailed comparison...")

            comparison = {
                'layer_name': layer_name,
                'hz_mean_width': hz_mean_width,
                'crown_mean_width': crown_mean_width,
                'hz_max_width': hz_max_width,
                'crown_max_width': crown_max_width,
                'width_improvement': width_improvement,
                'hz_range': hz_range,
                'crown_range': crown_range,
                'tighter_crown_ratio': tighter_crown / total_neurons,
                'tighter_hz_ratio': tighter_hz / total_neurons,
                'shape': hz_lb.shape,
                'total_neurons': total_neurons,
                'better_method': 'CROWN' if crown_mean_width < hz_mean_width else 'HybridZonotope'
            }

            self.layer_precision_comparison[layer_name] = comparison

            winner = "🥇 CROWN" if crown_mean_width < hz_mean_width else "🥇 HybridZonotope"
            print(f"\n🏆 Conclusion: {winner} (smaller average width)")
            print("="*80)

        except Exception as e:
            print(f"❌ [Precision Comparison] Failed for {layer_name}: {e}")
            import traceback
            traceback.print_exc()

    def _print_final_precision_summary(self):
        if not self.layer_precision_comparison:
            print("🔍 [Final Summary] No precision comparison data available")
            return

        print("\n" + "="*80)
        print("🏆 FINAL PRECISION COMPARISON SUMMARY")
        print("="*80)

        hz_wins = 0
        crown_wins = 0
        total_layers = len(self.layer_precision_comparison)

        for layer_name, comp in self.layer_precision_comparison.items():
            if comp['better_method'] == 'HybridZonotope':
                hz_wins += 1
            else:
                crown_wins += 1

            improvement = comp['width_improvement']
            winner_mark = "🥇" if comp['better_method'] == 'CROWN' else "🥈"

            print(f"{winner_mark} {layer_name:20s}: {improvement:+8.2%} improvement (CROWN vs HZ)")

        print("-" * 80)
        print(f"📊 Overall Results:")
        print(f"   Total Layers:        {total_layers}")
        print(f"   CROWN Wins:          {crown_wins} ({crown_wins/total_layers:.1%})")
        print(f"   HybridZonotope Wins: {hz_wins} ({hz_wins/total_layers:.1%})")

        avg_improvement = np.mean([comp['width_improvement'] for comp in self.layer_precision_comparison.values()])
        print(f"   Average Improvement: {avg_improvement:+.2%} (CROWN vs HZ)")

        if avg_improvement > 0:
            print("🏆 Overall Winner: CROWN (auto_LiRPA)")
        else:
            print("🏆 Overall Winner: HybridZonotope")

        print("="*80)

    def _abstract_constraint_solving_core(self, model, input_hz, method, sample_idx=0):

        model = model.pytorch_model
        hz = input_hz

        verification_core_start_time = time.time()
        layer_count = 0
        total_layer_time = 0.0

        all_layers = list(model.children())
        linear_layers = [i for i, layer in enumerate(all_layers) if isinstance(layer, nn.Linear)]
        last_linear_index = linear_layers[-1] if linear_layers else -1

        print(f"\n🕒 Starting layer-by-layer verification - method: {method}")
        print(f"🔧 Network structure analysis: total layers={len(all_layers)}, Linear layers={len(linear_layers)}")
        if self.enable_generator_merging and last_linear_index >= 0:
            print(f"🎯 Generator merging will be automatically enabled at layer {last_linear_index} (last Linear layer)")
        print("="*60)

        if self.enable_soundness_check:
            print(f"🔍 [Soundness] Computing concrete network inference...")

            if hasattr(hz, 'center_grid'):

                input_center = hz.center_grid.squeeze(-1)
            else:

                input_center = hz.center.squeeze(-1) if hz.center.dim() > 1 else hz.center

            self._compute_concrete_inference(input_center, model)

        if self.enable_layer_comparison or self.bab_config['enabled']:
            input_layer_name = f"input_layer"
            print(f"🔍 [Layer Bounds] Computing input layer bounds (layer_comparison={self.enable_layer_comparison}, bab_enabled={self.bab_config['enabled']})")

            original_input_center = self.input_center

            if original_input_center.dim() > 1:
                input_center_flat = original_input_center.view(-1)
            else:
                input_center_flat = original_input_center

            self.concrete_layer_values[input_layer_name] = {
                'values': input_center_flat.detach().clone(),
                'shape': original_input_center.shape,
                'layer_type': 'input',
                'layer_index': -1
            }
            print(f"✅ [Concrete] {input_layer_name}: {original_input_center.shape} -> flattened: {input_center_flat.shape}")
            print(f"    Range: [{input_center_flat.min():.6f}, {input_center_flat.max():.6f}]")

            hz_input_bounds = self._compute_hz_layer_bounds(hz, input_layer_name, "input")

            autolirpa_input_bounds = self._get_autolirpa_layer_bounds(input_layer_name, layer_index=0)

            if hz_input_bounds[0] is not None and autolirpa_input_bounds[0] is not None:
                self._compare_layer_precision(input_layer_name, hz_input_bounds, autolirpa_input_bounds)

        for layer in model.children():
            layer_name = f"layer_{layer_count}_{type(layer).__name__}"
            layer_type = type(layer).__name__.lower()
            print(f"📋 Processing layer {layer_count}: {type(layer).__name__}")

            autolirpa_index = layer_count + 1

            if isinstance(layer, nn.Linear):
                layer_start_time = time.time()

                W = layer.weight
                b = layer.bias
                hz.set_method(method)

                is_last_linear = (layer_count == last_linear_index)
                enable_merging = self.enable_generator_merging and is_last_linear

                if enable_merging:
                    print(f"🔧 Last Linear layer (layer {layer_count}): enabling generator merging optimization")
                    hz = hz.linear(W, b, enable_generator_merging=True, cosine_threshold=self.cosine_threshold)
                else:
                    if self.enable_generator_merging and not is_last_linear:
                        print(f"⏭️  Intermediate Linear layer (layer {layer_count}): skipping generator merging optimization")
                    hz = hz.linear(W, b, enable_generator_merging=False)

                layer_end_time = time.time()
                layer_duration = layer_end_time - layer_start_time
                total_layer_time += layer_duration
                print(f"⏱️  Linear layer processing time: {layer_duration:.4f} seconds")

            elif isinstance(layer, nn.Conv2d):
                layer_start_time = time.time()

                hz.set_method(method)
                print(f"🚀 Processing Conv2d layer: verifier method {method}, hz.method {hz.method}")
                hz = hz.conv(layer.weight, layer.bias, stride=layer.stride, padding=layer.padding,
                             dilation=layer.dilation, groups=layer.groups)

                layer_end_time = time.time()
                layer_duration = layer_end_time - layer_start_time
                total_layer_time += layer_duration
                print(f"⏱️  Conv2d layer processing time: {layer_duration:.4f} seconds")

            elif isinstance(layer, nn.ReLU):
                layer_start_time = time.time()

                hz.set_method(method)

                relu_layer_name = layer_name
                debug_applied_constraints = []
                relu_constraints_to_apply = []

                if hasattr(self, 'current_relu_constraints') and self.current_relu_constraints:
                    for constraint in self.current_relu_constraints:
                        constraint_layer = constraint['layer']

                        if (constraint_layer == relu_layer_name or
                            constraint_layer == layer_name or
                            (constraint_layer.startswith('layer_') and constraint_layer.endswith('_ReLU') and layer_name == constraint_layer)):
                            relu_constraints_to_apply.append(constraint)
                            debug_applied_constraints.append(f"ReLU[{constraint['neuron_idx']}]={constraint['constraint_type']}")

                if debug_applied_constraints:
                    print(f"🔒 [{layer_name}] applyingReLUconstraint: {debug_applied_constraints}")

                hz = hz.relu(
                    auto_lirpa_info=None,
                    relu_constraints=relu_constraints_to_apply
                )

                layer_end_time = time.time()
                layer_duration = layer_end_time - layer_start_time
                total_layer_time += layer_duration
                print(f"⏱️  ReLU layer processing time: {layer_duration:.4f} seconds")

            elif isinstance(layer, nn.Sigmoid):
                layer_start_time = time.time()

                hz.set_method(method)
                hz = hz.sigmoid_or_tanh('sigmoid')

                layer_end_time = time.time()
                layer_duration = layer_end_time - layer_start_time
                total_layer_time += layer_duration
                print(f"⏱️  Sigmoid layer processing time: {layer_duration:.4f} seconds")

            elif isinstance(layer, nn.MaxPool2d):
                layer_start_time = time.time()

                hz.set_method(method)

                for name, bounds in self.autolirpa_layer_bounds.items():
                    print(f"Layer {name} bounds: lower={bounds['lb'].shape}, upper={bounds['ub'].shape}")

                hz = hz.maxpool(
                    kernel_size=layer.kernel_size,
                    stride=layer.stride,
                    padding=layer.padding,
                    auto_lirpa_info=None

                )

                layer_end_time = time.time()
                layer_duration = layer_end_time - layer_start_time
                total_layer_time += layer_duration
                print(f"⏱️  MaxPool2d layer processing time: {layer_duration:.4f} seconds")

            elif isinstance(layer, nn.Flatten) or isinstance(layer, OnnxFlatten):
                layer_start_time = time.time()

                hz = HybridZonotopeOps.FlattenHybridZonotopeGridIntersection(hz)

                layer_end_time = time.time()
                layer_duration = layer_end_time - layer_start_time
                total_layer_time += layer_duration
                print(f"⏱️  Flatten layer processing time: {layer_duration:.4f} seconds")

            elif isinstance(layer, OperatorWrapper):
                layer_start_time = time.time()

                layer_type_str = type(layer).__name__.lower()
                op_type = getattr(layer, 'op_type', None)

                if layer_type_str == 'tanh' or op_type == 'Tanh':

                    print(f"📋 Processing layer {layer_count}: tanh (via OperatorWrapper)")
                    hz.set_method(method)
                    hz = hz.sigmoid_or_tanh('tanh')

                    layer_end_time = time.time()
                    layer_duration = layer_end_time - layer_start_time
                    total_layer_time += layer_duration
                    print(f"⏱️  Tanh layer processing time: {layer_duration:.4f} seconds")

                elif hasattr(layer, 'op_type') and layer.op_type in ["Add", "Sub", "Mul", "Div"]:

                        other = getattr(layer, 'other', None)
                        if other is None:
                            raise ValueError(f"OperatorWrapper {layer.op_type} missing 'other' attribute for scalar operation.")

                        hz.set_method(method)

                        if layer.op_type == "Add":
                            hz = hz.add(other)
                            print(f"Applied Add scalar operation with value: {other}")
                        elif layer.op_type == "Sub":
                            hz = hz.subtract(other)
                            print(f"Applied Sub scalar operation with value: {other}")
                        elif layer.op_type == "Mul":
                            hz = hz.multiply(other)
                            print(f"Applied Mul scalar operation with value: {other}")
                        elif layer.op_type == "Div":
                            if other == 0:
                                raise ValueError("Division by zero encountered in OperatorWrapper Div layer.")
                            hz = hz.divide(other)
                            print(f"Applied Div scalar operation with value: {other}")

                        layer_end_time = time.time()
                        layer_duration = layer_end_time - layer_start_time
                        total_layer_time += layer_duration
                        print(f"⏱️  {layer.op_type} operation processing time: {layer_duration:.4f} seconds")
                else:

                    layer_type_str = type(layer).__name__
                    op_type = getattr(layer, 'op_type', 'not_found')
                    raise NotImplementedError(
                        f"OperatorWrapper layer not supported in HybridZonotopeVerifier.\n"
                        f"  Layer type: {layer_type_str}\n"
                        f"  Layer repr: {repr(layer)}\n"
                        f"  op_type: {op_type}\n"
                        f"  Supported op_types: ['Add', 'Sub', 'Mul', 'Div']\n"
                        f"  Supported layer names: ['tanh']"
                    )

            else:
                raise NotImplementedError(f"Layer {layer} not supported in HybridZonotopeVerifier.")

            print("-" * 60)
            layer_count += 1

        print(f"✅ All {layer_count} layers processed")
        print(f"📊 Total layer-by-layer processing time: {total_layer_time:.4f} seconds")
        print("="*60)

        if isinstance(hz, HybridZonotopeGrid):
            hz_elem = HybridZonotopeOps.FlattenHybridZonotopeGridIntersection(hz)
        else:
            hz_elem = hz

        print("Output: ", hz_elem.n)
        verification_core_end_time = time.time()
        verification_core_time = verification_core_end_time - verification_core_start_time

        print("\n📈 Verification core time statistics:")
        print(f"📊 Total layer-by-layer processing time: {total_layer_time:.4f} seconds")
        print(f"⏱️  Other processing time: {verification_core_time - total_layer_time:.4f} seconds")
        print(f"🕒 Total verification core time: {verification_core_time:.4f} seconds")
        print("="*60)

        if self.enable_layer_comparison:
            self._print_final_precision_summary()

        if self.enable_soundness_check:
            self._print_final_soundness_summary()

        print(f"\n🔍 Computing output layer dimension-wise bounds (total {hz_elem.n} output neurons)")
        output_lbs, output_ubs = self._concretize_hz(hz_elem, method=method)

        print(f"🚀 Returning output layer HybridZonotope and bounds for two-stage verification")
        return hz_elem, output_lbs, output_ubs

    def _concretize_hz(self, hz_elem, method='hybridz', time_limit=500):

        print("="*60)
        spec_verification_start_time = time.time()

        print(f"Using GetLayerWiseBounds for {hz_elem.n} outputs, method={method}")

        lbs_tensor, ubs_tensor = HybridZonotopeOps.GetLayerWiseBounds(
            hz_elem.center, hz_elem.G_c, hz_elem.G_b,
            hz_elem.A_c, hz_elem.A_b, hz_elem.b,
            method=method, time_limit=time_limit
        )

        if isinstance(lbs_tensor, (int, float)):

            lbs_tensor = torch.tensor([lbs_tensor], dtype=hz_elem.dtype, device=hz_elem.device)
            ubs_tensor = torch.tensor([ubs_tensor], dtype=hz_elem.dtype, device=hz_elem.device)
            lbs = [lbs_tensor.item()]
            ubs = [ubs_tensor.item()]
        else:

            if not isinstance(lbs_tensor, torch.Tensor):
                lbs_tensor = torch.tensor(lbs_tensor, dtype=hz_elem.dtype, device=hz_elem.device)
                ubs_tensor = torch.tensor(ubs_tensor, dtype=hz_elem.dtype, device=hz_elem.device)

            lbs = lbs_tensor.cpu().numpy().flatten().tolist()
            ubs = ubs_tensor.cpu().numpy().flatten().tolist()

        for i in range(len(lbs)):
            print(f"✅ Output {i}: [{lbs[i]:.6f}, {ubs[i]:.6f}]")

        if not isinstance(lbs_tensor, torch.Tensor):
            lbs_tensor = torch.tensor(lbs, dtype=hz_elem.dtype, device=hz_elem.device)
            ubs_tensor = torch.tensor(ubs, dtype=hz_elem.dtype, device=hz_elem.device)

        spec_verification_end_time = time.time()
        print(f"🕒 Total Output Spec verification time: {spec_verification_end_time - spec_verification_start_time:.2f} seconds")
        print("="*60)

        return lbs_tensor, ubs_tensor

    def _single_result_verdict_hz(self, output_hz,
                                  output_lbs: torch.Tensor,
                                  output_ubs: torch.Tensor,
                                  output_constraints: Optional[List[List[float]]],
                                  true_label: Optional[int]) -> VerificationStatus:

        if output_constraints is not None:
            print(f"🔧 HZ verification: processing linear constraints ({len(output_constraints)} constraints)")
            for row in output_constraints:
                a = torch.tensor(row[:-1], device=output_lbs.device)
                b = row[-1]
                worst = torch.sum(torch.where(a>=0, a*output_lbs, a*output_ubs)) + b
                if worst < 0:
                    return VerificationStatus.UNSAT
            return VerificationStatus.SAT

        if true_label is not None:
            print(f"🔧 HZ verification: using two-stage verification strategy (conservative judgment first, then precise difference)")
            return self._classify_with_two_stage_strategy_hz(output_hz, output_lbs, output_ubs, true_label)

        return VerificationStatus.UNKNOWN

    def _classify_with_two_stage_strategy_hz(self, output_hz, output_lbs: torch.Tensor, output_ubs: torch.Tensor, true_label: int) -> VerificationStatus:

        if len(output_lbs) <= true_label:
            print(f"❌ true_label {true_label} exceeds output dimension {len(output_lbs)}")
            return VerificationStatus.UNKNOWN

        num_outputs = len(output_lbs)
        print(f"🔍 Starting two-stage HZ verification: true_label={true_label}, num_outputs={num_outputs}")

        print(f"📈 Stage 1: Conservative bound comparison")
        true_label_lb = output_lbs[true_label].item()

        conservative_safe = True
        for j in range(num_outputs):
            if j == true_label:
                continue
            other_ub = output_ubs[j].item()
            if true_label_lb <= other_ub:
                print(f"  ⚠️  Conservative judgment failed: output[{true_label}]_lb={true_label_lb:.6f} <= output[{j}]_ub={other_ub:.6f}")
                conservative_safe = False
                break
            else:
                print(f"  ✅ output[{true_label}]_lb={true_label_lb:.6f} > output[{j}]_ub={other_ub:.6f}")

        if conservative_safe:
            print(f"🎉 Stage 1 success: true_label lower bound greater than all other neuron upper bounds, directly judged robust")
            return VerificationStatus.SAT

        print(f"🔍 Stage 2: Precise difference verification (ERAN style)")
        return self._classify_with_difference_bounds_hz(output_hz, true_label)

    def _classify_with_difference_bounds_hz(self, output_hz, true_label: int) -> VerificationStatus:

        if output_hz.G_c.shape[0] <= true_label:
            print(f"❌ true_label {true_label} exceeds output dimension {output_hz.G_c.shape[0]}")
            return VerificationStatus.UNKNOWN

        num_outputs = output_hz.G_c.shape[0]
        print(f"🔍 Starting HZ difference verification: true_label={true_label}, num_outputs={num_outputs}")

        for j in range(num_outputs):
            if j == true_label:
                continue

            print(f"  Checking difference: output[{true_label}] - output[{j}]")

            try:

                diff_hz = HybridZonotopeOps.ConstructNeuronDifferenceHZ(output_hz, true_label, j)

                diff_lb, diff_ub = self._concretize_hz(diff_hz, method=self.method)

                if isinstance(diff_lb, torch.Tensor):
                    diff_lb_val = diff_lb.item() if diff_lb.numel() == 1 else diff_lb[0].item()
                    diff_ub_val = diff_ub.item() if diff_ub.numel() == 1 else diff_ub[0].item()
                else:
                    diff_lb_val = float(diff_lb)
                    diff_ub_val = float(diff_ub)

                print(f"    Difference range: [{diff_lb_val:.6f}, {diff_ub_val:.6f}]")

                if diff_lb_val <= 0:
                    print(f"    ❌ Difference output[{true_label}] - output[{j}] lower bound <= 0: {diff_lb_val:.6f}")
                    return VerificationStatus.UNSAT
                else:
                    print(f"    ✅ Difference output[{true_label}] - output[{j}] lower bound > 0: {diff_lb_val:.6f}")

            except Exception as e:
                print(f"    ⚠️  Error constructing difference zonotope: {e}")
                print(f"    Cannot complete HZ difference verification, returning UNKNOWN")
                return VerificationStatus.UNKNOWN

        print(f"  ✅ All HZ difference verifications passed")
        return VerificationStatus.SAT

    def _abstract_constraint_solving(self, input_lb: torch.Tensor, input_ub: torch.Tensor, sample_idx: int) -> VerificationStatus:

        print(f"   🔧 Creating HybridZonotope abstract domain")
        self.input_hz = HybridZonotopeGrid(
            input_lb=input_lb,
            input_ub=input_ub,
            method=self.method,
            time_limit=500,
            relaxation_ratio=self.relaxation_ratio,
            device=self.device
        )

        print(f"   🔍 Using method: {self.method}")
        verification_result = self._abstract_constraint_solving_core(model=self.model, input_hz=self.input_hz, method=self.method, sample_idx=sample_idx)

        if verification_result is not None and len(verification_result) == 3:
            output_hz, output_lbs, output_ubs = verification_result
        else:

            print(f"   ❌ Verification core returned abnormal value")
            return VerificationStatus.UNKNOWN

        self.apply_relu_constraints_to_bounds()

        if output_hz is not None:
            verdict = self._single_result_verdict_hz(
                output_hz,
                output_lbs,
                output_ubs,
                self.spec.output_spec.output_constraints if self.spec.output_spec.output_constraints is not None else None,
                self.spec.output_spec.labels[sample_idx].item() if self.spec.output_spec.labels is not None else None
            )
        else:

            verdict = VerificationStatus.UNKNOWN

        print(f"   📊 Verification result: {verdict.name}")
        return verdict

    def verify(self, proof=None, public_inputs=None):

        print_memory_usage("HybridZonotopeVerifier Start")
        print("🚀 Starting complete verification process - conforming to theoretical architecture design")

        if self.input_center is None:
            print("❌ Error: input_center is None. Cannot proceed with verification.")
            return {"verified": False, "error": "input_center is None"}

        num_samples = self.input_center.shape[0] if self.input_center.ndim > 1 else 1
        print(f"Total samples: {num_samples}")
        print(f"Input center shape: {self.input_center.shape}")
        print(f"Input boundary shapes: {self.input_lb.shape}, {self.input_ub.shape}")

        results = []
        for idx in range(num_samples):
            print_memory_usage(f"Sample {idx+1}")
            print(f"\n🔍 Processing sample {idx+1}/{num_samples}")
            print("="*80)

            center_input, true_label = self.get_sample_center_and_label(idx)
            if not self.check_clean_prediction(center_input, true_label, idx):

                print(f"⏭️  Skipping verification for sample {idx+1}")
                results.append(VerificationStatus.CLEAN_FAILURE)
                continue

            self.clean_prediction_stats['verification_attempted'] += 1


            if self.input_lb.shape[0] == 1:
                lb_i = self.input_lb[0] if self.input_lb.ndim > 1 else self.input_lb
                ub_i = self.input_ub[0] if self.input_ub.ndim > 1 else self.input_ub
            elif self.input_lb.shape[0] > idx:
                lb_i = self.input_lb[idx]
                ub_i = self.input_ub[idx]
            else:
                lb_i = self.input_lb[0] if self.input_lb.ndim > 1 else self.input_lb
                ub_i = self.input_ub[0] if self.input_ub.ndim > 1 else self.input_ub

            if self.use_auto_lirpa:
                input_example = (lb_i + ub_i) / 2.0
                if self._setup_auto_lirpa(input_example.unsqueeze(0)):
                    eps = getattr(self.spec.input_spec, 'epsilon', None)

                    if eps is not None and hasattr(self.dataset, 'std') and self.dataset.std is not None:

                        std_val = self.dataset.std
                        if isinstance(std_val, list):
                            if len(std_val) == 1:

                                eps_normalized = eps / std_val[0]
                                print(f"🔧 [Auto_LiRPA] Original eps: {eps}, Normalized eps: {eps_normalized} (divided by std[0]: {std_val[0]})")
                            else:

                                eps_normalized = eps / std_val[0]
                                print(f"🔧 [Auto_LiRPA] Original eps: {eps}, Normalized eps: {eps_normalized} (divided by std[0]: {std_val[0]}, full std: {std_val})")
                        elif isinstance(std_val, (int, float)):

                            eps_normalized = eps / std_val
                            print(f"🔧 [Auto_LiRPA] Original eps: {eps}, Normalized eps: {eps_normalized} (divided by std: {std_val})")
                        else:

                            eps_normalized = eps
                            print(f"🔧 [Auto_LiRPA] Using original eps: {eps} (std format not recognized: {type(std_val)})")
                        eps = eps_normalized
                    self._compute_autolirpa_bounds((lb_i, ub_i), eps=eps)

            print("🌟 Step 1: HybridZonotope abstract constraint solving")
            initial_verdict = self._abstract_constraint_solving(lb_i, ub_i, idx)

            if initial_verdict == VerificationStatus.SAT:
                self.clean_prediction_stats['verification_sat'] += 1

                print(f"✅ HybridZonotope verification successful - sample {idx+1} is safe")
                results.append(initial_verdict)
                continue
            elif initial_verdict == VerificationStatus.UNSAT:
                self.clean_prediction_stats['verification_unsat'] += 1
            else:
                self.clean_prediction_stats['verification_unknown'] += 1

            if initial_verdict == VerificationStatus.UNSAT:
                print(f"❌ HybridZonotope found potential violation - sample {idx+1}")
            else:
                print(f"❓ HybridZonotope result uncertain - sample {idx+1}")

            print("🌳 Automatically activating Specification Refinement BaB process")
            print("="*60)

            if self.bab_config['enabled']:

                refinement_verdict = self._spec_refinement_verification(lb_i, ub_i, idx)
                results.append(refinement_verdict)
            else:
                print("⚠️  BaB not enabled, returning initial verdict")
                results.append(initial_verdict)

        return self._all_results_verdict(results)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Abstract Constraint Transformer (ACT) - Unified Neural Network Verification Framework')
    
    # ================================================================================
    # ACT NATIVE PARAMETERS
    # ================================================================================
    
    # ACT Core Verifier Selection
    parser.add_argument('--verifier', type=str, default=None, 
                        choices=['interval', 'eran', 'abcrown', 'hybridz'],
                        help='Backend verification engine. "interval": ACT native interval analysis, "eran": ERAN external verifier, "abcrown": αβ-CROWN external verifier, "hybridz": ACT novel hybrid zonotope verifier')
    parser.add_argument('--method', type=str, default=None, 
                        help='Verification method. ERAN: [deepzono, refinezono, deeppoly, refinepoly], αβ-CROWN: [alpha, beta, alpha_beta], ACT-HybridZ: [hybridz, hybridz_relaxed, hybridz_relaxed_with_bab], ACT-Interval: [interval]')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                        help='Computation device (cpu or cuda)')

    # ACT Hybrid Zonotope Novel Features
    parser.add_argument('--relaxation_ratio', type=float, default=1.0,
                        help='ACT Hybrid Zonotope relaxation ratio: 0.0=full-precision MILP, 1.0=fully-relaxed LP, 0.0~1.0=partially-relaxed MILP+LP. Only applies to hybridz_relaxed method (hybridz_relaxed_with_bab forces 1.0)')
    parser.add_argument('--enable_generator_merging', action='store_true', default=False,
                        help='ACT innovation: Enable parallel generator merging optimization in the final linear layer (hybrid zonotope enhancement)')
    parser.add_argument('--cosine_threshold', type=float, default=0.95,
                        help='ACT innovation: Cosine similarity threshold for parallel generator detection (0.0-1.0, higher=stricter)')

    # ACT Specification Refinement (Branch-and-Bound) Framework
    parser.add_argument('--enable_spec_refinement', action='store_true', default=False,
                        help='ACT innovation: Enable specification refinement BaB verification. Automatically triggers when initial abstract verification returns UNKNOWN/UNSAT')
    parser.add_argument('--bab_max_depth', type=int, default=8,
                        help='ACT BaB: Maximum search depth')
    parser.add_argument('--bab_max_subproblems', type=int, default=500,
                        help='ACT BaB: Maximum number of subproblems')
    parser.add_argument('--bab_time_limit', type=float, default=300.0,
                        help='ACT BaB: Time limit in seconds')
    parser.add_argument('--bab_split_tolerance', type=float, default=1e-6,
                        help='ACT BaB: Split tolerance')
    parser.add_argument('--bab_verbose', action='store_true', default=True,
                        help='ACT BaB: Enable verbose output')

    # ================================================================================
    # EXTERNAL TOOL COMPATIBILITY PARAMETERS
    # ================================================================================
    # The following parameters are designed for compatibility with external verification
    # tools (ERAN and αβ-CROWN) to enable unified interface calls. Parameter names and
    # descriptions are adapted from:
    # 
    # - αβ-CROWN (α,β-CROWN): https://github.com/Verified-Intelligence/alpha-beta-CROWN
    #   Copyright (C) 2021-2025 The α,β-CROWN Team
    #   Licensed under BSD 3-Clause License
    #   Primary authors: Huan Zhang, Zhouxing Shi, Xiangru Zhong
    #
    # - ERAN: https://github.com/eth-sri/eran
    #   Copyright ETH Zurich
    #   Licensed under Apache 2.0 License
    #
    # These parameters enable seamless integration and unified command-line interface
    # across different verification backends while maintaining ACT's native capabilities.
    # ================================================================================

    # Input Specification (adapted from αβ-CROWN specification hierarchy)
    parser.add_argument('--input_lb', type=float, nargs='+', default=None,
                        help='Lower bounds for set-based input specification (box constraints)')
    parser.add_argument('--input_ub', type=float, nargs='+', default=None,
                        help='Upper bounds for set-based input specification (box constraints)')

    # File I/O (adapted from αβ-CROWN general hierarchy)
    parser.add_argument('--root_path', type=str, default="",
                        help='Root path prefix for relative file paths')

    # Model Configuration (adapted from αβ-CROWN model hierarchy)
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to neural network model file (ONNX format supported)')
    parser.add_argument('--load_model', type=str, default=None, 
                        help='Load pretrained model from specified path (αβ-CROWN compatibility)')

    # Data Configuration (adapted from αβ-CROWN data hierarchy)
    parser.add_argument("--start", type=int, default=0, 
                        help='Start from the i-th property in specified dataset')
    parser.add_argument("--end", type=int, default=10000, 
                        help='End with the (i-1)-th property in the dataset')
    parser.add_argument("--select_instance", type=int, nargs='+', default=None,
                        help='Select specific instances to verify (list of indices)')
    parser.add_argument('--num_outputs', type=int, default=10,
                        help="Number of output classes for classification problems")
    parser.add_argument("--mean", nargs='+', type=float, default=None,
                        help='Mean values for data preprocessing normalisation (single value or per-channel list)')
    parser.add_argument("--std", nargs='+', type=float, default=None,
                        help='Standard deviation values for data preprocessing normalisation (single value or per-channel list)')
    parser.add_argument('--pkl_path', type=str, default=None,
                        help="Load verification properties from pickle file (legacy oval20 dataset support)")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset name (mnist, cifar10, etc.) or path to CSV file")
    parser.add_argument("--anchor", type=str, default=None,
                        help="Anchor dataset path for data point anchoring in specifications")
    parser.add_argument("--filter_path", type=str, default=None,
                        help='Pickle file containing examples to skip during verification')
    parser.add_argument("--data_idx_file", type=str, default=None,
                        help='Text file with list of example IDs to run')

    # Specification Configuration (adapted from αβ-CROWN specification hierarchy)
    parser.add_argument("--spec_type", type=str, default='local_lp', 
                        choices=['local_lp', 'local_vnnlib', 'set_vnnlib', "set_box"],
                        help='Verification specification type: "local_lp"=Lp norm around data points, "local_vnnlib"=VNNLIB with anchor points, "set_vnnlib"=set-based VNNLIB (e.g., AcasXu), "set_box"=box constraints')
    parser.add_argument("--robustness_type", type=str, default="verified-acc",
                        choices=["verified-acc", "runnerup", "clean-acc", "specify-target", "all-positive"],
                        help='Robustness verification target: "verified-acc"=verify against all labels, "runnerup"=verify against runner-up labels only')
    parser.add_argument("--norm", type=str, default='inf', choices=['1', '2', 'inf'],
                        help='Lp-norm for epsilon perturbation in robustness verification')
    parser.add_argument("--epsilon", type=float, default=None,
                        help='Perturbation bound (Lp norm). If unset, dataset-specific defaults may apply')
    parser.add_argument("--epsilon_min", type=float, default=0.,
                        help='Minimum perturbation bound (typically 0.0)')
    parser.add_argument("--vnnlib_path", type=str, default=None,
                        help='Path to VNNLIB specification file (overrides Lp/robustness verification arguments)')
    parser.add_argument("--vnnlib_path_prefix", type=str, default='',
                        help='Prefix to add to VNNLIB specification paths (for correcting malformed CSV files)')
    parser.add_argument("--rhs_offset", type=float, default=None,
                        help='Offset to add to right-hand side of constraints (advanced usage)')

    parsed_args = parser.parse_args(sys.argv[1:])
    args_dict = vars(parsed_args)

    if args_dict["mean"] is None:

        if args_dict["dataset"] is None or (args_dict["dataset"] is not None and args_dict["dataset"].endswith('.csv')):
            args_dict["mean"] = [0.0]
        elif args_dict["dataset"] == 'mnist':
            args_dict["mean"] = [0.1307]
        elif args_dict["dataset"] in ['cifar', 'cifar10']:
            args_dict["mean"] = [0.4914, 0.4822, 0.4465]
        else:
            args_dict["mean"] = [0.0]

    if args_dict["std"] is None:

        if args_dict["dataset"] is None or (args_dict["dataset"] is not None and args_dict["dataset"].endswith('.csv')):
            args_dict["std"] = [1.0]
        elif args_dict["dataset"] == 'mnist':
            args_dict["std"] = [0.3081]
        elif args_dict["dataset"] in ['cifar', 'cifar10']:
            args_dict["std"] = [0.2023, 0.1994, 0.2010]
        else:
            args_dict["std"] = [1.0]

    print(f"Using mean: {args_dict['mean']}, std: {args_dict['std']}")

    model = Model(model_path=args_dict["model_path"],
                  device = args_dict["device"])

    # For VNNLIB spec types, use vnnlib_path instead of dataset path
    dataset_path_for_init = args_dict["dataset"]
    if args_dict["spec_type"] in ["local_vnnlib", "set_vnnlib"] and args_dict["vnnlib_path"] is not None:
        dataset_path_for_init = args_dict["vnnlib_path"]
        print(f"🔍 Using VNNLIB file as dataset path: {dataset_path_for_init}")

    dataset = Dataset(dataset_path=dataset_path_for_init,
                      anchor_csv_path=args_dict["anchor"],
                      device=args_dict["device"],
                      spec_type=args_dict["spec_type"],
                      start=args_dict["start"],
                      end=args_dict["end"],
                      num_outputs=args_dict["num_outputs"],
                      mean=args_dict["mean"],
                      std=args_dict["std"],
                      preprocess=True)

    input_spec = InputSpec(dataset = dataset,
                           norm = args_dict["norm"],
                           epsilon = args_dict["epsilon"],
                           vnnlib_path = args_dict["vnnlib_path"])

    output_spec = OutputSpec(dataset = dataset)

    spec = Spec(input_spec=input_spec,
                output_spec=output_spec, model=model)

    verifier_type = args_dict["verifier"]

    method = args_dict["method"]

    if verifier_type == 'eran' and method in ['deepzono', 'refinezono', 'deeppoly', 'refinepoly']:
        if dataset.dataset_path not in ["mnist", "cifar10", "acasxu"]:
            raise ValueError(f"ERAN verifier with method {method} is not supported for dataset {dataset.dataset_path}. \
                             Please use \'mnist\', \'cifar10\' or \'acasxu\'.")
        if args_dict["enable_spec_refinement"]:
            print("⚠️  ERAN verifier is an external verifier, does not support specification refinement BaB, automatically disabled")
        verifier = ERANVerifier(dataset, method, spec)
        verifier.verify(proof=None, public_inputs=None)

    elif verifier_type == 'abcrown' and method in ['alpha', 'beta', 'alpha_beta']:

        if dataset.dataset_path not in ["mnist", "cifar", "eran"]:
            raise ValueError(f"abCrown verifier with method {method} is not supported for dataset {dataset.dataset_path}. \
                             Please use \'mnist\', \'cifar\', 'eran'.")
        if args_dict["enable_spec_refinement"]:
            print("⚠️  ABCROWN verifier is an external verifier, does not support native specification refinement BaB, automatically disabled")
        verifier = ABCROWNVerifier(dataset, method, spec)
        verifier.verify(proof=None, public_inputs=None)

    elif verifier_type == 'interval':
        if method != 'interval':
            raise ValueError(f"Interval verifier only supports 'interval' method, got {method}.")
        verifier = IntervalVerifier(dataset, method, spec)

        if args_dict["enable_spec_refinement"]:
            print("🌳 Enabling specification refinement BaB verification")
            verifier.bab_config.update({
                'enabled': True,
                'max_depth': args_dict["bab_max_depth"],
                'max_subproblems': args_dict["bab_max_subproblems"],
                'time_limit': args_dict["bab_time_limit"],
                'split_tolerance': args_dict["bab_split_tolerance"],
                'verbose': args_dict["bab_verbose"]
            })
            print(f"   Maximum depth: {args_dict['bab_max_depth']}")
            print(f"   Maximum subproblems: {args_dict['bab_max_subproblems']}")
            print(f"   Time limit: {args_dict['bab_time_limit']} seconds")

        else:
            print("⚙️  Specification refinement BaB verification disabled")
            verifier.bab_config['enabled'] = False

        result = verifier.verify()
        if result == VerificationStatus.SAT:
            print("🎉 The property is satisfied.")
        elif result == VerificationStatus.UNSAT:
            print("❌ The property is not satisfied.")
        else:
            print("❓ The property status is unknown.")

    elif verifier_type == 'hybridz':
        if method not in ['interval', 'hybridz', 'hybridz_relaxed', 'hybridz_relaxed_with_bab']:
            raise ValueError(f"Hybrid Zonotope verifier only supports methods 'interval', 'hybridz', 'hybridz_relaxed', 'hybridz_relaxed_with_bab', got {method}.")

        if method == 'hybridz_relaxed_with_bab':

            relaxation_ratio = 1.0
            if args_dict["relaxation_ratio"] != 1.0:
                print(f"⚠️  hybridz_relaxed_with_bab method forces relaxation_ratio=1.0 (full relaxed LP), ignoring user setting of {args_dict['relaxation_ratio']}")
            else:
                print(f"🔧 hybridz_relaxed_with_bab method using relaxation_ratio=1.0 (full relaxed LP)")
        else:

            relaxation_ratio = args_dict["relaxation_ratio"]
            if method == 'hybridz_relaxed':
                print(f"🔧 hybridz_relaxed method using relaxation_ratio={relaxation_ratio}")

        verifier = HybridZonotopeVerifier(dataset, method, spec, args_dict["device"],
                                          relaxation_ratio,
                                          args_dict["enable_generator_merging"],
                                          args_dict["cosine_threshold"])

        if args_dict["enable_spec_refinement"]:
            if method == 'hybridz_relaxed_with_bab':
                print("🌳 Enabling specification refinement BaB verification (hybridz_relaxed_with_bab)")
                verifier.bab_config.update({
                    'enabled': True,
                    'max_depth': args_dict["bab_max_depth"],
                    'max_subproblems': args_dict["bab_max_subproblems"],
                    'time_limit': args_dict["bab_time_limit"],
                    'split_tolerance': args_dict["bab_split_tolerance"],
                    'verbose': args_dict["bab_verbose"]
                })
                print(f"   Maximum depth: {args_dict['bab_max_depth']}")
                print(f"   Maximum subproblems: {args_dict['bab_max_subproblems']}")
                print(f"   Time limit: {args_dict['bab_time_limit']} seconds")

            else:
                print(f"⚠️  Specification refinement BaB only supports hybridz_relaxed_with_bab method, current method is {method}, automatically disabled")
                verifier.bab_config['enabled'] = False
        else:
            print("⚙️  Specification refinement BaB verification disabled")
            verifier.bab_config['enabled'] = False

        start_time = time.time()
        result = verifier.verify()
        end_time = time.time()
        print(f"⏱️  Total verification time: {end_time - start_time:.2f} seconds")
        if result == VerificationStatus.SAT:
            print("🎉 The property is satisfied.")
        elif result == VerificationStatus.UNSAT:
            print("❌ The property is not satisfied.")
        else:
            print("❓ The property status is unknown.")
    else:
        raise ValueError(f"Unsupported verifier: {verifier_type}. Supported verifiers: 'eran', 'abcrown', 'interval', 'hybridz'.")

