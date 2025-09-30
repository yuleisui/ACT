#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###########################################################################
##   Abstract Constraint Transformer (ACT) - HybridZonotope Transformers ##
##                                                                       ##
##   doctormeeee (https://github.com/doctormeeee) and contributors       ##
##   Copyright (C) 2024-2025                                             ##
##                                                                       ##
###########################################################################

import torch
import torch.nn.functional as F
import os
import sys
import psutil
import time

import path_config

from abstract_constraint_solver.hybridz.hybridz_operations import HybridZonotopeOps

def setup_gurobi_license():
    if 'GRB_LICENSE_FILE' not in os.environ:
        if 'ACTHOME' in os.environ:
            license_path = os.path.join(os.environ['ACTHOME'], 'gurobi', 'gurobi.lic')
            print(f"[ACT] Using ACTHOME environment variable: {os.environ['ACTHOME']}")
        else:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            solver_dir = os.path.dirname(current_dir)  
            verifier_dir = os.path.dirname(solver_dir)  
            project_root = os.path.dirname(verifier_dir) 
            license_path = os.path.join(project_root, 'gurobi', 'gurobi.lic')
            print(f"[ACT] Auto-detecting project root from file location")
        
        license_path = os.path.abspath(license_path)
        
        if os.path.exists(license_path):
            os.environ['GRB_LICENSE_FILE'] = license_path
            print(f"[ACT] Gurobi license found and set: {license_path}")
        else:
            print(f"[WARN] Gurobi license not found at: {license_path}")
            print(f"[INFO] Please ensure gurobi.lic is placed in: {os.path.dirname(license_path)}")
    else:
        print(f"[ACT] Using existing Gurobi license: {os.environ['GRB_LICENSE_FILE']}")

setup_gurobi_license()

torch.set_printoptions(
    linewidth=500,
    threshold=10000,
    sci_mode=False,
    precision=4
)

def print_memory_usage(stage_name=""):
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024

    system_memory = psutil.virtual_memory()
    total_mb = system_memory.total / 1024 / 1024
    available_mb = system_memory.available / 1024 / 1024
    used_percent = (memory_mb / total_mb) * 100

    if torch.cuda.is_available():
        gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
        gpu_cached_mb = torch.cuda.memory_reserved() / 1024 / 1024

    return memory_mb

class HybridZonotopeElem:

    def __init__(self, center=None, G_c=None, G_b=None, A_c=None, A_b=None, b=None,
                    method='hybridz', time_limit=500, relaxation_ratio=1.0, dtype=torch.float32, device='cpu',
                    ci_mode=False):
        self.center = None
        self.G_c = None
        self.G_b = None
        self.A_c = None
        self.A_b = None
        self.b = None
        self.method = method
        self.time_limit = time_limit
        self.relaxation_ratio = relaxation_ratio
        self.ci_mode = ci_mode

        if center is None and G_c is None and G_b is None and A_c is None and A_b is None and b is None:

            self.center = torch.tensor([], dtype=dtype, device=device)
            self.G_c = torch.tensor([], dtype=dtype, device=device)
            self.G_b = torch.tensor([], dtype=dtype, device=device)
            self.A_c = torch.tensor([], dtype=dtype, device=device)
            self.A_b = torch.tensor([], dtype=dtype, device=device)
            self.b = torch.tensor([], dtype=dtype, device=device)
            return

        if center is not None:
            self.center = center.detach().clone().to(dtype=dtype, device=device)
        else:
            raise ValueError("Center cannot be None, please provide a valid center tensor.")
        if G_c is not None:
            self.G_c = G_c.detach().clone().to(dtype=dtype, device=device)
        else:
            raise ValueError("G_c cannot be None, please provide a valid generator tensor.")

        if G_b is not None:
            self.G_b = G_b.detach().clone().to(dtype=dtype, device=device)
        if A_c is not None:
            self.A_c = A_c.detach().clone().to(dtype=dtype, device=device)
        if A_b is not None:
            self.A_b = A_b.detach().clone().to(dtype=dtype, device=device)
        if b is not None:
            self.b = b.detach().clone().to(dtype=dtype, device=device)

        self.dtype = dtype
        self.device = device

        self.n = self.center.shape[0]

        self.ng = self.G_c.shape[1] if self.G_c is not None and self.G_c.numel() > 0 else 0
        self.nb = self.G_b.shape[1] if self.G_b is not None and self.G_b.numel() > 0 else 0
        if self.A_c is not None and self.A_b is not None and self.b is not None:
            self.nc = self.b.shape[0] if self.b.numel() > 0 else 0
        else:
            self.nc = 0

    def set_method(self, method, time_limit=None):
        valid_methods = ['interval', 'hybridz', 'hybridz_relaxed', 'hybridz_relaxed_with_bab']
        if method not in valid_methods:
            raise ValueError(f"Invalid method '{method}'. Valid methods: {valid_methods}")

        self.method = method
        if time_limit is not None:
            self.time_limit = time_limit
        print(f"HybridZonotopeElem method updated to: {self.method}")

    def GetInputHybridZElem(self):
        return HybridZonotopeElem(self.center, self.G_c, self.G_b, self.A_c, self.A_b, self.b,
                                  method=self.method, time_limit=self.time_limit, relaxation_ratio=self.relaxation_ratio,
                                  dtype=self.dtype, device=self.device, ci_mode=self.ci_mode)

    def linear(self, W, bias=None, enable_generator_merging=False, cosine_threshold=0.95):

        new_center = W @ self.center + (bias.reshape(-1, 1) if bias is not None else 0)
        new_G_c = W @ self.G_c
        new_G_b = W @ self.G_b

        if enable_generator_merging and new_G_c is not None and new_G_c.shape[1] > 1:

            new_center, new_G_c, new_G_b, new_A_c, new_A_b, new_b = HybridZonotopeOps.MergeParallelGenerators(
                new_center, new_G_c, new_G_b, self.A_c, self.A_b, self.b,
                cosine_threshold=cosine_threshold,
                dtype=self.dtype, device=self.device
            )

            return HybridZonotopeElem(new_center, new_G_c, new_G_b, new_A_c, new_A_b, new_b,
                                      method=self.method, time_limit=self.time_limit, relaxation_ratio=self.relaxation_ratio,
                                      dtype=self.dtype, device=self.device)
        else:
            if not enable_generator_merging:

                pass
            else:

                pass

            return HybridZonotopeElem(new_center, new_G_c, new_G_b, self.A_c, self.A_b, self.b,
                                      method=self.method, time_limit=self.time_limit, relaxation_ratio=self.relaxation_ratio,
                                      dtype=self.dtype, device=self.device)

    def add(self, scalar):
        new_center, new_G_c, new_G_b, new_A_c, new_A_b, new_b = HybridZonotopeOps.AddCore(
            self.center, self.G_c, self.G_b, self.A_c, self.A_b, self.b, scalar
        )
        return HybridZonotopeElem(new_center, new_G_c, new_G_b, new_A_c, new_A_b, new_b,
                                  method=self.method, time_limit=self.time_limit, relaxation_ratio=self.relaxation_ratio,
                                  dtype=self.dtype, device=self.device)

    def subtract(self, scalar):
        new_center, new_G_c, new_G_b, new_A_c, new_A_b, new_b = HybridZonotopeOps.SubtractCore(
            self.center, self.G_c, self.G_b, self.A_c, self.A_b, self.b, scalar
        )
        return HybridZonotopeElem(new_center, new_G_c, new_G_b, new_A_c, new_A_b, new_b,
                                  method=self.method, time_limit=self.time_limit, relaxation_ratio=self.relaxation_ratio,
                                  dtype=self.dtype, device=self.device)

    def multiply(self, scalar):
        new_center, new_G_c, new_G_b, new_A_c, new_A_b, new_b = HybridZonotopeOps.MultiplyCore(
            self.center, self.G_c, self.G_b, self.A_c, self.A_b, self.b, scalar
        )
        return HybridZonotopeElem(new_center, new_G_c, new_G_b, new_A_c, new_A_b, new_b,
                                  method=self.method, time_limit=self.time_limit, relaxation_ratio=self.relaxation_ratio,
                                  dtype=self.dtype, device=self.device)

    def divide(self, scalar):
        new_center, new_G_c, new_G_b, new_A_c, new_A_b, new_b = HybridZonotopeOps.DivideCore(
            self.center, self.G_c, self.G_b, self.A_c, self.A_b, self.b, scalar
        )
        return HybridZonotopeElem(new_center, new_G_c, new_G_b, new_A_c, new_A_b, new_b,
                                  method=self.method, time_limit=self.time_limit, relaxation_ratio=self.relaxation_ratio,
                                  dtype=self.dtype, device=self.device)

    def relu(self, auto_lirpa_info=None, relu_constraints=None):

        if relu_constraints:
            print(f"[HZ ReLU] Applying {len(relu_constraints)} ReLU constraints")
            return self._relu_with_constraints(relu_constraints)

        if auto_lirpa_info is not None:
            hz_verifier = auto_lirpa_info.get('hz_verifier')
            layer_name = auto_lirpa_info.get('layer_name')

        return self._relu_standard()

    def _relu_auto_lirpa_pre_run(self, stable_pos, stable_neg, unstable, hz_verifier, layer_name):
        result = HybridZonotopeOps.ReLUAutoLiRPAPreRun(
            stable_pos, stable_neg, unstable, hz_verifier, layer_name,
            self.n, self.dtype, self.device
        )

        if result is None:
            return self._relu_standard()

        new_center_list, new_G_c_list, new_G_b_list, new_A_c_list, new_A_b_list, new_b_list = result

        new_center = torch.cat(new_center_list, dim=0)
        new_G_c = HybridZonotopeOps.BlockDiagonalCat(new_G_c_list)
        new_G_b = HybridZonotopeOps.BlockDiagonalCat(new_G_b_list)

        non_empty_A_c = [item for item in new_A_c_list if item.shape[0] > 0]
        non_empty_A_b = [item for item in new_A_b_list if item.shape[0] > 0]
        non_empty_b = [item for item in new_b_list if item.shape[0] > 0]

        new_A_c = HybridZonotopeOps.BlockDiagonalCat(non_empty_A_c) if non_empty_A_c else torch.zeros(0, new_G_c.shape[1], dtype=self.dtype, device=self.device)
        new_A_b = HybridZonotopeOps.BlockDiagonalCat(non_empty_A_b) if non_empty_A_b else torch.zeros(0, new_G_b.shape[1], dtype=self.dtype, device=self.device)
        new_b = torch.cat(non_empty_b, dim=0) if non_empty_b else torch.zeros(0, dtype=self.dtype, device=self.device)

        abstract_transformer_hz_elem = HybridZonotopeElem(
            new_center, new_G_c, new_G_b, new_A_c, new_A_b, new_b,
            method=self.method, time_limit=self.time_limit, relaxation_ratio=self.relaxation_ratio,
            dtype=self.dtype, device=self.device
        )
        input_hz_elem = self.GetInputHybridZElem()

        estimated_memory_gb, use_memory_optimized, debug_info = HybridZonotopeOps.MemoryUsageEstimationIntersection(
            abstract_transformer_hz_elem, input_hz_elem, memory_threshold_gb=1.0
        )
        HybridZonotopeOps.PrintMemoryEstimation(estimated_memory_gb, use_memory_optimized, debug_info, memory_threshold_gb=1.0)

        if use_memory_optimized:
            new_hz_elem = HybridZonotopeOps.ActivationOutputIntersectionElemMemoryOptimized(abstract_transformer_hz_elem, input_hz_elem)
        else:
            new_hz_elem = HybridZonotopeOps.ActivationOutputIntersectionElem(abstract_transformer_hz_elem, input_hz_elem)

        return new_hz_elem

    def _relu_with_constraints(self, relu_constraints):

        print(f"[HZ Elem] Starting ReLU constraint transformation, constraint count: {len(relu_constraints)}")

        lb, ub = HybridZonotopeOps.GetLayerWiseBounds(
            self.center, self.G_c, self.G_b, self.A_c, self.A_b, self.b,
            self.method, self.time_limit
        )

        constraints_applied = 0
        for constraint in relu_constraints:
            neuron_idx = constraint['neuron_idx']
            constraint_type = constraint['constraint_type']

            if neuron_idx < lb.numel():

                if constraint_type == 'inactive':

                    new_ub = min(ub[neuron_idx].item(), 0.0)
                    ub[neuron_idx] = new_ub

                    constraints_applied += 1
                elif constraint_type == 'active':

                    new_lb = max(lb[neuron_idx].item(), 0.0)
                    lb[neuron_idx] = new_lb

                    constraints_applied += 1
            else:

                pass

        new_center_list, new_G_c_list, new_G_b_list, new_A_c_list, new_A_b_list, new_b_list = HybridZonotopeOps.ReLUStandard(
            (lb, ub), self.dtype, self.device, method=self.method, relaxation_ratio=self.relaxation_ratio
        )

        new_center = torch.cat(new_center_list, dim=0)
        new_G_c = HybridZonotopeOps.BlockDiagonalCat(new_G_c_list)
        new_G_b = HybridZonotopeOps.BlockDiagonalCat(new_G_b_list)
        new_A_c = HybridZonotopeOps.BlockDiagonalCat(new_A_c_list)
        new_A_b = HybridZonotopeOps.BlockDiagonalCat(new_A_b_list)
        new_b = torch.cat(new_b_list, dim=0)

        abstract_transformer_hz_elem = HybridZonotopeElem(
            new_center, new_G_c, new_G_b, new_A_c, new_A_b, new_b,
            method=self.method, time_limit=self.time_limit, relaxation_ratio=self.relaxation_ratio,
            dtype=self.dtype, device=self.device
        )

        input_hz_elem = self.GetInputHybridZElem()

        estimated_memory_gb, use_memory_optimized, debug_info = HybridZonotopeOps.MemoryUsageEstimationIntersection(
            abstract_transformer_hz_elem, input_hz_elem, memory_threshold_gb=1.0
        )
        HybridZonotopeOps.PrintMemoryEstimation(estimated_memory_gb, use_memory_optimized, debug_info, memory_threshold_gb=1.0)

        if use_memory_optimized:
            new_hz_elem = HybridZonotopeOps.ActivationOutputIntersectionElemMemoryOptimized(abstract_transformer_hz_elem, input_hz_elem)
        else:
            new_hz_elem = HybridZonotopeOps.ActivationOutputIntersectionElem(abstract_transformer_hz_elem, input_hz_elem)

        return new_hz_elem

    def _relu_standard(self):

        print_memory_usage("Elem ReLU Start")

        lb, ub = HybridZonotopeOps.GetLayerWiseBounds(
            self.center, self.G_c, self.G_b, self.A_c, self.A_b, self.b,
            self.method, self.time_limit
        )

        new_center_list, new_G_c_list, new_G_b_list, new_A_c_list, new_A_b_list, new_b_list = HybridZonotopeOps.ReLUStandard(
            (lb, ub), self.dtype, self.device, method=self.method, relaxation_ratio=self.relaxation_ratio
        )

        new_center = torch.cat(new_center_list, dim=0)
        new_G_c = HybridZonotopeOps.BlockDiagonalCat(new_G_c_list)
        new_G_b = HybridZonotopeOps.BlockDiagonalCat(new_G_b_list)
        new_A_c = HybridZonotopeOps.BlockDiagonalCat(new_A_c_list)
        new_A_b = HybridZonotopeOps.BlockDiagonalCat(new_A_b_list)
        new_b = torch.cat(new_b_list, dim=0)

        abstract_transformer_hz_elem = HybridZonotopeElem(
            new_center, new_G_c, new_G_b, new_A_c, new_A_b, new_b,
            method=self.method, time_limit=self.time_limit, relaxation_ratio=self.relaxation_ratio,
            dtype=self.dtype, device=self.device
        )
        input_hz_elem = self.GetInputHybridZElem()

        estimated_memory_gb, use_memory_optimized, debug_info = HybridZonotopeOps.MemoryUsageEstimationIntersection(
            abstract_transformer_hz_elem, input_hz_elem, memory_threshold_gb=1.0
        )
        HybridZonotopeOps.PrintMemoryEstimation(estimated_memory_gb, use_memory_optimized, debug_info, memory_threshold_gb=1.0)

        if use_memory_optimized:
            new_hz_elem = HybridZonotopeOps.ActivationOutputIntersectionElemMemoryOptimized(abstract_transformer_hz_elem, input_hz_elem)
        else:
            new_hz_elem = HybridZonotopeOps.ActivationOutputIntersectionElem(abstract_transformer_hz_elem, input_hz_elem)

        return new_hz_elem

    def sigmoid_or_tanh(self, func_type):
        if func_type not in ['sigmoid', 'tanh']:
            raise ValueError(f"Unsupported function type: {func_type}. Supported: 'sigmoid', 'tanh'.")

        lb, ub = HybridZonotopeOps.GetLayerWiseBounds(self.center, self.G_c, self.G_b, self.A_c, self.A_b, self.b, self.method, self.time_limit, ci_mode=self.ci_mode)
        mid = (lb + ub) / 2

        new_center_list, new_G_c_list, new_G_b_list, new_A_c_list, new_A_b_list, new_b_list = \
            HybridZonotopeOps.SigmoidTanhStandard((lb, ub), mid, func_type,
                                                  self.dtype, self.device,
                                                  self.method, self.relaxation_ratio)

        new_center = torch.cat(new_center_list, dim=0)
        new_G_c = HybridZonotopeOps.BlockDiagonalCat(new_G_c_list)
        new_G_b = HybridZonotopeOps.BlockDiagonalCat(new_G_b_list)
        new_A_c = HybridZonotopeOps.BlockDiagonalCat(new_A_c_list)
        new_A_b = HybridZonotopeOps.BlockDiagonalCat(new_A_b_list)
        new_b = torch.cat(new_b_list, dim=0)

        abstract_transformer_hz_elem = HybridZonotopeElem(new_center, new_G_c, new_G_b, new_A_c, new_A_b, new_b,
                                                          method=self.method, time_limit=self.time_limit, relaxation_ratio=self.relaxation_ratio,
                                                          dtype=self.dtype, device=self.device)
        input_hz_elem = self.GetInputHybridZElem()

        estimated_memory_gb, use_memory_optimized, debug_info = HybridZonotopeOps.MemoryUsageEstimationIntersection(
            abstract_transformer_hz_elem, input_hz_elem, memory_threshold_gb=1.0
        )
        HybridZonotopeOps.PrintMemoryEstimation(estimated_memory_gb, use_memory_optimized, debug_info, memory_threshold_gb=1.0)

        if use_memory_optimized:
            new_hz_elem = HybridZonotopeOps.ActivationOutputIntersectionElemMemoryOptimized(abstract_transformer_hz_elem, input_hz_elem)
        else:
            new_hz_elem = HybridZonotopeOps.ActivationOutputIntersectionElem(abstract_transformer_hz_elem, input_hz_elem)

        return new_hz_elem

class HybridZonotopeGrid:
    def __init__(self, center_grid=None, G_c_grid=None, G_b_grid=None, A_c_tensor=None,
                 A_b_tensor=None, b_tensor=None, input_lb=None, input_ub=None, method='hybridz', time_limit=500, relaxation_ratio=1.0, dtype=torch.float32, device='cpu',
                 ci_mode=False):
        if (input_lb is not None or input_ub is not None) and (
            center_grid is not None or G_c_grid is not None or G_b_grid is not None or
            A_c_tensor is not None or A_b_tensor is not None or b_tensor is not None
        ):
            raise ValueError("HybridZonotopeGrid: cannot pass in grid params and input_lb/input_ub at the same time. Please choose either of them.")

        self.method = method
        self.time_limit = time_limit
        self.relaxation_ratio = relaxation_ratio
        self.dtype = dtype
        self.device = device
        self.ci_mode = ci_mode

        if input_lb is not None and input_ub is not None:

            self.center_grid, self.G_c_grid, self.G_b_grid, self.A_c_tensor, self.A_b_tensor, self.b_tensor = (
                HybridZonotopeGrid.fromInputBounds(input_lb, input_ub, dtype=dtype, device=device)
            )
        else:

            if center_grid is None or G_c_grid is None or G_b_grid is None:
                raise ValueError("center_grid, G_c_grid, G_b_grid cannot be None.")
            self.center_grid = center_grid.detach().to(dtype=dtype, device=device)
            self.G_c_grid = G_c_grid.detach().to(dtype=dtype, device=device)
            self.G_b_grid = G_b_grid.detach().to(dtype=dtype, device=device)
            self.A_c_tensor = A_c_tensor.detach().to(dtype=dtype, device=device) if A_c_tensor is not None else None
            self.A_b_tensor = A_b_tensor.detach().to(dtype=dtype, device=device) if A_b_tensor is not None else None
            self.b_tensor = b_tensor.detach().to(dtype=dtype, device=device) if b_tensor is not None else None

        self.C, self.H, self.W = self.center_grid.shape[:3]
        self.n = self.C * self.H * self.W
        self.ng = self.G_c_grid.shape[3]
        self.nb = self.G_b_grid.shape[3]
        self.nc = self.b_tensor.shape[0] if self.b_tensor is not None and self.b_tensor.numel() > 0 else 0

    def set_method(self, method, time_limit=None):
        valid_methods = ['interval', 'hybridz', 'hybridz_relaxed', 'hybridz_relaxed_with_bab']
        if method not in valid_methods:
            raise ValueError(f"Invalid method '{method}'. Valid methods: {valid_methods}")

        self.method = method
        if time_limit is not None:
            self.time_limit = time_limit
        print(f"HybridZonotopeGrid method updated to: {self.method}")

    @staticmethod
    def fromInputBounds(input_lb=None, input_ub=None, dtype=torch.float32, device='cpu'):

        assert input_lb is not None and input_ub is not None, "Input bounds cannot be None."
        assert input_lb.shape == input_ub.shape

        if input_lb.dim() == 1:
            input_lb = input_lb.unsqueeze(0).unsqueeze(0)
            input_ub = input_ub.unsqueeze(0).unsqueeze(0)

        elif input_lb.dim() == 2:
            input_lb = input_lb.unsqueeze(0)
            input_ub = input_ub.unsqueeze(0)

        elif input_lb.dim() == 3:

            pass
        else:
            raise ValueError(f"Unsupported input dimension: {input_lb.dim()}D. Supports 1D, 2D, 3D, 4D inputs.")

        C, H, W = input_lb.shape
        n = C * H * W
        ng = n
        nb = 0
        nc = 0

        center_tensor = ((input_lb + input_ub) / 2).to(dtype=dtype, device=device)
        radius_tensor = ((input_ub - input_lb) / 2).to(dtype=dtype, device=device)

        center_grid = center_tensor.unsqueeze(-1)

        G_c_grid = torch.zeros(C, H, W, ng, dtype=dtype, device=device)
        flat_radius = radius_tensor.reshape(-1)
        for idx in range(n):
            c = idx // (H * W)
            h = (idx % (H * W)) // W
            w = idx % W
            G_c_grid[c, h, w, idx] = flat_radius[idx]

        G_b_grid = torch.zeros(C, H, W, nb, dtype=dtype, device=device)

        A_c_tensor = torch.zeros((nc, ng), dtype=dtype, device=device)
        A_b_tensor = torch.zeros((nc, nb), dtype=dtype, device=device)
        b_tensor = torch.zeros((nc, 1), dtype=dtype, device=device)

        return center_grid, G_c_grid, G_b_grid, A_c_tensor, A_b_tensor, b_tensor

    def GetInputHybridZGrid(self):
        return HybridZonotopeGrid(self.center_grid, self.G_c_grid, self.G_b_grid, self.A_c_tensor, self.A_b_tensor, self.b_tensor,
                                  method=self.method, time_limit=self.time_limit, relaxation_ratio=self.relaxation_ratio,
                                  dtype=self.dtype, device=self.device, ci_mode=self.ci_mode)

    def add(self, scalar):
        new_center_grid, new_G_c_grid, new_G_b_grid, new_A_c_tensor, new_A_b_tensor, new_b_tensor = HybridZonotopeOps.AddCore(
            self.center_grid, self.G_c_grid, self.G_b_grid, self.A_c_tensor, self.A_b_tensor, self.b_tensor, scalar
        )
        return HybridZonotopeGrid(new_center_grid, new_G_c_grid, new_G_b_grid,
                                  new_A_c_tensor, new_A_b_tensor, new_b_tensor,
                                  method=self.method, time_limit=self.time_limit, relaxation_ratio=self.relaxation_ratio,
                                  dtype=self.dtype, device=self.device)

    def subtract(self, scalar):
        new_center_grid, new_G_c_grid, new_G_b_grid, new_A_c_tensor, new_A_b_tensor, new_b_tensor = HybridZonotopeOps.SubtractCore(
            self.center_grid, self.G_c_grid, self.G_b_grid, self.A_c_tensor, self.A_b_tensor, self.b_tensor, scalar
        )
        return HybridZonotopeGrid(new_center_grid, new_G_c_grid, new_G_b_grid,
                                  new_A_c_tensor, new_A_b_tensor, new_b_tensor,
                                  method=self.method, time_limit=self.time_limit, relaxation_ratio=self.relaxation_ratio,
                                  dtype=self.dtype, device=self.device)

    def multiply(self, scalar):
        new_center_grid, new_G_c_grid, new_G_b_grid, new_A_c_tensor, new_A_b_tensor, new_b_tensor = HybridZonotopeOps.MultiplyCore(
            self.center_grid, self.G_c_grid, self.G_b_grid, self.A_c_tensor, self.A_b_tensor, self.b_tensor, scalar
        )
        return HybridZonotopeGrid(new_center_grid, new_G_c_grid, new_G_b_grid,
                                  new_A_c_tensor, new_A_b_tensor, new_b_tensor,
                                  method=self.method, time_limit=self.time_limit, relaxation_ratio=self.relaxation_ratio,
                                  dtype=self.dtype, device=self.device)

    def divide(self, scalar):
        new_center_grid, new_G_c_grid, new_G_b_grid, new_A_c_tensor, new_A_b_tensor, new_b_tensor = HybridZonotopeOps.DivideCore(
            self.center_grid, self.G_c_grid, self.G_b_grid, self.A_c_tensor, self.A_b_tensor, self.b_tensor, scalar
        )
        return HybridZonotopeGrid(new_center_grid, new_G_c_grid, new_G_b_grid,
                                  new_A_c_tensor, new_A_b_tensor, new_b_tensor,
                                  method=self.method, time_limit=self.time_limit, relaxation_ratio=self.relaxation_ratio,
                                  dtype=self.dtype, device=self.device)

    def PreActivationGetFlattenedTensor(self):

        N = self.C * self.H * self.W
        flat_center = self.center_grid.reshape(N, *self.center_grid.shape[3:])
        flat_G_c = self.G_c_grid.reshape(N, *self.G_c_grid.shape[3:])

        if self.nb == 0:
            flat_G_b = self.G_b_grid.reshape(N, 0)
        else:
            flat_G_b = self.G_b_grid.reshape(N, *self.G_b_grid.shape[3:])
        return flat_center, flat_G_c, flat_G_b

    def GetReshapedTensor(self, flat_center, flat_G_c, flat_G_b, split=2):
        if split == 1:

            return flat_center.reshape(self.C, self.H, self.W, *flat_center.shape[1:]), \
                   flat_G_c.reshape(self.C, self.H, self.W, *flat_G_c.shape[1:]), \
                   flat_G_b.reshape(self.C, self.H, self.W, *flat_G_b.shape[1:])
        else:
            N = self.C * self.H * self.W
            assert flat_center.shape[0] == N * split, f"flat_center.shape[0]={flat_center.shape[0]}, expected {N}*{split}"
            center_grid = flat_center.reshape(self.C, self.H, self.W, split, *flat_center.shape[1:])
            G_c_grid = flat_G_c.reshape(self.C, self.H, self.W, split, *flat_G_c.shape[1:])
            G_b_grid = flat_G_b.reshape(self.C, self.H, self.W, split, *flat_G_b.shape[1:])
            return center_grid, G_c_grid, G_b_grid

    def relu(self, auto_lirpa_info=None, relu_constraints=None):

        if relu_constraints:

            return self._relu_with_constraints(relu_constraints)

        if auto_lirpa_info is not None:
            hz_verifier = auto_lirpa_info.get('hz_verifier')
            layer_name = auto_lirpa_info.get('layer_name')

        return self._relu_standard()

    def _relu_with_constraints(self, relu_constraints):

        print(f"[HZ Grid] Starting ReLU constraint transformation")

        flat_center, flat_G_c, flat_G_b = self.PreActivationGetFlattenedTensor()

        lb, ub = HybridZonotopeOps.GetLayerWiseBounds(
            flat_center, flat_G_c, flat_G_b,
            self.A_c_tensor, self.A_b_tensor, self.b_tensor,
            self.method, self.time_limit
        )

        for constraint in relu_constraints:
            neuron_idx = constraint['neuron_idx']
            constraint_type = constraint['constraint_type']

            if neuron_idx < lb.numel():
                if constraint_type == 'inactive':

                    ub[neuron_idx] = min(ub[neuron_idx].item(), 0.0)

                elif constraint_type == 'active':

                    lb[neuron_idx] = max(lb[neuron_idx].item(), 0.0)

        n_neurons = len(lb)
        batch_size = min(1024, n_neurons)

        all_new_center_list, all_new_G_c_list, all_new_G_b_list = [], [], []
        all_A_c_list, all_A_b_list, all_b_list = [], [], []

        for batch_start in range(0, n_neurons, batch_size):
            batch_end = min(batch_start + batch_size, n_neurons)
            batch_lb = lb[batch_start:batch_end]
            batch_ub = ub[batch_start:batch_end]

            batch_center_list, batch_G_c_list, batch_G_b_list, batch_A_c_list, batch_A_b_list, batch_b_list = HybridZonotopeOps.ReLUStandard(
                (batch_lb, batch_ub), self.dtype, self.device, method=self.method, relaxation_ratio=self.relaxation_ratio
            )

            all_new_center_list.extend(batch_center_list)
            all_new_G_c_list.extend(batch_G_c_list)
            all_new_G_b_list.extend(batch_G_b_list)
            all_A_c_list.extend(batch_A_c_list)
            all_A_b_list.extend(batch_A_b_list)
            all_b_list.extend(batch_b_list)

            del batch_center_list, batch_G_c_list, batch_G_b_list
            del batch_A_c_list, batch_A_b_list, batch_b_list
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        new_flat_center = torch.cat(all_new_center_list, dim=0)

        new_flat_G_c = HybridZonotopeOps.BlockDiagonalCat(all_new_G_c_list)
        new_flat_G_b = HybridZonotopeOps.BlockDiagonalCat(all_new_G_b_list)

        new_center_grid, new_G_c_grid, new_G_b_grid = self.GetReshapedTensor(
            new_flat_center, new_flat_G_c, new_flat_G_b, split=2
        )

        new_A_c_tensor = HybridZonotopeOps.BlockDiagonalCat(all_A_c_list) if all_A_c_list else torch.zeros(0, new_flat_G_c.shape[1], dtype=self.dtype, device=self.device)
        new_A_b_tensor = HybridZonotopeOps.BlockDiagonalCat(all_A_b_list) if all_A_b_list else torch.zeros(0, new_flat_G_b.shape[1], dtype=self.dtype, device=self.device)
        new_b_tensor = torch.cat(all_b_list, dim=0) if all_b_list else torch.zeros(0, 1, dtype=self.dtype, device=self.device)

        abstract_transformer_hz_grid = HybridZonotopeGrid(
            new_center_grid, new_G_c_grid, new_G_b_grid, new_A_c_tensor, new_A_b_tensor, new_b_tensor,
            method=self.method, time_limit=self.time_limit, relaxation_ratio=self.relaxation_ratio,
            dtype=self.dtype, device=self.device
        )

        input_hz_grid = self.GetInputHybridZGrid()
        new_hz_grid = HybridZonotopeOps.ActivationOutputIntersectionGrid(abstract_transformer_hz_grid, input_hz_grid)
        return new_hz_grid

    def _relu_standard(self):
        print_memory_usage("Grid ReLU Start")

        flat_center, flat_G_c, flat_G_b = self.PreActivationGetFlattenedTensor()

        start_time = time.time()

        lb, ub = HybridZonotopeOps.GetLayerWiseBounds(
            flat_center, flat_G_c, flat_G_b, self.A_c_tensor, self.A_b_tensor, self.b_tensor,
            self.method, self.time_limit
        )

        end_time = time.time()
        print(f"ReLU Grid Pre-activation bounds computed in {end_time - start_time:.4f} seconds")

        n_neurons = len(lb)
        batch_size = min(1024, n_neurons)
        print(f"Memory-optimized ReLU: processing {n_neurons} neurons in batches of {batch_size}")

        all_new_center_list, all_new_G_c_list, all_new_G_b_list = [], [], []
        all_A_c_list, all_A_b_list, all_b_list = [], [], []

        for batch_start in range(0, n_neurons, batch_size):
            batch_end = min(batch_start + batch_size, n_neurons)
            batch_lb = lb[batch_start:batch_end]
            batch_ub = ub[batch_start:batch_end]

            print(f"Processing batch {batch_start//batch_size + 1}/{(n_neurons + batch_size - 1)//batch_size}: neurons {batch_start}-{batch_end-1}")
            print_memory_usage(f"Batch {batch_start//batch_size + 1}")

            batch_center_list, batch_G_c_list, batch_G_b_list, batch_A_c_list, batch_A_b_list, batch_b_list = HybridZonotopeOps.ReLUStandard(
                (batch_lb, batch_ub), self.dtype, self.device, method=self.method, relaxation_ratio=self.relaxation_ratio
            )

            all_new_center_list.extend(batch_center_list)
            all_new_G_c_list.extend(batch_G_c_list)
            all_new_G_b_list.extend(batch_G_b_list)
            all_A_c_list.extend(batch_A_c_list)
            all_A_b_list.extend(batch_A_b_list)
            all_b_list.extend(batch_b_list)

            del batch_center_list, batch_G_c_list, batch_G_b_list
            del batch_A_c_list, batch_A_b_list, batch_b_list
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        print(f"Grid ReLU batch processing completed: {len(all_new_center_list)} total elements")

        print("Constructing block diagonal matrices in memory-efficient way...")

        new_flat_center = torch.cat(all_new_center_list, dim=0)

        new_flat_G_c = HybridZonotopeOps.BlockDiagonalCat(all_new_G_c_list)
        new_flat_G_b = HybridZonotopeOps.BlockDiagonalCat(all_new_G_b_list)

        new_center_grid, new_G_c_grid, new_G_b_grid = self.GetReshapedTensor(
            new_flat_center, new_flat_G_c, new_flat_G_b, split=2
        )

        new_A_c_tensor = HybridZonotopeOps.BlockDiagonalCat(all_A_c_list) if all_A_c_list else torch.zeros(0, new_flat_G_c.shape[1], dtype=self.dtype, device=self.device)
        new_A_b_tensor = HybridZonotopeOps.BlockDiagonalCat(all_A_b_list) if all_A_b_list else torch.zeros(0, new_flat_G_b.shape[1], dtype=self.dtype, device=self.device)
        new_b_tensor = torch.cat(all_b_list, dim=0) if all_b_list else torch.zeros(0, 1, dtype=self.dtype, device=self.device)

        print("Here in Grid ReLU standard mode, constructing new HybridZonotopeGrid for abstract tranformer.\n\n")
        abstract_transformer_hz_grid = HybridZonotopeGrid(
            new_center_grid, new_G_c_grid, new_G_b_grid, new_A_c_tensor, new_A_b_tensor, new_b_tensor,
            method=self.method, time_limit=self.time_limit, relaxation_ratio=self.relaxation_ratio,
            dtype=self.dtype, device=self.device
        )

        print("Here in Grid ReLU standard mode, constructing new HybridZonotopeGrid for the input hz.\n\n")
        input_hz_grid = self.GetInputHybridZGrid()
        new_hz_grid = HybridZonotopeOps.ActivationOutputIntersectionGrid(abstract_transformer_hz_grid, input_hz_grid)

        return new_hz_grid

    def _relu_auto_lirpa_pre_run(self, stable_pos, stable_neg, unstable, hz_verifier, layer_name):

        flat_center, flat_G_c, flat_G_b = self.PreActivationGetFlattenedTensor()
        N = flat_center.shape[0]

        result = HybridZonotopeOps.ReLUAutoLiRPAPreRun(
            stable_pos, stable_neg, unstable, hz_verifier, layer_name,
            N, self.dtype, self.device
        )

        if result is None:
            return self._relu_standard()

        new_center_list, new_G_c_list, new_G_b_list, A_c_list, A_b_list, b_list = result

        new_flat_center = torch.cat(new_center_list, dim=0)
        new_flat_G_c = HybridZonotopeOps.BlockDiagonalCat(new_G_c_list)
        new_flat_G_b = HybridZonotopeOps.BlockDiagonalCat(new_G_b_list)

        new_center_grid, new_G_c_grid, new_G_b_grid = self.GetReshapedTensor(
            new_flat_center, new_flat_G_c, new_flat_G_b, split=2
        )

        new_A_c_tensor = HybridZonotopeOps.BlockDiagonalCat(A_c_list) if A_c_list else torch.zeros(0, new_flat_G_c.shape[1], dtype=self.dtype, device=self.device)
        new_A_b_tensor = HybridZonotopeOps.BlockDiagonalCat(A_b_list) if A_b_list else torch.zeros(0, new_flat_G_b.shape[1], dtype=self.dtype, device=self.device)
        new_b_tensor = torch.cat(b_list, dim=0) if b_list else torch.zeros(0, 1, dtype=self.dtype, device=self.device)

        abstract_transformer_hz_grid = HybridZonotopeGrid(
            new_center_grid, new_G_c_grid, new_G_b_grid, new_A_c_tensor, new_A_b_tensor, new_b_tensor,
            method=self.method, time_limit=self.time_limit, relaxation_ratio=self.relaxation_ratio,
            dtype=self.dtype, device=self.device
        )
        input_hz_grid = self.GetInputHybridZGrid()
        new_hz_grid = HybridZonotopeOps.ActivationOutputIntersectionGrid(abstract_transformer_hz_grid, input_hz_grid)

        return new_hz_grid

    def sigmoid_or_tanh(self, func_type):

        if func_type not in ['sigmoid', 'tanh']:
            raise ValueError(f"Unsupported function type: {func_type}. Supported: 'sigmoid', 'tanh'.")

        print(f"Computing bounds for entire feature map ({self.C}x{self.H}x{self.W}) for {func_type}")
        flat_center, flat_G_c, flat_G_b = self.PreActivationGetFlattenedTensor()

        lb, ub = HybridZonotopeOps.GetLayerWiseBounds(
            flat_center, flat_G_c, flat_G_b,
            self.A_c_tensor, self.A_b_tensor, self.b_tensor,
            self.method, self.time_limit
        )
        mid = (lb + ub) / 2

        N = flat_center.shape[0]

        new_center_list, new_G_c_list, new_G_b_list, new_A_c_list, new_A_b_list, new_b_list = \
            HybridZonotopeOps.SigmoidTanhStandard((lb, ub), mid, func_type,
                                                  self.dtype, self.device,
                                                  self.method, self.relaxation_ratio)

        new_flat_center = torch.cat(new_center_list, dim=0)
        new_flat_G_c = HybridZonotopeOps.BlockDiagonalCat(new_G_c_list)
        new_flat_G_b = HybridZonotopeOps.BlockDiagonalCat(new_G_b_list)

        new_center_grid, new_G_c_grid, new_G_b_grid = self.GetReshapedTensor(
            new_flat_center, new_flat_G_c, new_flat_G_b, split=2
        )

        new_A_c_tensor = HybridZonotopeOps.BlockDiagonalCat(new_A_c_list) if new_A_c_list else torch.zeros(0, new_flat_G_c.shape[1], dtype=self.dtype, device=self.device)
        new_A_b_tensor = HybridZonotopeOps.BlockDiagonalCat(new_A_b_list) if new_A_b_list else torch.zeros(0, new_flat_G_b.shape[1], dtype=self.dtype, device=self.device)
        new_b_tensor = torch.cat(new_b_list, dim=0) if new_b_list else torch.zeros(0, 1, dtype=self.dtype, device=self.device)

        abstract_transformer_hz_grid = HybridZonotopeGrid(
            new_center_grid, new_G_c_grid, new_G_b_grid, new_A_c_tensor, new_A_b_tensor, new_b_tensor,
            method=self.method, time_limit=self.time_limit, relaxation_ratio=self.relaxation_ratio,
            dtype=self.dtype, device=self.device
        )
        input_hz_grid = self.GetInputHybridZGrid()
        new_hz_grid = HybridZonotopeOps.ActivationOutputIntersectionGrid(abstract_transformer_hz_grid, input_hz_grid)
        return new_hz_grid

    def conv(self, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):

        center = self.center_grid.permute(3, 0, 1, 2)
        new_center = F.conv2d(center, weight, bias, stride, padding, dilation, groups)
        new_center = new_center.permute(1, 2, 3, 0)

        _, _, _, ng = self.G_c_grid.shape
        G_c_out_list = []
        for g in range(ng):
            Gc = self.G_c_grid[..., g].unsqueeze(0)
            out = F.conv2d(Gc, weight, None, stride, padding, dilation, groups)
            G_c_out_list.append(out)

        new_G_c = torch.stack(G_c_out_list, dim=-1).squeeze(0)

        _, _, _, nb = self.G_b_grid.shape
        if nb > 0:
            G_b_out_list = []
            for b in range(nb):
                Gb = self.G_b_grid[..., b].unsqueeze(0)
                out = F.conv2d(Gb, weight, None, stride, padding, dilation, groups)
                G_b_out_list.append(out)
            new_G_b = torch.stack(G_b_out_list, dim=-1).squeeze(0)
        else:

            C_out, H_out, W_out = new_center.shape[:3]
            new_G_b = torch.zeros(C_out, H_out, W_out, 0, dtype=self.dtype, device=self.device)

        return HybridZonotopeGrid(
            new_center, new_G_c, new_G_b,
            self.A_c_tensor, self.A_b_tensor, self.b_tensor,
            method=self.method, time_limit=self.time_limit, relaxation_ratio=self.relaxation_ratio,
            dtype=self.dtype, device=self.device
        )

    def maxpool(self, kernel_size, stride=1, padding=0, auto_lirpa_info=None):

        if auto_lirpa_info is not None:
            bounded_model = auto_lirpa_info.get('bounded_model')
            layer_name = auto_lirpa_info.get('layer_name')
            layer_bounds = auto_lirpa_info.get('layer_bounds', {})
            print(f"MaxPool using auto_LiRPA optimization for {layer_name}")

        else:
            print(" MaxPool using standard computation")

        kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        stride = (stride, stride) if isinstance(stride, int) else stride
        padding = (padding, padding) if isinstance(padding, int) else padding

        C_in, H_in, W_in = self.C, self.H, self.W
        H_ker, W_ker = kernel_size

        C_out = C_in
        H_out = (H_in - H_ker + 2 * padding[0]) // stride[0] + 1
        W_out = (W_in - W_ker + 2 * padding[1]) // stride[1] + 1

        print("Using serial MILP MaxPool...")
        return self._maxpool_serial(kernel_size, stride, padding)

    def _maxpool_serial(self, kernel_size, stride, padding):

        C_in, H_in, W_in = self.C, self.H, self.W
        H_ker, W_ker = kernel_size
        C_out = C_in
        H_out = (H_in - H_ker + 2 * padding[0]) // stride[0] + 1
        W_out = (W_in - W_ker + 2 * padding[1]) // stride[1] + 1

        print(f"Computing bounds for entire input feature map ({C_in}x{H_in}x{W_in}) before MaxPool")
        flat_center, flat_G_c, flat_G_b = self.PreActivationGetFlattenedTensor()

        input_lb, input_ub = HybridZonotopeOps.GetLayerWiseBounds(
            flat_center, flat_G_c, flat_G_b,
            self.A_c_tensor, self.A_b_tensor, self.b_tensor,
            self.method, self.time_limit
        )

        input_lb_grid = input_lb.reshape(C_in, H_in, W_in)
        input_ub_grid = input_ub.reshape(C_in, H_in, W_in)

        print(f"✅ Bounds computed, now performing MaxPool pooling with bounds-guided selection")

        new_center_grid = torch.zeros(C_out, H_out, W_out, 1, dtype=self.center_grid.dtype, device=self.center_grid.device)
        new_G_c_grid = torch.zeros(C_out, H_out, W_out, self.ng, dtype=self.G_c_grid.dtype, device=self.G_c_grid.device)
        new_G_b_grid = torch.zeros(C_out, H_out, W_out, self.nb, dtype=self.G_b_grid.dtype, device=self.G_b_grid.device)

        for c_out in range(C_out):
            for h_out in range(H_out):
                for w_out in range(W_out):
                    max_lb = None
                    max_ub = None
                    best_h_in = None
                    best_w_in = None

                    for kh in range(H_ker):
                        for kw in range(W_ker):
                            h_in = h_out * stride[0] - padding[0] + kh
                            w_in = w_out * stride[1] - padding[1] + kw

                            if not (0 <= h_in < H_in and 0 <= w_in < W_in):
                                continue

                            lb_current = input_lb_grid[c_out, h_in, w_in].item()
                            ub_current = input_ub_grid[c_out, h_in, w_in].item()

                            if max_ub is None or ub_current > max_ub:
                                max_lb = lb_current
                                max_ub = ub_current
                                best_h_in = h_in
                                best_w_in = w_in
                            elif ub_current == max_ub and lb_current > max_lb:
                                max_lb = lb_current
                                best_h_in = h_in
                                best_w_in = w_in

                    if best_h_in is not None and best_w_in is not None:
                        new_center_grid[c_out, h_out, w_out] = self.center_grid[c_out, best_h_in, best_w_in]
                        new_G_c_grid[c_out, h_out, w_out] = self.G_c_grid[c_out, best_h_in, best_w_in]
                        new_G_b_grid[c_out, h_out, w_out] = self.G_b_grid[c_out, best_h_in, best_w_in]

        return HybridZonotopeGrid(
            new_center_grid, new_G_c_grid, new_G_b_grid,
            self.A_c_tensor, self.A_b_tensor, self.b_tensor,
            method=self.method, time_limit=self.time_limit, relaxation_ratio=self.relaxation_ratio,
            dtype=self.dtype, device=self.device
        )

    def _maxpool_auto_lirpa_based(self, kernel_size, stride, padding, auto_lirpa_info):

        try:
            bounded_model = auto_lirpa_info.get('bounded_model')
            layer_name = auto_lirpa_info.get('layer_name')
            layer_bounds = auto_lirpa_info.get('layer_bounds', {})
            verifier = auto_lirpa_info.get('verifier')

            if not bounded_model or not layer_bounds:
                print("⚠️  Insufficient auto_LiRPA info, falling back to MILP")
                if not bounded_model:
                    print("  Missing: bounded_model")
                if not layer_bounds:
                    print("  Missing: layer_bounds")
                return None

            print(f"auto_LiRPA guided element selection for {layer_name}")
            print(f"Available layer bounds: {list(layer_bounds.keys())}")

            return self._maxpool_auto_lirpa_selection(kernel_size, stride, padding, bounded_model, layer_bounds)

        except Exception as e:
            print(f"⚠️  auto_LiRPA guided MaxPool failed: {e}")
            return None

    def _maxpool_auto_lirpa_selection(self, kernel_size, stride, padding, bounded_model, layer_bounds):

        print("Learning MaxPool element selection from auto_LiRPA...")

        C_in, H_in, W_in = self.C, self.H, self.W
        H_ker, W_ker = kernel_size
        C_out = C_in
        H_out = (H_in - H_ker + 2 * padding[0]) // stride[0] + 1
        W_out = (W_in - W_ker + 2 * padding[1]) // stride[1] + 1

        print(f"MaxPool mapping: {C_in}×{H_in}×{W_in} → {C_out}×{H_out}×{W_out}")

        try:

            input_bounds, output_bounds = self._extract_maxpool_bounds(
                layer_bounds, C_in, H_in, W_in, C_out, H_out, W_out, kernel_size, stride, padding
            )

            if input_bounds is None or output_bounds is None:
                print("⚠️  Cannot find matching input/output bounds")
                return None

            selection_mapping = self._analyze_selection_pattern(
                input_bounds, output_bounds, kernel_size, stride, padding
            )

            return self._apply_selection_pattern(selection_mapping, C_out, H_out, W_out)

        except Exception as e:
            print(f"❌ Learning from auto_LiRPA failed: {e}")
            return None

    def _extract_maxpool_bounds(self, layer_bounds, C_in, H_in, W_in, C_out, H_out, W_out, kernel_size, stride, padding):
        input_target_size = C_in * H_in * W_in
        output_target_size = C_out * H_out * W_out

        print(f"Looking for input bounds (size {input_target_size}) and output bounds (size {output_target_size})")

        layer_keys = list(layer_bounds.keys())
        input_bounds = None
        output_bounds = None

        for i, key in enumerate(layer_keys):
            bounds = layer_bounds[key]
            shape = bounds.get('shape', [0])

            if len(shape) >= 4:
                total_elements = shape[1] * shape[2] * shape[3]
                actual_shape = (shape[1], shape[2], shape[3])
            elif len(shape) >= 2:
                total_elements = shape[1]
                actual_shape = (shape[1],)
            else:
                total_elements = shape[0] if len(shape) > 0 else 0
                actual_shape = shape

            if total_elements == input_target_size:
                print(f"Found potential input layer: {key} (size={total_elements})")

                if i + 1 < len(layer_keys):
                    next_key = layer_keys[i + 1]
                    next_bounds = layer_bounds[next_key]
                    next_shape = next_bounds.get('shape', [0])

                    if len(next_shape) >= 4:
                        next_total_elements = next_shape[1] * next_shape[2] * next_shape[3]
                    elif len(next_shape) >= 2:
                        next_total_elements = next_shape[1]
                    else:
                        next_total_elements = next_shape[0] if len(next_shape) > 0 else 0

                    print(f"Checking next layer {next_key}: size={next_total_elements}, target_output={output_target_size}")

                    if next_total_elements == output_target_size:
                        print(f"✅ Found input-output pair: {key} → {next_key}")

                        lower_tensor = bounds['lower'].squeeze(0) if bounds['lower'].shape[0] == 1 else bounds['lower']
                        upper_tensor = bounds['upper'].squeeze(0) if bounds['upper'].shape[0] == 1 else bounds['upper']
                        input_bounds = {
                            'lower': lower_tensor.view(C_in, H_in, W_in),
                            'upper': upper_tensor.view(C_in, H_in, W_in)
                        }

                        next_lower_tensor = next_bounds['lower'].squeeze(0) if next_bounds['lower'].shape[0] == 1 else next_bounds['lower']
                        next_upper_tensor = next_bounds['upper'].squeeze(0) if next_bounds['upper'].shape[0] == 1 else next_bounds['upper']
                        output_bounds = {
                            'lower': next_lower_tensor.view(C_out, H_out, W_out),
                            'upper': next_upper_tensor.view(C_out, H_out, W_out)
                        }

                        break

                if input_bounds and output_bounds:
                    break

        if input_bounds and output_bounds:
            print("Successfully extracted both input and output bounds!")

            self._debug_print_maxpool_bounds(input_bounds, output_bounds, kernel_size, stride, padding)
            return input_bounds, output_bounds
        else:
            print("❌ Could not find matching input-output pair")
            return None, None

    def _analyze_selection_pattern(self, input_bounds, output_bounds, kernel_size, stride, padding):
        input_lb = input_bounds['lower']
        input_ub = input_bounds['upper']
        output_lb = output_bounds['lower']
        output_ub = output_bounds['upper']

        H_ker, W_ker = kernel_size
        C_out, H_out, W_out = output_lb.shape
        C_in, H_in, W_in = input_lb.shape

        selection_mapping = {}

        print("Analyzing selection pattern by direct boundary matching...")

        for c in range(C_out):
            for h_out in range(H_out):
                for w_out in range(W_out):
                    target_lb = output_lb[c, h_out, w_out].item()
                    target_ub = output_ub[c, h_out, w_out].item()

                    found_match = False

                    for kh in range(H_ker):
                        for kw in range(W_ker):
                            h_in = h_out * stride[0] - padding[0] + kh
                            w_in = w_out * stride[1] - padding[1] + kw

                            if 0 <= h_in < H_in and 0 <= w_in < W_in:
                                elem_lb = input_lb[c, h_in, w_in].item()
                                elem_ub = input_ub[c, h_in, w_in].item()

                                lb_match = abs(elem_lb - target_lb) < 1e-6
                                ub_match = abs(elem_ub - target_ub) < 1e-6

                                if lb_match and ub_match:
                                    selection_mapping[(c, h_out, w_out)] = (c, h_in, w_in)
                                    found_match = True
                                    break

                        if found_match:
                            break

                    if not found_match:
                        for kh in range(H_ker):
                            for kw in range(W_ker):
                                h_in = h_out * stride[0] - padding[0] + kh
                                w_in = w_out * stride[1] - padding[1] + kw

                                if 0 <= h_in < H_in and 0 <= w_in < W_in:
                                    elem_ub = input_ub[c, h_in, w_in].item()

                                    if abs(elem_ub - target_ub) < 1e-6:
                                        selection_mapping[(c, h_out, w_out)] = (c, h_in, w_in)
                                        found_match = True
                                        break

                            if found_match:
                                break

        print(f"✅ Found selection mapping for {len(selection_mapping)} positions")
        return selection_mapping

    def _apply_selection_pattern(self, selection_mapping, C_out, H_out, W_out):
        print("Applying learned selection pattern...")

        new_center_grid = torch.zeros(C_out, H_out, W_out, 1, dtype=self.dtype, device=self.device)
        new_G_c_grid = torch.zeros(C_out, H_out, W_out, self.ng, dtype=self.dtype, device=self.device)
        new_G_b_grid = torch.zeros(C_out, H_out, W_out, self.nb, dtype=self.dtype, device=self.device)

        copied_count = 0
        for (c_out, h_out, w_out), (c_in, h_in, w_in) in selection_mapping.items():
            print("Selected position:", (c_out, h_out, w_out), "from input position:", (c_in, h_in, w_in))

            new_center_grid[c_out, h_out, w_out] = self.center_grid[c_in, h_in, w_in]
            new_G_c_grid[c_out, h_out, w_out] = self.G_c_grid[c_in, h_in, w_in]
            new_G_b_grid[c_out, h_out, w_out] = self.G_b_grid[c_in, h_in, w_in]
            copied_count += 1

        print(f"✅ Copied {copied_count} elements based on auto_LiRPA selection pattern")

        return HybridZonotopeGrid(
            new_center_grid, new_G_c_grid, new_G_b_grid,
            self.A_c_tensor, self.A_b_tensor, self.b_tensor,
            method=self.method, time_limit=self.time_limit, relaxation_ratio=self.relaxation_ratio,
            dtype=self.dtype, device=self.device
        )

    def _debug_print_maxpool_bounds(self, input_bounds, output_bounds, kernel_size, stride, padding):
        input_lb = input_bounds['lower']
        input_ub = input_bounds['upper']
        output_lb = output_bounds['lower']
        output_ub = output_bounds['upper']

        C_in, H_in, W_in = input_lb.shape
        C_out, H_out, W_out = output_lb.shape
        H_ker, W_ker = kernel_size

        print("\n" + "="*80)
        print("MaxPool Boundary Analysis - Validating auto_LiRPA Selection Strategy")
        print("="*80)
        print(f"Input size: {C_in}×{H_in}×{W_in}, Output size: {C_out}×{H_out}×{W_out}")
        print(f"Pooling parameters: kernel={kernel_size}, stride={stride}, padding={padding}")

        c = 0
        print(f"\nChannel {c} boundary information:")

        print("\n🔢 Input boundaries:")
        print("Position format: (h,w) [lower_bound, upper_bound]")
        for h in range(H_in):
            for w in range(W_in):
                lb = input_lb[c, h, w].item()
                ub = input_ub[c, h, w].item()
                print(f"({h},{w}) [{lb:.3f}, {ub:.3f}]", end="  ")
            print()

        print("\nOutput boundaries:")
        for h_out in range(H_out):
            for w_out in range(W_out):
                lb = output_lb[c, h_out, w_out].item()
                ub = output_ub[c, h_out, w_out].item()
                print(f"({h_out},{w_out}) [{lb:.3f}, {ub:.3f}]", end="  ")
            print()

        print(f"\nDetailed pooling window analysis:")
        for h_out in range(H_out):
            for w_out in range(W_out):
                print(f"\nOutput position ({h_out},{w_out}):")
                print(f"Output boundary: [{output_lb[c, h_out, w_out].item():.3f}, {output_ub[c, h_out, w_out].item():.3f}]")
                print(f"Corresponding pooling window:")

                for kh in range(H_ker):
                    for kw in range(W_ker):
                        h_in = h_out * stride[0] - padding[0] + kh
                        w_in = w_out * stride[1] - padding[1] + kw

                        if 0 <= h_in < H_in and 0 <= w_in < W_in:
                            lb = input_lb[c, h_in, w_in].item()
                            ub = input_ub[c, h_in, w_in].item()

                            lb_match = abs(lb - output_lb[c, h_out, w_out].item()) < 1e-6
                            ub_match = abs(ub - output_ub[c, h_out, w_out].item()) < 1e-6
                            match_str = ""
                            if lb_match and ub_match:
                                match_str = " ← Perfect match!"
                            elif ub_match:
                                match_str = " ← Upper bound match"
                            elif lb_match:
                                match_str = " ← 📉 Lower bound match"

                            print(f"Position({h_in},{w_in}): [{lb:.3f}, {ub:.3f}]{match_str}")

        print("="*80 + "\n")
