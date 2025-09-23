import torch
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from scipy.optimize import linprog
import torch.nn.functional as F

import os
import sys
import psutil
import gc

def setup_gurobi_license():
    if 'GRB_LICENSE_FILE' not in os.environ:
        if 'ACTHOME' in os.environ:
            license_path = os.path.join(os.environ['ACTHOME'], 'gurobi', 'gurobi.lic')
            print(f"[ACT] Using ACTHOME environment variable: {os.environ['ACTHOME']}")
        else:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.join(current_dir, '..')
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
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'modules', 'abcrown', 'auto_LiRPA')))
from auto_LiRPA import BoundedModule, PerturbationLpNorm, BoundedTensor

import time

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
                    method='hybridz', time_limit=500, relaxation_ratio=1.0, dtype=torch.float32, device='cpu'):
        self.center = None
        self.G_c = None
        self.G_b = None
        self.A_c = None
        self.A_b = None
        self.b = None
        self.method = method
        self.time_limit = time_limit
        self.relaxation_ratio = relaxation_ratio

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
                                  dtype=self.dtype, device=self.device)

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
            print(f"🔒 [HZ ReLU] Applying {len(relu_constraints)} ReLU constraints")
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

        print(f"🔒 [HZ Elem] Starting ReLU constraint transformation, constraint count: {len(relu_constraints)}")

        lb, ub = HybridZonotopeOps.GetLayerWiseBounds(
            self.center, self.G_c, self.G_b, self.A_c, self.A_b, self.b,
            self.method, self.time_limit
        )

        original_unstable = ((lb < 0) & (ub > 0)).sum().item()

        constraints_applied = 0
        for constraint in relu_constraints:
            neuron_idx = constraint['neuron_idx']
            constraint_type = constraint['constraint_type']

            if neuron_idx < lb.numel():
                old_lb = lb[neuron_idx].item()
                old_ub = ub[neuron_idx].item()

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

        modified_unstable = ((lb < 0) & (ub > 0)).sum().item()

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

        start_time = time.time()

        lb, ub = HybridZonotopeOps.GetLayerWiseBounds(
            self.center, self.G_c, self.G_b, self.A_c, self.A_b, self.b,
            self.method, self.time_limit
        )
        end_time = time.time()

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

        lb, ub = HybridZonotopeOps.GetLayerWiseBounds(self.center, self.G_c, self.G_b, self.A_c, self.A_b, self.b, self.method, self.time_limit)
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
                 A_b_tensor=None, b_tensor=None, input_lb=None, input_ub=None, method='hybridz', time_limit=500, relaxation_ratio=1.0, dtype=torch.float32, device='cpu'):
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
                                  dtype=self.dtype, device=self.device)

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

        print(f"🔒 [HZ Grid] Starting ReLU constraint transformation")

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
        print(f"🧠 Memory-optimized ReLU: processing {n_neurons} neurons in batches of {batch_size}")

        all_new_center_list, all_new_G_c_list, all_new_G_b_list = [], [], []
        all_A_c_list, all_A_b_list, all_b_list = [], [], []

        for batch_start in range(0, n_neurons, batch_size):
            batch_end = min(batch_start + batch_size, n_neurons)
            batch_lb = lb[batch_start:batch_end]
            batch_ub = ub[batch_start:batch_end]

            print(f"  Processing batch {batch_start//batch_size + 1}/{(n_neurons + batch_size - 1)//batch_size}: neurons {batch_start}-{batch_end-1}")
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

        print("🔧 Constructing block diagonal matrices in memory-efficient way...")

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

        print(f"🚀 Computing bounds for entire feature map ({self.C}x{self.H}x{self.W}) for {func_type}")
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
            print(f"🚀 MaxPool using auto_LiRPA optimization for {layer_name}")

        else:
            print("⚙️  MaxPool using standard computation")

        kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        stride = (stride, stride) if isinstance(stride, int) else stride
        padding = (padding, padding) if isinstance(padding, int) else padding

        C_in, H_in, W_in = self.C, self.H, self.W
        H_ker, W_ker = kernel_size

        C_out = C_in
        H_out = (H_in - H_ker + 2 * padding[0]) // stride[0] + 1
        W_out = (W_in - W_ker + 2 * padding[1]) // stride[1] + 1

        print("🔧 Using serial MILP MaxPool...")
        return self._maxpool_serial(kernel_size, stride, padding)

    def _maxpool_serial(self, kernel_size, stride, padding):

        C_in, H_in, W_in = self.C, self.H, self.W
        H_ker, W_ker = kernel_size
        C_out = C_in
        H_out = (H_in - H_ker + 2 * padding[0]) // stride[0] + 1
        W_out = (W_in - W_ker + 2 * padding[1]) // stride[1] + 1

        print(f"🚀 Computing bounds for entire input feature map ({C_in}x{H_in}x{W_in}) before MaxPool")
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
                    print("   Missing: bounded_model")
                if not layer_bounds:
                    print("   Missing: layer_bounds")
                return None

            print(f"� auto_LiRPA guided element selection for {layer_name}")
            print(f"📋 Available layer bounds: {list(layer_bounds.keys())}")

            return self._maxpool_auto_lirpa_selection(kernel_size, stride, padding, bounded_model, layer_bounds)

        except Exception as e:
            print(f"⚠️  auto_LiRPA guided MaxPool failed: {e}")
            return None

    def _maxpool_auto_lirpa_selection(self, kernel_size, stride, padding, bounded_model, layer_bounds):

        print("🧠 Learning MaxPool element selection from auto_LiRPA...")

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
                print(f"🎯 Found potential input layer: {key} (size={total_elements})")

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

                    print(f"  Checking next layer {next_key}: size={next_total_elements}, target_output={output_target_size}")

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
            print("🎯 Successfully extracted both input and output bounds!")

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
        print("🎯 Applying learned selection pattern...")

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
        print(f"🔧 Pooling parameters: kernel={kernel_size}, stride={stride}, padding={padding}")

        c = 0
        print(f"\n📋 Channel {c} boundary information:")

        print("\n🔢 Input boundaries:")
        print("Position format: (h,w) [lower_bound, upper_bound]")
        for h in range(H_in):
            for w in range(W_in):
                lb = input_lb[c, h, w].item()
                ub = input_ub[c, h, w].item()
                print(f"({h},{w}) [{lb:.3f}, {ub:.3f}]", end="  ")
            print()

        print("\n🎯 Output boundaries:")
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
                print(f"  Output boundary: [{output_lb[c, h_out, w_out].item():.3f}, {output_ub[c, h_out, w_out].item():.3f}]")
                print(f"  Corresponding pooling window:")

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
                                match_str = " ← 🎯 Perfect match!"
                            elif ub_match:
                                match_str = " ← 📈 Upper bound match"
                            elif lb_match:
                                match_str = " ← 📉 Lower bound match"

                            print(f"    Position({h_in},{w_in}): [{lb:.3f}, {ub:.3f}]{match_str}")

        print("="*80 + "\n")

class HybridZonotopeOps:

    @staticmethod
    def MemoryUsageEstimationIntersection(abstract_transformer_hz, input_hz, memory_threshold_gb=1.0):

        Z_n, Z_ng, Z_nb = abstract_transformer_hz.n, abstract_transformer_hz.ng, abstract_transformer_hz.nb
        Z_nc = abstract_transformer_hz.A_c.shape[0] if abstract_transformer_hz.A_c is not None and abstract_transformer_hz.A_c.numel() > 0 else 0

        Y_n, Y_ng, Y_nb = input_hz.n, input_hz.ng, input_hz.nb
        Y_nc = input_hz.A_c.shape[0] if input_hz.A_c is not None and input_hz.A_c.numel() > 0 else 0

        center_memory = (Z_n + Y_n) * 4
        G_c_memory = (Z_n * Z_ng + Y_n * Y_ng) * 4
        G_b_memory = (Z_n * Z_nb + Y_n * Y_nb) * 4
        A_c_memory = (Z_nc * Z_ng + Y_nc * Y_ng) * 4
        R_matrix_memory = (Y_n * Z_n) * 4

        total_memory_bytes = center_memory + G_c_memory + G_b_memory + A_c_memory + R_matrix_memory
        estimated_memory_gb = total_memory_bytes / (1024**3)
        use_memory_optimized = estimated_memory_gb > memory_threshold_gb

        debug_info = {
            'estimated_memory_gb': estimated_memory_gb,
            'Z_dims': (Z_n, Z_ng, Z_nb, Z_nc),
            'Y_dims': (Y_n, Y_ng, Y_nb, Y_nc),
            'memory_breakdown': {
                'center_memory_gb': center_memory / (1024**3),
                'G_c_memory_gb': G_c_memory / (1024**3),
                'G_b_memory_gb': G_b_memory / (1024**3),
                'A_c_memory_gb': A_c_memory / (1024**3),
                'R_matrix_memory_gb': R_matrix_memory / (1024**3)
            }
        }

        return estimated_memory_gb, use_memory_optimized, debug_info

    @staticmethod
    def PrintMemoryEstimation(estimated_memory_gb, use_memory_optimized, debug_info, memory_threshold_gb=1.0):

        Z_n, Z_ng, Z_nb, Z_nc = debug_info['Z_dims']
        Y_n, Y_ng, Y_nb, Y_nc = debug_info['Y_dims']

        if estimated_memory_gb >= 0.001:
            memory_str = f"{estimated_memory_gb:.3f} GB"
        else:
            memory_mb = estimated_memory_gb * 1024
            memory_str = f"{memory_mb:.2f} MB"

        if use_memory_optimized:

            pass
        else:

            pass

    @staticmethod
    def ConstructNeuronDifferenceHZ(output_hz_elem, target_neuron, other_neuron):

        n_outputs = output_hz_elem.n

        W = torch.zeros((1, n_outputs), dtype=output_hz_elem.dtype, device=output_hz_elem.device)
        W[0, target_neuron] = 1.0
        W[0, other_neuron] = -1.0

        diff_hz = output_hz_elem.linear(W, bias=None)

        return diff_hz

    @staticmethod
    def AddCore(center, G_c, G_b, A_c, A_b, b, scalar):
        new_center = center + scalar
        return new_center, G_c, G_b, A_c, A_b, b

    @staticmethod
    def SubtractCore(center, G_c, G_b, A_c, A_b, b, scalar):
        new_center = center - scalar
        return new_center, G_c, G_b, A_c, A_b, b

    @staticmethod
    def MultiplyCore(center, G_c, G_b, A_c, A_b, b, scalar):
        new_center = center * scalar
        new_G_c = G_c * scalar
        new_G_b = G_b * scalar
        return new_center, new_G_c, new_G_b, A_c, A_b, b

    @staticmethod
    def DivideCore(center, G_c, G_b, A_c, A_b, b, scalar):
        if abs(scalar) < 1e-10:
            raise ValueError("Division by zero or near-zero scalar is undefined.")

        new_center = center / scalar
        new_G_c = G_c / scalar
        new_G_b = G_b / scalar
        return new_center, new_G_c, new_G_b, A_c, A_b, b

    @staticmethod
    def ComputeIntervalElemBounds(flat_center, flat_G_c, flat_G_b):
        Gc_term = torch.sum(torch.abs(flat_G_c), dim=1, keepdim=True)
        Gb_term = torch.sum(torch.abs(flat_G_b), dim=1, keepdim=True)
        radius = Gc_term + Gb_term
        lb = (flat_center.squeeze() - radius.squeeze())
        ub = (flat_center.squeeze() + radius.squeeze())
        return lb, ub

    @staticmethod
    def ComputeGenericZElemBounds(flat_center, flat_G_c):
        Gc_term = torch.sum(torch.abs(flat_G_c), dim=1, keepdim=True)
        lb = (flat_center.squeeze() - Gc_term.squeeze())
        ub = (flat_center.squeeze() + Gc_term.squeeze())
        return lb, ub

    @staticmethod
    def ComputeConstrainedZElemBoundsLinprog(center, Gc, Ac, b, time_limit=500):

        if not hasattr(center, 'detach'):
            center = torch.tensor(center, dtype=torch.float32)
        if not hasattr(Gc, 'detach'):
            Gc = torch.tensor(Gc, dtype=torch.float32)
        if Ac is not None and not hasattr(Ac, 'detach'):
            Ac = torch.tensor(Ac, dtype=torch.float32)
        if b is not None and not hasattr(b, 'detach'):
            b = torch.tensor(b, dtype=torch.float32)

        if center.dim() == 0:
            center = center.unsqueeze(0)
        if center.dim() == 1 and Gc.dim() == 1:
            Gc = Gc.unsqueeze(0)

        N = center.shape[0]
        ng = Gc.shape[1]

        to_np = lambda x: x.detach().cpu().numpy() if hasattr(x, 'detach') else np.asarray(x)
        center_np = to_np(center).flatten()
        Gc_np = to_np(Gc)
        A_eq = to_np(Ac) if Ac is not None else np.zeros((0, ng))
        b_eq = to_np(b).flatten() if b is not None else np.zeros(0)

        bounds = [(-1, 1)] * ng

        lb = np.zeros(N)
        ub = np.zeros(N)

        for i in range(N):

            c = -Gc_np[i, :]

            try:
                res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
                if res.success:
                    ub[i] = center_np[i] - res.fun
                else:
                    ub[i] = float('inf')
            except:
                ub[i] = float('inf')

            try:
                res = linprog(-c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
                if res.success:
                    lb[i] = center_np[i] + res.fun
                else:
                    lb[i] = -float('inf')
            except:
                lb[i] = -float('inf')

        lb_tensor = torch.tensor(lb, dtype=center.dtype, device=center.device)
        ub_tensor = torch.tensor(ub, dtype=center.dtype, device=center.device)

        if lb_tensor.numel() == 1:
            return lb_tensor.item(), ub_tensor.item()
        else:
            return lb_tensor, ub_tensor

    @staticmethod
    def ComputeConstrainedZElemBoundsSMT(center_dim, Gc_dim, Ac_dim, b_dim, time_limit=500, return_counterexample=False, property_constraint=None):

        try:
            import z3
        except ImportError:
            print("⚠️  Z3 not installed, falling back to Gurobi solver")

            if property_constraint is not None:
                print("❌ [SMT] Safety property verification requires Z3, but Z3 is not installed")
                return None, None, "UNKNOWN"
            elif return_counterexample:

                lb, ub = HybridZonotopeOps.ComputeConstrainedZElemBoundsGurobi(center_dim, Gc_dim, Ac_dim, b_dim, time_limit)
                return lb, ub, None
            else:

                return HybridZonotopeOps.ComputeConstrainedZElemBoundsGurobi(center_dim, Gc_dim, Ac_dim, b_dim, time_limit)

        if property_constraint is not None:
            print(f"🔍 [SMT] Entering safety property verification mode: {property_constraint}")
            violation_result = HybridZonotopeOps._SolvePropertyViolationSMT(center_dim, Gc_dim, Ac_dim, b_dim, property_constraint, time_limit)

            if violation_result is True:

                print("🚨 [SMT] Safety property violation found - unsafe configuration exists")
                return None, None, "UNSAFE"
            elif violation_result is False:

                print("✅ [SMT] Safety property verified - constraint system is unsatisfiable")
                return None, None, "SAFE"
            else:

                print("⚠️  [SMT] Solver result unknown - possible timeout or solver issue")
                return None, None, "UNKNOWN"

        print(f"📊 [SMT] Entering boundary computation mode (return_counterexample={return_counterexample})")

        if not hasattr(center_dim, 'detach'):
            center_dim = torch.tensor(center_dim, dtype=torch.float32)
        if not hasattr(Gc_dim, 'detach'):
            Gc_dim = torch.tensor(Gc_dim, dtype=torch.float32)
        if Ac_dim is not None and not hasattr(Ac_dim, 'detach'):
            Ac_dim = torch.tensor(Ac_dim, dtype=torch.float32)
        if b_dim is not None and not hasattr(b_dim, 'detach'):
            b_dim = torch.tensor(b_dim, dtype=torch.float32)

        to_np = lambda x: x.detach().cpu().numpy() if hasattr(x, 'detach') else np.asarray(x)
        center_np = to_np(center_dim).flatten()
        Gc_np = to_np(Gc_dim)
        Ac_np = to_np(Ac_dim) if Ac_dim is not None else np.zeros((0, Gc_np.shape[1]))
        b_np = to_np(b_dim).flatten() if b_dim is not None else np.zeros(0)

        N = len(center_np)
        ng = Gc_np.shape[1]
        nc = len(b_np)

        print(f"🔧 [SMT] Computing bounds for {N} neurons...")

        lb = np.zeros(N)
        ub = np.zeros(N)
        counterexamples = {} if return_counterexample else None

        for neuron_idx in range(N):
                neuron_lb, neuron_ub = HybridZonotopeOps._SolveSingleNeuronSMT(
                    center_np, Gc_np, Ac_np, b_np, neuron_idx, time_limit
                )

                lb[neuron_idx] = neuron_lb
                ub[neuron_idx] = neuron_ub

        lb_tensor = torch.tensor(lb, dtype=center_dim.dtype, device=center_dim.device)
        ub_tensor = torch.tensor(ub, dtype=center_dim.dtype, device=center_dim.device)

        if lb_tensor.numel() == 1:

            result_lb = lb_tensor.item()
            result_ub = ub_tensor.item()
        else:

            result_lb = lb_tensor
            result_ub = ub_tensor

        if return_counterexample:

            return result_lb, result_ub, counterexamples
        else:

            return result_lb, result_ub

    @staticmethod
    def _SolvePropertyViolationSMT(flat_center, flat_G_c, A_c_tensor, b_tensor, property_constraint, time_limit=30):

        try:
            import z3
        except ImportError:
            print("⚠️  Z3 not installed, cannot perform SMT counterexample generation")
            return None

        if not hasattr(flat_center, 'detach'):
            flat_center = torch.tensor(flat_center, dtype=torch.float32)
        if not hasattr(flat_G_c, 'detach'):
            flat_G_c = torch.tensor(flat_G_c, dtype=torch.float32)
        if A_c_tensor is not None and not hasattr(A_c_tensor, 'detach'):
            A_c_tensor = torch.tensor(A_c_tensor, dtype=torch.float32)
        if b_tensor is not None and not hasattr(b_tensor, 'detach'):
            b_tensor = torch.tensor(b_tensor, dtype=torch.float32)

        to_np = lambda x: x.detach().cpu().numpy() if hasattr(x, 'detach') else np.asarray(x)
        center_np = to_np(flat_center).flatten()
        Gc_np = to_np(flat_G_c)
        Ac_np = to_np(A_c_tensor) if A_c_tensor is not None and A_c_tensor.numel() > 0 else np.zeros((0, Gc_np.shape[1]))
        b_np = to_np(b_tensor).flatten() if b_tensor is not None and b_tensor.numel() > 0 else np.zeros(0)

        n_outputs, ng = Gc_np.shape
        nc = len(b_np)

        print(f"🔧 [SMT] Using unified SMT framework to solve property violation...")
        print(f"   Parameters: n_outputs={n_outputs}, ng={ng}, nc={nc}")

        solver = z3.Solver()
        solver.set("timeout", time_limit * 1000)

        eps_c_vars = [z3.Real(f"eps_c_{i}") for i in range(ng)]

        for i, eps_var in enumerate(eps_c_vars):
            solver.add(eps_var >= -1.0)
            solver.add(eps_var <= 1.0)

        if nc > 0:
            for i in range(nc):
                constraint_expr = 0
                for j in range(ng):
                    if abs(Ac_np[i, j]) > 1e-12:
                        constraint_expr += float(Ac_np[i, j]) * eps_c_vars[j]

                if isinstance(constraint_expr, (int, float)) and constraint_expr == 0:

                    if abs(b_np[i]) > 1e-12:
                        print(f"   ❌ [SMT] Constraint {i} unsatisfiable: 0 == {b_np[i]}")
                        return None
                else:

                    solver.add(constraint_expr == float(b_np[i]))

        output_exprs = []
        for i in range(n_outputs):
            output_expr = float(center_np[i])
            for j in range(ng):
                if abs(Gc_np[i, j]) > 1e-12:
                    output_expr += float(Gc_np[i, j]) * eps_c_vars[j]
            output_exprs.append(output_expr)

        if "output_max_other > output[" in property_constraint:

            import re
            match = re.search(r'output\[(\d+)\]', property_constraint)
            if match:
                true_label = int(match.group(1))
                print(f"   🎯 [SMT] Classification violation: finding max(others) > output[{true_label}]")

                other_constraints = []
                for k in range(n_outputs):
                    if k != true_label:
                        other_constraints.append(output_exprs[k] > output_exprs[true_label])

                if other_constraints:
                    solver.add(z3.Or(other_constraints))
                else:
                    print(f"   ❌ [SMT] Only one output class, cannot construct violation constraint")
                    return None
            else:
                print(f"   ❌ [SMT] Cannot parse true label from: {property_constraint}")
                return None

        elif "linear_constraints_violated" in property_constraint:

            print(f"   🎯 [SMT] Linear constraint violation: specific violation condition not yet implemented")

            return None

        else:
            print(f"   ❌ [SMT] Unrecognized property constraint type: {property_constraint}")
            return None

        print(f"   🔍 [SMT] startingsolving...")
        check_result = solver.check()

        if check_result == z3.sat:
            print(f"   ✅ [SMT] Found solution satisfying violation constraint")
            model = solver.model()

            eps_c_values = []
            for i, eps_var in enumerate(eps_c_vars):
                val = model[eps_var]
                if val is not None:
                    eps_c_values.append(float(val.as_decimal(6)))
                else:
                    eps_c_values.append(0.0)

            print(f"   📊 [SMT] Found eps_c solution satisfying violation condition: {eps_c_values[:5]}..." if len(eps_c_values) > 5 else f"   📊 [SMT] eps_c solution: {eps_c_values}")

            print(f"   ✅ [SMT] Satisfiability check passed: solution violating safety property exists")
            return True

        elif check_result == z3.unsat:
            print(f"   ✅ [SMT] constraint unsatisfiable, safety property verified")
            return False
        else:
            print(f"   ⚠️  [SMT] solving timed out or unknown result: {check_result}")
            return None

    @staticmethod
    def _SolveSingleNeuronSMT(center_np, Gc_np, Ac_np, b_np, neuron_idx, time_limit):

        import z3

        ng = Gc_np.shape[1]
        nc = len(b_np)

        lb_solver = z3.Optimize()
        lb_solver.set(timeout=int(time_limit * 1000))

        eps_c = [z3.Real(f'eps_c_{i}') for i in range(ng)]

        for i in range(ng):
            lb_solver.add(eps_c[i] >= -1.0)
            lb_solver.add(eps_c[i] <= 1.0)

        for row in range(nc):
            constraint_expr = 0
            for col in range(ng):
                if abs(Ac_np[row, col]) > 1e-12:
                    constraint_expr += Ac_np[row, col] * eps_c[col]
            lb_solver.add(constraint_expr == b_np[row])

        objective = center_np[neuron_idx]
        for i in range(ng):
            if abs(Gc_np[neuron_idx, i]) > 1e-12:
                objective += Gc_np[neuron_idx, i] * eps_c[i]

        lb_solver.minimize(objective)
        lb_result = lb_solver.check()

        if lb_result == z3.sat:
            lb_model = lb_solver.model()
            lower_bound = float(center_np[neuron_idx])
            lb_eps_values = []

            for i in range(ng):
                eps_val = lb_model.eval(eps_c[i], model_completion=True)
                eps_float = float(eps_val.as_fraction()) if eps_val is not None else 0.0
                lb_eps_values.append(eps_float)
                lower_bound += float(Gc_np[neuron_idx, i]) * eps_float
        else:
            lower_bound = float('-inf')
            lb_eps_values = None

        ub_solver = z3.Optimize()
        ub_solver.set(timeout=int(time_limit * 1000))

        eps_c_ub = [z3.Real(f'eps_c_ub_{i}') for i in range(ng)]

        for i in range(ng):
            ub_solver.add(eps_c_ub[i] >= -1.0)
            ub_solver.add(eps_c_ub[i] <= 1.0)

        for row in range(nc):
            constraint_expr = 0
            for col in range(ng):
                if abs(Ac_np[row, col]) > 1e-12:
                    constraint_expr += Ac_np[row, col] * eps_c_ub[col]
            ub_solver.add(constraint_expr == b_np[row])

        objective_ub = center_np[neuron_idx]
        for i in range(ng):
            if abs(Gc_np[neuron_idx, i]) > 1e-12:
                objective_ub += Gc_np[neuron_idx, i] * eps_c_ub[i]

        ub_solver.maximize(objective_ub)
        ub_result = ub_solver.check()

        if ub_result == z3.sat:
            ub_model = ub_solver.model()
            upper_bound = float(center_np[neuron_idx])
            ub_eps_values = []

            for i in range(ng):
                eps_val = ub_model.eval(eps_c_ub[i], model_completion=True)
                eps_float = float(eps_val.as_fraction()) if eps_val is not None else 0.0
                ub_eps_values.append(eps_float)
                upper_bound += float(Gc_np[neuron_idx, i]) * eps_float
        else:
            upper_bound = float('inf')
            ub_eps_values = None

        return lower_bound, upper_bound

    @staticmethod
    def ComputeConstrainedZElemBoundsGurobi(center_dim, Gc_dim, Ac_dim, b_dim, time_limit=500):

        if not hasattr(center_dim, 'detach'):
            center_dim = torch.tensor(center_dim, dtype=torch.float32)
        if not hasattr(Gc_dim, 'detach'):
            Gc_dim = torch.tensor(Gc_dim, dtype=torch.float32)
        if Ac_dim is not None and not hasattr(Ac_dim, 'detach'):
            Ac_dim = torch.tensor(Ac_dim, dtype=torch.float32)
        if b_dim is not None and not hasattr(b_dim, 'detach'):
            b_dim = torch.tensor(b_dim, dtype=torch.float32)

        if center_dim.dim() == 0:
            center_dim = center_dim.unsqueeze(0)
        if center_dim.dim() == 1 and Gc_dim.dim() == 1:
            Gc_dim = Gc_dim.unsqueeze(0)

        N = center_dim.shape[0]
        ng = Gc_dim.shape[1] if Gc_dim.numel() > 0 else 0
        nc = Ac_dim.shape[0] if Ac_dim is not None and Ac_dim.numel() > 0 else 0

        to_np = lambda x: x.detach().cpu().numpy() if hasattr(x, 'detach') else np.asarray(x)
        center_np = to_np(center_dim).flatten()
        Gc_np = to_np(Gc_dim)
        Ac_np = to_np(Ac_dim) if Ac_dim is not None else np.zeros((0, ng))
        b_np = to_np(b_dim).flatten() if b_dim is not None else np.zeros(0)

        env = gp.Env(empty=True)
        env.setParam('OutputFlag', 0)
        env.setParam('LogToConsole', 0)
        env.start()
        model = gp.Model(env=env)
        model.setParam('OutputFlag', 0)

        is_output_layer = (N <= 20)

        model.setParam('Threads', 0)
        model.setParam('Method', 2)
        model.setParam('Crossover', 0)
        model.setParam('BarHomogeneous', 1)

        if is_output_layer:

            model.setParam('TimeLimit', min(480, time_limit * N))
            model.setParam('Presolve', 2)
            model.setParam('NumericFocus', 1)
            model.setParam('ScaleFlag', 2)

        else:

            model.setParam('TimeLimit', time_limit * N)
            model.setParam('Presolve', 2)
            model.setParam('NumericFocus', 1)
            model.setParam('Aggregate', 1)
            model.setParam('AggFill', 20)

        eps_c = model.addVars(ng, lb=-1, ub=1, vtype=GRB.CONTINUOUS, name="eps_c")

        eps_tolerance = 1e-12
        constraint_start_time = time.time()
        added_constraints = 0

        Ac_nonzero = np.nonzero(np.abs(Ac_np) > eps_tolerance)

        for row in range(nc):

            row_terms = []

            ac_indices = np.where(np.abs(Ac_np[row, :]) > eps_tolerance)[0]
            for col in ac_indices:
                coeff = Ac_np[row, col]
                row_terms.append(coeff * eps_c[col])

            if row_terms:
                model.addConstr(gp.quicksum(row_terms) == b_np[row], name=f"constraint_{row}")
                added_constraints += 1

            if (row + 1) % 100 == 0 or row == nc - 1:
                elapsed = time.time() - constraint_start_time

        lb = np.zeros(N)
        ub = np.zeros(N)

        for neuron_idx in range(N):

            obj_coeffs_Gc = Gc_np[neuron_idx, :]

            nonzero_indices_Gc = np.where(np.abs(obj_coeffs_Gc) > 1e-12)[0]
            nonzero_coeffs_Gc = obj_coeffs_Gc[nonzero_indices_Gc]

            solve_start = time.time()
            obj_terms = [center_np[neuron_idx]]
            if len(nonzero_indices_Gc) > 0:
                obj_terms.extend([Gc_np[neuron_idx, i] * eps_c[i] for i in nonzero_indices_Gc])

            obj = gp.quicksum(obj_terms)

            if len(obj_terms) == 1:

                lb[neuron_idx] = center_np[neuron_idx]
                ub[neuron_idx] = center_np[neuron_idx]
                solve_time = time.time() - solve_start

                continue

            solve_start = time.time()

            model.setObjective(obj, GRB.MAXIMIZE)
            model.optimize()
            if model.status == GRB.OPTIMAL:
                ub[neuron_idx] = model.objVal
            else:
                ub[neuron_idx] = float('inf')

            model.setObjective(obj, GRB.MINIMIZE)
            model.optimize()
            if model.status == GRB.OPTIMAL:
                lb[neuron_idx] = model.objVal
            else:
                lb[neuron_idx] = -float('inf')

            solve_time = time.time() - solve_start

        model.dispose()
        env.dispose()

        lb_tensor = torch.tensor(lb, dtype=center_dim.dtype, device=center_dim.device)
        ub_tensor = torch.tensor(ub, dtype=center_dim.dtype, device=center_dim.device)

        if lb_tensor.numel() == 1:
            return lb_tensor.item(), ub_tensor.item()
        else:
            return lb_tensor, ub_tensor

    @staticmethod
    def ComputeHybridZElemBoundsGurobi(center_dim, Gc_dim, Gb_dim, Ac_dim, Ab_dim, b_dim, time_limit=500):

        if not hasattr(center_dim, 'detach'):
            center_dim = torch.tensor(center_dim, dtype=torch.float32)
        if not hasattr(Gc_dim, 'detach'):
            Gc_dim = torch.tensor(Gc_dim, dtype=torch.float32)
        if not hasattr(Gb_dim, 'detach'):
            Gb_dim = torch.tensor(Gb_dim, dtype=torch.float32)
        if Ac_dim is not None and not hasattr(Ac_dim, 'detach'):
            Ac_dim = torch.tensor(Ac_dim, dtype=torch.float32)
        if Ab_dim is not None and not hasattr(Ab_dim, 'detach'):
            Ab_dim = torch.tensor(Ab_dim, dtype=torch.float32)
        if b_dim is not None and not hasattr(b_dim, 'detach'):
            b_dim = torch.tensor(b_dim, dtype=torch.float32)

        if center_dim.dim() == 0:
            center_dim = center_dim.unsqueeze(0)
        if center_dim.dim() == 1 and Gc_dim.dim() == 1:
            Gc_dim = Gc_dim.unsqueeze(0)
            Gb_dim = Gb_dim.unsqueeze(0)

        N = center_dim.shape[0]
        ng = Gc_dim.shape[1] if Gc_dim.numel() > 0 else 0
        nb = Gb_dim.shape[1] if Gb_dim.numel() > 0 else 0
        nc = Ac_dim.shape[0] if Ac_dim is not None and Ac_dim.numel() > 0 else 0

        to_np = lambda x: x.detach().cpu().numpy() if hasattr(x, 'detach') else np.asarray(x)
        center_np = to_np(center_dim).flatten()
        Gc_np = to_np(Gc_dim)
        Gb_np = to_np(Gb_dim)
        Ac_np = to_np(Ac_dim) if Ac_dim is not None else np.zeros((0, ng))
        Ab_np = to_np(Ab_dim) if Ab_dim is not None else np.zeros((0, nb))
        b_np = to_np(b_dim).flatten() if b_dim is not None else np.zeros(0)

        total_constraint_terms = nc * (ng + nb)
        print(f"📊 MILP problem size analysis:")
        print(f"   - Neuron count N: {N}")
        print(f"   - Continuous variables ng: {ng}")
        print(f"   - Binary variables nb: {nb} ⚠️")
        print(f"   - Constraint count nc: {nc}")
        print(f"   - Total constraint terms: {total_constraint_terms:,}")
        print(f"   - Binary complexity: 2^{nb} = {2**min(nb, 20):,}...")

        if nc > 0:
            Ac_density = np.count_nonzero(Ac_np) / (nc * ng) if ng > 0 else 0
            Ab_density = np.count_nonzero(Ab_np) / (nc * nb) if nb > 0 else 0
            b_range = (np.min(b_np), np.max(b_np)) if nc > 0 else (0, 0)
            print(f"🔍 Constraint matrix analysis:")
            print(f"   - Ac nonzero density: {Ac_density:.4f} ({np.count_nonzero(Ac_np)}/{nc*ng})")
            print(f"   - Ab nonzero density: {Ab_density:.4f} ({np.count_nonzero(Ab_np)}/{nc*nb})")
            print(f"   - b value range: [{b_range[0]:.6f}, {b_range[1]:.6f}]")
            print(f"   - Ac coefficient range: [{np.min(Ac_np):.6f}, {np.max(Ac_np):.6f}]")
            print(f"   - Ab coefficient range: [{np.min(Ab_np):.6f}, {np.max(Ab_np):.6f}]")

        print(f"🔧 Building Gurobi model...")
        print(f"🔧 Model construction start time: {time.time()}")
        model_start_time = time.time()

        import gc
        gc.collect()
        print(f"🧹 Memory defragmentation done")

        try:

            env = gp.Env(empty=True)
            env.setParam('OutputFlag', 0)
            env.setParam('LogToConsole', 0)

            env.start()
            model = gp.Model(env=env)
            print(f"✅ Gurobi new environment + model created successfully")
        except Exception as e:
            print(f"❌ Gurobi model creation failed: {e}")
            raise

        model.setParam('OutputFlag', 0)
        model.setParam('LogToConsole', 0)

        model.setParam('Threads', 0)
        model.setParam('Aggregate', 1)
        model.setParam('AggFill', 20)

        is_output_layer = (N <= 20)

        if nb > 200 and not is_output_layer:
            print(f"⚡ Ultra large-scale binary problem (nb={nb}) - sparse matrix focused config")
            model.setParam('TimeLimit', min(300, time_limit))
            model.setParam('Method', 3)
            model.setParam('MIPFocus', 2)
            model.setParam('Presolve', 2)
            model.setParam('Cuts', 3)
            model.setParam('Heuristics', 0.05)
            model.setParam('Threads', 0)

            model.setParam('NumericFocus', 1)
            model.setParam('ScaleFlag', 2)
        elif nb > 200 and is_output_layer:
            print(f"🎯 Output layer ultra large-scale binary problem (nb={nb}, N={N}) - 32-core parallel feasibility-first config")
            model.setParam('TimeLimit', 480)
            model.setParam('Method', 3)
            model.setParam('MIPFocus', 3)
            model.setParam('Presolve', 1)
            model.setParam('Cuts', 1)
            model.setParam('Heuristics', 0.2)

            model.setParam('Threads', 0)

            model.setParam('NumericFocus', 2)
            model.setParam('FeasibilityTol', 1e-6)
            model.setParam('OptimalityTol', 1e-6)
        elif nb > 50:
            print(f"🎯 Large-scale binary problem (nb={nb}) - 32-core parallel accuracy-first config")
            model.setParam('TimeLimit', time_limit * N)
            model.setParam('Threads', 0)

            model.setParam('MIPFocus', 1)
            model.setParam('Presolve', 2)
            model.setParam('Cuts', 2)
            model.setParam('Heuristics', 0.1)

        else:
            print(f"🚀 Small/medium-scale binary problem (nb={nb}) - 32-core parallel standard config")
            model.setParam('TimeLimit', time_limit * N)
            model.setParam('Threads', 0)
            model.setParam('Presolve', 2)
            model.setParam('MIPFocus', 1)

        print(f"🔧 Adding variables: {ng} continuous + {nb} binary...")
        var_start_time = time.time()
        eps_c = model.addVars(ng, lb=-1, ub=1, vtype=GRB.CONTINUOUS, name="eps_c")
        eps_b = model.addVars(nb, vtype=GRB.BINARY, name="eps_b")
        var_time = time.time() - var_start_time
        print(f"✅ Variable addition complete ({var_time:.2f}s)")

        constraint_start_time = time.time()

        eps_tolerance = 1e-12

        Ac_nonzero = np.nonzero(np.abs(Ac_np) > eps_tolerance)
        Ab_nonzero = np.nonzero(np.abs(Ab_np) > eps_tolerance)

        for row in range(nc):

            row_terms = []

            ac_indices = np.where(np.abs(Ac_np[row, :]) > eps_tolerance)[0]
            for col in ac_indices:
                coeff = Ac_np[row, col]
                row_terms.append(coeff * eps_c[col])

            ab_indices = np.where(np.abs(Ab_np[row, :]) > eps_tolerance)[0]
            for col in ab_indices:
                coeff = Ab_np[row, col]
                row_terms.append(coeff * (2 * eps_b[col] - 1))

            if row_terms:
                model.addConstr(gp.quicksum(row_terms) == b_np[row], name=f"constraint_{row}")

            if (row + 1) % 100 == 0 or row == nc - 1:
                elapsed = time.time() - constraint_start_time

        constraint_time = time.time() - constraint_start_time

        print(f"🔧 Updating model...")
        update_start_time = time.time()
        model.update()
        update_time = time.time() - update_start_time
        print(f"✅ Model update complete ({update_time:.2f}s)")
        model_build_time = time.time() - model_start_time
        print(f"📊 Sparse model build total time: {model_build_time:.2f}s")

        lb = np.zeros(N)
        ub = np.zeros(N)

        print(f"🚀 Starting optimization for {N} neurons...")
        opt_start_time = time.time()

        recent_times = []

        for neuron_idx in range(N):
            neuron_start_time = time.time()

            obj_coeffs_Gc = Gc_np[neuron_idx, :]
            obj_coeffs_Gb = Gb_np[neuron_idx, :]

            nonzero_indices_Gc = np.where(np.abs(obj_coeffs_Gc) > 1e-12)[0]
            nonzero_indices_Gb = np.where(np.abs(obj_coeffs_Gb) > 1e-12)[0]
            nonzero_coeffs_Gc = obj_coeffs_Gc[nonzero_indices_Gc]
            nonzero_coeffs_Gb = obj_coeffs_Gb[nonzero_indices_Gb]

            solve_start = time.time()
            obj_terms = [center_np[neuron_idx]]

            if len(nonzero_indices_Gc) > 0:
                obj_terms.extend([Gc_np[neuron_idx, i] * eps_c[i] for i in nonzero_indices_Gc])

            if len(nonzero_indices_Gb) > 0:
                obj_terms.extend([Gb_np[neuron_idx, i] * (2 * eps_b[i] - 1) for i in nonzero_indices_Gb])

            obj = gp.quicksum(obj_terms)

            if len(obj_terms) == 1:

                lb[neuron_idx] = center_np[neuron_idx]
                ub[neuron_idx] = center_np[neuron_idx]
                solve_time = time.time() - solve_start

                continue

            model.setObjective(obj, GRB.MAXIMIZE)
            model.optimize()
            if model.status == GRB.OPTIMAL:
                ub[neuron_idx] = model.objVal
            elif model.status == GRB.TIME_LIMIT:
                ub[neuron_idx] = model.objVal if hasattr(model, 'objVal') else float('inf')
            else:
                ub[neuron_idx] = float('inf')
                print(f"⚠️ [Gurobi] Neuron {neuron_idx} upper bound solve failed: status={model.status}")

            model.setObjective(obj, GRB.MINIMIZE)
            model.optimize()
            if model.status == GRB.OPTIMAL:
                lb[neuron_idx] = model.objVal
            elif model.status == GRB.TIME_LIMIT:
                lb[neuron_idx] = model.objVal if hasattr(model, 'objVal') else -float('inf')
            else:
                lb[neuron_idx] = -float('inf')
                print(f"⚠️ [Gurobi] Neuron {neuron_idx} lower bound solve failed: status={model.status}")

            solve_time = time.time() - solve_start

            neuron_time = time.time() - neuron_start_time
            recent_times.append(neuron_time)
            if len(recent_times) > 3:
                recent_times.pop(0)
        total_opt_time = time.time() - opt_start_time
        print(f"✅ All neuron optimizations complete ({total_opt_time:.2f}s)")

        model.dispose()
        env.dispose()
        print(f"🧹 Gurobi environment cleaned up")

        lb_tensor = torch.tensor(lb, dtype=center_dim.dtype, device=center_dim.device)
        ub_tensor = torch.tensor(ub, dtype=center_dim.dtype, device=center_dim.device)

        if lb_tensor.numel() == 1:
            return lb_tensor.item(), ub_tensor.item()
        else:
            return lb_tensor, ub_tensor

    @staticmethod
    def GetLayerWiseBounds(flat_center, flat_G_c, flat_G_b, A_c_tensor, A_b_tensor, b_tensor, method='hybridz', time_limit=500, num_workers=4):
        print_memory_usage("GetLayerWiseBounds Start")
        N = flat_center.shape[0]
        print(f"🚀 Processing {N} neurons with method={method}")

        flat_center = flat_center.detach()
        flat_G_c = flat_G_c.detach()
        flat_G_b = flat_G_b.detach()
        A_c_tensor = A_c_tensor.detach() if A_c_tensor is not None else None
        A_b_tensor = A_b_tensor.detach() if A_b_tensor is not None else None
        b_tensor = b_tensor.detach() if b_tensor is not None else None

        print("ReLU pre-activation bounding: N =", N, ", method =", method)

        if method == 'interval':

            return HybridZonotopeOps.ComputeIntervalElemBounds(flat_center, flat_G_c, flat_G_b)

        elif method == 'hybridz_relaxed':
            print("🎭 Using mixed strategy (relaxed LP + exact MILP) bounds")

            ng = flat_G_c.shape[1] if flat_G_c.numel() > 0 else 0
            nb = flat_G_b.shape[1] if flat_G_b.numel() > 0 else 0
            nc = A_c_tensor.shape[0] if A_c_tensor is not None and A_c_tensor.numel() > 0 else 0

            print(f"🎭 Mixed strategy bounds: N={N}, ng={ng}, nb={nb}, nc={nc}")

            if nb == 0:
                if nc == 0:
                    print("🚀 Fully relaxed: No constraints, using Generic bounds")
                    return HybridZonotopeOps.ComputeGenericZElemBounds(flat_center, flat_G_c)
                else:
                    print(f"🚀 Fully relaxed: Using LP bounds with {nc} constraints")
                    constraint_complexity = nc * ng
                    gurobi_threshold = 5000

                    if constraint_complexity <= gurobi_threshold:
                        return HybridZonotopeOps.ComputeConstrainedZElemBoundsLinprog(flat_center, flat_G_c, A_c_tensor, b_tensor, time_limit)
                    else:
                        return HybridZonotopeOps.ComputeConstrainedZElemBoundsGurobi(flat_center, flat_G_c, A_c_tensor, b_tensor, time_limit)
            else:

                print(f"🎭 Mixed strategy: Using MILP bounds with {nb} binary variables")
                return HybridZonotopeOps.ComputeHybridZElemBoundsGurobi(flat_center, flat_G_c, flat_G_b, A_c_tensor, A_b_tensor, b_tensor, time_limit)

        elif method == 'hybridz_relaxed_with_bab':
            print("🌳 Using Hybrid Strategy: Gurobi bounds + SMT counterexamples")

            ng = flat_G_c.shape[1] if flat_G_c.numel() > 0 else 0
            nb = flat_G_b.shape[1] if flat_G_b.numel() > 0 else 0
            nc = A_c_tensor.shape[0] if A_c_tensor is not None and A_c_tensor.numel() > 0 else 0

            print(f"🌳 Hybrid bounds: N={N}, ng={ng}, nb={nb}, nc={nc}")

            if nb > 0:
                print("⚠️  WARNING: Binary variables detected in BaB mode! Converting to fully relaxed LP...")
                print(f"   Binary variables (nb={nb}) will be treated as continuous [-1, 1]")

            if nc == 0:
                print("🚀 No constraints detected! Using Generic bounds")
                return HybridZonotopeOps.ComputeGenericZElemBounds(flat_center, flat_G_c)
            else:
                print(f"📊 Phase 1: Using Gurobi for fast constrained bounds: {N} neurons, {nc} constraints")

                print("   📋 Note: SMT counterexample generation available on-demand for output layer")
                return HybridZonotopeOps.ComputeConstrainedZElemBoundsGurobi(
                    flat_center, flat_G_c, A_c_tensor, b_tensor, time_limit
                )

        elif method == 'hybridz':
            print("Here, in Grid pre-activation bounding method:", method)

            ng = flat_G_c.shape[1] if flat_G_c.numel() > 0 else 0
            nb = flat_G_b.shape[1] if flat_G_b.numel() > 0 else 0
            nc = A_c_tensor.shape[0] if A_c_tensor is not None and A_c_tensor.numel() > 0 else 0

            print(f"🚀 Ultra-vectorized bounds: N={N}, ng={ng}, nb={nb}, nc={nc}")

            if nb == 0 and nc == 0:
                print(f"🚀 No constraints detected! Using ultra-fast Generic bounds for {N} neurons")
                return HybridZonotopeOps.ComputeGenericZElemBounds(flat_center, flat_G_c)

            elif nb == 0:

                constraint_complexity = nc * ng
                gurobi_threshold = 5000

                if constraint_complexity <= gurobi_threshold:
                    print(f"Using linprog for LP bounds: {N} neurons, {nc} constraints (complexity: {constraint_complexity})")
                    return HybridZonotopeOps.ComputeConstrainedZElemBoundsLinprog(flat_center, flat_G_c, A_c_tensor, b_tensor, time_limit)
                else:
                    print(f"Using Gurobi for LP bounds: {N} neurons, {nc} constraints (complexity: {constraint_complexity})")
                    return HybridZonotopeOps.ComputeConstrainedZElemBoundsGurobi(flat_center, flat_G_c, A_c_tensor, b_tensor, time_limit)

            else:
                print(f"Using integrated MILP bounds computation for {N} neurons")
                return HybridZonotopeOps.ComputeHybridZElemBoundsGurobi(flat_center, flat_G_c, flat_G_b, A_c_tensor, A_b_tensor, b_tensor, time_limit)

        raise ValueError(f"Unsupported method: {method}. Supported methods: 'interval', 'hybridz', 'hybridz_relaxed', 'hybridz_relaxed_with_bab'")

    @staticmethod
    def BlockDiagonalCat(tensor_list):
        if not tensor_list:
            return torch.empty(0, 0)

        if len(tensor_list) == 1:
            return tensor_list[0]

        total_rows = sum(t.shape[0] for t in tensor_list)
        total_cols = sum(t.shape[1] for t in tensor_list)

        if total_cols == 0:
            return torch.zeros(total_rows, 0, dtype=tensor_list[0].dtype, device=tensor_list[0].device)

        if total_rows == 0:
            return torch.zeros(0, total_cols, dtype=tensor_list[0].dtype, device=tensor_list[0].device)

        result = torch.zeros(total_rows, total_cols, dtype=tensor_list[0].dtype, device=tensor_list[0].device)

        row_offset = 0
        col_offset = 0

        for tensor in tensor_list:
            rows, cols = tensor.shape

            if rows > 0 and cols > 0:
                row_end = row_offset + rows
                col_end = col_offset + cols
                result[row_offset:row_end, col_offset:col_end] = tensor

            row_offset += rows
            col_offset += cols

        return result

    @staticmethod
    def ReLUElem(lb, ub, dtype=torch.float32, device='cpu'):
        if lb >= 0:

            new_center = torch.tensor([[(lb+ub)/2], [(lb+ub)/2]], dtype=dtype, device=device)

            new_G_c = torch.tensor([[(ub-lb)/2],
                                    [(ub-lb)/2]], dtype=dtype, device=device)

            new_G_b = torch.zeros(2, 0, dtype=dtype, device=device)
            new_A_c = torch.zeros(0, 1, dtype=dtype, device=device)
            new_A_b = torch.zeros(0, 0, dtype=dtype, device=device)
            new_b = torch.zeros(0, 1, dtype=dtype, device=device)

        elif ub <= 0:

            new_center = torch.tensor([[(lb+ub)/2], [0]], dtype=dtype, device=device)

            new_G_c = torch.tensor([[(ub-lb)/2],
                                    [0]], dtype=dtype, device=device)

            new_G_b = torch.zeros(2, 0, dtype=dtype, device=device)
            new_A_c = torch.zeros(0, 1, dtype=dtype, device=device)
            new_A_b = torch.zeros(0, 0, dtype=dtype, device=device)
            new_b = torch.zeros(0, 1, dtype=dtype, device=device)

        else:

            new_center, new_G_c, new_G_b, new_A_c, new_A_b, new_b = HybridZonotopeOps.ReLUUnionHybridZ(lb, ub, dtype, device)

        return new_center, new_G_c, new_G_b, new_A_c, new_A_b, new_b

    @staticmethod
    def ReLUElemRelaxed(lb, ub, dtype=torch.float32, device='cpu'):

        if lb >= 0:

            new_center = torch.tensor([[(lb+ub)/2], [(lb+ub)/2]], dtype=dtype, device=device)

            new_G_c = torch.tensor([[(ub-lb)/2],
                                    [(ub-lb)/2]], dtype=dtype, device=device)

            new_G_b = torch.zeros(2, 0, dtype=dtype, device=device)
            new_A_c = torch.zeros(0, 1, dtype=dtype, device=device)
            new_A_b = torch.zeros(0, 0, dtype=dtype, device=device)
            new_b = torch.zeros(0, 1, dtype=dtype, device=device)

        elif ub <= 0:

            new_center = torch.tensor([[(lb+ub)/2], [0]], dtype=dtype, device=device)

            new_G_c = torch.tensor([[(ub-lb)/2],
                                    [0]], dtype=dtype, device=device)

            new_G_b = torch.zeros(2, 0, dtype=dtype, device=device)
            new_A_c = torch.zeros(0, 1, dtype=dtype, device=device)
            new_A_b = torch.zeros(0, 0, dtype=dtype, device=device)
            new_b = torch.zeros(0, 1, dtype=dtype, device=device)

        else:

            exact_center, exact_G_c, exact_G_b, exact_A_c, exact_A_b, exact_b = HybridZonotopeOps.ReLUUnionHybridZ(lb, ub, dtype, device)

            new_G_c = torch.cat([exact_G_c, exact_G_b], dim=1)
            new_G_b = torch.zeros(2, 0, dtype=dtype, device=device)

            new_A_c = torch.cat([exact_A_c, exact_A_b], dim=1)
            new_A_b = torch.zeros(exact_A_b.shape[0], 0, dtype=dtype, device=device)

            new_center = exact_center
            new_b = exact_b

        return new_center, new_G_c, new_G_b, new_A_c, new_A_b, new_b

    @staticmethod
    def SigmoidTanhElem(lb, ub, mid, func_type, dtype=torch.float32, device='cpu'):

        if lb < mid < 0 < ub:
            doubleTangentsHZ_1 = HybridZonotopeOps.DoubleTangentsHybridZ(func_type, lb, mid, dtype, device)
            singleTangentHZ_2 = HybridZonotopeOps.SingleTangentHybridZ(func_type, mid, ub, dtype, device)
            unioned_center, unioned_G_c, unioned_G_b, unioned_A_c, unioned_A_b, unioned_b = HybridZonotopeOps.SigmoidTanhUnionHybridZ(doubleTangentsHZ_1, singleTangentHZ_2)

        elif lb < 0 < mid < ub:
            singleTangentHZ_1 = HybridZonotopeOps.SingleTangentHybridZ(func_type, lb, mid, dtype, device)
            doubleTangentsHZ_2 = HybridZonotopeOps.DoubleTangentsHybridZ(func_type, mid, ub, dtype, device)
            unioned_center, unioned_G_c, unioned_G_b, unioned_A_c, unioned_A_b, unioned_b = HybridZonotopeOps.SigmoidTanhUnionHybridZ(singleTangentHZ_1, doubleTangentsHZ_2)

        else:
            doubleTangentsHZ_1 = HybridZonotopeOps.DoubleTangentsHybridZ(func_type, lb, mid, dtype, device)
            doubleTangentsHZ_2 = HybridZonotopeOps.DoubleTangentsHybridZ(func_type, mid, ub, dtype, device)
            unioned_center, unioned_G_c, unioned_G_b, unioned_A_c, unioned_A_b, unioned_b = HybridZonotopeOps.SigmoidTanhUnionHybridZ(doubleTangentsHZ_1, doubleTangentsHZ_2)

        return unioned_center, unioned_G_c, unioned_G_b, unioned_A_c, unioned_A_b, unioned_b

    @staticmethod
    def SigmoidTanhElemRelaxed(lb, ub, mid, func_type, dtype=torch.float32, device='cpu'):

        original_center, original_G_c, original_G_b, original_A_c, original_A_b, original_b = HybridZonotopeOps.SigmoidTanhElem(lb, ub, mid, func_type, dtype, device)

        if original_G_b.shape[1] == 0:
            return original_center, original_G_c, original_G_b, original_A_c, original_A_b, original_b

        new_G_c = torch.cat([original_G_c, original_G_b], dim=1)
        new_G_b = torch.zeros(original_G_b.shape[0], 0, dtype=dtype, device=device)

        new_A_c = torch.cat([original_A_c, original_A_b], dim=1)
        new_A_b = torch.zeros(original_A_b.shape[0], 0, dtype=dtype, device=device)

        new_center = original_center
        new_b = original_b

        return new_center, new_G_c, new_G_b, new_A_c, new_A_b, new_b

    @staticmethod
    def ReLUUnionHybridZ(lb, ub, dtype=torch.float32, device='cpu'):

        new_center = torch.tensor([[0],
                                   [0]], dtype=dtype, device=device)

        new_G_c = torch.tensor([[lb/2, 0, 0, ub/2],
                                [0, 0, 0, ub/2]],
                               dtype=dtype, device=device)

        new_G_b = torch.tensor([[(lb-ub)/2],
                                [-ub/2]], dtype=dtype, device=device)

        new_A_c = torch.tensor([[1,1,0,0],
                                [0,0,1,1]], dtype=dtype, device=device)

        new_A_b = torch.tensor([[1],
                                [-1]], dtype=dtype, device=device)

        new_b = torch.tensor([[1],
                              [1]], dtype=dtype, device=device)

        return new_center, new_G_c, new_G_b, new_A_c, new_A_b, new_b

    @staticmethod
    def DoubleTangentsHybridZ(activation : str, x1 : float, x2 : float, dtype=torch.float32, device='cpu'):

        if activation == 'tanh':
            f_x1 = torch.tanh(torch.tensor(x1))
            k_x1 = 1 - f_x1 ** 2
            f_x2 = torch.tanh(torch.tensor(x2))
            k_x2 = 1 - f_x2 ** 2
        elif activation == 'sigmoid':
            f_x1 = torch.sigmoid(torch.tensor(x1))
            k_x1 = f_x1 * (1 - f_x1)
            f_x2 = torch.sigmoid(torch.tensor(x2))
            k_x2 = f_x2 * (1 - f_x2)
        else:
            raise ValueError(f"Unsupported activation function: {activation}. Supported: 'tanh', 'sigmoid'.")

        grad_diff = abs(k_x2 - k_x1)
        numerical_eps = 1e-8

        if grad_diff < numerical_eps:

            return HybridZonotopeOps.SingleTangentHybridZ(activation, x1, x2, dtype, device)

        x1_center = (x1 + x2) / 2
        x2_center = (f_x1 + f_x2) / 2

        new_center = torch.tensor([[x1_center], [x2_center]], dtype=dtype, device=device)

        G_x11 = ((f_x2 - f_x1 + k_x2*x1 - k_x1*x2) / (k_x2 - k_x1) - x1) / 2
        G_x12 = (k_x2 * (f_x2 - f_x1 + k_x2*x1 - k_x1*x2) / (k_x2 - k_x1) - k_x2*x1) / 2

        G_x21 = ((f_x2 - f_x1 + k_x1*x1 - k_x2*x2) / (k_x1 - k_x2) - x1) / 2
        G_x22 = (k_x1 * (f_x2 - f_x1 + k_x1*x1 - k_x2*x2) / (k_x1 - k_x2) - k_x1*x1) / 2

        values_to_check = [G_x11, G_x12, G_x21, G_x22]
        for i, val in enumerate(values_to_check):
            if not torch.isfinite(torch.tensor(val)):

                return HybridZonotopeOps.SingleTangentHybridZ(activation, x1, x2, dtype, device)

        new_G_c = torch.tensor([[G_x11, G_x21], [G_x12, G_x22]], dtype=dtype, device=device)

        return HybridZonotopeElem(new_center, new_G_c, method='hybridz', dtype=dtype, device=device)

    @staticmethod
    def SingleTangentHybridZ(activation : str, x1 : float, x2 : float, dtype=torch.float32, device='cpu'):

        if activation == 'tanh':
            f_x1 = torch.tanh(torch.tensor(x1))
            k_x1 = 1 - f_x1 ** 2
            f_x2 = torch.tanh(torch.tensor(x2))
            k_x2 = 1 - f_x2 ** 2
        elif activation == 'sigmoid':
            f_x1 = torch.sigmoid(torch.tensor(x1))
            k_x1 = f_x1 * (1 - f_x1)
            f_x2 = torch.sigmoid(torch.tensor(x2))
            k_x2 = f_x2 * (1 - f_x2)
        else:
            raise ValueError(f"Unsupported activation function: {activation}. Supported: 'tanh', 'sigmoid'.")

        k_opt = min(k_x1, k_x2)

        x1_center = (x1 + x2) / 2
        x2_center = (f_x1 + f_x2) / 2
        new_center = torch.tensor([[x1_center], [x2_center]], dtype=dtype, device=device)

        G_x11 = 0
        G_x12 = (f_x2 - f_x1 - k_opt*x2 + k_opt*x1) / 2

        G_x21 = (x2 - x1) / 2
        G_x22 = (k_opt * x2 - k_opt * x1) / 2

        new_G_c = torch.tensor([[G_x11, G_x21], [G_x12, G_x22]], dtype=dtype, device=device)

        return HybridZonotopeElem(new_center, new_G_c, method='hybridz', dtype=dtype, device=device)

    @staticmethod
    def SigmoidTanhUnionHybridZ(hz1 : HybridZonotopeElem, hz2 : HybridZonotopeElem):

        Gc_x_11 = hz1.G_c[0, 0]
        Gc_x_12 = hz1.G_c[1, 0]

        Gc_x_21 = hz1.G_c[0, 1]
        Gc_x_22 = hz1.G_c[1, 1]

        Gc_y_11 = hz2.G_c[0, 0]
        Gc_y_12 = hz2.G_c[1, 0]

        Gc_y_21 = hz2.G_c[0, 1]
        Gc_y_22 = hz2.G_c[1, 1]

        c_x_1 = hz1.center[0, 0]
        c_x_2 = hz1.center[1, 0]

        c_y_1 = hz2.center[0, 0]
        c_y_2 = hz2.center[1, 0]

        new_G_c = torch.tensor([[-Gc_x_11, -Gc_x_21, 0, 0, -Gc_y_11, -Gc_y_21, 0, 0],
                                   [-Gc_x_12, -Gc_x_22, 0, 0, -Gc_y_12, -Gc_y_22, 0, 0]], dtype=hz1.dtype, device=hz1.device)

        new_G_b = torch.tensor([[(c_x_1-c_y_1-Gc_x_11+Gc_y_11-Gc_x_21+Gc_y_21)/2],
                                [(c_x_2-c_y_2-Gc_x_12+Gc_y_12-Gc_x_22+Gc_y_22)/2]], dtype=hz1.dtype, device=hz1.device)

        new_center = torch.tensor([[(c_x_1+c_y_1+Gc_x_11+Gc_y_11+Gc_x_21+Gc_y_21)/2],
                                   [(c_x_2+c_y_2+Gc_x_12+Gc_y_12+Gc_x_22+Gc_y_22)/2]], dtype=hz1.dtype, device=hz1.device)

        new_A_c = torch.tensor([[1,0,1,0,0,0,0,0],
                                [0,1,0,1,0,0,0,0],
                                [0,0,0,0,1,0,1,0],
                                [0,0,0,0,0,1,0,1]], dtype=hz1.dtype, device=hz1.device)

        new_A_b = torch.tensor([[1], [1], [-1], [-1]], dtype=hz1.dtype, device=hz1.device)

        new_b = torch.tensor([[1], [1], [1], [1]], dtype=hz1.dtype, device=hz1.device)

        return new_center, new_G_c, new_G_b, new_A_c, new_A_b, new_b

    @staticmethod
    def FlattenHybridZonotopeGridIntersection(hz: HybridZonotopeGrid):
        if hz.center_grid.ndim == 4:
            N = hz.center_grid.shape[0] * hz.center_grid.shape[1] * hz.center_grid.shape[2]
            flat_center = hz.center_grid.reshape(N, *hz.center_grid.shape[3:])
            flat_G_c = hz.G_c_grid.reshape(N, *hz.G_c_grid.shape[3:])

            if hz.nb == 0:
                flat_G_b = hz.G_b_grid.reshape(N, 0)
            else:
                flat_G_b = hz.G_b_grid.reshape(N, *hz.G_b_grid.shape[3:])
        elif hz.center_grid.ndim == 5:
            N = hz.center_grid.shape[0] * hz.center_grid.shape[1] * hz.center_grid.shape[2] * hz.center_grid.shape[3]
            flat_center = hz.center_grid.reshape(N, *hz.center_grid.shape[4:])
            flat_G_c = hz.G_c_grid.reshape(N, *hz.G_c_grid.shape[4:])

            if hz.nb == 0:
                flat_G_b = hz.G_b_grid.reshape(N, 0)
            else:
                flat_G_b = hz.G_b_grid.reshape(N, *hz.G_b_grid.shape[4:])
        else:
            raise ValueError(f"Unsupported center_grid shape: {hz.center_grid.shape}. Expected 4D or 5D tensor.")

        return HybridZonotopeElem(flat_center, flat_G_c, flat_G_b, hz.A_c_tensor, hz.A_b_tensor, hz.b_tensor,
                                  method=hz.method, time_limit=hz.time_limit, relaxation_ratio=hz.relaxation_ratio,
                                  dtype=hz.dtype, device=hz.device)

    @staticmethod
    def ReshapeHybridZonotopeGrid(hz: HybridZonotopeGrid, center, G_c, G_b):
        C, H, W = hz.C, hz.H, hz.W
        center_grid = center.reshape(C, H, W, *center.shape[1:])
        G_c_grid = G_c.reshape(C, H, W, *G_c.shape[1:])
        G_b_grid = G_b.reshape(C, H, W, *G_b.shape[1:])

        return center_grid, G_c_grid, G_b_grid

    @staticmethod
    def ExtractStableReLUOutput(center, G_c, G_b, dtype, device):

        total_neurons = center.shape[0] // 2
        dim = total_neurons

        perm_indices = list(range(0, 2*dim, 2)) + list(range(1, 2*dim, 2))
        P = torch.eye(2*dim, dtype=dtype, device=device)[perm_indices]

        permuted_center = P @ center
        permuted_G_c = P @ G_c
        permuted_G_b = P @ G_b

        output_filter = torch.cat([torch.zeros(dim, dim, dtype=dtype, device=device),
                                 torch.eye(dim, dtype=dtype, device=device)], dim=1)

        filtered_center = output_filter @ permuted_center
        filtered_G_c = output_filter @ permuted_G_c
        filtered_G_b = output_filter @ permuted_G_b

        return filtered_center, filtered_G_c, filtered_G_b

    @staticmethod
    def ActivationOutputIntersectionElem(abstract_transormer_hz : HybridZonotopeElem, input_hz : HybridZonotopeElem):

        dim = input_hz.n

        perm_indices = list(range(0, 2*dim, 2)) + list(range(1, 2*dim, 2))
        P = torch.eye(2*dim, dtype=input_hz.dtype, device=input_hz.device)[perm_indices]

        R = torch.cat([torch.eye(dim, dtype=input_hz.dtype, device=input_hz.device),
                       torch.zeros(dim, dim, dtype=input_hz.dtype, device=input_hz.device)], dim=1)

        Z_center, Y_center = P @ abstract_transormer_hz.center, input_hz.center
        Z_G_c, Y_G_c = P @ abstract_transormer_hz.G_c, input_hz.G_c
        Z_G_b, Y_G_b = P @ abstract_transormer_hz.G_b, input_hz.G_b
        Z_A_c, Y_A_c = abstract_transormer_hz.A_c, input_hz.A_c
        Z_A_b, Y_A_b = abstract_transormer_hz.A_b, input_hz.A_b
        Z_b, Y_b = abstract_transormer_hz.b, input_hz.b

        Z_n, Y_n = abstract_transormer_hz.n, input_hz.n
        Z_ng, Y_ng = abstract_transormer_hz.ng, input_hz.ng
        Z_nb, Y_nb = abstract_transormer_hz.nb, input_hz.nb
        Z_nc, Y_nc = abstract_transormer_hz.nc, input_hz.nc

        print(f"Z_n: {Z_n}, Y_n: {Y_n}, Z_ng: {Z_ng}, Y_ng: {Y_ng}, Z_nb: {Z_nb}, Y_nb: {Y_nb}, Z_nc: {Z_nc}, Y_nc: {Y_nc}")

        new_Gc = torch.cat([Z_G_c, torch.zeros(Z_n, Y_ng, device=Z_G_c.device)], dim=1)

        new_Gb = torch.cat([Z_G_b, torch.zeros(Z_n, Y_nb, device=Z_G_b.device)], dim=1)

        new_center = Z_center

        Ac_top = torch.cat([torch.cat([Z_A_c, torch.zeros(Z_nc, Y_ng, device=Z_A_c.device)], dim=1),
                            torch.cat([torch.zeros(Y_nc, Z_ng, device=Y_A_c.device), Y_A_c], dim=1)], dim=0)
        Ac_bottom = torch.cat([R @ Z_G_c, -Y_G_c], dim=1)
        new_Ac = torch.cat([Ac_top, Ac_bottom], dim=0)

        Ab_top = torch.cat([torch.cat([Z_A_b, torch.zeros(Z_nc, Y_nb, device=Z_A_b.device)], dim=1),
                            torch.cat([torch.zeros(Y_nc, Z_nb, device=Y_A_b.device), Y_A_b], dim=1)], dim=0)
        Ab_bottom = torch.cat([R @ Z_G_b, -Y_G_b], dim=1)
        new_Ab = torch.cat([Ab_top, Ab_bottom], dim=0)

        new_b = torch.cat([Z_b, Y_b, Y_center - R @ Z_center], dim=0)

        output_filter = torch.cat([torch.zeros(dim, dim, dtype=input_hz.dtype, device=input_hz.device),
                                   torch.eye(dim, dtype=input_hz.dtype, device=input_hz.device)], dim=1)

        new_center = output_filter @ new_center
        new_Gc = output_filter @ new_Gc
        new_Gb = output_filter @ new_Gb

        return HybridZonotopeElem(new_center, new_Gc, new_Gb, new_Ac, new_Ab, new_b,
                                  method=input_hz.method, time_limit=input_hz.time_limit,
                                  dtype=input_hz.dtype, device=input_hz.device)

    @staticmethod
    def ActivationOutputIntersectionElemMemoryOptimized(abstract_transormer_hz : HybridZonotopeElem, input_hz : HybridZonotopeElem):
        dim = input_hz.n

        print_memory_usage("Intersection start")

        print("🔧 Computing permutation indices...")
        perm_indices = list(range(0, 2*dim, 2)) + list(range(1, 2*dim, 2))
        print(f"🔧 Permutation indices computed, length: {len(perm_indices)}")

        chunk_size = min(512, dim // 4) if dim > 1024 else dim
        print(f"🔧 Using chunk_size={chunk_size} for intersection operations")

        Z_center = abstract_transormer_hz.center[perm_indices]

        Z_G_c = abstract_transormer_hz.G_c[perm_indices]

        Z_G_b = abstract_transormer_hz.G_b[perm_indices]

        Y_center, Y_G_c, Y_G_b = input_hz.center, input_hz.G_c, input_hz.G_b
        Z_A_c, Y_A_c = abstract_transormer_hz.A_c, input_hz.A_c
        Z_A_b, Y_A_b = abstract_transormer_hz.A_b, input_hz.A_b
        Z_b, Y_b = abstract_transormer_hz.b, input_hz.b

        Z_n, Y_n = abstract_transormer_hz.n, input_hz.n
        Z_ng, Y_ng = abstract_transormer_hz.ng, input_hz.ng
        Z_nb, Y_nb = abstract_transormer_hz.nb, input_hz.nb
        Z_nc, Y_nc = abstract_transormer_hz.nc, input_hz.nc

        print_memory_usage("After P matrix operations")

        if Y_ng > 0:

            new_Gc = torch.zeros(Z_n, Z_ng + Y_ng, device=Z_G_c.device, dtype=Z_G_c.dtype)
            new_Gc[:, :Z_ng] = Z_G_c

        else:
            new_Gc = Z_G_c

        if Y_nb > 0:

            new_Gb = torch.zeros(Z_n, Z_nb + Y_nb, device=Z_G_b.device, dtype=Z_G_b.dtype)
            new_Gb[:, :Z_nb] = Z_G_b

        else:
            new_Gb = Z_G_b

        new_center = Z_center

        del Z_G_c, Z_G_b
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        print_memory_usage("After Gc/Gb construction and cleanup")

        print("🔧 Step 4: Constructing Ac_bottom without storing intermediate R_Z_Gc...")

        temp_R_Z_Gc = abstract_transormer_hz.G_c[perm_indices][:dim]
        Ac_bottom = torch.cat([temp_R_Z_Gc, -Y_G_c], dim=1)
        del temp_R_Z_Gc
        print(f"✅ Step 4 completed, Ac_bottom shape: {Ac_bottom.shape}")

        print("🔧 Step 5: Constructing Ab_bottom without storing intermediate R_Z_Gb...")

        if abstract_transormer_hz.G_b.shape[1] > 0:
            temp_R_Z_Gb = abstract_transormer_hz.G_b[perm_indices][:dim]
            Ab_bottom = torch.cat([temp_R_Z_Gb, -Y_G_b], dim=1)
            del temp_R_Z_Gb
        else:
            Ab_bottom = torch.cat([torch.zeros(dim, 0, device=Y_G_b.device, dtype=Y_G_b.dtype), -Y_G_b], dim=1)
        print(f"✅ Step 5 completed, Ab_bottom shape: {Ab_bottom.shape}")

        print("🔧 Step 6: Constructing new_b without storing intermediate R_Z_center...")

        temp_R_Z_center = Z_center[:dim]
        new_b = torch.cat([Z_b, Y_b, Y_center - temp_R_Z_center], dim=0)
        del temp_R_Z_center
        print(f"✅ Step 6 completed, new_b shape: {new_b.shape}")

        print_memory_usage("After optimized R matrix operations")

        print("🔧 Step 7: Constructing constraint matrices with minimal memory footprint...")

        total_nc = Z_nc + Y_nc
        total_ng = Z_ng + Y_ng
        total_nb = Z_nb + Y_nb
        total_rows_Ac = total_nc + dim
        total_rows_Ab = total_nc + dim

        new_Ac = torch.zeros(total_rows_Ac, total_ng, device=new_Gc.device, dtype=new_Gc.dtype)
        new_Ab = torch.zeros(total_rows_Ab, total_nb, device=new_Gb.device, dtype=new_Gb.dtype)

        row_idx = 0

        if Z_nc > 0:
            new_Ac[row_idx:row_idx+Z_nc, :Z_ng] = Z_A_c

            new_Ab[row_idx:row_idx+Z_nc, :Z_nb] = Z_A_b

            row_idx += Z_nc

        if Y_nc > 0:

            new_Ac[row_idx:row_idx+Y_nc, Z_ng:] = Y_A_c

            new_Ab[row_idx:row_idx+Y_nc, Z_nb:] = Y_A_b
            row_idx += Y_nc

        new_Ac[row_idx:, :] = Ac_bottom
        new_Ab[row_idx:, :] = Ab_bottom

        del Ac_bottom, Ab_bottom
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        print_memory_usage("After constraint matrix construction")

        print("🔧 Step 8: Applying output filter (taking second half)...")

        new_center = new_center[dim:]
        new_Gc = new_Gc[dim:]
        new_Gb = new_Gb[dim:]
        print(f"✅ Step 8 completed, final shapes: center{new_center.shape}, Gc{new_Gc.shape}, Gb{new_Gb.shape}")

        del Z_center, Y_center, Y_G_c, Y_G_b, Z_b, Y_b
        del Z_A_c, Z_A_b, Y_A_c, Y_A_b
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        print_memory_usage("Intersection completed")

        return HybridZonotopeElem(new_center, new_Gc, new_Gb, new_Ac, new_Ab, new_b,
                                  method=input_hz.method, time_limit=input_hz.time_limit,
                                  dtype=input_hz.dtype, device=input_hz.device)

    @staticmethod
    def ActivationOutputIntersectionGrid(abstract_transormer_hz_grid : HybridZonotopeGrid, input_hz_grid : HybridZonotopeGrid):
        input_C, input_H, input_W = input_hz_grid.C, input_hz_grid.H, input_hz_grid.W

        abstract_transormer_hz = HybridZonotopeOps.FlattenHybridZonotopeGridIntersection(abstract_transormer_hz_grid)
        input_hz = HybridZonotopeOps.FlattenHybridZonotopeGridIntersection(input_hz_grid)

        estimated_memory_gb, use_memory_optimized, debug_info = HybridZonotopeOps.MemoryUsageEstimationIntersection(
            abstract_transormer_hz, input_hz, memory_threshold_gb=1.0
        )
        HybridZonotopeOps.PrintMemoryEstimation(estimated_memory_gb, use_memory_optimized, debug_info, memory_threshold_gb=1.0)

        if use_memory_optimized:
            new_hz_elem = HybridZonotopeOps.ActivationOutputIntersectionElemMemoryOptimized(abstract_transormer_hz, input_hz)
        else:
            new_hz_elem = HybridZonotopeOps.ActivationOutputIntersectionElem(abstract_transormer_hz, input_hz)
        new_center, new_Gc, new_Gb, new_Ac, new_Ab, new_b = new_hz_elem.center, new_hz_elem.G_c, new_hz_elem.G_b, \
                                                                new_hz_elem.A_c, new_hz_elem.A_b, new_hz_elem.b

        N = input_C * input_H * input_W
        assert new_center.shape[0] == N, f"Expected {N} elements, got {new_center.shape[0]}"

        center_grid, G_c_grid, G_b_grid = HybridZonotopeOps.ReshapeHybridZonotopeGrid(
            input_hz_grid, new_center, new_Gc, new_Gb
        )

        return HybridZonotopeGrid(center_grid, G_c_grid, G_b_grid, new_Ac, new_Ab, new_b,
                                  method=input_hz.method, time_limit=input_hz.time_limit, relaxation_ratio=input_hz.relaxation_ratio,
                                  dtype=input_hz.dtype, device=input_hz.device)

    @staticmethod
    def ReLUAutoLiRPAPreRun(stable_pos, stable_neg, unstable, hz_verifier, layer_name,
                           n_neurons, dtype, device):

        lirpa_lb, lirpa_ub = hz_verifier._get_activation_bounds(layer_name)
        lirpa_lb_flat = lirpa_lb.view(-1)
        lirpa_ub_flat = lirpa_ub.view(-1)

        if len(lirpa_lb_flat) != n_neurons:
            print(f"⚠️  Dimension mismatch: HZ={n_neurons}, auto_LiRPA={len(lirpa_lb_flat)}, falling back")
            return None

        print(f"🎯 ReLU optimization: {len(stable_pos)} stable+, {len(stable_neg)} stable-, {len(unstable)} unstable")

        new_center_list, new_G_c_list, new_G_b_list, new_A_c_list, new_A_b_list, new_b_list = [], [], [], [], [], []

        for i in range(n_neurons):
            lb_i, ub_i = lirpa_lb_flat[i], lirpa_ub_flat[i]

            if i in stable_pos:

                lb_i_optimized = max(lb_i, 1e-6)
                ub_i_optimized = max(ub_i, lb_i_optimized + 1e-6)
                print(f"  Neuron {i}: stable+ [{lb_i:.6f}, {ub_i:.6f}] -> [{lb_i_optimized:.6f}, {ub_i_optimized:.6f}]")
            elif i in stable_neg:

                ub_i_optimized = min(ub_i, -1e-6)
                lb_i_optimized = min(lb_i, ub_i_optimized - 1e-6)
                print(f"  Neuron {i}: stable- [{lb_i:.6f}, {ub_i:.6f}] -> [{lb_i_optimized:.6f}, {ub_i_optimized:.6f}]")
            else:

                lb_i_optimized, ub_i_optimized = lb_i, ub_i
                print(f"  Neuron {i}: unstable [{lb_i:.6f}, {ub_i:.6f}] (needs binary var)")

            new_center_i, new_G_c_i, new_G_b_i, new_A_c_i, new_A_b_i, new_b_i = HybridZonotopeOps.ReLUElem(
                lb_i_optimized, ub_i_optimized, dtype=dtype, device=device
            )

            new_center_list.append(new_center_i)
            new_G_c_list.append(new_G_c_i)
            new_G_b_list.append(new_G_b_i)
            new_A_c_list.append(new_A_c_i)
            new_A_b_list.append(new_A_b_i)
            new_b_list.append(new_b_i)

        print(f"✅ Optimized ReLU: reduced constraints from {n_neurons} potential to {len(unstable)} actual binary variables")

        return new_center_list, new_G_c_list, new_G_b_list, new_A_c_list, new_A_b_list, new_b_list

    @staticmethod
    def ReLUStandard(pre_activation_bounds, dtype, device, method='hybridz', relaxation_ratio=1.0):

        lb, ub = pre_activation_bounds
        n_neurons = len(lb)

        if method == 'hybridz_relaxed':

            n_relaxed = int(n_neurons * relaxation_ratio)
            n_exact = n_neurons - n_relaxed

            if relaxation_ratio == 1.0:
                print(f"🚀 Using fully relaxed ReLU (LP instead of MILP) for all {n_neurons} neurons")
            elif relaxation_ratio == 0.0:
                print(f"🎯 Using fully exact ReLU (MILP) for all {n_neurons} neurons")
            else:
                print(f"🎭 Using mixed ReLU strategy: {n_relaxed} relaxed (LP) + {n_exact} exact (MILP) out of {n_neurons} neurons (ratio={relaxation_ratio:.1f})")
        elif method == 'hybridz_relaxed_with_bab':

            print(f"🌳 BaB mode: Using fully relaxed ReLU (LP) for all {n_neurons} neurons")
        else:
            print(f"Using standard ReLU (MILP) for {n_neurons} neurons")

        new_center_list, new_G_c_list, new_G_b_list, new_A_c_list, new_A_b_list, new_b_list = [], [], [], [], [], []

        for i in range(n_neurons):
            lb_i, ub_i = lb[i], ub[i]

            use_relaxed = False
            if method == 'hybridz_relaxed':
                if relaxation_ratio >= 1.0:
                    use_relaxed = True
                elif relaxation_ratio <= 0.0:
                    use_relaxed = False
                else:

                    use_relaxed = (i % 100) < (relaxation_ratio * 100)
            elif method == 'hybridz_relaxed_with_bab':

                use_relaxed = True

            if use_relaxed:
                new_center_i, new_G_c_i, new_G_b_i, new_A_c_i, new_A_b_i, new_b_i = HybridZonotopeOps.ReLUElemRelaxed(
                    lb_i, ub_i, dtype=dtype, device=device
                )
            else:
                new_center_i, new_G_c_i, new_G_b_i, new_A_c_i, new_A_b_i, new_b_i = HybridZonotopeOps.ReLUElem(
                    lb_i, ub_i, dtype=dtype, device=device
                )

            new_center_list.append(new_center_i)
            new_G_c_list.append(new_G_c_i)
            new_G_b_list.append(new_G_b_i)
            new_A_c_list.append(new_A_c_i)
            new_A_b_list.append(new_A_b_i)
            new_b_list.append(new_b_i)

        return new_center_list, new_G_c_list, new_G_b_list, new_A_c_list, new_A_b_list, new_b_list

    @staticmethod
    def SigmoidTanhStandard(pre_activation_bounds, mid_values, func_type, dtype, device, method='hybridz', relaxation_ratio=1.0):

        lb, ub = pre_activation_bounds
        n_neurons = len(lb)

        if method == 'hybridz_relaxed':

            n_relaxed = int(n_neurons * relaxation_ratio)
            n_exact = n_neurons - n_relaxed

            if relaxation_ratio == 1.0:
                print(f"🚀 Using fully relaxed {func_type.upper()} (LP instead of MILP) for all {n_neurons} neurons")
            elif relaxation_ratio == 0.0:
                print(f"🎯 Using fully exact {func_type.upper()} (MILP) for all {n_neurons} neurons")
            else:
                print(f"🎭 Using mixed {func_type.upper()} strategy: {n_relaxed} relaxed (LP) + {n_exact} exact (MILP) out of {n_neurons} neurons (ratio={relaxation_ratio:.1f})")
        else:
            print(f"Using standard {func_type.upper()} (MILP) for {n_neurons} neurons")

        new_center_list, new_G_c_list, new_G_b_list, new_A_c_list, new_A_b_list, new_b_list = [], [], [], [], [], []

        for i in range(n_neurons):
            lb_i, ub_i, mid_i = lb[i], ub[i], mid_values[i]

            use_relaxed = False
            if method == 'hybridz_relaxed':
                if relaxation_ratio >= 1.0:
                    use_relaxed = True
                elif relaxation_ratio <= 0.0:
                    use_relaxed = False
                else:

                    use_relaxed = (i % 100) < (relaxation_ratio * 100)

            if use_relaxed:
                new_center_i, new_G_c_i, new_G_b_i, new_A_c_i, new_A_b_i, new_b_i = HybridZonotopeOps.SigmoidTanhElemRelaxed(
                    lb_i, ub_i, mid_i, func_type, dtype=dtype, device=device
                )
            else:
                new_center_i, new_G_c_i, new_G_b_i, new_A_c_i, new_A_b_i, new_b_i = HybridZonotopeOps.SigmoidTanhElem(
                    lb_i, ub_i, mid_i, func_type, dtype=dtype, device=device
                )

            new_center_list.append(new_center_i)
            new_G_c_list.append(new_G_c_i)
            new_G_b_list.append(new_G_b_i)
            new_A_c_list.append(new_A_c_i)
            new_A_b_list.append(new_A_b_i)
            new_b_list.append(new_b_i)

        return new_center_list, new_G_c_list, new_G_b_list, new_A_c_list, new_A_b_list, new_b_list

    @staticmethod
    def MergeParallelGenerators(center, G_c, G_b, A_c, A_b, b, cosine_threshold=0.95, dtype=torch.float32, device='cpu'):

        if G_c is None or G_c.numel() == 0:
            print("🔍 No continuous generators to merge")
            return center, G_c, G_b, A_c, A_b, b

        n, ng = G_c.shape
        nc = A_c.shape[0] if A_c is not None else 0

        if ng <= 1:
            print("🔍 Only one continuous generator, no merging needed")
            return center, G_c, G_b, A_c, A_b, b

        print(f"🔍 Analyzing {ng} continuous generators for parallel merging (threshold={cosine_threshold:.3f})")

        if A_c is not None and A_c.numel() > 0:

            combined_matrix = torch.cat([G_c, A_c], dim=0)
            print(f"📐 Combined matrix shape: {combined_matrix.shape} (Gc: {G_c.shape}, Ac: {A_c.shape})")
        else:

            combined_matrix = G_c
            print(f"📐 Analyzing Gc only: {G_c.shape} (no constraints)")

        def cosine_similarity(vec1, vec2):
            dot_product = torch.dot(vec1.flatten(), vec2.flatten())
            norm1 = torch.norm(vec1)
            norm2 = torch.norm(vec2)
            if norm1 < 1e-12 or norm2 < 1e-12:
                return 0.0
            return dot_product / (norm1 * norm2)

        merge_pairs = []
        merged_indices = set()

        for i in range(ng):
            if i in merged_indices:
                continue

            for j in range(i + 1, ng):
                if j in merged_indices:
                    continue

                cos_sim = cosine_similarity(combined_matrix[:, i], combined_matrix[:, j])

                if abs(cos_sim) > cosine_threshold:
                    merge_pairs.append((i, j, cos_sim.item()))
                    merged_indices.add(j)
                    print(f"  📋 Found parallel generators: col {i} ↔ col {j} (cosine={cos_sim:.4f})")

        if not merge_pairs:
            print("✅ No parallel generators found, returning original matrices")
            return center, G_c, G_b, A_c, A_b, b

        print(f"🎯 Found {len(merge_pairs)} parallel generator pairs to merge")

        remaining_indices = []
        for i in range(ng):
            if i not in merged_indices:
                remaining_indices.append(i)

        new_ng = len(remaining_indices)
        G_c_merged = torch.zeros(n, new_ng, dtype=dtype, device=device)
        A_c_merged = torch.zeros(nc, new_ng, dtype=dtype, device=device) if A_c is not None else None

        col_idx = 0
        merge_dict = {}

        for orig_idx in remaining_indices:
            G_c_merged[:, col_idx] = G_c[:, orig_idx]
            if A_c_merged is not None:
                A_c_merged[:, col_idx] = A_c[:, orig_idx]
            merge_dict[orig_idx] = col_idx
            col_idx += 1

        for i, j, cos_sim in merge_pairs:
            if i in merge_dict:
                target_col = merge_dict[i]

                G_c_merged[:, target_col] += G_c[:, j]
                if A_c_merged is not None:
                    A_c_merged[:, target_col] += A_c[:, j]
                print(f"  ✅ Merged col {j} into col {i} (cosine={cos_sim:.4f}) -> new_col {target_col}")

        print(f"🎉 Generator merging completed: {ng} -> {new_ng} generators (reduction: {ng-new_ng})")
        print(f"   📊 Compression ratio: {(ng-new_ng)/ng*100:.1f}% reduction")

        return center, G_c_merged, G_b, A_c_merged, A_b, b

    @staticmethod
    def GetCounterexampleForOutputLayer(flat_center, flat_G_c, A_c_tensor, b_tensor, property_constraint, time_limit=30):

        try:
            print(f"🔧 [SMT] Counterexample generation for output layer starting...")
            print(f"   Output layer shape: center{flat_center.shape}, G_c{flat_G_c.shape}")
            print(f"   Constraint shape: A_c{A_c_tensor.shape}, b{b_tensor.shape}")
            print(f"   Property constraint: {property_constraint}")

            result = HybridZonotopeOps.ComputeConstrainedZElemBoundsSMT(
                flat_center, flat_G_c, A_c_tensor, b_tensor,
                time_limit=time_limit,
                property_constraint=property_constraint
            )

            if len(result) == 3:
                lb, ub, status = result
                if status == "UNSAFE":
                    print("🚨 [SMT] Safety property violated – unsafe configuration found")
                    return "VIOLATION_FOUND"
                elif status == "SAFE":
                    print("✅ [SMT] Safety property verified – constraint unsatisfiable")
                    return "SAFE_VERIFIED"
                else:
                    print("⚠️  [SMT] Solving result unknown")
                    return "UNKNOWN_RESULT"
            else:
                return None

        except Exception as e:
            print(f"   ❌ [SMT] Counterexample generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None

