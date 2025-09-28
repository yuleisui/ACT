#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#########################################################################
##   Abstract Constraint Transformer (ACT) - HybridZonotope Operations ##
##                                                                     ##
##   doctormeeee (https://github.com/doctormeeee) and contributors     ##
##   Copyright (C) 2024-2025                                           ##
##                                                                     ##
#########################################################################

import torch
import torch.nn.functional as F
import os
import sys
import psutil
import time
import numpy as np
from scipy.optimize import linprog

try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    print("Warning: Gurobi not available. HybridZonotopeOps will use alternative solvers.")
    GUROBI_AVAILABLE = False

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
            print("Z3 not installed, falling back to Gurobi solver")

            if property_constraint is not None:
                print("[SMT] Safety property verification requires Z3, but Z3 is not installed")
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

                print("[SMT] Safety property violation found - unsafe configuration exists")
                return None, None, "UNSAFE"
            elif violation_result is False:

                print("SMT] Safety property verified - constraint system is unsatisfiable")
                return None, None, "SAFE"
            else:

                print("[SMT] Solver result unknown - possible timeout or solver issue")
                return None, None, "UNKNOWN"

        print(f"[SMT] Entering boundary computation mode (return_counterexample={return_counterexample})")

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

        print(f"[SMT] Computing bounds for {N} neurons...")

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
            print("Z3 not installed, cannot perform SMT counterexample generation")
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

        print(f"[SMT] Using unified SMT framework to solve property violation...")
        print(f"Parameters: n_outputs={n_outputs}, ng={ng}, nc={nc}")

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
                        print(f"[SMT] Constraint {i} unsatisfiable: 0 == {b_np[i]}")
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
                print(f"[SMT] Classification violation: finding max(others) > output[{true_label}]")

                other_constraints = []
                for k in range(n_outputs):
                    if k != true_label:
                        other_constraints.append(output_exprs[k] > output_exprs[true_label])

                if other_constraints:
                    solver.add(z3.Or(other_constraints))
                else:
                    print(f"[SMT] Only one output class, cannot construct violation constraint")
                    return None
            else:
                print(f"[SMT] Cannot parse true label from: {property_constraint}")
                return None

        elif "linear_constraints_violated" in property_constraint:

            print(f"[SMT] Linear constraint violation: specific violation condition not yet implemented")

            return None

        else:
            print(f"[SMT] Unrecognized property constraint type: {property_constraint}")
            return None

        print(f"🔍 [SMT] startingsolving...")
        check_result = solver.check()

        if check_result == z3.sat:
            print(f"[SMT] Found solution satisfying violation constraint")
            model = solver.model()

            eps_c_values = []
            for i, eps_var in enumerate(eps_c_vars):
                val = model[eps_var]
                if val is not None:
                    eps_c_values.append(float(val.as_decimal(6)))
                else:
                    eps_c_values.append(0.0)

            print(f"[SMT] Found eps_c solution satisfying violation condition: {eps_c_values[:5]}..." if len(eps_c_values) > 5 else f"   [SMT] eps_c solution: {eps_c_values}")

            print(f"[SMT] Satisfiability check passed: solution violating safety property exists")
            return True

        elif check_result == z3.unsat:
            print(f"[SMT] constraint unsatisfiable, safety property verified")
            return False
        else:
            print(f"[SMT] solving timed out or unknown result: {check_result}")
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
        print(f"MILP problem size analysis:")
        print(f"- Neuron count N: {N}")
        print(f"- Continuous variables ng: {ng}")
        print(f"- Binary variables nb: {nb}")
        print(f"- Constraint count nc: {nc}")
        print(f"- Total constraint terms: {total_constraint_terms:,}")
        print(f"- Binary complexity: 2^{nb} = {2**min(nb, 20):,}...")

        if nc > 0:
            Ac_density = np.count_nonzero(Ac_np) / (nc * ng) if ng > 0 else 0
            Ab_density = np.count_nonzero(Ab_np) / (nc * nb) if nb > 0 else 0
            b_range = (np.min(b_np), np.max(b_np)) if nc > 0 else (0, 0)
            print(f"🔍 Constraint matrix analysis:")
            print(f"- Ac nonzero density: {Ac_density:.4f} ({np.count_nonzero(Ac_np)}/{nc*ng})")
            print(f"- Ab nonzero density: {Ab_density:.4f} ({np.count_nonzero(Ab_np)}/{nc*nb})")
            print(f"- b value range: [{b_range[0]:.6f}, {b_range[1]:.6f}]")
            print(f"- Ac coefficient range: [{np.min(Ac_np):.6f}, {np.max(Ac_np):.6f}]")
            print(f"- Ab coefficient range: [{np.min(Ab_np):.6f}, {np.max(Ab_np):.6f}]")

        print(f"Building Gurobi model...")
        print(f"Model construction start time: {time.time()}")
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
            print(f"Gurobi new environment + model created successfully")
        except Exception as e:
            print(f"Gurobi model creation failed: {e}")
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
            print(f"Output layer ultra large-scale binary problem (nb={nb}, N={N})")
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
            print(f"Large-scale binary problem (nb={nb})")
            model.setParam('TimeLimit', time_limit * N)
            model.setParam('Threads', 0)

            model.setParam('MIPFocus', 1)
            model.setParam('Presolve', 2)
            model.setParam('Cuts', 2)
            model.setParam('Heuristics', 0.1)

        else:
            print(f"Small/medium-scale binary problem (nb={nb}) ")
            model.setParam('TimeLimit', time_limit * N)
            model.setParam('Threads', 0)
            model.setParam('Presolve', 2)
            model.setParam('MIPFocus', 1)

        print(f"Adding variables: {ng} continuous + {nb} binary...")
        var_start_time = time.time()
        eps_c = model.addVars(ng, lb=-1, ub=1, vtype=GRB.CONTINUOUS, name="eps_c")
        eps_b = model.addVars(nb, vtype=GRB.BINARY, name="eps_b")
        var_time = time.time() - var_start_time
        print(f"Variable addition complete ({var_time:.2f}s)")

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

        print(f"Updating model...")
        update_start_time = time.time()
        model.update()
        update_time = time.time() - update_start_time
        print(f"Model update complete ({update_time:.2f}s)")
        model_build_time = time.time() - model_start_time
        print(f"Sparse model build total time: {model_build_time:.2f}s")

        lb = np.zeros(N)
        ub = np.zeros(N)

        print(f"Starting optimization for {N} neurons...")
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
                print(f"[Gurobi] Neuron {neuron_idx} upper bound solve failed: status={model.status}")

            model.setObjective(obj, GRB.MINIMIZE)
            model.optimize()
            if model.status == GRB.OPTIMAL:
                lb[neuron_idx] = model.objVal
            elif model.status == GRB.TIME_LIMIT:
                lb[neuron_idx] = model.objVal if hasattr(model, 'objVal') else -float('inf')
            else:
                lb[neuron_idx] = -float('inf')
                print(f"[Gurobi] Neuron {neuron_idx} lower bound solve failed: status={model.status}")

            solve_time = time.time() - solve_start

            neuron_time = time.time() - neuron_start_time
            recent_times.append(neuron_time)
            if len(recent_times) > 3:
                recent_times.pop(0)
        total_opt_time = time.time() - opt_start_time
        print(f"All neuron optimizations complete ({total_opt_time:.2f}s)")

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
    def GetLayerWiseBounds(flat_center, flat_G_c, flat_G_b, A_c_tensor, A_b_tensor, b_tensor, method='hybridz', time_limit=500, num_workers=4, ci_mode=False):
        print_memory_usage("GetLayerWiseBounds Start")
        N = flat_center.shape[0]
        print(f"Processing {N} neurons with method={method}")

        flat_center = flat_center.detach()
        flat_G_c = flat_G_c.detach()
        flat_G_b = flat_G_b.detach()
        A_c_tensor = A_c_tensor.detach() if A_c_tensor is not None else None
        A_b_tensor = A_b_tensor.detach() if A_b_tensor is not None else None
        b_tensor = b_tensor.detach() if b_tensor is not None else None

        print("ReLU pre-activation bounding: N =", N, ", method =", method)

        if method == 'interval':
            print("Using Interval Arithmetic (no external solver required)")
            return HybridZonotopeOps.ComputeIntervalElemBounds(flat_center, flat_G_c, flat_G_b)

        elif method == 'hybridz_relaxed':
            print("Using mixed strategy (relaxed LP + exact MILP) bounds")

            ng = flat_G_c.shape[1] if flat_G_c.numel() > 0 else 0
            nb = flat_G_b.shape[1] if flat_G_b.numel() > 0 else 0
            nc = A_c_tensor.shape[0] if A_c_tensor is not None and A_c_tensor.numel() > 0 else 0

            print(f"Mixed strategy bounds: N={N}, ng={ng}, nb={nb}, nc={nc}")

            if nb == 0:
                if nc == 0:
                    print("Fully relaxed: No constraints, using Generic bounds")
                    print("Using Generic Zonotope Bounds (no external solver required)")
                    return HybridZonotopeOps.ComputeGenericZElemBounds(flat_center, flat_G_c)
                else:
                    print(f"Fully relaxed: Using LP bounds with {nc} constraints")
                    constraint_complexity = nc * ng
                    gurobi_threshold = 5000

                    if ci_mode:
                        print(f"CI mode: Forcing linprog for LP bounds with {nc} constraints")
                        print("Using scipy.linprog (CI mode - no commercial license required)")
                        return HybridZonotopeOps.ComputeConstrainedZElemBoundsLinprog(flat_center, flat_G_c, A_c_tensor, b_tensor, time_limit)
                    elif constraint_complexity <= gurobi_threshold:
                        print("Using scipy.linprog (low complexity LP)")
                        return HybridZonotopeOps.ComputeConstrainedZElemBoundsLinprog(flat_center, flat_G_c, A_c_tensor, b_tensor, time_limit)
                    else:
                        print("Using Gurobi (high complexity LP - commercial solver)")
                        return HybridZonotopeOps.ComputeConstrainedZElemBoundsGurobi(flat_center, flat_G_c, A_c_tensor, b_tensor, time_limit)
            else:

                print(f"Mixed strategy: Using MILP bounds with {nb} binary variables")
                print("Using Gurobi (MILP with binary variables - commercial solver)")
                return HybridZonotopeOps.ComputeHybridZElemBoundsGurobi(flat_center, flat_G_c, flat_G_b, A_c_tensor, A_b_tensor, b_tensor, time_limit)

        elif method == 'hybridz_relaxed_with_bab':
            print("Using Hybrid Strategy: Gurobi bounds + SMT counterexamples")

            ng = flat_G_c.shape[1] if flat_G_c.numel() > 0 else 0
            nb = flat_G_b.shape[1] if flat_G_b.numel() > 0 else 0
            nc = A_c_tensor.shape[0] if A_c_tensor is not None and A_c_tensor.numel() > 0 else 0

            print(f"Hybrid bounds: N={N}, ng={ng}, nb={nb}, nc={nc}")

            if nb > 0:
                print("WARNING: Binary variables detected in BaB mode! Converting to fully relaxed LP...")
                print(f"Binary variables (nb={nb}) will be treated as continuous [-1, 1]")

            if nc == 0:
                print("No constraints detected! Using Generic bounds")
                print("Using Generic Zonotope Bounds (no external solver required)")
                return HybridZonotopeOps.ComputeGenericZElemBounds(flat_center, flat_G_c)
            else:
                print(f"Phase 1: Using Gurobi for fast constrained bounds: {N} neurons, {nc} constraints")
                print("Using Gurobi (BaB phase 1 - commercial solver)")
                print("  📋 Note: SMT counterexample generation available on-demand for output layer")
                return HybridZonotopeOps.ComputeConstrainedZElemBoundsGurobi(
                    flat_center, flat_G_c, A_c_tensor, b_tensor, time_limit
                )

        elif method == 'hybridz':
            print("Here, in Grid pre-activation bounding method:", method)

            ng = flat_G_c.shape[1] if flat_G_c.numel() > 0 else 0
            nb = flat_G_b.shape[1] if flat_G_b.numel() > 0 else 0
            nc = A_c_tensor.shape[0] if A_c_tensor is not None and A_c_tensor.numel() > 0 else 0

            print(f"Ultra-vectorized bounds: N={N}, ng={ng}, nb={nb}, nc={nc}")

            if nb == 0 and nc == 0:
                print(f"No constraints detected! Using ultra-fast Generic bounds for {N} neurons")
                print("Using Generic Zonotope Bounds (no external solver required)")
                return HybridZonotopeOps.ComputeGenericZElemBounds(flat_center, flat_G_c)

            elif nb == 0:

                constraint_complexity = nc * ng
                gurobi_threshold = 5000

                if ci_mode:
                    print(f"CI mode: Forcing linprog for LP bounds: {N} neurons, {nc} constraints")
                    print("Using scipy.linprog (CI mode - no commercial license required)")
                    return HybridZonotopeOps.ComputeConstrainedZElemBoundsLinprog(flat_center, flat_G_c, A_c_tensor, b_tensor, time_limit)
                elif constraint_complexity <= gurobi_threshold:
                    print(f"Using linprog for LP bounds: {N} neurons, {nc} constraints (complexity: {constraint_complexity})")
                    print("Using scipy.linprog (low complexity LP)")
                    return HybridZonotopeOps.ComputeConstrainedZElemBoundsLinprog(flat_center, flat_G_c, A_c_tensor, b_tensor, time_limit)
                else:
                    print(f"Using Gurobi for LP bounds: {N} neurons, {nc} constraints (complexity: {constraint_complexity})")
                    print("Using Gurobi (high complexity LP - commercial solver)")
                    return HybridZonotopeOps.ComputeConstrainedZElemBoundsGurobi(flat_center, flat_G_c, A_c_tensor, b_tensor, time_limit)

            else:
                print(f"Using integrated MILP bounds computation for {N} neurons")
                print("Using Gurobi (MILP with binary variables - commercial solver)")
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
        
        from hybridz_transformers import HybridZonotopeElem

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

        
        from hybridz_transformers import HybridZonotopeElem
        return HybridZonotopeElem(new_center, new_G_c, method='hybridz', dtype=dtype, device=device)

    @staticmethod
    def SingleTangentHybridZ(activation : str, x1 : float, x2 : float, dtype=torch.float32, device='cpu'):
        
        from hybridz_transformers import HybridZonotopeElem

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

          
        from hybridz_transformers import HybridZonotopeElem
        return HybridZonotopeElem(new_center, new_G_c, method='hybridz', dtype=dtype, device=device)

    @staticmethod
    def SigmoidTanhUnionHybridZ(hz1 : "HybridZonotopeElem", hz2 : "HybridZonotopeElem"):

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
    def FlattenHybridZonotopeGridIntersection(hz: "HybridZonotopeGrid"):
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

        from hybridz_transformers import HybridZonotopeElem
        return HybridZonotopeElem(flat_center, flat_G_c, flat_G_b, hz.A_c_tensor, hz.A_b_tensor, hz.b_tensor,
                                  method=hz.method, time_limit=hz.time_limit, relaxation_ratio=hz.relaxation_ratio,
                                  dtype=hz.dtype, device=hz.device)

    @staticmethod
    def ReshapeHybridZonotopeGrid(hz: "HybridZonotopeGrid", center, G_c, G_b):
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
    def ActivationOutputIntersectionElem(abstract_transormer_hz : "HybridZonotopeElem", input_hz : "HybridZonotopeElem"):

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

        
        from hybridz_transformers import HybridZonotopeElem
        return HybridZonotopeElem(new_center, new_Gc, new_Gb, new_Ac, new_Ab, new_b,
                                  method=input_hz.method, time_limit=input_hz.time_limit,
                                  dtype=input_hz.dtype, device=input_hz.device)

    @staticmethod
    def ActivationOutputIntersectionElemMemoryOptimized(abstract_transormer_hz : "HybridZonotopeElem", input_hz : "HybridZonotopeElem"):
        dim = input_hz.n

        print_memory_usage("Intersection start")

        print("Computing permutation indices...")
        perm_indices = list(range(0, 2*dim, 2)) + list(range(1, 2*dim, 2))
        print(f"Permutation indices computed, length: {len(perm_indices)}")

        chunk_size = min(512, dim // 4) if dim > 1024 else dim
        print(f"Using chunk_size={chunk_size} for intersection operations")

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

        print("Step 4: Constructing Ac_bottom without storing intermediate R_Z_Gc...")

        temp_R_Z_Gc = abstract_transormer_hz.G_c[perm_indices][:dim]
        Ac_bottom = torch.cat([temp_R_Z_Gc, -Y_G_c], dim=1)
        del temp_R_Z_Gc
        print(f"Step 4 completed, Ac_bottom shape: {Ac_bottom.shape}")

        print("Step 5: Constructing Ab_bottom without storing intermediate R_Z_Gb...")

        if abstract_transormer_hz.G_b.shape[1] > 0:
            temp_R_Z_Gb = abstract_transormer_hz.G_b[perm_indices][:dim]
            Ab_bottom = torch.cat([temp_R_Z_Gb, -Y_G_b], dim=1)
            del temp_R_Z_Gb
        else:
            Ab_bottom = torch.cat([torch.zeros(dim, 0, device=Y_G_b.device, dtype=Y_G_b.dtype), -Y_G_b], dim=1)
        print(f"Step 5 completed, Ab_bottom shape: {Ab_bottom.shape}")

        print("Step 6: Constructing new_b without storing intermediate R_Z_center...")

        temp_R_Z_center = Z_center[:dim]
        new_b = torch.cat([Z_b, Y_b, Y_center - temp_R_Z_center], dim=0)
        del temp_R_Z_center
        print(f"Step 6 completed, new_b shape: {new_b.shape}")

        print_memory_usage("After optimized R matrix operations")

        print("Step 7: Constructing constraint matrices with minimal memory footprint...")

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

        print("Step 8: Applying output filter (taking second half)...")

        new_center = new_center[dim:]
        new_Gc = new_Gc[dim:]
        new_Gb = new_Gb[dim:]
        print(f"Step 8 completed, final shapes: center{new_center.shape}, Gc{new_Gc.shape}, Gb{new_Gb.shape}")

        del Z_center, Y_center, Y_G_c, Y_G_b, Z_b, Y_b
        del Z_A_c, Z_A_b, Y_A_c, Y_A_b
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        print_memory_usage("Intersection completed")

        
        from hybridz_transformers import HybridZonotopeElem
        return HybridZonotopeElem(new_center, new_Gc, new_Gb, new_Ac, new_Ab, new_b,
                                  method=input_hz.method, time_limit=input_hz.time_limit,
                                  dtype=input_hz.dtype, device=input_hz.device)

    @staticmethod
    def ActivationOutputIntersectionGrid(abstract_transormer_hz_grid : "HybridZonotopeGrid", input_hz_grid : "HybridZonotopeGrid"):
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

        
        from hybridz_transformers import HybridZonotopeGrid
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
            print(f"Dimension mismatch: HZ={n_neurons}, auto_LiRPA={len(lirpa_lb_flat)}, falling back")
            return None

        print(f"ReLU optimization: {len(stable_pos)} stable+, {len(stable_neg)} stable-, {len(unstable)} unstable")

        new_center_list, new_G_c_list, new_G_b_list, new_A_c_list, new_A_b_list, new_b_list = [], [], [], [], [], []

        for i in range(n_neurons):
            lb_i, ub_i = lirpa_lb_flat[i], lirpa_ub_flat[i]

            if i in stable_pos:

                lb_i_optimized = max(lb_i, 1e-6)
                ub_i_optimized = max(ub_i, lb_i_optimized + 1e-6)
                print(f"Neuron {i}: stable+ [{lb_i:.6f}, {ub_i:.6f}] -> [{lb_i_optimized:.6f}, {ub_i_optimized:.6f}]")
            elif i in stable_neg:

                ub_i_optimized = min(ub_i, -1e-6)
                lb_i_optimized = min(lb_i, ub_i_optimized - 1e-6)
                print(f"Neuron {i}: stable- [{lb_i:.6f}, {ub_i:.6f}] -> [{lb_i_optimized:.6f}, {ub_i_optimized:.6f}]")
            else:

                lb_i_optimized, ub_i_optimized = lb_i, ub_i
                print(f"Neuron {i}: unstable [{lb_i:.6f}, {ub_i:.6f}] (needs binary var)")

            new_center_i, new_G_c_i, new_G_b_i, new_A_c_i, new_A_b_i, new_b_i = HybridZonotopeOps.ReLUElem(
                lb_i_optimized, ub_i_optimized, dtype=dtype, device=device
            )

            new_center_list.append(new_center_i)
            new_G_c_list.append(new_G_c_i)
            new_G_b_list.append(new_G_b_i)
            new_A_c_list.append(new_A_c_i)
            new_A_b_list.append(new_A_b_i)
            new_b_list.append(new_b_i)

        print(f"Optimized ReLU: reduced constraints from {n_neurons} potential to {len(unstable)} actual binary variables")

        return new_center_list, new_G_c_list, new_G_b_list, new_A_c_list, new_A_b_list, new_b_list

    @staticmethod
    def ReLUStandard(pre_activation_bounds, dtype, device, method='hybridz', relaxation_ratio=1.0):

        lb, ub = pre_activation_bounds
        n_neurons = len(lb)

        if method == 'hybridz_relaxed':

            n_relaxed = int(n_neurons * relaxation_ratio)
            n_exact = n_neurons - n_relaxed

            if relaxation_ratio == 1.0:
                print(f"Using fully relaxed ReLU (LP instead of MILP) for all {n_neurons} neurons")
            elif relaxation_ratio == 0.0:
                print(f"Using fully exact ReLU (MILP) for all {n_neurons} neurons")
            else:
                print(f"Using mixed ReLU strategy: {n_relaxed} relaxed (LP) + {n_exact} exact (MILP) out of {n_neurons} neurons (ratio={relaxation_ratio:.1f})")
        elif method == 'hybridz_relaxed_with_bab':

            print(f"BaB mode: Using fully relaxed ReLU (LP) for all {n_neurons} neurons")
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
                print(f"Using fully relaxed {func_type.upper()} (LP instead of MILP) for all {n_neurons} neurons")
            elif relaxation_ratio == 0.0:
                print(f"Using fully exact {func_type.upper()} (MILP) for all {n_neurons} neurons")
            else:
                print(f"Using mixed {func_type.upper()} strategy: {n_relaxed} relaxed (LP) + {n_exact} exact (MILP) out of {n_neurons} neurons (ratio={relaxation_ratio:.1f})")
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
            print(f"Combined matrix shape: {combined_matrix.shape} (Gc: {G_c.shape}, Ac: {A_c.shape})")
        else:

            combined_matrix = G_c
            print(f"Analyzing Gc only: {G_c.shape} (no constraints)")

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
                    print(f"📋 Found parallel generators: col {i} ↔ col {j} (cosine={cos_sim:.4f})")

        if not merge_pairs:
            print("No parallel generators found, returning original matrices")
            return center, G_c, G_b, A_c, A_b, b

        print(f"Found {len(merge_pairs)} parallel generator pairs to merge")

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
                print(f"Merged col {j} into col {i} (cosine={cos_sim:.4f}) -> new_col {target_col}")

        print(f"Generator merging completed: {ng} -> {new_ng} generators (reduction: {ng-new_ng})")
        print(f"Compression ratio: {(ng-new_ng)/ng*100:.1f}% reduction")

        return center, G_c_merged, G_b, A_c_merged, A_b, b

    @staticmethod
    def GetCounterexampleForOutputLayer(flat_center, flat_G_c, A_c_tensor, b_tensor, property_constraint, time_limit=30):

        try:
            print(f"[SMT] Counterexample generation for output layer starting...")
            print(f"Output layer shape: center{flat_center.shape}, G_c{flat_G_c.shape}")
            print(f"Constraint shape: A_c{A_c_tensor.shape}, b{b_tensor.shape}")
            print(f"Property constraint: {property_constraint}")

            print("Using Z3 SMT Solver (counterexample generation)")
            result = HybridZonotopeOps.ComputeConstrainedZElemBoundsSMT(
                flat_center, flat_G_c, A_c_tensor, b_tensor,
                time_limit=time_limit,
                property_constraint=property_constraint
            )

            if len(result) == 3:
                lb, ub, status = result
                if status == "UNSAFE":
                    print("[SMT] Safety property violated – unsafe configuration found")
                    return "VIOLATION_FOUND"
                elif status == "SAFE":
                    print("[SMT] Safety property verified – constraint unsatisfiable")
                    return "SAFE_VERIFIED"
                else:
                    print("[SMT] Solving result unknown")
                    return "UNKNOWN_RESULT"
            else:
                return None

        except Exception as e:
            print(f"[SMT] Counterexample generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None
