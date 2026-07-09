
from __future__ import annotations
from typing import Optional, TYPE_CHECKING
import numpy as np
import os
from act.back_end.solver.solver_base import (
    Solver,
    SolverCaps,
    _empty_blockdiag,
    _problem,
    _make_problem_n1,
)
from act.util.path_config import get_project_root

if TYPE_CHECKING:
    from act.back_end.core import Bounds
    from act.back_end.solver.solver_base import BatchLPProblem, BatchLPSolution

try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    print("Warning: Gurobi not available. Some operations will use alternative solvers.")
    GUROBI_AVAILABLE = False


def is_gurobi_available() -> bool:
    return bool(GUROBI_AVAILABLE)

def setup_gurobi_license():
    """Setup Gurobi license path based on current folder layout."""
    if 'GRB_LICENSE_FILE' not in os.environ:
        if 'ACTHOME' in os.environ:
            license_path = os.path.join(os.environ['ACTHOME'], 'modules', 'gurobi', 'gurobi.lic')
            print(f"[ACT] Using ACTHOME environment variable: {os.path.relpath(os.environ['ACTHOME'])}")
        else:
            project_root = get_project_root()
            license_path = os.path.join(project_root, 'modules', 'gurobi', 'gurobi.lic')
            print(f"[ACT] Auto-detecting project root: {os.path.relpath(project_root)}")
        
        license_path = os.path.abspath(license_path)
        
        if os.path.exists(license_path):
            os.environ['GRB_LICENSE_FILE'] = license_path
            print(f"[ACT] Gurobi license found: {os.path.relpath(license_path)}")
        else:
            print(f"[WARN] Gurobi license not found: {os.path.relpath(license_path)}")
            print(f"[INFO] Please place gurobi.lic in: {os.path.relpath(os.path.dirname(license_path))}")
    else:
        print(f"[ACT] Using existing Gurobi license: {os.path.relpath(os.environ['GRB_LICENSE_FILE'])}")

setup_gurobi_license()


class GurobiSolver(Solver):
    """Gurobi backend for exact LP/MILP solving (CPU-only)."""

    def capabilities(self) -> SolverCaps:
        return SolverCaps(False)

    def __init__(self):
        if not GUROBI_AVAILABLE:
            raise RuntimeError("gurobipy is not available in this environment.")

    @staticmethod
    def compute_bounds(hz) -> 'Bounds':
        from act.back_end.core import Bounds
        import torch
        n = int(hz.c.shape[0])
        p = int(hz.Gc.shape[1])
        q = int(hz.Gb.shape[1])
        c_np = hz.c.detach().cpu().numpy().astype("float64").reshape(-1)
        Gc_np = hz.Gc.detach().cpu().numpy().astype("float64")
        Gb_np = hz.Gb.detach().cpu().numpy().astype("float64")
        Ac_np = hz.Ac.detach().cpu().numpy().astype("float64")
        Ab_np = hz.Ab.detach().cpu().numpy().astype("float64")
        b_np = hz.b.detach().cpu().numpy().astype("float64").reshape(-1)
        nc = Ac_np.shape[0]
        LB = np.empty((n,), dtype=np.float64)
        UB = np.empty((n,), dtype=np.float64)
        for i in range(n):
            m = gp.Model(f"hz_dim_{i}")
            m.Params.OutputFlag = 0
            xi_c = m.addMVar(p, lb=-1.0, ub=1.0, name="xi_c")
            xi_b = m.addMVar(q, vtype=GRB.BINARY, name="xi_b") if q > 0 else None
            if nc > 0:
                if xi_b is not None:
                    for r in range(nc):
                        m.addConstr(Ac_np[r] @ xi_c + Ab_np[r] @ xi_b == b_np[r])
                else:
                    for r in range(nc):
                        m.addConstr(Ac_np[r] @ xi_c == b_np[r])
            obj_c = Gc_np[i]
            obj_b = Gb_np[i] if q > 0 else np.zeros(0)
            if xi_b is not None:
                m.setObjective(obj_c @ xi_c + obj_b @ xi_b, GRB.MINIMIZE)
            else:
                m.setObjective(obj_c @ xi_c, GRB.MINIMIZE)
            m.optimize()
            LB[i] = c_np[i] + (m.ObjVal if m.Status == GRB.OPTIMAL else 0.0)
            if xi_b is not None:
                m.setObjective(obj_c @ xi_c + obj_b @ xi_b, GRB.MAXIMIZE)
            else:
                m.setObjective(obj_c @ xi_c, GRB.MAXIMIZE)
            m.optimize()
            UB[i] = c_np[i] + (m.ObjVal if m.Status == GRB.OPTIMAL else 0.0)
        dtype, device = hz.c.dtype, hz.c.device
        return Bounds(lb=torch.from_numpy(LB).to(device=device, dtype=dtype),
                      ub=torch.from_numpy(UB).to(device=device, dtype=dtype))

    def solve_batch(
        self,
        problem: "BatchLPProblem",  # noqa: F821
        timelimit: Optional[float] = None,
    ) -> "BatchLPSolution":  # noqa: F821
        """Solve a batch of N independent LPs.

        N=1: builds a single gp.Model, solves, returns BatchLPSolution[N=1].
        N>1: raises NotImplementedError — Gurobi's multi-scenario API does not
             expose truly parallel solving for varying constraint matrices.
        """
        from act.back_end.solver.solver_base import BatchLPSolution
        import torch

        if problem.N != 1:
            raise NotImplementedError(
                f"GurobiSolver.solve_batch: N={problem.N} not supported. "
                f"Gurobi does not expose a truly parallel multi-LP API for "
                f"varying constraint matrices. Use TorchLPSolver for N>1, "
                f"or constrain BaB to bab_max_batch_size=1 and "
                f"verify_lp_batched is skipped (set lp_enabled=False)."
            )

        nvars = problem.nvars
        lb = problem.lb[0].cpu().numpy().astype(np.float64)
        ub = problem.ub[0].cpu().numpy().astype(np.float64)

        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()
        m = gp.Model("verify_batch_n1", env=env)
        x = m.addMVar(nvars, lb=lb, ub=ub, name="x")

        # Decompose block-diagonal sparse: for N=1 the block is the full matrix.
        if problem.m_eq > 0:
            A_eq = (
                problem.A_eq_blockdiag.to_dense()[: problem.m_eq, :nvars]
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            b_eq = problem.b_eq[0].cpu().numpy().astype(np.float64)
            m.addConstr(A_eq @ x == b_eq)

        if problem.m_le > 0:
            A_le = (
                problem.A_le_blockdiag.to_dense()[: problem.m_le, :nvars]
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            b_le = problem.b_le[0].cpu().numpy().astype(np.float64)
            m.addConstr(A_le @ x <= b_le)

        obj_c = problem.obj_c[0].cpu().numpy().astype(np.float64)
        obj_const = float(problem.obj_const[0].item())
        sense = GRB.MINIMIZE if problem.sense == "min" else GRB.MAXIMIZE
        m.setObjective(obj_c @ x + obj_const, sense)

        if timelimit is not None:
            m.Params.TimeLimit = float(timelimit)
        m.optimize()

        dtype = problem.lb.dtype
        device = problem.lb.device

        if m.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
            status = "SAT"
            x_val = torch.as_tensor(x.X, dtype=dtype, device=device).unsqueeze(0)
            max_viol = torch.zeros(1, dtype=dtype, device=device)
        elif m.Status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
            status = "UNSAT"
            x_val = torch.zeros_like(problem.lb)
            max_viol = torch.full((1,), float("inf"), dtype=dtype, device=device)
        else:
            status = "UNKNOWN"
            x_val = torch.zeros_like(problem.lb)
            max_viol = torch.full((1,), float("nan"), dtype=dtype, device=device)

        return BatchLPSolution(
            statuses=(status,),
            x=x_val,
            max_viol=max_viol,
        )


# --- Self-tests (run with: python -m act.back_end.solver.solver_gurobi) ---

import importlib.util as _ilu

_HAS_GUROBI = _ilu.find_spec("gurobipy") is not None


def _test_solve_batch_n1_sat():
    if not _HAS_GUROBI:
        print("SKIP  _test_solve_batch_n1_sat (no gurobipy)")
        return
    import torch
    from act.back_end.solver.solver_base import BatchLPProblem

    nvars = 2
    p = BatchLPProblem(
        nvars=nvars, m_eq=0, m_le=0,
        lb=torch.tensor([[0.0, 0.0]]),
        ub=torch.tensor([[1.0, 1.0]]),
        A_eq_blockdiag=_empty_blockdiag(1, 0, nvars),
        b_eq=torch.zeros(1, 0),
        A_le_blockdiag=_empty_blockdiag(1, 0, nvars),
        b_le=torch.zeros(1, 0),
        obj_c=torch.tensor([[1.0, 0.0]]),
        obj_const=torch.zeros(1),
        sense="min",
    )
    sol = GurobiSolver().solve_batch(p)
    assert sol.statuses == ("SAT",), f"expected SAT, got {sol.statuses}"
    assert sol.x.shape == (1, nvars)
    assert float(sol.x[0, 0]) < 0.01, f"expected x[0]~0, got {sol.x[0,0]}"


def _test_solve_batch_n1_infeasible():
    if not _HAS_GUROBI:
        print("SKIP  _test_solve_batch_n1_infeasible (no gurobipy)")
        return
    import torch
    from act.back_end.solver.solver_base import BatchLPProblem

    nvars = 1
    A_le_vals = torch.tensor([-1.0])
    A_le_idx = torch.tensor([[0], [0]], dtype=torch.long)
    A_le_sp = torch.sparse_coo_tensor(A_le_idx, A_le_vals, (1, 1))

    p = BatchLPProblem(
        nvars=nvars, m_eq=0, m_le=1,
        lb=torch.tensor([[0.0]]),
        ub=torch.tensor([[1.0]]),
        A_eq_blockdiag=_empty_blockdiag(1, 0, nvars),
        b_eq=torch.zeros(1, 0),
        A_le_blockdiag=A_le_sp,
        b_le=torch.tensor([[-3.0]]),
        obj_c=torch.zeros(1, nvars),
        obj_const=torch.zeros(1),
    )
    sol = GurobiSolver().solve_batch(p)
    assert sol.statuses == ("UNSAT",), f"expected UNSAT, got {sol.statuses}"


def _test_solve_batch_n_greater_than_1_raises():
    if not _HAS_GUROBI:
        print("SKIP  _test_solve_batch_n_greater_than_1_raises (no gurobipy)")
        return
    p = _problem(N=2, nvars=3)
    try:
        GurobiSolver().solve_batch(p)
        raise AssertionError("expected NotImplementedError for N=2")
    except NotImplementedError as e:
        msg = str(e)
        assert "N=2" in msg, f"diagnostic message missing N=2: {msg}"
        assert "TorchLPSolver" in msg, f"diagnostic message missing TorchLPSolver: {msg}"


if __name__ == "__main__":
    import sys

    tests = [
        _test_solve_batch_n1_sat,
        _test_solve_batch_n1_infeasible,
        _test_solve_batch_n_greater_than_1_raises,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            name = t.__name__
            if not _HAS_GUROBI:
                pass
            else:
                print(f"PASS  {name}")
        except Exception as e:
            print(f"FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(failed)
