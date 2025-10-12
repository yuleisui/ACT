# verif_status.py - Verification status, results, and core verification functions
import torch, numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from act.back_end.core import Bounds, Con, ConSet
from act.back_end.analyze import analyze
from act.back_end.cons_exportor import export_to_solver, to_numpy
from act.back_end.solver_base import Solver, SolveStatus

# Import specification classes from front_end
from act.front_end.specs import InKind, InputSpec, OutKind, OutputSpec

class VerifStatus: CERTIFIED="CERTIFIED"; COUNTEREXAMPLE="COUNTEREXAMPLE"; UNKNOWN="UNKNOWN"
@dataclass
class VerifResult:
    status: str
    ce_x: Optional[np.ndarray]=None; ce_y: Optional[np.ndarray]=None
    model_stats: Dict[str, Any]=field(default_factory=dict)

def seed_from_input_spec(I: InputSpec) -> Bounds:
    if I.kind==InKind.BOX:       return Bounds(I.lb.clone(), I.ub.clone())
    if I.kind==InKind.LINF_BALL: return Bounds(I.center - torch.tensor(I.eps, dtype=I.center.dtype, device=I.center.device),
                                               I.center + torch.tensor(I.eps, dtype=I.center.dtype, device=I.center.device))
    if I.kind==InKind.LIN_POLY:  raise ValueError("LIN_POLY needs a seed box")
    raise NotImplementedError

def add_input_spec(globalC: ConSet, input_ids: List[int], I: InputSpec):
    if I.kind==InKind.BOX:       globalC.add_box(-1, input_ids, Bounds(I.lb, I.ub))
    elif I.kind==InKind.LINF_BALL:
        e=torch.tensor(I.eps, dtype=I.center.dtype, device=I.center.device)
        globalC.add_box(-1, input_ids, Bounds(I.center-e, I.center+e))
    elif I.kind==InKind.LIN_POLY:
        globalC.replace(Con("INEQ", tuple(input_ids), {"tag": "in:linpoly", "A": I.A, "b": I.b}))
    else: raise NotImplementedError

def materialise_input_poly(globalC: ConSet, solver: Solver):
    for con in globalC.S.values():
        if con.meta.get("tag")=="in:linpoly":
            A=to_numpy(con.meta["A"]); b=to_numpy(con.meta["b"]); vids=list(con.var_ids)
            for i in range(A.shape[0]): solver.add_lin_le(vids, list(A[i,:]), float(b[i]))

def _new_scalar_var(solver: Solver) -> int:
    old=solver.n; solver.add_vars(1); return old

def add_negated_output_spec_to_solver(solver: Solver, out_ids: List[int], O: OutputSpec):
    if O.kind==OutKind.LINEAR_LE:
        coeffs=list(to_numpy(O.c)); solver.add_lin_ge(out_ids, coeffs, float(O.d + 1e-6))
    elif O.kind==OutKind.TOP1_ROBUST:
        t=int(O.y_true); v=_new_scalar_var(solver)
        for j, oj in enumerate(out_ids):
            if j==t: continue
            solver.add_lin_ge([v, oj, out_ids[t]],[1.0,-1.0,1.0],0.0)
        solver.add_lin_ge([v],[1.0],0.0)
    elif O.kind==OutKind.MARGIN_ROBUST:
        t=int(O.y_true); delta=float(O.margin); v=_new_scalar_var(solver)
        for j, oj in enumerate(out_ids):
            if j==t: continue
            solver.add_lin_ge([v, oj, out_ids[t]],[1.0,-1.0,1.0],-delta)
        solver.add_lin_ge([v],[1.0],0.0)
    else: raise NotImplementedError

@torch.no_grad()
def verify_once(net, entry_id, input_ids, output_ids, input_spec, output_spec, seed_bounds, solver: Solver,
                timelimit: Optional[float]=None, maximize_violation: bool=False) -> VerifResult:
    before, after, globalC = analyze(net, entry_id, seed_bounds)
    add_input_spec(globalC, input_ids, input_spec)
    export_to_solver(globalC, solver, objective=None, sense="min")
    materialise_input_poly(globalC, solver)
    add_negated_output_spec_to_solver(solver, output_ids, output_spec)
    if maximize_violation and output_spec.kind in (OutKind.TOP1_ROBUST, OutKind.MARGIN_ROBUST):
        solver.set_objective_linear([solver.n-1], [1.0], 0.0, sense="max")
    else:
        solver.set_objective_linear([], [], 0.0, sense="min")
    solver.optimize(timelimit)
    st=solver.status()
    if st in (SolveStatus.OPTIMAL, SolveStatus.FEASIBLE) and solver.has_solution():
        return VerifResult(VerifStatus.COUNTEREXAMPLE,
                           ce_x=solver.get_values(input_ids),
                           ce_y=solver.get_values(output_ids),
                           model_stats={"status":st,"ncons":len(globalC.S)})
    if st==SolveStatus.INFEASIBLE:
        return VerifResult(VerifStatus.CERTIFIED, model_stats={"status":st,"ncons":len(globalC.S)})
    return VerifResult(VerifStatus.UNKNOWN, model_stats={"status":st})
