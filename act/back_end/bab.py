# bab.py - Branch-and-bound verification with verification status and results
import time, heapq, numpy as np, torch
from dataclasses import dataclass, field
from typing import Optional, List, Callable, Dict, Any
from act.back_end.core import Bounds, Con, ConSet
from act.back_end.solver.solver_base import Solver, SolveStatus
from act.front_end.device_manager import get_default_device, get_default_dtype

# Import specification classes from front_end
from act.front_end.specs import InKind, InputSpec, OutKind, OutputSpec

# -----------------------------------------------------------------------------
# Verification Status and Results (formerly from verify_status.py)
# -----------------------------------------------------------------------------

class VerifStatus: 
    CERTIFIED="CERTIFIED"; COUNTEREXAMPLE="COUNTEREXAMPLE"; UNKNOWN="UNKNOWN"

@dataclass
class VerifResult:
    status: str
    ce_x: Optional[np.ndarray]=None; ce_y: Optional[np.ndarray]=None
    model_stats: Dict[str, Any]=field(default_factory=dict)

def seed_from_input_spec(I: InputSpec) -> Bounds:
    if I.kind==InKind.BOX:       return Bounds(I.lb.clone(), I.ub.clone())
    if I.kind==InKind.LINF_BALL: 
        device = get_default_device()
        dtype = get_default_dtype()
        eps_tensor = torch.tensor(I.eps, dtype=dtype, device=device)
        return Bounds(I.center - eps_tensor, I.center + eps_tensor)
    if I.kind==InKind.LIN_POLY:  raise ValueError("LIN_POLY needs a seed box")
    raise NotImplementedError

def add_input_spec(globalC: ConSet, input_ids: List[int], I: InputSpec):
    if I.kind==InKind.BOX:       globalC.add_box(-1, input_ids, Bounds(I.lb, I.ub))
    elif I.kind==InKind.LINF_BALL:
        device = get_default_device()
        dtype = get_default_dtype()
        e = torch.tensor(I.eps, dtype=dtype, device=device)
        globalC.add_box(-1, input_ids, Bounds(I.center-e, I.center+e))
    elif I.kind==InKind.LIN_POLY:
        globalC.replace(Con("INEQ", tuple(input_ids), {"tag": "in:linpoly", "A": I.A, "b": I.b}))
    else: raise NotImplementedError

def materialise_input_poly(globalC: ConSet, solver: Solver):
    from act.back_end.cons_exportor import to_numpy  # Import here to avoid circular import
    for con in globalC.S.values():
        if con.meta.get("tag")=="in:linpoly":
            A=to_numpy(con.meta["A"]); b=to_numpy(con.meta["b"]); vids=list(con.var_ids)
            for i in range(A.shape[0]): solver.add_lin_le(vids, list(A[i,:]), float(b[i]))

def _new_scalar_var(solver: Solver) -> int:
    old=solver.n; solver.add_vars(1); return old

def add_negated_output_spec_to_solver(solver: Solver, out_ids: List[int], O: OutputSpec):
    from act.back_end.cons_exportor import to_numpy  # Import here to avoid circular import
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
    from act.back_end.analyze import analyze  # Import here to avoid circular import
    from act.back_end.cons_exportor import export_to_solver  # Import here to avoid circular import
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

# -----------------------------------------------------------------------------
# Branch-and-Bound Verification
# -----------------------------------------------------------------------------

@dataclass
class BabNode:
    box: Bounds; depth: int; score: float
    candidate_ce: Optional[np.ndarray]=None
    lower_certified: bool=False
    def __lt__(self, other): return self.score > other.score  # max-heap

def width_sum(B: Bounds) -> float: return float(torch.sum(B.ub - B.lb).item())
def choose_split_dim(B: Bounds) -> int: return int(torch.argmax(B.ub - B.lb).item())
def branch(B: Bounds, d: int) -> tuple[Bounds, Bounds]:
    mid=0.5*(B.lb[d]+B.ub[d]); Lb=B.lb.clone(); Ub=B.ub.clone(); Lb2=B.lb.clone(); Ub2=B.ub.clone()
    Ub[d]=mid; Lb2[d]=mid; return Bounds(Lb,Ub), Bounds(Lb2,Ub2)

@torch.no_grad()
def forward_model_eval(model_fn: Callable[[torch.Tensor], torch.Tensor], x_np: np.ndarray) -> np.ndarray:
    device = get_default_device()
    dtype = get_default_dtype()
    x = torch.from_numpy(x_np).to(device=device, dtype=dtype)
    y = model_fn(x).detach().cpu().numpy()
    return y

def violates_property(y: np.ndarray, O: OutputSpec) -> bool:
    if O.kind==OutKind.LINEAR_LE:   return float(np.dot(np.asarray(O.c,float), y)) >= float(O.d) + 1e-8
    if O.kind==OutKind.TOP1_ROBUST: t=int(O.y_true); return np.max(y[np.arange(len(y))!=t]-y[t]) >= 0.0
    if O.kind==OutKind.MARGIN_ROBUST: t=int(O.y_true); return np.max(y[np.arange(len(y))!=t]-y[t]) >= float(O.margin)
    raise NotImplementedError

def solve_and_validate(net, entry_id, input_ids, output_ids, input_spec, output_spec, node: BabNode, solver, model_fn, tlim=None):
    res=verify_once(net, entry_id, input_ids, output_ids, input_spec, output_spec, node.box, solver, tlim, maximize_violation=True)
    if res.status==VerifStatus.CERTIFIED: node.lower_certified=True; return "CERT", node, res
    if res.status==VerifStatus.UNKNOWN:   return "UNKNOWN", node, res
    x_ce=res.ce_x; y_ce=forward_model_eval(model_fn, x_ce)
    if violates_property(y_ce, output_spec): node.candidate_ce=x_ce; return "TRUE_CE", node, res
    else:                                    node.candidate_ce=x_ce; return "FALSE_CE", node, res

def verify_bab(net, entry_id, input_ids: List[int], output_ids: List[int],
               input_spec, output_spec, root_box: Bounds, solver, model_fn,
               max_depth=20, max_nodes=2000, time_budget_s=300.0) -> VerifResult:
    pq: List[BabNode]=[]; heapq.heappush(pq, BabNode(root_box, 0, score=width_sum(root_box)))
    start=time.time(); processed=0
    while pq and (time.time()-start) < time_budget_s and processed < max_nodes:
        node=heapq.heappop(pq); processed+=1
        status, node, res = solve_and_validate(net, entry_id, input_ids, output_ids, input_spec, output_spec, node, solver, model_fn)
        if status=="TRUE_CE":
            y = forward_model_eval(model_fn, node.candidate_ce)
            return VerifResult(VerifStatus.COUNTEREXAMPLE, ce_x=node.candidate_ce, ce_y=y, model_stats={"nodes":processed})
        if status=="CERT": continue
        # FALSE_CE or UNKNOWN -> refine by branching
        if node.depth >= max_depth: continue
        d=choose_split_dim(node.box); Lbox,Rbox=branch(node.box, d)
        heapq.heappush(pq, BabNode(Lbox, node.depth+1, width_sum(Lbox)))
        heapq.heappush(pq, BabNode(Rbox, node.depth+1, width_sum(Rbox)))
    return VerifResult(VerifStatus.CERTIFIED, model_stats={"nodes":processed})
