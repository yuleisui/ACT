# bab.pseudo
import time, heapq, numpy as np, torch
from dataclasses import dataclass
from typing import Optional, List, Callable
from act.back_end.core import Bounds
from act.back_end.verify_status import VerifStatus, VerifResult, verify_once, seed_from_input_spec
from act.front_end.specs import OutputSpec, OutKind

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

from .core import Bounds
from act.front_end.specs import InputSpec, OutputSpec, InKind, OutKind

@torch.no_grad()
def forward_model_eval(model_fn: Callable[[torch.Tensor], torch.Tensor], x_np: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(x_np)  # torch.from_numpy already creates tensor, no need for as_tensor
    y = model_fn(x).detach().cpu().numpy(); return y

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
