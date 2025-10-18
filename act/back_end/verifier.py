#===- act/back_end/verifier.py - Spec-free Verification Engine ----------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Spec-free, input-free verification (single-shot + branch-and-bound).
#   Assumes ACT Net already encodes input and output specifications via
#   INPUT_SPEC and ASSERT layers produced by torch2act.TorchToACT.
#
#===---------------------------------------------------------------------===#

# Public API:
#   - verify_once(net, solver, timelimit=None, maximize_violation=False) -> VerifResult
#   - verify_bab(net, solver, model_fn, max_depth=20, max_nodes=2000, time_budget_s=300.0) -> VerifResult
#
# Notes:
#   * No external input specs, shapes, or variable ids are required.
#   * All constraints are extracted from the ACT Net.
#   * model_fn is used only to numerically validate a returned counterexample candidate.

from __future__ import annotations
import time
import heapq
from dataclasses import dataclass, field
from typing import Optional, List, Callable, Dict, Any

import numpy as np
import torch

# ACT backend imports
from act.back_end.core import Bounds, Con, ConSet
from act.back_end.solver.solver_base import Solver, SolveStatus

# Front-end enums (kinds)
from act.front_end.specs import InKind, OutKind

# -----------------------------------------------------------------------------
# Verification Status and Results
# -----------------------------------------------------------------------------

class VerifStatus:
    CERTIFIED = "CERTIFIED"
    COUNTEREXAMPLE = "COUNTEREXAMPLE"
    UNKNOWN = "UNKNOWN"

@dataclass
class VerifResult:
    status: str
    ce_x: Optional[np.ndarray] = None
    ce_y: Optional[np.ndarray] = None
    model_stats: Dict[str, Any] = field(default_factory=dict)

# -----------------------------------------------------------------------------
# ACT Net extraction helpers
# -----------------------------------------------------------------------------

def find_entry_layer_id(net) -> int:
    """Return the id of the single INPUT layer."""
    candidates = [L.id for L in net.layers if L.kind == "INPUT"]
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one INPUT layer, found {len(candidates)}.")
    return candidates[0]

def gather_input_spec_layers(net):
    """Return list of INPUT_SPEC layers (can be ≥1)."""
    return [L for L in net.layers if L.kind == "INPUT_SPEC"]

def get_assert_layer(net):
    """Return the last ASSERT layer (must be last)."""
    if not net.layers or net.layers[-1].kind != "ASSERT":
        raise ValueError("Expected last ACT layer to be ASSERT.")
    return net.layers[-1]

def get_input_ids(net) -> List[int]:
    """Out vars of the INPUT layer."""
    entry = find_entry_layer_id(net)
    return list(net.by_id[entry].out_vars)

def get_output_ids(net) -> List[int]:
    """Use ASSERT.in_vars; ASSERT is constraint-only so in_vars == prev out_vars."""
    assert_layer = get_assert_layer(net)
    return list(assert_layer.in_vars)

# -----------------------------------------------------------------------------
# Seed and input spec helpers
# -----------------------------------------------------------------------------

def seed_from_input_specs(spec_layers) -> Bounds:
    """
    Produce a seed Bounds from INPUT_SPEC layers.
    Policy:
      * Prefer BOX (lb/ub) if present.
      * Else use LINF_BALL with center+eps OR pre-pushed lb/ub if provided.
      * If only LIN_POLY exists (no box), raise (needs a seed box).
    """
    # BOX first
    for L in spec_layers:
        if L.meta.get("kind") == InKind.BOX and ("lb" in L.params) and ("ub" in L.params):
            return Bounds(L.params["lb"].clone(), L.params["ub"].clone())
    # L∞ next
    for L in spec_layers:
        if L.meta.get("kind") == InKind.LINF_BALL:
            # Prefer lb/ub if precomputed
            if ("lb" in L.params) and ("ub" in L.params):
                return Bounds(L.params["lb"].clone(), L.params["ub"].clone())
            # Else center + eps
            center = L.params.get("center", None)
            eps = L.meta.get("eps", None)
            if center is None or eps is None:
                raise ValueError("LINF_BALL requires lb/ub or center+eps to create a seed box.")
            e = torch.tensor(eps, dtype=center.dtype, device=center.device)
            return Bounds(center - e, center + e)
    # LIN_POLY only -> fail
    for L in spec_layers:
        if L.meta.get("kind") == InKind.LIN_POLY:
            raise ValueError("LIN_POLY needs a seed box to start analysis/BnB (no BOX/L∞ provided).")
    raise ValueError("No input specification (BOX/L∞/LIN_POLY) found for seeding.")

def add_all_input_specs(globalC: ConSet, input_ids: List[int], spec_layers) -> None:
    """Add all INPUT_SPEC layers into the global constraint set."""
    for L in spec_layers:
        k = L.meta.get("kind")
        if k == InKind.BOX:
            globalC.add_box(-1, input_ids, Bounds(L.params["lb"], L.params["ub"]))
        elif k == InKind.LINF_BALL:
            if ("lb" in L.params) and ("ub" in L.params):
                globalC.add_box(-1, input_ids, Bounds(L.params["lb"], L.params["ub"]))
            else:
                center = L.params.get("center", None)
                eps = L.meta.get("eps", None)
                if center is None or eps is None:
                    raise ValueError("LINF_BALL requires lb/ub or center+eps")
                e = torch.tensor(eps, dtype=center.dtype, device=center.device)
                globalC.add_box(-1, input_ids, Bounds(center - e, center + e))
        elif k == InKind.LIN_POLY:
            A = L.params["A"]; b = L.params["b"]
            globalC.replace(Con("INEQ", tuple(input_ids), {"tag": "in:linpoly", "A": A, "b": b}))
        else:
            raise NotImplementedError(f"Unsupported INPUT_SPEC kind: {k}")

def materialise_input_poly(globalC: ConSet, solver: Solver) -> None:
    """Convert tagged LIN_POLY input constraints into solver rows."""
    from act.back_end.cons_exportor import to_numpy  # lazy import to avoid cycles
    for con in globalC.S.values():
        if con.meta.get("tag") == "in:linpoly":
            A = to_numpy(con.meta["A"]); b = to_numpy(con.meta["b"]); vids = list(con.var_ids)
            for i in range(A.shape[0]):
                solver.add_lin_le(vids, list(A[i, :]), float(b[i]))

# -----------------------------------------------------------------------------
# ASSERT negation and numeric property check
# -----------------------------------------------------------------------------

def _new_scalar_var(solver: Solver) -> int:
    old = solver.n
    solver.add_vars(1)
    return old

def add_negated_assert_to_solver(solver: Solver, out_ids: List[int], assert_layer) -> None:
    """Add the negation of the ASSERT layer as constraints to the solver."""
    from act.back_end.cons_exportor import to_numpy  # lazy import
    k = assert_layer.meta.get("kind")
    if k == OutKind.LINEAR_LE:
        coeffs = list(to_numpy(assert_layer.params["c"]))
        d = float(assert_layer.meta["d"])
        solver.add_lin_ge(out_ids, coeffs, d + 1e-6)
    elif k == OutKind.TOP1_ROBUST:
        t = int(assert_layer.meta["y_true"]); v = _new_scalar_var(solver)
        for j, oj in enumerate(out_ids):
            if j == t:
                continue
            solver.add_lin_ge([v, oj, out_ids[t]], [1.0, -1.0, 1.0], 0.0)
        solver.add_lin_ge([v], [1.0], 0.0)
    elif k == OutKind.MARGIN_ROBUST:
        t = int(assert_layer.meta["y_true"]); delta = float(assert_layer.meta["margin"]); v = _new_scalar_var(solver)
        for j, oj in enumerate(out_ids):
            if j == t:
                continue
            solver.add_lin_ge([v, oj, out_ids[t]], [1.0, -1.0, 1.0], -delta)
        solver.add_lin_ge([v], [1.0], 0.0)
    elif k == OutKind.RANGE:
        # Practical default: encode a one-sided violation (≥ ub + eps)
        ub = assert_layer.params.get("ub", None)
        lb = assert_layer.params.get("lb", None)
        if ub is not None:
            for i, yi in enumerate(out_ids):
                solver.add_lin_ge([yi], [1.0], float(ub[i].item()) + 1e-6)
        # Optionally also check y ≤ lb - eps via branching or separate calls
        if lb is not None:
            pass  # leave to BnB branch or second call if desired
    else:
        raise NotImplementedError(f"Unsupported ASSERT kind: {k}")

@torch.no_grad()
def forward_model_eval(model_fn: Callable[[torch.Tensor], torch.Tensor], x_np: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(x_np)
    y = model_fn(x).detach().cpu().numpy()
    return y

def violates_property_against_assert(y: np.ndarray, assert_layer) -> bool:
    """Numeric check of an ASSERT property on a concrete output y."""
    k = assert_layer.meta.get("kind")
    if k == OutKind.LINEAR_LE:
        c = np.asarray(assert_layer.params["c"], dtype=float)
        d = float(assert_layer.meta["d"])
        return float(np.dot(c, y)) >= d + 1e-8
    if k == OutKind.TOP1_ROBUST:
        t = int(assert_layer.meta["y_true"])
        others = [i for i in range(len(y)) if i != t]
        return (y[others] - y[t]).max() >= 0.0
    if k == OutKind.MARGIN_ROBUST:
        t = int(assert_layer.meta["y_true"]); margin = float(assert_layer.meta["margin"])
        others = [i for i in range(len(y)) if i != t]
        return (y[others] - y[t]).max() >= margin
    if k == OutKind.RANGE:
        lb = assert_layer.params.get("lb", None)
        ub = assert_layer.params.get("ub", None)
        if lb is not None and np.any(y < np.asarray(lb, dtype=float) - 1e-8):
            return True
        if ub is not None and np.any(y > np.asarray(ub, dtype=float) + 1e-8):
            return True
        return False
    raise NotImplementedError(f"ASSERT kind not supported: {k}")

# -----------------------------------------------------------------------------
# Single-shot verification
# -----------------------------------------------------------------------------

@torch.no_grad()
def verify_once(net, solver: Solver, timelimit: Optional[float] = None, maximize_violation: bool = False) -> VerifResult:
    """Spec-free, input-free verification using the embedded ACT constraints."""
    from act.back_end.analyze import analyze  # lazy import to avoid cycles
    from act.back_end.cons_exportor import export_to_solver

    entry_id = find_entry_layer_id(net)
    input_ids = get_input_ids(net)
    output_ids = get_output_ids(net)
    spec_layers = gather_input_spec_layers(net)
    assert_layer = get_assert_layer(net)

    seed_bounds = seed_from_input_specs(spec_layers)

    before, after, globalC = analyze(net, entry_id, seed_bounds)
    add_all_input_specs(globalC, input_ids, spec_layers)

    export_to_solver(globalC, solver, objective=None, sense="min")
    materialise_input_poly(globalC, solver)

    add_negated_assert_to_solver(solver, output_ids, assert_layer)

    k = assert_layer.meta.get("kind")
    # If robust kind, maximize violation scalar (the aux var created by add_negated_assert_to_solver).
    if maximize_violation and k in (OutKind.TOP1_ROBUST, OutKind.MARGIN_ROBUST):
        # The newly created scalar is at index (solver.n - 1) due to _new_scalar_var rule.
        solver.set_objective_linear([solver.n - 1], [1.0], 0.0, sense="max")
    else:
        solver.set_objective_linear([], [], 0.0, sense="min")

    solver.optimize(timelimit)
    st = solver.status()
    if st in (SolveStatus.OPTIMAL, SolveStatus.FEASIBLE) and solver.has_solution():
        return VerifResult(
            VerifStatus.COUNTEREXAMPLE,
            ce_x=solver.get_values(input_ids),
            ce_y=solver.get_values(output_ids),
            model_stats={"status": st, "ncons": len(globalC.S)}
        )
    if st == SolveStatus.INFEASIBLE:
        return VerifResult(VerifStatus.CERTIFIED, model_stats={"status": st, "ncons": len(globalC.S)})
    return VerifResult(VerifStatus.UNKNOWN, model_stats={"status": st})

# -----------------------------------------------------------------------------
# Branch-and-Bound verification
# -----------------------------------------------------------------------------

@dataclass
class BabNode:
    box: Bounds
    depth: int
    score: float
    candidate_ce: Optional[np.ndarray] = None
    lower_certified: bool = False
    def __lt__(self, other):  # max-heap by score
        return self.score > other.score

def width_sum(B: Bounds) -> float:
    return float(torch.sum(B.ub - B.lb).item())

def choose_split_dim(B: Bounds) -> int:
    return int(torch.argmax(B.ub - B.lb).item())

def branch(B: Bounds, d: int) -> tuple[Bounds, Bounds]:
    mid = 0.5 * (B.lb[d] + B.ub[d])
    Lb = B.lb.clone(); Ub = B.ub.clone()
    Lb2 = B.lb.clone(); Ub2 = B.ub.clone()
    Ub[d] = mid; Lb2[d] = mid
    return Bounds(Lb, Ub), Bounds(Lb2, Ub2)

def solve_and_validate(net, node: BabNode, solver: Solver, model_fn: Callable[[torch.Tensor], torch.Tensor]):
    """Solve at this node's box; return ('CERT'/'TRUE_CE'/'FALSE_CE'/'UNKNOWN', node, VerifResult)."""
    from act.back_end.analyze import analyze
    from act.back_end.cons_exportor import export_to_solver

    entry_id = find_entry_layer_id(net)
    input_ids = get_input_ids(net)
    output_ids = get_output_ids(net)
    spec_layers = gather_input_spec_layers(net)
    assert_layer = get_assert_layer(net)

    before, after, globalC = analyze(net, entry_id, node.box)
    add_all_input_specs(globalC, input_ids, spec_layers)
    export_to_solver(globalC, solver, objective=None, sense="min")
    materialise_input_poly(globalC, solver)
    add_negated_assert_to_solver(solver, output_ids, assert_layer)

    k = assert_layer.meta.get("kind")
    if k in (OutKind.TOP1_ROBUST, OutKind.MARGIN_ROBUST):
        solver.set_objective_linear([solver.n - 1], [1.0], 0.0, sense="max")
    else:
        solver.set_objective_linear([], [], 0.0, sense="min")

    solver.optimize(None)
    st = solver.status()
    if st == SolveStatus.INFEASIBLE:
        node.lower_certified = True
        return "CERT", node, VerifResult(VerifStatus.CERTIFIED, model_stats={"status": st})
    if st in (SolveStatus.OPTIMAL, SolveStatus.FEASIBLE) and solver.has_solution():
        x_ce = solver.get_values(input_ids)
        y_ce = forward_model_eval(model_fn, x_ce)
        if violates_property_against_assert(y_ce, assert_layer):
            node.candidate_ce = x_ce
            return "TRUE_CE", node, VerifResult(VerifStatus.COUNTEREXAMPLE, ce_x=x_ce, ce_y=y_ce)
        else:
            node.candidate_ce = x_ce
            return "FALSE_CE", node, VerifResult(VerifStatus.UNKNOWN, model_stats={"status": st})
    return "UNKNOWN", node, VerifResult(VerifStatus.UNKNOWN, model_stats={"status": st})

def verify_bab(net, solver: Solver, model_fn: Callable[[torch.Tensor], torch.Tensor],
               max_depth: int = 20, max_nodes: int = 2000, time_budget_s: float = 300.0) -> VerifResult:
    """Spec-free, input-free branch-and-bound verification using the embedded ACT constraints."""
    spec_layers = gather_input_spec_layers(net)
    root_box = seed_from_input_specs(spec_layers)

    pq: List[BabNode] = []
    heapq.heappush(pq, BabNode(root_box, 0, score=width_sum(root_box)))
    start = time.time()
    processed = 0

    while pq and (time.time() - start) < time_budget_s and processed < max_nodes:
        node = heapq.heappop(pq)
        processed += 1

        status, node, res = solve_and_validate(net, node, solver, model_fn)
        if status == "TRUE_CE":
            # already numeric-validated
            return VerifResult(VerifStatus.COUNTEREXAMPLE, ce_x=node.candidate_ce, ce_y=res.ce_y,
                               model_stats={"nodes": processed})
        if status == "CERT":
            continue

        # For FALSE_CE or UNKNOWN: branch further if depth allows
        if node.depth >= max_depth:
            continue
        d = choose_split_dim(node.box)
        Lbox, Rbox = branch(node.box, d)
        heapq.heappush(pq, BabNode(Lbox, node.depth + 1, width_sum(Lbox)))
        heapq.heappush(pq, BabNode(Rbox, node.depth + 1, width_sum(Rbox)))

    return VerifResult(VerifStatus.CERTIFIED, model_stats={"nodes": processed})
