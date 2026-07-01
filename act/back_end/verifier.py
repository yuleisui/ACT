#===- act/back_end/verifier.py - Spec-free Verification Engine ----------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Spec-free, input-free verification. Assumes the ACT Net already encodes
#   both input and output specifications via INPUT_SPEC and ASSERT layers
#   (produced by torch2act.TorchToACT).
#
# Architecture — verify_once:
#   1. Seed [B, *input_shape] bounds from INPUT_SPEC layers (no CSP).
#   2. analyze() propagates batched bounds through every TF op.
#   3. Read pre-encoded [B*M, n_out] linear-form C / [B, M] thresholds / M
#      from the ASSERT layer params (produced upstream by
#      OutputSpec.encode_linear at FE construction time).
#   4. INTERVAL CERTIFICATION: one tensor pass computes margin_max under
#      output bounds; sample b is CERTIFIED iff every M lane passes.
#   5. CONCRETE FALSIFICATION (when model_fn given): one batched forward at
#      box centre; samples whose concrete output meets-or-exceeds threshold
#      become FALSIFIED. Remaining samples are UNKNOWN.
#   6. Return List[VerifyResult] of length B (one per input lane).
#
#===---------------------------------------------------------------------===#

# Public API:
#   - verify_once(net, *, model_fn=None) -> List[VerifyResult]
#       Pure-tensor batched single-shot verifier.
#   - setup_and_solve_batch(net, input_bounds_per_b, solver, timelimit=None)
#       Batch-native CSP setup helper used by LP and BaB refinement.
#   - find_entry_layer_id / get_input_ids / get_output_ids /
#     gather_input_spec_layers / get_assert_layer / seed_from_input_specs /
#     add_all_input_specs (helpers).
#
# Notes:
#   * Spec-free verification: all constraints extracted from ACT Net layers.
#   * verify_once returns one VerifyResult per lane (len(result) == B).
#   * INPUT_SPEC constraints (including LIN_POLY) are propagated through
#     analyze(); they enter via add_all_input_specs into entry_fact.cons.
#     LIN_POLY constraints are not consumed by verify_once's interval
#     certification; they are preserved for the batch-native solver path.

from __future__ import annotations
from typing import Optional, List, Callable, Dict, Any, TYPE_CHECKING, cast

import torch
import copy

# ACT backend imports
from act.back_end.core import Bounds, Con, ConSet, Fact, Net
from act.back_end.solver.solver_base import Solver, SolveStatus, BatchLPSolution
from act.back_end.layer_schema import LayerKind
from act.back_end.utils import validate_constraints

if TYPE_CHECKING:
    from act.back_end.analyze import AnalyzeCache

# Front-end enums (kinds)
from act.front_end.specs import InKind, OutKind

# Verification types (canonical location: act/util/stats.py)
from act.util.stats import VerifyStatus, VerifyResult

# -----------------------------------------------------------------------------
# Sequential per-sample slicing (for B>1 BaB)
# -----------------------------------------------------------------------------

def _slice_first_dim(value: Any, sample_idx: int, expected_b: int) -> Any:
    if isinstance(value, torch.Tensor) and value.dim() >= 1 and value.shape[0] == expected_b:
        return value[sample_idx:sample_idx + 1]
    return value


def slice_net_to_sample(net: Net, sample_idx: int) -> Net:
    from act.front_end.spec_creator_base import LabeledInputTensor

    mutable_kinds = {
        LayerKind.INPUT.value,
        LayerKind.INPUT_SPEC.value,
        LayerKind.ASSERT.value,
    }
    layers = []
    for layer in net.layers:
        if layer.kind not in mutable_kinds:
            layers.append(layer)
            continue
        layer2 = copy.copy(layer)
        layer2.params = dict(layer.params)
        layer2.in_vars = list(layer.in_vars)
        layer2.out_vars = list(layer.out_vars)
        layer2.cache = dict(layer.cache)
        layers.append(layer2)
    net2 = copy.copy(net)
    net2.layers = layers
    net2.preds = net.preds
    net2.succs = net.succs
    net2.by_id = {layer.id: layer for layer in layers}

    entry_id = find_entry_layer_id(net2)
    input_layer = net2.by_id[entry_id]
    shape = input_layer.params.get("shape") or []
    shape_t = tuple(shape) if isinstance(shape, (list, tuple)) else ()
    B = int(shape_t[0]) if shape_t else 1
    if shape_t and int(shape_t[0]) == B:
        input_layer.params["shape"] = (1,) + tuple(shape_t[1:])
    li = input_layer.params.get("labeled_input")
    if isinstance(li, LabeledInputTensor):
        new_tensor = _slice_first_dim(li.tensor, sample_idx, B)
        new_label = _slice_first_dim(li.label, sample_idx, B) if li.label is not None else None
        input_layer.__dict__["params"]["labeled_input"] = LabeledInputTensor(
            tensor=new_tensor, label=new_label,
        )

    for spec_layer in gather_input_spec_layers(net2):
        for key in ("center", "eps", "lb", "ub", "A", "b"):
            val = spec_layer.params.get(key)
            if val is not None:
                spec_layer.params[key] = _slice_first_dim(val, sample_idx, B)

    assert_layer = get_assert_layer(net2)
    m_raw = assert_layer.params.get("M", 1)
    if isinstance(m_raw, torch.Tensor):
        m_rows = int(m_raw.item())
    elif isinstance(m_raw, int):
        m_rows = m_raw
    else:
        raise ValueError(f"ASSERT M must be int or tensor, got {m_raw!r}")
    for key in ("y_true", "margin", "c", "d", "lb", "ub"):
        val = assert_layer.params.get(key)
        if val is not None:
            assert_layer.params[key] = _slice_first_dim(val, sample_idx, B)
    # C is [B*M, n_out] — first dim is B*M not B, so slice rows manually
    c_big = assert_layer.params.get("C")
    if isinstance(c_big, torch.Tensor) and c_big.shape[0] == B * m_rows:
        assert_layer.params["C"] = c_big[sample_idx * m_rows:(sample_idx + 1) * m_rows]
    thresholds = assert_layer.params.get("thresholds")
    if isinstance(thresholds, torch.Tensor) and thresholds.shape[0] == B:
        assert_layer.params["thresholds"] = thresholds[sample_idx:sample_idx + 1]

    return net2


# -----------------------------------------------------------------------------
# ACT Net extraction helpers
# -----------------------------------------------------------------------------

def find_entry_layer_id(net) -> int:
    """Return the id of the single INPUT layer."""
    candidates = [L.id for L in net.layers if L.kind == LayerKind.INPUT.value]
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one INPUT layer, found {len(candidates)}.")
    return candidates[0]

def get_input_ids(net) -> List[int]:
    """Return input variable IDs (out_vars of INPUT layer)."""
    entry = find_entry_layer_id(net)
    return list(net.by_id[entry].out_vars)

def get_output_ids(net) -> List[int]:
    """Return output variable IDs (in_vars of ASSERT layer)."""
    assert_layer = net.layers[-1]
    if assert_layer.kind != LayerKind.ASSERT.value:
        raise ValueError("Expected last layer to be ASSERT.")
    return list(assert_layer.in_vars)

def gather_input_spec_layers(net):
    """Return list of INPUT_SPEC layers."""
    return [L for L in net.layers if L.kind == LayerKind.INPUT_SPEC.value]

def get_assert_layer(net):
    """Return the ASSERT layer (must be last)."""
    assert_layer = net.layers[-1]
    if assert_layer.kind != LayerKind.ASSERT.value:
        raise ValueError("Expected last layer to be ASSERT.")
    return assert_layer

# -----------------------------------------------------------------------------
# Seed and input spec helpers
# -----------------------------------------------------------------------------

def seed_from_input_specs(spec_layers) -> Bounds:
    """
    Create seed Bounds from INPUT_SPEC layers.
    Prefers BOX, then LINF_BALL, raises if only LIN_POLY exists.
    
    Note: This extracts only box bounds for seeding abstract interpretation.
    All constraints (including LIN_POLY) are added via add_all_input_specs().
    """
    # BOX first
    for spec_layer in spec_layers:
        if spec_layer.params.get("kind") == InKind.BOX and "lb" in spec_layer.params and "ub" in spec_layer.params:
            return Bounds(spec_layer.params["lb"].clone(), spec_layer.params["ub"].clone())
    
    # LINF_BALL next
    for spec_layer in spec_layers:
        if spec_layer.params.get("kind") == InKind.LINF_BALL:
            if "lb" in spec_layer.params and "ub" in spec_layer.params:
                return Bounds(spec_layer.params["lb"].clone(), spec_layer.params["ub"].clone())
            center = spec_layer.params.get("center")
            eps = spec_layer.params.get("eps")
            if center is not None and eps is not None:
                e = eps.to(device=center.device, dtype=center.dtype) if torch.is_tensor(eps) else center.new_tensor(eps)
                return Bounds(center - e, center + e)

    # LP_EMBEDDING seeds the enclosing box; finite-p precision is recovered by
    # the dual input contribution, which reads p_norm/perturbed_positions.
    for spec_layer in spec_layers:
        if spec_layer.params.get("kind") == InKind.LP_EMBEDDING:
            if "lb" in spec_layer.params and "ub" in spec_layer.params:
                return Bounds(spec_layer.params["lb"].clone(), spec_layer.params["ub"].clone())
            center = spec_layer.params.get("center")
            eps = spec_layer.params.get("eps")
            if center is None or eps is None:
                raise ValueError("LP_EMBEDDING requires center/eps or lb/ub for seeding.")
            e = eps.to(device=center.device, dtype=center.dtype) if torch.is_tensor(eps) else center.new_tensor(eps)
            lb = center.clone()
            ub = center.clone()
            positions = spec_layer.params.get("perturbed_positions")
            if positions is None:
                mask = torch.ones(center.shape[:-1], device=center.device, dtype=torch.bool)
            else:
                pos_t = positions.to(device=center.device) if torch.is_tensor(positions) else torch.as_tensor(positions, device=center.device)
                if pos_t.dtype == torch.bool:
                    if tuple(pos_t.shape) == tuple(center.shape[:-1]):
                        mask = pos_t.to(dtype=torch.bool)
                    else:
                        view_shape = [1] * (center.dim() - 1)
                        view_shape[-1] = center.shape[-2]
                        mask = pos_t.reshape(view_shape).expand(center.shape[:-1]).to(dtype=torch.bool)
                else:
                    mask = torch.zeros(center.shape[:-1], device=center.device, dtype=torch.bool)
                    mask.index_fill_(-1, pos_t.to(dtype=torch.long).flatten(), True)
            expanded = mask.unsqueeze(-1).expand_as(center)
            return Bounds(torch.where(expanded, center - e, lb), torch.where(expanded, center + e, ub))
    
    # LIN_POLY only -> error
    if any(spec_layer.params.get("kind") == InKind.LIN_POLY for spec_layer in spec_layers):
        raise ValueError("LIN_POLY requires a seed box (BOX or LINF_BALL).")
    
    raise ValueError("No valid input specification found for seeding.")

def add_all_input_specs(globalC: ConSet, input_ids: List[int], spec_layers) -> None:
    """
    Add all INPUT_SPEC constraints to constraint set.
    
    This function adds:
    - BOX constraints (box bounds)
    - LINF_BALL constraints (converted to box)
    - LP_EMBEDDING/LIN_POLY constraints (box seed or linear polytope A·x ≤ b)
    
    The LIN_POLY constraints are tagged with "in:linpoly" and will be
    exported by export_to_batch_problem() in cons_exportor.py.
    """
    for L in spec_layers:
        k = L.params.get("kind")
        if k == InKind.BOX:
            globalC.add_box(-1, input_ids, Bounds(L.params["lb"], L.params["ub"]))
        elif k == InKind.LINF_BALL:
            if "lb" in L.params and "ub" in L.params:
                globalC.add_box(-1, input_ids, Bounds(L.params["lb"], L.params["ub"]))
            else:
                center = L.params["center"]
                eps = L.params["eps"]
                e = eps.to(device=center.device, dtype=center.dtype) if torch.is_tensor(eps) else center.new_tensor(eps)
                globalC.add_box(-1, input_ids, Bounds(center - e, center + e))
        elif k == InKind.LP_EMBEDDING:
            if "lb" in L.params and "ub" in L.params:
                globalC.add_box(-1, input_ids, Bounds(L.params["lb"], L.params["ub"]))
            else:
                center = L.params["center"]
                eps = L.params["eps"]
                e = eps.to(device=center.device, dtype=center.dtype) if torch.is_tensor(eps) else center.new_tensor(eps)
                globalC.add_box(-1, input_ids, Bounds(center - e, center + e))
        elif k == InKind.LIN_POLY:
            A, b = L.params["A"], L.params["b"]
            globalC.replace(Con("INEQ", tuple(input_ids), {"tag": "in:linpoly", "A": A, "b": b}))
        else:
            raise NotImplementedError(f"Unsupported INPUT_SPEC kind: {k}")




@torch.no_grad()
def setup_and_solve_batch(
    net,
    input_bounds_per_b: Bounds,
    solver: Solver,
    timelimit: Optional[float] = None,
    *,
    cache: Optional["AnalyzeCache"] = None,
) -> BatchLPSolution:
    """[BATCHED-API] Orchestrate analyze → export_to_batch_problem → solve_batch.

    ``input_bounds_per_b`` must already be a tensor-view batch
    ``[B, *input_shape]``; B=1 is just
    the length-one batch case, not a scalar special case.
    """
    from act.back_end.analyze import analyze
    from act.back_end.cons_exportor import export_to_batch_problem

    if input_bounds_per_b.lb.dim() < 2 or input_bounds_per_b.ub.dim() < 2:
        raise ValueError(
            f"setup_and_solve_batch: input_bounds_per_b must be batched "
            f"[B, *input_shape], got lb={tuple(input_bounds_per_b.lb.shape)} "
            f"ub={tuple(input_bounds_per_b.ub.shape)}"
        )

    entry_id = find_entry_layer_id(net)
    input_ids = get_input_ids(net)
    spec_layers = gather_input_spec_layers(net)
    assert_layer = get_assert_layer(net)

    entry_fact = Fact(bounds=input_bounds_per_b, cons=ConSet())
    add_all_input_specs(entry_fact.cons, input_ids, spec_layers)

    _before, after, globalC = analyze(net, entry_id, entry_fact, cache=cache)
    validate_constraints(globalC, after, net)

    problem = export_to_batch_problem(
        net=net,
        globalC=globalC,
        assert_layer=assert_layer,
        input_box_per_b=input_bounds_per_b,
    )
    solution = solver.solve_batch(problem, timelimit=timelimit)

    expected_n = int(input_bounds_per_b.lb.shape[0])
    if len(solution.statuses) != expected_n:
        raise ValueError(
            f"setup_and_solve_batch: solver returned {len(solution.statuses)} "
            f"statuses for B={expected_n}"
        )
    valid_statuses = {SolveStatus.SAT, SolveStatus.UNSAT, SolveStatus.UNKNOWN}
    unexpected = [status for status in solution.statuses if status not in valid_statuses]
    if unexpected:
        raise ValueError(
            f"setup_and_solve_batch: unexpected solver statuses {unexpected}"
        )
    if solution.max_viol.shape != (expected_n,):
        raise ValueError(
            f"setup_and_solve_batch: max_viol shape "
            f"{tuple(solution.max_viol.shape)} != ({expected_n},)"
        )
    return solution


@torch.no_grad()
def verify_lp_batched(
    net,
    solver_factory: Callable[[], Solver],
    timelimit: Optional[float] = None,
) -> List[VerifyResult]:
    """[BATCHED-API] Run one native batched LP verification pass.

    The ACT net supplies a batched INPUT_SPEC seed ``[B, *input_shape]`` and a
    batched ASSERT layer. ``setup_and_solve_batch`` solves all B LPs at once;
    this function decodes each solver lane to a ``VerifyResult`` and validates
    SAT candidates concretely before reporting FALSIFIED.
    """
    import importlib

    spec_layers = gather_input_spec_layers(net)
    seed_bounds = seed_from_input_specs(spec_layers)
    if seed_bounds.lb.dim() < 2 or seed_bounds.ub.dim() < 2:
        raise ValueError(
            f"verify_lp_batched: seed bounds must be [B, *input_shape], "
            f"got lb={tuple(seed_bounds.lb.shape)} ub={tuple(seed_bounds.ub.shape)}"
        )
    batch_size = int(seed_bounds.lb.shape[0])
    solver = solver_factory()
    solution = setup_and_solve_batch(
        net,
        Bounds(seed_bounds.lb.clone(), seed_bounds.ub.clone()),
        solver,
        timelimit=timelimit,
    )
    if len(solution.statuses) != batch_size:
        raise ValueError(
            f"verify_lp_batched: solver returned {len(solution.statuses)} "
            f"statuses for B={batch_size}"
        )
    if solution.x.dim() != 2 or solution.x.shape[0] != batch_size:
        raise ValueError(
            f"verify_lp_batched: solution.x must be [B, nvars], got "
            f"shape={tuple(solution.x.shape)} for B={batch_size}"
        )

    input_ids = get_input_ids(net)
    input_index = torch.tensor(input_ids, device=solution.x.device, dtype=torch.long)
    x_candidates = solution.x.index_select(1, input_index).reshape_as(seed_bounds.lb)
    assert_layer = get_assert_layer(net)

    sat_mask = torch.tensor(
        [status in (SolveStatus.SAT, "FEASIBLE") for status in solution.statuses],
        device=x_candidates.device,
        dtype=torch.bool,
    )
    violations = torch.zeros(batch_size, device=x_candidates.device, dtype=torch.bool)
    if bool(sat_mask.any().item()):
        bab_module = importlib.import_module("act.back_end.bab.bab")
        sat_idx = torch.where(sat_mask)[0]
        checked_sat = bab_module.check_violations_batched(
            net, x_candidates.index_select(0, sat_idx), assert_layer,
        )
        if checked_sat.shape != (int(sat_idx.numel()),):
            raise ValueError(
                f"verify_lp_batched: check_violations_batched returned "
                f"shape={tuple(checked_sat.shape)} expected ({int(sat_idx.numel())},)"
            )
        violations.scatter_(
            0, sat_idx, checked_sat.to(device=x_candidates.device, dtype=torch.bool),
        )

    results: List[VerifyResult] = []
    x_cpu = x_candidates.detach().cpu()
    max_viol_cpu = solution.max_viol.detach().cpu()
    for lane, status in enumerate(solution.statuses):
        metadata: Dict[str, Any] = {
            "lane": lane,
            "B": batch_size,
            "solver_status": status,
            "max_viol": float(max_viol_cpu[lane].item()),
        }
        if status in (SolveStatus.SAT, "FEASIBLE"):
            if bool(violations[lane].item()):
                results.append(
                    VerifyResult(
                        VerifyStatus.FALSIFIED,
                        counterexample=x_cpu[lane].clone(),
                        metadata=metadata,
                    )
                )
            else:
                metadata["validation"] = "no_verified_violation"
                results.append(VerifyResult(VerifyStatus.UNKNOWN, metadata=metadata))
        elif status in (SolveStatus.UNSAT, "INFEASIBLE"):
            results.append(VerifyResult(VerifyStatus.CERTIFIED, metadata=metadata))
        elif status == "TIMEOUT":
            results.append(VerifyResult(VerifyStatus.TIMEOUT, metadata=metadata))
        elif status == SolveStatus.UNKNOWN:
            results.append(VerifyResult(VerifyStatus.UNKNOWN, metadata=metadata))
        else:
            raise ValueError(f"verify_lp_batched: unexpected solver status {status!r}")
    return results


# -----------------------------------------------------------------------------
# Single-shot verification
# -----------------------------------------------------------------------------


def _get_output_layer_bounds(net, after: Dict[int, Fact]) -> Bounds:
    """Return the Bounds tensor produced by the network's output layer.

    The output layer is the unique predecessor of the ASSERT layer; the
    returned Bounds is shaped ``[B, n_out]``.
    """
    assert_layer = get_assert_layer(net)
    pred_ids = net.preds.get(assert_layer.id, [])
    if len(pred_ids) != 1:
        raise ValueError(
            f"ASSERT layer {assert_layer.id} must have exactly one "
            f"predecessor (the network output), got predecessors={pred_ids}"
        )
    return after[pred_ids[0]].bounds


@torch.no_grad()
def verify_once(
    net,
    *,
    model_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> List[VerifyResult]:
    """Single-shot, pure-tensor batched verifier.

    Pipeline:

      1. Seed bounds from INPUT_SPEC layers (already shaped ``[B, *input_shape]``).
      2. ``analyze`` propagates batched bounds through every layer.
      3. Read pre-encoded ``C`` / ``thresholds`` / ``M`` from the ASSERT
         layer params (encoding lives in ``OutputSpec.encode_linear`` on the
         front-end; verify_once does no kind-dispatch).
      4. INTERVAL CERTIFICATION: in one tensor pass, compute the
         per-row interval upper bound of ``C @ y`` and compare to the
         per-lane threshold; ALL of a sample's M lanes must pass for that
         sample to be CERTIFIED.
      5. CONCRETE FALSIFICATION (only if ``model_fn`` given): evaluate the
         model at the box centre; any sample where a lane's concrete
         margin meets-or-exceeds the threshold is FALSIFIED.
      6. Remaining samples are UNKNOWN.

    Args:
        net: an ACT ``Net`` whose first layer is INPUT, last layer is ASSERT,
            and whose INPUT_SPEC layers carry already-batchified
            ``[B, *input_shape]`` lb/ub.
        model_fn: optional callable mapping ``x: [B, *input_shape] ->
            [B, n_out]`` for concrete falsification. If omitted, the
            FALSIFIED status is never produced (FALSIFIED requires evidence).

    Returns:
        ``List[VerifyResult]`` of length ``B`` (one per input lane). Each
        result carries ``status`` plus a ``metadata['lane'] = i`` and any
        ``counterexample`` (a ``torch.Tensor`` of shape ``[*input_shape]``)
        for FALSIFIED lanes.
    """
    from act.back_end.analyze import analyze
    from act.back_end.transfer_functions import get_transfer_function

    # 1. Extract structure and seed.
    entry_id = find_entry_layer_id(net)
    input_ids = get_input_ids(net)
    output_ids = get_output_ids(net)
    spec_layers = gather_input_spec_layers(net)
    assert_layer = get_assert_layer(net)

    seed_bounds = seed_from_input_specs(spec_layers)
    if seed_bounds.lb.dim() < 2:
        raise ValueError(
            f"verify_once: INPUT_SPEC seed must be batched [B, *input_shape], "
            f"got dim={seed_bounds.lb.dim()} shape={tuple(seed_bounds.lb.shape)}. "
            f"Use VerifiableModel._merge_specs_to_batch (front-end) or manually "
            f"expand INPUT_SPEC lb/ub to [B, ...] before calling verify_once."
        )
    B = seed_bounds.lb.shape[0]

    # Dual standalone dispatch: when ``--solver dual`` is set (dual moved
    # dual from the --tf-mode axis to the --solver axis), route through
    # DualSolver.evaluate_spec instead of analyze() + interval cert. LP/Gurobi
    # path remains authoritative for the LP-feeding TFs (interval/hybridz).
    # ``ensure_active_tf`` still self-heals the TF default for interval/hybridz
    # callers; ``is_dual_solver_active`` reads the orthogonal solver-mode global.
    from act.back_end.transfer_functions import ensure_active_tf, is_dual_solver_active
    active_tf = ensure_active_tf("interval")

    if is_dual_solver_active():
        from act.back_end.solver.solver_dual import DualSolver
        from act.front_end.specs import OutputSpec

        def _unbatch(val: Any) -> Any:
            # ASSERT params are pre-batchified ([B, ...]) by FE; OutputSpec
            # constructor expects unbroadcasted scalar/1-D form. Single-property
            # batch verification: all rows share the same spec, so row 0 is the
            # canonical form. Per-sample-varying spec support is a future task.
            if isinstance(val, torch.Tensor) and val.dim() >= 1 and val.shape[0] == B:
                return val[0]
            return val

        out_spec = OutputSpec(
            kind=assert_layer.params.get("kind"),
            c=_unbatch(assert_layer.params.get("c")),
            d=_unbatch(assert_layer.params.get("d")),
            y_true=assert_layer.params.get("y_true"),
            margin=_unbatch(assert_layer.params.get("margin")),
            lb=_unbatch(assert_layer.params.get("lb")),
            ub=_unbatch(assert_layer.params.get("ub")),
        )
        num_classes = len(output_ids)
        # DualSolver is now self-contained: no tf parameter, evaluate_spec
        # computes its own forward bounds internally from the net.
        result = DualSolver().evaluate_spec(net, out_spec, num_classes=num_classes)
        return result.to_verify_results()

    # 2. Build entry_fact (with all INPUT_SPEC constraints) and analyze.
    entry_fact = Fact(bounds=seed_bounds, cons=ConSet())
    add_all_input_specs(entry_fact.cons, input_ids, spec_layers)
    _before, after, _globalC = analyze(net, entry_id, entry_fact)

    # 3. Pull output bounds (pre-ASSERT layer's Fact).
    output_bounds = _get_output_layer_bounds(net, after)
    output_lb = output_bounds.lb
    output_ub = output_bounds.ub
    if output_lb.dim() != 2 or output_lb.shape[0] != B:
        raise ValueError(
            f"verify_once: output bounds must be [B={B}, n_out], got "
            f"shape={tuple(output_lb.shape)}. Some TF op on this network's "
            f"path collapsed the leading batch dimension."
        )
    n_out = output_lb.shape[1]
    if n_out != len(output_ids):
        raise ValueError(
            f"verify_once: output_lb has n_out={n_out} but ASSERT.in_vars "
            f"has length {len(output_ids)}"
        )
    device = output_lb.device
    dtype = output_lb.dtype

    # 4. Read pre-encoded ASSERT params (produced by OutputSpec.encode_linear
    # at FE construction time). Dispatch on ``kind`` because UNSAFE_LINEAR
    # has EXISTS-row safety semantics while the four other kinds (LINEAR_LE,
    # TOP1_ROBUST, MARGIN_ROBUST, RANGE) share an ALL-rows form.
    C = assert_layer.params["C"].to(device=device, dtype=dtype)
    thresholds = assert_layer.params["thresholds"].to(device=device, dtype=dtype)
    M = int(assert_layer.params["M"])
    kind = assert_layer.params.get("kind")
    is_unsafe_linear = kind == OutKind.UNSAFE_LINEAR
    assert C.dim() == 2 and C.shape == (B * M, n_out), (
        f"verify_once: ASSERT params['C'].shape={tuple(C.shape)} "
        f"expected ({B * M}, {n_out})"
    )
    assert thresholds.shape == (B, M), (
        f"verify_once: ASSERT params['thresholds'].shape="
        f"{tuple(thresholds.shape)} expected ({B}, {M})"
    )

    C_pos = C.clamp(min=0)
    C_neg = C.clamp(max=0)
    lb_exp = output_lb.repeat_interleave(M, dim=0)
    ub_exp = output_ub.repeat_interleave(M, dim=0)

    if is_unsafe_linear:
        # UNSAFE polytope = {y : C y <= d}. Property is SAFE iff for all y in
        # the box, EXISTS row i with c_i @ y > d_i (i.e. y leaves the polytope
        # on row i). Sound under-approximation: EXISTS row i such that
        # min_{y in box} (c_i @ y) > d_i. min(c_i @ y) = c_i_pos @ lb + c_i_neg @ ub.
        margin_min = (C_pos * lb_exp + C_neg * ub_exp).sum(dim=-1)
        certified = (margin_min.view(B, M) > thresholds).any(dim=-1)
    else:
        # LINEAR_LE / TOP1_ROBUST / MARGIN_ROBUST / RANGE: certified iff for
        # all y in the box, ALL rows max_y (c_i @ y) < d_i.
        margin_max = (C_pos * ub_exp + C_neg * lb_exp).sum(dim=-1)
        certified = (margin_max.view(B, M) < thresholds).all(dim=-1)

    # 5. Concrete falsification (optional).
    falsified = torch.zeros(B, dtype=torch.bool, device=device)
    counterexamples: List[Optional[torch.Tensor]] = [None] * B
    if model_fn is not None:
        x_center = 0.5 * (seed_bounds.lb + seed_bounds.ub)
        y_concrete = model_fn(x_center)
        if y_concrete.dim() != 2 or y_concrete.shape != (B, n_out):
            raise ValueError(
                f"verify_once: model_fn returned shape "
                f"{tuple(y_concrete.shape)}, expected ({B}, {n_out})"
            )
        y_concrete = y_concrete.to(device=device, dtype=dtype)
        C_view = C.view(B, M, n_out)
        concrete_violation = torch.einsum("bmn,bn->bm", C_view, y_concrete)
        if is_unsafe_linear:
            # Concrete y is in the UNSAFE polytope iff ALL rows c_i @ y <= d_i;
            # that is the violation condition for UNSAFE_LINEAR.
            falsified = (~certified) & (
                (concrete_violation <= thresholds).all(dim=-1)
            )
        else:
            # ALL-rows kinds: FALSIFIED iff ANY lane's concrete margin
            # meets-or-exceeds threshold.
            falsified = (~certified) & (
                (concrete_violation >= thresholds).any(dim=-1)
            )
        if falsified.any():
            x_center_cpu = x_center.detach().cpu()
            # B1 (oracle-verified): single sync via .tolist() replaces B per-element .item() syncs.
            # torch.where returns ascending indices; lane order is preserved.
            for i in torch.where(falsified)[0].tolist():
                counterexamples[i] = x_center_cpu[i].clone()

    # 6. Assemble per-lane results.
    results: List[VerifyResult] = []
    cert_list = certified.tolist()
    fals_list = falsified.tolist()
    for i in range(B):
        meta: Dict[str, Any] = {"lane": i, "B": B, "M": M}
        if cert_list[i]:
            results.append(
                VerifyResult(VerifyStatus.CERTIFIED, metadata=meta)
            )
        elif fals_list[i]:
            results.append(
                VerifyResult(
                    VerifyStatus.FALSIFIED,
                    counterexample=counterexamples[i],
                    metadata=meta,
                )
            )
        else:
            results.append(
                VerifyResult(VerifyStatus.UNKNOWN, metadata=meta)
            )
    return results


#===---------------------------------------------------------------------===#
# Self-contained ASSERT-encoding + verify_once test battery.
# Run via: python -m act.back_end.verifier
#===---------------------------------------------------------------------===#


def _test_build_top1_robust_drops_y_true_row() -> None:  # pragma: no cover
    # Encoding is row-deletion, not masking: every row is e_j - e_{y_true}
    # for j != y_true, hence M = K-1 and Frobenius row norm = sqrt(2).
    from act.front_end.specs import OutputSpec, OutKind

    K = 5
    out = OutputSpec(
        kind=OutKind.TOP1_ROBUST, y_true=torch.tensor([0, 2, 4])
    ).encode_linear(
        B=3, n_out=K, device=torch.device("cpu"), dtype=torch.float32,
    )
    assert out["M"] == K - 1, f"expected M=K-1=4, got {out['M']}"
    assert out["C"].shape == (3 * (K - 1), K), (
        f"expected C.shape == (B*M, K) == (12, 5), got "
        f"{tuple(out['C'].shape)}"
    )
    row_norms = out["C"].norm(dim=1)
    assert (row_norms > 0).all(), (
        f"found a zero row in C (y_true row was masked, not dropped): "
        f"norms={row_norms.tolist()}"
    )
    expected_norm = torch.full_like(row_norms, 2.0).sqrt()
    assert torch.allclose(row_norms, expected_norm), (
        f"every row should be e_j - e_{{y_true}} with ||.||=sqrt(2); "
        f"got norms={row_norms.tolist()}"
    )


def _test_build_linear_le_threshold_is_d_unchanged() -> None:  # pragma: no cover
    from act.front_end.specs import OutputSpec, OutKind

    out = OutputSpec(
        kind=OutKind.LINEAR_LE,
        c=torch.tensor([1.0, -1.0]),
        d=torch.tensor(0.5),
    ).encode_linear(
        B=3, n_out=2, device=torch.device("cpu"), dtype=torch.float32,
    )
    assert out["M"] == 1
    assert tuple(out["C"].shape) == (3, 2)
    assert tuple(out["thresholds"].shape) == (3, 1)
    assert torch.allclose(
        out["thresholds"],
        torch.full((3, 1), 0.5, dtype=torch.float32),
    ), f"thresholds mismatch: {out['thresholds'].tolist()}"


def _test_build_margin_robust_threshold_is_negated_margin() -> None:  # pragma: no cover
    from act.front_end.specs import OutputSpec, OutKind

    out = OutputSpec(
        kind=OutKind.MARGIN_ROBUST,
        y_true=torch.tensor([1]),
        margin=torch.tensor(0.1),
    ).encode_linear(
        B=1, n_out=4, device=torch.device("cpu"), dtype=torch.float32,
    )
    assert out["M"] == 3
    expected = torch.full((1, 3), -0.1, dtype=torch.float32)
    assert torch.allclose(out["thresholds"], expected), (
        f"thresholds should be -margin; got {out['thresholds'].tolist()}"
    )


def _test_build_range_interleaves_pm_e_rows() -> None:  # pragma: no cover
    from act.front_end.specs import OutputSpec, OutKind

    out = OutputSpec(
        kind=OutKind.RANGE,
        lb=torch.tensor([-1.0, -1.0, -1.0]),
        ub=torch.tensor([1.0, 1.0, 1.0]),
    ).encode_linear(
        B=2, n_out=3, device=torch.device("cpu"), dtype=torch.float32,
    )
    assert out["M"] == 6
    expected = torch.tensor(
        [
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    C_per_sample = out["C"].view(2, 6, 3)
    for b in range(2):
        assert torch.allclose(C_per_sample[b], expected), (
            f"sample {b}: rows mismatch.\n got={C_per_sample[b].tolist()}\n"
            f" want={expected.tolist()}"
        )


def _test_interval_margin_certification_shape() -> None:  # pragma: no cover
    # margin_max = sum_k (max(C_k,0)*ub_k + min(C_k,0)*lb_k) over [B*M, n_out];
    # per-sample CERTIFIED iff every M lane satisfies margin_max < threshold.
    from act.front_end.specs import OutputSpec, OutKind

    B, n_out = 2, 5
    output_lb = torch.zeros(B, n_out)
    output_ub = torch.ones(B, n_out)
    out = OutputSpec(
        kind=OutKind.TOP1_ROBUST, y_true=torch.tensor([0, 1])
    ).encode_linear(
        B=B, n_out=n_out, device=torch.device("cpu"), dtype=torch.float32,
    )
    M = out["M"]
    assert M == 4

    C = out["C"]
    C_pos = C.clamp(min=0)
    C_neg = C.clamp(max=0)
    lb_exp = output_lb.repeat_interleave(M, dim=0)
    ub_exp = output_ub.repeat_interleave(M, dim=0)
    margin_max = (C_pos * ub_exp + C_neg * lb_exp).sum(dim=-1)
    assert tuple(margin_max.shape) == (B * M,), (
        f"margin_max shape {tuple(margin_max.shape)} != (B*M,) == ({B * M},)"
    )

    cert_per_sample = (
        margin_max.view(B, M) < out["thresholds"]
    ).all(dim=-1)
    assert tuple(cert_per_sample.shape) == (B,), (
        f"per-sample cert shape {tuple(cert_per_sample.shape)} != ({B},)"
    )


def _make_dense_net_box_test(  # pragma: no cover
    B: int,
    n_in: int,
    n_out: int,
    weight: torch.Tensor,
    bias: torch.Tensor,
    lb_in: torch.Tensor,
    ub_in: torch.Tensor,
    assert_params: Dict[str, Any],
):
    # assert_params is high-level (kind + y_true/margin/c/d/lb/ub); lift to
    # encoded form via OutputSpec.encode_linear to match the production
    # OutputSpecLayer.to_act_layers path.
    from act.back_end.core import Layer, Net
    from act.front_end.specs import OutputSpec

    in_v = list(range(n_in))
    out_v = list(range(n_in, n_in + n_out))

    spec_kwargs = {
        k: assert_params[k] for k in ("y_true", "margin", "c", "d", "lb", "ub")
        if k in assert_params
    }
    out_spec = OutputSpec(kind=assert_params["kind"], **spec_kwargs)
    encoded = out_spec.encode_linear(
        B=B, n_out=n_out, device=weight.device, dtype=weight.dtype,
    )

    layers = [
        Layer(
            id=0,
            kind=LayerKind.INPUT.value,
            params={"shape": (B, n_in), "dtype": str(weight.dtype)},
            in_vars=[],
            out_vars=in_v,
        ),
        Layer(
            id=1,
            kind=LayerKind.INPUT_SPEC.value,
            params={"kind": "BOX", "lb": lb_in, "ub": ub_in},
            in_vars=in_v,
            out_vars=in_v,
        ),
        Layer(
            id=2,
            kind=LayerKind.DENSE.value,
            params={
                "weight": weight,
                "in_features": n_in,
                "out_features": n_out,
                "weight_pos": weight.clamp(min=0),
                "weight_neg": weight.clamp(max=0),
                "bias": bias,
                "input_shape": (n_in,),
            },
            in_vars=in_v,
            out_vars=out_v,
        ),
        Layer(
            id=3,
            kind=LayerKind.ASSERT.value,
            params=encoded,
            in_vars=out_v,
            out_vars=out_v,
        ),
    ]
    preds = {0: [], 1: [0], 2: [1], 3: [2]}
    succs = {0: [1], 1: [2], 2: [3], 3: []}
    return Net(layers=layers, preds=preds, succs=succs)


def _make_attn_dual_planar_net(  # pragma: no cover
    B: int, L: int, D: int, H: int,
    center: torch.Tensor, eps: float,
    assert_d: float,
    *,
    mask: "torch.Tensor | None" = None,
    clamp_alpha: bool = False,
) -> "tuple[Net, dict[str, Any]]":
    """Build INPUT -> Q/K DENSE projections -> ATT_SCORES(dual_planar) -> ASSERT.

    Exercises the real ``analyze()`` -> ``tf_att_scores`` ->
    ``att_scores_dual_planar``/``LinearBounds`` -> ``cons_exportor``'s
    ``att_dual_planar:`` export path end-to-end, not direct unit calls into
    ``interval_tf/tf_attention.py``. The ``q_lb``/``k_lb`` baked onto the
    ATT_SCORES layer are seeded from the same box as INPUT_SPEC and pushed
    through the same ``Wq``/``Wk`` as the DENSE Q/K layers, so the result is
    a faithful (not synthetic) attention-score relaxation of this network.
    """
    from act.back_end.core import Layer
    from act.back_end.interval_tf.tf_attention import LinearBounds
    from act.front_end.specs import OutputSpec

    n_in = L * D
    in_v = list(range(n_in))
    lb_in = center - eps
    ub_in = center + eps

    Wq = torch.randn(H, D, dtype=center.dtype, generator=torch.Generator().manual_seed(11)) * 0.3
    Wk = torch.randn(H, D, dtype=center.dtype, generator=torch.Generator().manual_seed(12)) * 0.3

    eye = torch.eye(D, dtype=center.dtype)
    center3 = center.reshape(B, L, D)
    radius3 = torch.full((B, L, D), eps, dtype=center.dtype)
    seed_w = radius3.unsqueeze(-1) * eye
    emb_lb = LinearBounds(
        seed_w, seed_w.clone(), center3.clone(), center3.clone(),
        p=float("inf"), eps=1.0, perturbed_words=1,
    )
    q_lb = emb_lb.matmul(Wq)
    k_lb = emb_lb.matmul(Wk)

    q_vars = list(range(n_in, n_in + L * H))
    k_vars = list(range(n_in + L * H, n_in + 2 * L * H))
    score_vars = list(range(n_in + 2 * L * H, n_in + 2 * L * H + L * L))

    def block_diag_proj(W: torch.Tensor) -> torch.Tensor:
        full = torch.zeros(L * H, n_in, dtype=center.dtype)
        for t in range(L):
            full[t * H:(t + 1) * H, t * D:(t + 1) * D] = W
        return full

    Wq_full, Wk_full = block_diag_proj(Wq), block_diag_proj(Wk)

    layers = [
        Layer(
            id=0, kind=LayerKind.INPUT.value,
            params={"shape": (B, n_in), "dtype": str(center.dtype)},
            in_vars=[], out_vars=in_v,
        ),
        Layer(
            id=1, kind=LayerKind.INPUT_SPEC.value,
            params={"kind": "BOX", "lb": lb_in, "ub": ub_in},
            in_vars=in_v, out_vars=in_v,
        ),
        Layer(
            id=2, kind=LayerKind.DENSE.value,
            params={
                "weight": Wq_full, "in_features": n_in, "out_features": L * H,
                "weight_pos": Wq_full.clamp(min=0), "weight_neg": Wq_full.clamp(max=0),
                "bias": torch.zeros(L * H, dtype=center.dtype), "input_shape": (n_in,),
            },
            in_vars=in_v, out_vars=q_vars,
        ),
        Layer(
            id=3, kind=LayerKind.DENSE.value,
            params={
                "weight": Wk_full, "in_features": n_in, "out_features": L * H,
                "weight_pos": Wk_full.clamp(min=0), "weight_neg": Wk_full.clamp(max=0),
                "bias": torch.zeros(L * H, dtype=center.dtype), "input_shape": (n_in,),
            },
            in_vars=in_v, out_vars=k_vars,
        ),
        Layer(
            id=4, kind=LayerKind.ATT_SCORES.value,
            params={
                "dk": float(H) ** 0.5, "q_vars": tuple(q_vars), "k_vars": tuple(k_vars),
                "q_src": 2, "k_src": 3,
                "attn_mode": "dual_planar", "q_lb": q_lb, "k_lb": k_lb, "head_size": H,
                "mask": mask, "clamp_alpha": clamp_alpha,
            },
            in_vars=q_vars + k_vars, out_vars=score_vars,
        ),
    ]
    n_scores = len(score_vars)
    out_spec = OutputSpec(
        kind="LINEAR_LE", c=torch.ones(n_scores, dtype=center.dtype),
        d=torch.tensor(assert_d, dtype=center.dtype),
    )
    encoded = out_spec.encode_linear(B=B, n_out=n_scores, device=center.device, dtype=center.dtype)
    layers.append(
        Layer(id=5, kind=LayerKind.ASSERT.value, params=encoded, in_vars=score_vars, out_vars=score_vars)
    )

    preds = {0: [], 1: [0], 2: [1], 3: [1], 4: [2, 3], 5: [4]}
    succs = {0: [1], 1: [2, 3], 2: [4], 3: [4], 4: [5], 5: []}
    net = Net(layers=layers, preds=preds, succs=succs)
    info: "dict[str, Any]" = {
        "Wq": Wq, "Wk": Wk, "lb_in": lb_in, "ub_in": ub_in,
        "score_id": 4, "n_in": n_in, "L": L, "D": D, "H": H,
    }
    return net, info


def _test_att_scores_dual_planar_analyze_soundness() -> None:  # pragma: no cover
    # Real `analyze()` worklist (not a direct LinearBounds unit call): the
    # propagated box for the ATT_SCORES(dual_planar) layer must bracket the
    # true concrete scaled-Q.K^T value for every sampled point in the box.
    from act.back_end.analyze import analyze
    from act.util.device_manager import get_default_dtype

    dtype = get_default_dtype()
    B, L, D, H = 1, 3, 4, 2
    torch.manual_seed(20)
    center = torch.randn(B, L * D, dtype=dtype) * 0.1
    eps = 0.05
    net, info = _make_attn_dual_planar_net(B, L, D, H, center, eps, assert_d=100.0)

    entry_fact = Fact(bounds=Bounds(info["lb_in"].clone(), info["ub_in"].clone()), cons=ConSet())
    _before, after, _globalC = analyze(net, 0, entry_fact)
    bounds = after[info["score_id"]].bounds

    Wq, Wk = info["Wq"], info["Wk"]
    l_box, u_box = info["lb_in"], info["ub_in"]
    n_samples = 100

    def concrete_scores(x: torch.Tensor) -> torch.Tensor:
        x3 = x.reshape(B, L, D)
        s = (x3 @ Wq.t()) @ (x3 @ Wk.t()).transpose(-1, -2) / (H ** 0.5)
        return s.reshape(B, -1)

    true_min = concrete_scores(l_box).clone()
    true_max = true_min.clone()
    for _ in range(n_samples):
        x = l_box + torch.rand_like(l_box) * (u_box - l_box)
        s = concrete_scores(x)
        true_min = torch.minimum(true_min, s)
        true_max = torch.maximum(true_max, s)
    assert (bounds.lb <= true_min + 1e-6).all(), "analyze(): unsound lower bound on ATT_SCORES(dual_planar)"
    assert (bounds.ub >= true_max - 1e-6).all(), "analyze(): unsound upper bound on ATT_SCORES(dual_planar)"


def _test_att_scores_dual_planar_verify_once_certified() -> None:  # pragma: no cover
    # End-to-end `verify_once()` through the dual-planar attention path with
    # a threshold far above the true score range -> CERTIFIED.
    from act.util.device_manager import get_default_dtype
    from act.util.stats import VerifyStatus

    dtype = get_default_dtype()
    B, L, D, H = 1, 3, 4, 2
    torch.manual_seed(21)
    center = torch.randn(B, L * D, dtype=dtype) * 0.1
    eps = 0.05
    net, _info = _make_attn_dual_planar_net(B, L, D, H, center, eps, assert_d=100.0)

    results = verify_once(net)
    assert len(results) == B
    assert results[0].status == VerifyStatus.CERTIFIED, f"expected CERTIFIED, got {results[0].status}"


def _test_att_scores_dual_planar_lp_export_solve() -> None:  # pragma: no cover
    # End-to-end LP export+solve through `cons_exportor`'s
    # `att_dual_planar:` handler (not reachable from any other test): a
    # tight threshold near the true score range exercises a real SAT/UNKNOWN
    # decision from TorchLPSolver, proving the export glue round-trips.
    from act.back_end.solver.solver_torchlp import TorchLPSolver
    from act.util.device_manager import get_default_dtype

    dtype = get_default_dtype()
    B, L, D, H = 1, 3, 4, 2
    torch.manual_seed(22)
    center = torch.randn(B, L * D, dtype=dtype) * 0.1
    eps = 0.05
    net, info = _make_attn_dual_planar_net(B, L, D, H, center, eps, assert_d=0.0)

    solution = setup_and_solve_batch(
        net, Bounds(info["lb_in"].clone(), info["ub_in"].clone()), TorchLPSolver(),
    )
    assert solution.statuses[0] in (SolveStatus.SAT, SolveStatus.UNKNOWN), (
        f"unexpected solver status {solution.statuses[0]!r}"
    )
    assert tuple(solution.x.shape)[0] == B
    assert float(solution.max_viol[0].item()) < 1.0, (
        f"LP residual too large: {float(solution.max_viol[0].item())}"
    )


def _test_att_scores_dual_planar_masked_and_clamp_alpha_soundness() -> None:  # pragma: no cover
    # Real `analyze()` with an additive mask and the clamp_alpha warm-start
    # variant both engaged -- exercises `fuse_attention_planes`'s
    # `clamp_alpha` branch and `att_scores_dual_planar`'s `mask is not None`
    # branch, neither hit by the unmasked/default tests above.
    from act.back_end.analyze import analyze
    from act.util.device_manager import get_default_dtype

    dtype = get_default_dtype()
    B, L, D, H = 1, 3, 4, 2
    torch.manual_seed(23)
    center = torch.randn(B, L * D, dtype=dtype) * 0.1
    eps = 0.05
    mask = torch.zeros(B, L, L, dtype=dtype)
    mask[0, 0, 1] = -5.0
    net, info = _make_attn_dual_planar_net(
        B, L, D, H, center, eps, assert_d=100.0, mask=mask, clamp_alpha=True,
    )

    entry_fact = Fact(bounds=Bounds(info["lb_in"].clone(), info["ub_in"].clone()), cons=ConSet())
    _before, after, _globalC = analyze(net, 0, entry_fact)
    bounds = after[info["score_id"]].bounds

    Wq, Wk = info["Wq"], info["Wk"]
    l_box, u_box = info["lb_in"], info["ub_in"]
    n_samples = 100

    def concrete_masked_scores(x: torch.Tensor) -> torch.Tensor:
        x3 = x.reshape(B, L, D)
        s = (x3 @ Wq.t()) @ (x3 @ Wk.t()).transpose(-1, -2) / (H ** 0.5)
        return (s + mask).reshape(B, -1)

    true_min = concrete_masked_scores(l_box).clone()
    true_max = true_min.clone()
    for _ in range(n_samples):
        x = l_box + torch.rand_like(l_box) * (u_box - l_box)
        s = concrete_masked_scores(x)
        true_min = torch.minimum(true_min, s)
        true_max = torch.maximum(true_max, s)
    assert (bounds.lb <= true_min + 1e-6).all(), "masked/clamp_alpha: unsound lower bound"
    assert (bounds.ub >= true_max - 1e-6).all(), "masked/clamp_alpha: unsound upper bound"


def _make_mini_transformer_block_net(  # pragma: no cover
    B: int, L: int, D: int, center: torch.Tensor, eps: float,
) -> "tuple[Net, dict[str, Any]]":
    """Build a real explicit-attention block: MHA_SPLIT(Q/K/V) -> ATT_SCORES
    (plain McCormick box mode, not dual_planar) -> CONCAT -> SOFTMAX ->
    ATT_MIX -> MHA_JOIN -> LAYERNORM(variant='no_var', broadcast gamma).

    Mirrors the per-position/per-feature decomposition torch2act's BERT
    graph builder uses (one MHA_SPLIT per query/key position, one ATT_MIX
    per value feature), at the smallest size (L=2 positions) that still
    requires the CONCAT-of-two-scores -> SOFTMAX -> two-feature ATT_MIX/
    MHA_JOIN path. None of these layer kinds have any other producer in
    the codebase (no NetFactory family, no torch2act path on this branch),
    so this is the only real (non-direct-unit-call) exercise of them.
    """
    from act.back_end.core import Layer
    from act.front_end.specs import OutputSpec

    n_in = L * D
    in_v = list(range(n_in))
    lb_in = center - eps
    ub_in = center + eps

    gen = torch.Generator().manual_seed(40)
    Wq = torch.randn(D, D, dtype=center.dtype, generator=gen) * 0.3
    Wk = torch.randn(D, D, dtype=center.dtype, generator=torch.Generator().manual_seed(41)) * 0.3
    Wv = torch.randn(D, D, dtype=center.dtype, generator=torch.Generator().manual_seed(42)) * 0.3

    layers = [
        Layer(id=0, kind=LayerKind.INPUT.value, params={"shape": (B, n_in), "dtype": str(center.dtype)}, in_vars=[], out_vars=in_v),
        Layer(id=1, kind=LayerKind.INPUT_SPEC.value, params={"kind": "BOX", "lb": lb_in, "ub": ub_in}, in_vars=in_v, out_vars=in_v),
    ]
    preds: "dict[int, list[int]]" = {0: [], 1: [0]}
    succs: "dict[int, list[int]]" = {0: [1], 1: []}
    next_id = 2
    next_var = n_in

    def alloc(n: int) -> "list[int]":
        nonlocal next_var
        v = list(range(next_var, next_var + n))
        next_var += n
        return v

    def add_layer(kind: str, params: "dict[str, Any]", in_vars: "list[int]", out_vars: "list[int]", pred_ids: "list[int]") -> int:
        nonlocal next_id
        layers.append(Layer(id=next_id, kind=kind, params=params, in_vars=in_vars, out_vars=out_vars))
        lid = next_id
        next_id += 1
        preds[lid] = pred_ids
        succs.setdefault(lid, [])
        for p in pred_ids:
            succs[p].append(lid)
        return lid

    mha_split = lambda W, role, **extra: add_layer(  # noqa: E731 - local convenience, not module API
        LayerKind.MHA_SPLIT.value,
        {"weight": W, "input_shape": (B, L, D), "hidden_size": D, "role": role, **extra},
        in_v, alloc(D if role != "value" else L), [1],
    )

    q_id = mha_split(Wq, "query", position=0)
    q_vars = layers[q_id].out_vars
    k_ids = [mha_split(Wk, "key", position=p) for p in range(L)]
    k_vars_per_pos = [layers[kid].out_vars for kid in k_ids]

    score_ids = []
    score_vars_flat: "list[int]" = []
    for kid, kv in zip(k_ids, k_vars_per_pos):
        sv = alloc(1)
        sid = add_layer(
            LayerKind.ATT_SCORES.value,
            {"dk": float(D) ** 0.5, "q_vars": q_vars, "k_vars": kv, "q_src": q_id, "k_src": kid},
            q_vars + kv, sv, [q_id, kid],
        )
        score_ids.append(sid)
        score_vars_flat += sv
    cat_vars = alloc(L)
    cat_id = add_layer(LayerKind.CONCAT.value, {"concat_dim": -1}, score_vars_flat, cat_vars, score_ids)
    sm_vars = alloc(L)
    sm_id = add_layer(LayerKind.SOFTMAX.value, {"axis": -1}, cat_vars, sm_vars, [cat_id])

    v_ids = [mha_split(Wv, "value", feature=f) for f in range(D)]
    v_vars_per_feature = [layers[vid].out_vars for vid in v_ids]

    mix_ids = []
    mix_vars_flat: "list[int]" = []
    for vid, vv in zip(v_ids, v_vars_per_feature):
        mv = alloc(1)
        mid = add_layer(
            LayerKind.ATT_MIX.value,
            {"rowsize": L, "w_vars": sm_vars, "v_vars": vv, "w_src": sm_id, "v_src": vid},
            sm_vars + vv, mv, [sm_id, vid],
        )
        mix_ids.append(mid)
        mix_vars_flat += mv
    join_vars = alloc(D)
    join_id = add_layer(LayerKind.MHA_JOIN.value, {}, mix_vars_flat, join_vars, mix_ids)

    # gamma.numel()==1 != D forces the broadcast-repeat branch.
    gamma = torch.tensor([1.5], dtype=center.dtype)
    beta = torch.tensor([0.1], dtype=center.dtype)
    ln_vars = alloc(D)
    ln_id = add_layer(
        LayerKind.LAYERNORM.value, {"gamma": gamma, "beta": beta, "variant": "no_var"},
        join_vars, ln_vars, [join_id],
    )

    out_spec = OutputSpec(kind="LINEAR_LE", c=torch.ones(D, dtype=center.dtype), d=torch.tensor(100.0, dtype=center.dtype))
    encoded = out_spec.encode_linear(B=B, n_out=D, device=center.device, dtype=center.dtype)
    assert_id = add_layer(LayerKind.ASSERT.value, encoded, ln_vars, ln_vars, [ln_id])

    net = Net(layers=layers, preds=preds, succs=succs)
    info: "dict[str, Any]" = {
        "Wq": Wq, "Wk": Wk, "Wv": Wv, "gamma": gamma, "beta": beta,
        "lb_in": lb_in, "ub_in": ub_in, "ln_id": ln_id, "n_in": n_in,
    }
    return net, info


def _test_mini_transformer_block_analyze_soundness() -> None:  # pragma: no cover
    # Real `analyze()` through MHA_SPLIT -> ATT_SCORES(box) -> SOFTMAX ->
    # ATT_MIX -> MHA_JOIN -> LAYERNORM(no_var, broadcast gamma): the
    # propagated box must bracket the true concrete forward pass.
    from act.back_end.analyze import analyze
    from act.util.device_manager import get_default_dtype

    dtype = get_default_dtype()
    B, L, D = 1, 2, 2
    torch.manual_seed(43)
    center = torch.randn(B, L * D, dtype=dtype) * 0.1
    eps = 0.05
    net, info = _make_mini_transformer_block_net(B, L, D, center, eps)

    entry_fact = Fact(bounds=Bounds(info["lb_in"].clone(), info["ub_in"].clone()), cons=ConSet())
    _before, after, _globalC = analyze(net, 0, entry_fact)
    bounds = after[info["ln_id"]].bounds

    Wq, Wk, Wv = info["Wq"], info["Wk"], info["Wv"]
    gamma, beta = info["gamma"], info["beta"]
    l_box, u_box = info["lb_in"], info["ub_in"]

    def concrete_forward(x: torch.Tensor) -> torch.Tensor:
        x3 = x.reshape(B, L, D)
        q = (x3 @ Wq.t())[:, 0, :]
        scores = torch.cat(
            [(q * (x3 @ Wk.t())[:, p, :]).sum(-1, keepdim=True) / (D ** 0.5) for p in range(L)], dim=-1,
        )
        probs = torch.softmax(scores, dim=-1)
        v_all = x3 @ Wv.t()
        mixed = torch.cat([(probs * v_all[:, :, f]).sum(-1, keepdim=True) for f in range(D)], dim=-1)
        centered = mixed - mixed.mean(dim=-1, keepdim=True)
        return centered * gamma.repeat(D) + beta.repeat(D)

    n_samples = 150
    true_min = concrete_forward(l_box).clone()
    true_max = true_min.clone()
    for _ in range(n_samples):
        x = l_box + torch.rand_like(l_box) * (u_box - l_box)
        y = concrete_forward(x)
        true_min = torch.minimum(true_min, y)
        true_max = torch.maximum(true_max, y)
    assert (bounds.lb <= true_min + 1e-6).all(), "mini transformer block: unsound lower bound"
    assert (bounds.ub >= true_max - 1e-6).all(), "mini transformer block: unsound upper bound"


def _test_mha_split_edge_cases_and_mask_add() -> None:  # pragma: no cover
    # Direct calls to the production transfer functions for the branches
    # the full-block Net above can't reach structurally: MHA_SPLIT with no
    # "weight" param (passthrough), MHA_SPLIT with no "role" (flatten), and
    # MASK_ADD (an unrelated single-layer op with no other test coverage).
    from act.back_end.core import Layer
    from act.back_end.interval_tf.tf_transformer import tf_mha_split, tf_mask_add
    from act.util.device_manager import get_default_dtype

    dtype = get_default_dtype()
    Bin = Bounds(torch.tensor([[-1.0, 2.0]], dtype=dtype), torch.tensor([[1.0, 3.0]], dtype=dtype))

    passthrough = tf_mha_split(Layer(id=0, kind=LayerKind.MHA_SPLIT.value, params={}, in_vars=[0, 1], out_vars=[0, 1]), Bin)
    assert torch.equal(passthrough.bounds.lb, Bin.lb) and torch.equal(passthrough.bounds.ub, Bin.ub), (
        "MHA_SPLIT with no weight must passthrough Bin unchanged"
    )

    W = torch.eye(2, dtype=dtype)
    flat = tf_mha_split(
        Layer(
            id=1, kind=LayerKind.MHA_SPLIT.value,
            params={"weight": W, "input_shape": (1, 1, 2), "hidden_size": 2}, in_vars=[0, 1], out_vars=[0, 1],
        ),
        Bin,
    )
    assert flat.bounds.lb.shape == (1, 2) and flat.bounds.ub.shape == (1, 2), "MHA_SPLIT flatten-role output shape"

    M = torch.tensor([[0.5, -0.5]], dtype=dtype)
    masked = tf_mask_add(Layer(id=2, kind=LayerKind.MASK_ADD.value, params={"M": M}, in_vars=[0, 1], out_vars=[0, 1]), Bin)
    assert torch.allclose(masked.bounds.lb, Bin.lb + M) and torch.allclose(masked.bounds.ub, Bin.ub + M), (
        "MASK_ADD must shift both bounds by M"
    )


def _test_new_elementwise_tf_soundness() -> None:  # pragma: no cover
    # Direct calls to the 5 new interval_tf/tf_mlp.py transfer functions
    # (ERF, SQRT, SIN, COS, QUANTIZE) -- no NetFactory family or other
    # producer generates these layer kinds, so this is their only exercise.
    # Each assertion samples the true concrete function over the box and
    # checks the propagated interval brackets it.
    from act.back_end.core import Layer
    from act.back_end.interval_tf.tf_mlp import tf_erf, tf_sqrt, tf_sin, tf_cos, tf_quantize
    from act.util.device_manager import get_default_dtype

    dtype = get_default_dtype()

    def assert_sound(name: str, lo: torch.Tensor, hi: torch.Tensor, l_box: torch.Tensor, u_box: torch.Tensor, fn, n: int = 150) -> None:
        true_min = fn(l_box).clone()
        true_max = true_min.clone()
        for _ in range(n):
            x = l_box + torch.rand_like(l_box) * (u_box - l_box)
            y = fn(x)
            true_min = torch.minimum(true_min, y)
            true_max = torch.maximum(true_max, y)
        assert (lo <= true_min + 1e-6).all(), f"{name}: unsound lower bound"
        assert (hi >= true_max - 1e-6).all(), f"{name}: unsound upper bound"

    l_erf = torch.tensor([[-1.0, 0.5]], dtype=dtype)
    u_erf = torch.tensor([[1.0, 2.0]], dtype=dtype)
    erf_out = tf_erf(Layer(id=0, kind=LayerKind.ERF.value, params={}, in_vars=[0, 1], out_vars=[0, 1]), Bounds(l_erf, u_erf))
    assert_sound("erf", erf_out.bounds.lb, erf_out.bounds.ub, l_erf, u_erf, torch.erf)

    # Box straddles negative -> exercises the min-clamp in tf_sqrt.
    l_sqrt = torch.tensor([[-1.0, 0.5]], dtype=dtype)
    u_sqrt = torch.tensor([[2.0, 3.0]], dtype=dtype)
    sqrt_out = tf_sqrt(Layer(id=1, kind=LayerKind.SQRT.value, params={}, in_vars=[0, 1], out_vars=[0, 1]), Bounds(l_sqrt, u_sqrt))
    assert_sound(
        "sqrt", sqrt_out.bounds.lb, sqrt_out.bounds.ub, l_sqrt, u_sqrt,
        lambda x: torch.sqrt(torch.clamp(x, min=0.0)),
    )

    # SIN/COS: narrow (no critical point), has-max, has-min, full-period(>=2pi).
    sin_cases = {"narrow": (0.1, 0.5), "has_max": (1.0, 2.0), "has_min": (-2.0, -1.0), "full_period": (0.0, 7.0)}
    for name, (lv, uv) in sin_cases.items():
        lb = torch.tensor([[lv]], dtype=dtype)
        ub = torch.tensor([[uv]], dtype=dtype)
        out = tf_sin(Layer(id=2, kind=LayerKind.SIN.value, params={}, in_vars=[0], out_vars=[0]), Bounds(lb, ub))
        assert_sound(f"sin[{name}]", out.bounds.lb, out.bounds.ub, lb, ub, torch.sin)

    cos_cases = {"narrow": (0.1, 0.5), "has_max": (-0.5, 0.5), "has_min": (2.5, 3.5), "full_period": (0.0, 7.0)}
    for name, (lv, uv) in cos_cases.items():
        lb = torch.tensor([[lv]], dtype=dtype)
        ub = torch.tensor([[uv]], dtype=dtype)
        out = tf_cos(Layer(id=3, kind=LayerKind.COS.value, params={}, in_vars=[0], out_vars=[0]), Bounds(lb, ub))
        assert_sound(f"cos[{name}]", out.bounds.lb, out.bounds.ub, lb, ub, torch.cos)

    scale = torch.tensor([0.1], dtype=dtype)
    zero_point = torch.tensor([0.0], dtype=dtype)
    l_q = torch.tensor([[-1.0, 0.5]], dtype=dtype)
    u_q = torch.tensor([[1.0, 2.0]], dtype=dtype)
    q_out = tf_quantize(
        Layer(
            id=4, kind=LayerKind.QUANTIZE.value,
            params={"scale": scale, "zero_point": zero_point, "qmin": -128, "qmax": 127},
            in_vars=[0, 1], out_vars=[0, 1],
        ),
        Bounds(l_q, u_q),
    )

    def quantize_concrete(x: torch.Tensor) -> torch.Tensor:
        code = torch.clamp(torch.round(x / scale), min=-128 - zero_point, max=127 - zero_point)
        return scale * code

    assert_sound("quantize", q_out.bounds.lb, q_out.bounds.ub, l_q, u_q, quantize_concrete)


def _make_dual_att_cores_net(  # pragma: no cover
    B: int, L: int, D: int, center: torch.Tensor, eps: float, assert_d: float,
) -> "tuple[Net, dict[str, Any]]":
    """DENSE Q/K/V -> ATT_SCORES -> SOFTMAX -> ATT_MIX -> CONCAT -> LAYERNORM -> GELU.

    The dual attention path (dual_tf/tf_transformer.py) consumes the bilinear
    cores ATT_SCORES (Q.Kt) / ATT_MIX (probs.V) with q_src/k_src/w_src/v_src
    reading predecessor boxes; it stubs MHA_SPLIT/MHA_JOIN. So Q/K/V come from
    DENSE (which dual supports) rather than the interval MHA_SPLIT decomposition,
    giving a net the DualSolver can run end to end. Non-degenerate dims (L,D>1)
    avoid the size-1 shape class.
    """
    from act.back_end.core import Layer
    from act.front_end.specs import OutputSpec

    n_in = L * D
    in_v = list(range(n_in))
    lb_in, ub_in = center - eps, center + eps
    Wq = torch.randn(D, D, dtype=center.dtype, generator=torch.Generator().manual_seed(71)) * 0.2
    Wk = torch.randn(D, D, dtype=center.dtype, generator=torch.Generator().manual_seed(72)) * 0.2
    Wv = torch.randn(D, D, dtype=center.dtype, generator=torch.Generator().manual_seed(73)) * 0.2

    layers = [
        Layer(id=0, kind=LayerKind.INPUT.value, params={"shape": (B, n_in), "dtype": str(center.dtype)}, in_vars=[], out_vars=in_v),
        Layer(id=1, kind=LayerKind.INPUT_SPEC.value, params={"kind": "BOX", "lb": lb_in, "ub": ub_in}, in_vars=in_v, out_vars=in_v),
    ]
    preds: "dict[int, list[int]]" = {0: [], 1: [0]}
    succs: "dict[int, list[int]]" = {0: [1], 1: []}
    next_id, next_var = 2, n_in

    def alloc(n: int) -> "list[int]":
        nonlocal next_var
        v = list(range(next_var, next_var + n)); next_var += n
        return v

    def add(kind, params, in_vars, out_vars, pred_ids) -> int:
        nonlocal next_id
        layers.append(Layer(id=next_id, kind=kind, params=params, in_vars=in_vars, out_vars=out_vars))
        lid = next_id; next_id += 1
        preds[lid] = pred_ids
        succs.setdefault(lid, [])
        for p in pred_ids:
            succs[p].append(lid)
        return lid

    def dense_pos(W, pos) -> int:
        full = torch.zeros(D, n_in, dtype=center.dtype)
        full[:, pos * D:(pos + 1) * D] = W
        return add(LayerKind.DENSE.value, {
            "weight": full, "in_features": n_in, "out_features": D,
            "weight_pos": full.clamp(min=0), "weight_neg": full.clamp(max=0),
            "bias": torch.zeros(D, dtype=center.dtype), "input_shape": (n_in,),
        }, in_v, alloc(D), [1])

    def dense_value_feature(W, feat) -> int:
        full = torch.zeros(L, n_in, dtype=center.dtype)
        for p in range(L):
            full[p, p * D:(p + 1) * D] = W[feat]
        return add(LayerKind.DENSE.value, {
            "weight": full, "in_features": n_in, "out_features": L,
            "weight_pos": full.clamp(min=0), "weight_neg": full.clamp(max=0),
            "bias": torch.zeros(L, dtype=center.dtype), "input_shape": (n_in,),
        }, in_v, alloc(L), [1])

    q_ids = [dense_pos(Wq, p) for p in range(L)]
    k_ids = [dense_pos(Wk, p) for p in range(L)]
    v_ids = [dense_value_feature(Wv, f) for f in range(D)]

    score_ids = []
    score_vars: "list[int]" = []
    for kp in range(L):
        sv = alloc(1)
        sid = add(LayerKind.ATT_SCORES.value, {
            "dk": float(D) ** 0.5,
            "q_vars": layers[q_ids[0]].out_vars, "k_vars": layers[k_ids[kp]].out_vars,
            "q_src": q_ids[0], "k_src": k_ids[kp],
        }, layers[q_ids[0]].out_vars + layers[k_ids[kp]].out_vars, sv, [q_ids[0], k_ids[kp]])
        score_ids.append(sid); score_vars += sv
    cat_id = add(LayerKind.CONCAT.value, {"concat_dim": -1}, score_vars, alloc(L), score_ids)
    sm_id = add(LayerKind.SOFTMAX.value, {"axis": -1}, layers[cat_id].out_vars, alloc(L), [cat_id])
    mix_ids = []
    mix_vars: "list[int]" = []
    for f in range(D):
        mv = alloc(1)
        mid = add(LayerKind.ATT_MIX.value, {
            "rowsize": L, "w_vars": layers[sm_id].out_vars, "v_vars": layers[v_ids[f]].out_vars,
            "w_src": sm_id, "v_src": v_ids[f],
        }, layers[sm_id].out_vars + layers[v_ids[f]].out_vars, mv, [sm_id, v_ids[f]])
        mix_ids.append(mid); mix_vars += mv
    join_id = add(LayerKind.CONCAT.value, {"concat_dim": -1}, mix_vars, alloc(D), mix_ids)
    gamma = torch.ones(D, dtype=center.dtype)
    beta = torch.zeros(D, dtype=center.dtype)
    ln_id = add(LayerKind.LAYERNORM.value, {"gamma": gamma, "beta": beta, "variant": "no_var"}, layers[join_id].out_vars, alloc(D), [join_id])
    gelu_id = add(LayerKind.GELU.value, {}, layers[ln_id].out_vars, alloc(D), [ln_id])

    out_spec = OutputSpec(kind="LINEAR_LE", c=torch.ones(D, dtype=center.dtype), d=torch.tensor(assert_d, dtype=center.dtype))
    enc = out_spec.encode_linear(B=B, n_out=D, device=center.device, dtype=center.dtype)
    add(LayerKind.ASSERT.value, enc, layers[gelu_id].out_vars, layers[gelu_id].out_vars, [gelu_id])

    net = Net(layers=layers, preds=preds, succs=succs)
    info: "dict[str, Any]" = {"Wq": Wq, "Wk": Wk, "Wv": Wv, "lb_in": lb_in, "ub_in": ub_in, "out_id": gelu_id, "L": L, "D": D}
    return net, info


def _make_dual_matmul_net(  # pragma: no cover
    B: int, I: int, K: int, J: int, center: torch.Tensor, eps: float, assert_d: float,
) -> "tuple[Net, dict[str, Any]]":
    """DENSE X [I,K] + DENSE Y [K,J] -> MATMUL -> SOFTMAX -> LAYERNORM -> GELU.

    The ONNX import lowers attention Q.Kt / probs.V to a generic var x var
    MATMUL, a distinct dual kernel (forward_matmul / backward_matmul) from the
    scalar ATT_SCORES/ATT_MIX cores. This net exercises that batched-bilinear
    path end to end through the DualSolver.
    """
    from act.back_end.core import Layer
    from act.front_end.specs import OutputSpec

    n_in = center.shape[1]
    in_v = list(range(n_in))
    lb_in, ub_in = center - eps, center + eps
    Wx = torch.randn(I * K, n_in, dtype=center.dtype, generator=torch.Generator().manual_seed(81)) * 0.2
    Wy = torch.randn(K * J, n_in, dtype=center.dtype, generator=torch.Generator().manual_seed(82)) * 0.2

    layers = [
        Layer(id=0, kind=LayerKind.INPUT.value, params={"shape": (B, n_in), "dtype": str(center.dtype)}, in_vars=[], out_vars=in_v),
        Layer(id=1, kind=LayerKind.INPUT_SPEC.value, params={"kind": "BOX", "lb": lb_in, "ub": ub_in}, in_vars=in_v, out_vars=in_v),
    ]
    x_vars = list(range(n_in, n_in + I * K))
    y_vars = list(range(n_in + I * K, n_in + I * K + K * J))
    z_vars = list(range(n_in + I * K + K * J, n_in + I * K + K * J + I * J))

    def dense(W, out_vars, n_out):
        return {
            "weight": W, "in_features": n_in, "out_features": n_out,
            "weight_pos": W.clamp(min=0), "weight_neg": W.clamp(max=0),
            "bias": torch.zeros(n_out, dtype=center.dtype), "input_shape": (n_in,),
        }

    layers.append(Layer(id=2, kind=LayerKind.DENSE.value, params=dense(Wx, x_vars, I * K), in_vars=in_v, out_vars=x_vars))
    layers.append(Layer(id=3, kind=LayerKind.DENSE.value, params=dense(Wy, y_vars, K * J), in_vars=in_v, out_vars=y_vars))
    layers.append(Layer(id=4, kind=LayerKind.MATMUL.value, params={"x_vars": x_vars, "y_vars": y_vars, "x_shape": (I, K), "y_shape": (K, J)}, in_vars=x_vars + y_vars, out_vars=z_vars))
    sm_vars = list(range(z_vars[-1] + 1, z_vars[-1] + 1 + I * J))
    layers.append(Layer(id=5, kind=LayerKind.SOFTMAX.value, params={"axis": -1}, in_vars=z_vars, out_vars=sm_vars))
    gamma = torch.ones(I * J, dtype=center.dtype)
    beta = torch.zeros(I * J, dtype=center.dtype)
    ln_vars = list(range(sm_vars[-1] + 1, sm_vars[-1] + 1 + I * J))
    layers.append(Layer(id=6, kind=LayerKind.LAYERNORM.value, params={"gamma": gamma, "beta": beta, "variant": "no_var"}, in_vars=sm_vars, out_vars=ln_vars))
    gelu_vars = list(range(ln_vars[-1] + 1, ln_vars[-1] + 1 + I * J))
    layers.append(Layer(id=7, kind=LayerKind.GELU.value, params={}, in_vars=ln_vars, out_vars=gelu_vars))
    out_spec = OutputSpec(kind="LINEAR_LE", c=torch.ones(I * J, dtype=center.dtype), d=torch.tensor(assert_d, dtype=center.dtype))
    enc = out_spec.encode_linear(B=B, n_out=I * J, device=center.device, dtype=center.dtype)
    layers.append(Layer(id=8, kind=LayerKind.ASSERT.value, params=enc, in_vars=gelu_vars, out_vars=gelu_vars))
    preds = {0: [], 1: [0], 2: [1], 3: [1], 4: [2, 3], 5: [4], 6: [5], 7: [6], 8: [7]}
    succs = {0: [1], 1: [2, 3], 2: [4], 3: [4], 4: [5], 5: [6], 6: [7], 7: [8], 8: []}
    net = Net(layers=layers, preds=preds, succs=succs)
    info: "dict[str, Any]" = {"Wx": Wx, "Wy": Wy, "lb_in": lb_in, "ub_in": ub_in, "z_id": 4, "I": I, "K": K, "J": J}
    return net, info


def _dual_forward_box(net, lb_in, ub_in, layer_id):  # pragma: no cover
    """Run the dual forward pass and return the (lb, ub) box at ``layer_id``."""
    from act.back_end.dual_tf.tf_forward import compute_forward_bounds

    bounds_dict = compute_forward_bounds(net, lb_in.clone(), ub_in.clone(), post_activation=False)
    box = bounds_dict[layer_id]
    return box.lb, box.ub


def _test_dual_transformer_att_cores() -> None:  # pragma: no cover
    # Dual attention scalar cores end to end: the dual FORWARD pass
    # (forward_attention/softmax/layernorm/gelu) box must bracket the concrete
    # attention output, and the dual BACKWARD pass (DualSolver.evaluate_spec)
    # must CERTIFY a loose bound yet NOT certify a bound below the true range
    # (proving the certified bound is used, not vacuous).
    from act.back_end.transfer_functions import set_solver_mode, get_solver_mode
    from act.util.device_manager import get_default_dtype
    from act.util.stats import VerifyStatus

    dtype = get_default_dtype()
    B, L, D = 1, 2, 2
    torch.manual_seed(90)
    center = torch.randn(B, L * D, dtype=dtype) * 0.05
    eps = 0.02
    net, info = _make_dual_att_cores_net(B, L, D, center, eps, assert_d=100.0)
    Wq, Wk, Wv = info["Wq"], info["Wk"], info["Wv"]
    l_box, u_box = info["lb_in"], info["ub_in"]

    def concrete_gelu_out(x: torch.Tensor) -> torch.Tensor:
        x3 = x.reshape(B, L, D)
        q0 = x3[:, 0, :] @ Wq.t()
        scores = torch.cat([(q0 * (x3[:, kp, :] @ Wk.t())).sum(-1, keepdim=True) / (D ** 0.5) for kp in range(L)], dim=-1)
        probs = torch.softmax(scores, dim=-1)
        v = torch.stack([x3[:, p, :] @ Wv.t() for p in range(L)], dim=1)
        ctx = torch.cat([(probs * v[:, :, f]).sum(-1, keepdim=True) for f in range(D)], dim=-1)
        normed = ctx - ctx.mean(dim=-1, keepdim=True)
        return torch.nn.functional.gelu(normed)

    lb, ub = _dual_forward_box(net, l_box, u_box, info["out_id"])
    assert torch.isfinite(lb).all() and torch.isfinite(ub).all(), "dual att-cores forward box must be finite"
    assert (lb <= ub + 1e-9).all(), "dual att-cores forward box lb must not exceed ub"
    concrete_sum_max = float(concrete_gelu_out(l_box).sum(-1).item())
    for _ in range(120):
        x = l_box + torch.rand_like(l_box) * (u_box - l_box)
        concrete_sum_max = max(concrete_sum_max, float(concrete_gelu_out(x).sum(-1).item()))

    prev = get_solver_mode()
    try:
        set_solver_mode("dual")
        loose = verify_once(net)
        assert loose[0].status == VerifyStatus.CERTIFIED, f"dual att-cores: loose bound expected CERTIFIED, got {loose[0].status}"
        assert concrete_sum_max <= 100.0 + 1e-6, (
            f"dual att-cores: certified d=100 contradicted by concrete sum {concrete_sum_max}"
        )
        net_tight, _ = _make_dual_att_cores_net(B, L, D, center, eps, assert_d=concrete_sum_max - 1.0)
        tight = verify_once(net_tight)
        assert tight[0].status != VerifyStatus.CERTIFIED, (
            f"dual att-cores: threshold below range must NOT certify, got {tight[0].status}"
        )
    finally:
        set_solver_mode(prev)


def _test_dual_transformer_matmul() -> None:  # pragma: no cover
    # Dual batched-bilinear MATMUL core (the ONNX attention lowering) end to
    # end: forward_matmul box brackets the concrete X@Y (through softmax/
    # layernorm/gelu), and backward_matmul via DualSolver certifies a loose
    # bound but not a below-range one.
    from act.back_end.transfer_functions import set_solver_mode, get_solver_mode
    from act.util.device_manager import get_default_dtype
    from act.util.stats import VerifyStatus

    dtype = get_default_dtype()
    B, I, K, J = 1, 2, 2, 2
    torch.manual_seed(91)
    center = torch.randn(B, 3, dtype=dtype) * 0.05
    eps = 0.02
    net, info = _make_dual_matmul_net(B, I, K, J, center, eps, assert_d=100.0)
    Wx, Wy = info["Wx"], info["Wy"]
    l_box, u_box = info["lb_in"], info["ub_in"]

    def concrete_matmul_z(x: torch.Tensor) -> torch.Tensor:
        X = (x @ Wx.t()).reshape(B, I, K)
        Y = (x @ Wy.t()).reshape(B, K, J)
        return (X @ Y).reshape(B, I * J)

    lb, ub = _dual_forward_box(net, l_box, u_box, info["z_id"])
    n_samples = 120
    true_min = concrete_matmul_z(l_box).clone()
    true_max = true_min.clone()
    for _ in range(n_samples):
        x = l_box + torch.rand_like(l_box) * (u_box - l_box)
        z = concrete_matmul_z(x)
        true_min = torch.minimum(true_min, z)
        true_max = torch.maximum(true_max, z)
    # MATMUL forward box is the sound four-corner McCormick envelope; it must
    # bracket the concrete X@Y (the layernorm/gelu that follow are checked via
    # the end-to-end certified bound below, not this pre-softmax box).
    assert (lb <= true_min + 1e-6).all(), "dual MATMUL forward: unsound lower bound"
    assert (ub >= true_max - 1e-6).all(), "dual MATMUL forward: unsound upper bound"

    prev = get_solver_mode()
    try:
        set_solver_mode("dual")
        loose = verify_once(net)
        assert loose[0].status == VerifyStatus.CERTIFIED, f"dual MATMUL: loose expected CERTIFIED, got {loose[0].status}"
        net_tight, info_t = _make_dual_matmul_net(B, I, K, J, center, eps, assert_d=-50.0)
        tight = verify_once(net_tight)
        assert tight[0].status != VerifyStatus.CERTIFIED, (
            f"dual MATMUL: threshold below range must NOT certify, got {tight[0].status}"
        )
    finally:
        set_solver_mode(prev)


def _test_dual_lp_embedding_finite_p() -> None:  # pragma: no cover
    # Finite-p LP_EMBEDDING input spec (p_norm=2) verified through the dual
    # solver: exercises seed_from_input_specs' LP_EMBEDDING center/eps/
    # perturbed_positions seeding AND solver_dual's exact per-word Lp-ball
    # dual-norm input contribution (_resolve_perturbation_norm ->
    # _dual_norm_exponent -> _dual_norm_contribution / _perturbed_block_slices),
    # the finite-p path box/L-inf specs never reach.
    from act.back_end.core import Layer
    from act.back_end.transfer_functions import set_solver_mode, get_solver_mode
    from act.front_end.specs import OutputSpec, InKind
    from act.util.device_manager import get_default_dtype
    from act.util.stats import VerifyStatus

    dtype = get_default_dtype()
    B, L, D = 1, 2, 2
    n_in = L * D
    torch.manual_seed(97)
    center3 = torch.randn(B, L, D, dtype=dtype) * 0.1
    eps = 0.05
    in_v = list(range(n_in))
    d_v = list(range(n_in, n_in + 2))
    W = torch.randn(2, n_in, dtype=dtype) * 0.2

    def build(assert_d: float) -> Net:
        layers = [
            Layer(id=0, kind=LayerKind.INPUT.value, params={"shape": (B, L, D), "dtype": str(dtype)}, in_vars=[], out_vars=in_v),
            Layer(id=1, kind=LayerKind.INPUT_SPEC.value, params={"kind": InKind.LP_EMBEDDING, "center": center3, "eps": torch.tensor([eps], dtype=dtype), "p_norm": 2.0, "perturbed_positions": torch.tensor([0])}, in_vars=in_v, out_vars=in_v),
            Layer(id=2, kind=LayerKind.DENSE.value, params={"weight": W, "in_features": n_in, "out_features": 2, "weight_pos": W.clamp(min=0), "weight_neg": W.clamp(max=0), "bias": torch.zeros(2, dtype=dtype), "input_shape": (n_in,)}, in_vars=in_v, out_vars=d_v),
        ]
        enc = OutputSpec(kind="LINEAR_LE", c=torch.ones(2, dtype=dtype), d=torch.tensor(assert_d, dtype=dtype)).encode_linear(B=B, n_out=2, device=torch.device("cpu"), dtype=dtype)
        layers.append(Layer(id=3, kind=LayerKind.ASSERT.value, params=enc, in_vars=d_v, out_vars=d_v))
        return Net(layers=layers, preds={0: [], 1: [0], 2: [1], 3: [2]}, succs={0: [1], 1: [2], 2: [3], 3: []})

    prev = get_solver_mode()
    try:
        set_solver_mode("dual")
        loose = verify_once(build(100.0))
        assert loose[0].status == VerifyStatus.CERTIFIED, f"dual LP_EMBEDDING: loose expected CERTIFIED, got {loose[0].status}"
        tight = verify_once(build(-100.0))
        assert tight[0].status != VerifyStatus.CERTIFIED, (
            f"dual LP_EMBEDDING: threshold below range must NOT certify, got {tight[0].status}"
        )
    finally:
        set_solver_mode(prev)


def _test_dual_smooth_activations() -> None:  # pragma: no cover
    # Dual backward for the new smooth activations (ERF/SQRT/SIN/COS/QUANTIZE
    # in dual_tf/tf_smooth.py): a DENSE -> activation -> ASSERT net run through
    # the DualSolver must CERTIFY a loose bound, exercising each activation's
    # forward relaxation + backward routing.
    from act.back_end.core import Layer
    from act.back_end.transfer_functions import set_solver_mode, get_solver_mode
    from act.util.device_manager import get_default_dtype
    from act.util.stats import VerifyStatus

    dtype = get_default_dtype()
    B, n = 1, 3

    def build(act_kind: str, act_params: "dict[str, Any]") -> Net:
        center = torch.full((B, n), 0.7, dtype=dtype)
        lb_in, ub_in = center - 0.05, center + 0.05
        in_v = list(range(n))
        d_v = list(range(n, 2 * n))
        o_v = list(range(2 * n, 3 * n))
        W = torch.eye(n, dtype=dtype)
        layers = [
            Layer(id=0, kind=LayerKind.INPUT.value, params={"shape": (B, n), "dtype": str(dtype)}, in_vars=[], out_vars=in_v),
            Layer(id=1, kind=LayerKind.INPUT_SPEC.value, params={"kind": "BOX", "lb": lb_in, "ub": ub_in}, in_vars=in_v, out_vars=in_v),
            Layer(id=2, kind=LayerKind.DENSE.value, params={"weight": W, "in_features": n, "out_features": n, "weight_pos": W, "weight_neg": W * 0, "bias": torch.zeros(n, dtype=dtype), "input_shape": (n,)}, in_vars=in_v, out_vars=d_v),
            Layer(id=3, kind=act_kind, params=act_params, in_vars=d_v, out_vars=o_v),
        ]
        from act.front_end.specs import OutputSpec
        enc = OutputSpec(kind="LINEAR_LE", c=torch.ones(n, dtype=dtype), d=torch.tensor(100.0, dtype=dtype)).encode_linear(B=B, n_out=n, device=torch.device("cpu"), dtype=dtype)
        layers.append(Layer(id=4, kind=LayerKind.ASSERT.value, params=enc, in_vars=o_v, out_vars=o_v))
        return Net(layers=layers, preds={0: [], 1: [0], 2: [1], 3: [2], 4: [3]}, succs={0: [1], 1: [2], 2: [3], 3: [4], 4: []})

    cases = [
        (LayerKind.ERF.value, {}),
        (LayerKind.SQRT.value, {}),
        (LayerKind.SIN.value, {}),
        (LayerKind.COS.value, {}),
        (LayerKind.QUANTIZE.value, {"scale": torch.tensor([0.1], dtype=dtype), "zero_point": torch.tensor([0.0], dtype=dtype), "qmin": -128, "qmax": 127}),
    ]
    prev = get_solver_mode()
    try:
        set_solver_mode("dual")
        for kind, params in cases:
            r = verify_once(build(kind, params))
            assert r[0].status == VerifyStatus.CERTIFIED, f"dual {kind}: expected CERTIFIED, got {r[0].status}"
    finally:
        set_solver_mode(prev)


def _test_dual_mha_split_join_not_implemented() -> None:  # pragma: no cover
    # The dual path deliberately stubs the MHA split/join reshape family
    # (only the ATT_SCORES/ATT_MIX scalar cores + MATMUL are relaxed); the
    # stubs must raise NotImplementedError so a mis-lowered net fails loudly.
    from act.back_end.core import Layer
    from act.back_end.dual_tf.tf_transformer import forward_mha, backward_mha

    dummy = Layer(id=0, kind=LayerKind.MHA_SPLIT.value, params={}, in_vars=[0], out_vars=[0])
    raised_fwd = False
    try:
        forward_mha(dummy, [], [], [], [], False, torch.device("cpu"), torch.get_default_dtype())
    except NotImplementedError:
        raised_fwd = True
    assert raised_fwd, "forward_mha must raise NotImplementedError"
    raised_bwd = False
    try:
        backward_mha(dummy, torch.zeros(1, 1), {}, [])
    except NotImplementedError:
        raised_bwd = True
    assert raised_bwd, "backward_mha must raise NotImplementedError"


def _test_setup_and_solve_batch_b1_smoke() -> None:  # pragma: no cover
    from act.back_end.solver.solver_torchlp import TorchLPSolver
    from act.util.device_manager import get_default_device, get_default_dtype

    device = get_default_device()
    dtype = get_default_dtype()
    weight = torch.ones(1, 1, device=device, dtype=dtype)
    bias = torch.zeros(1, device=device, dtype=dtype)
    lb_in = torch.full((1, 1), 1.0, device=device, dtype=dtype)
    ub_in = torch.full((1, 1), 2.0, device=device, dtype=dtype)
    net = _make_dense_net_box_test(
        B=1, n_in=1, n_out=1, weight=weight, bias=bias,
        lb_in=lb_in, ub_in=ub_in,
        assert_params={
            "kind": OutKind.LINEAR_LE,
            "c": torch.ones(1, device=device, dtype=dtype),
            "d": torch.tensor(0.0, device=device, dtype=dtype),
        },
    )

    solution = setup_and_solve_batch(
        net,
        Bounds(lb_in.clone(), ub_in.clone()),
        TorchLPSolver(),
    )
    assert solution.statuses == (SolveStatus.SAT,), f"got {solution.statuses}"
    assert tuple(solution.max_viol.shape) == (1,)
    assert float(solution.max_viol[0].item()) <= 1e-4


def _test_setup_and_solve_batch_b_greater_than_1() -> None:  # pragma: no cover
    from act.back_end.solver.solver_torchlp import TorchLPSolver
    from act.util.device_manager import get_default_device, get_default_dtype

    device = get_default_device()
    dtype = get_default_dtype()

    B = 4
    weight = torch.ones(1, 1, device=device, dtype=dtype)
    bias = torch.zeros(1, device=device, dtype=dtype)
    lb_in = torch.tensor([[1.0], [1.25], [1.5], [1.75]], device=device, dtype=dtype)
    ub_in = torch.tensor([[2.0], [2.25], [2.5], [2.75]], device=device, dtype=dtype)
    net = _make_dense_net_box_test(
        B=B, n_in=1, n_out=1, weight=weight, bias=bias,
        lb_in=lb_in, ub_in=ub_in,
        assert_params={
            "kind": OutKind.LINEAR_LE,
            "c": torch.ones(1, device=device, dtype=dtype),
            "d": torch.tensor(0.0, device=device, dtype=dtype),
        },
    )

    solution = setup_and_solve_batch(
        net,
        Bounds(lb_in.clone(), ub_in.clone()),
        TorchLPSolver(),
    )

    assert solution.statuses == (SolveStatus.SAT,) * B, (
        f"expected {B} SAT statuses, got {solution.statuses}"
    )
    assert tuple(solution.x.shape) == (B, solution.x.shape[1]), (
        f"solution.x should retain leading batch B={B}, got "
        f"{tuple(solution.x.shape)}"
    )
    for i in range(B):
        assert float(solution.max_viol[i].item()) <= 1e-4, (
            f"batch lane {i}: max_viol "
            f"{float(solution.max_viol[i].item())} > 1e-4"
        )



def _test_verify_once_b3_all_certified() -> None:  # pragma: no cover
    # Zero DENSE -> abstract output is singleton {0}, well below d=10.
    # End-to-end check that the [B*M, n_out] cert pass folds to per-sample.
    from act.util.device_manager import get_default_device, get_default_dtype
    from act.util.stats import VerifyStatus

    device = get_default_device()
    dtype = get_default_dtype()

    B, n_in, n_out = 3, 4, 2
    W = torch.zeros(n_out, n_in, device=device, dtype=dtype)
    b = torch.zeros(n_out, device=device, dtype=dtype)
    lb_in = torch.full((B, n_in), -1.0, device=device, dtype=dtype)
    ub_in = torch.full((B, n_in), 1.0, device=device, dtype=dtype)

    net = _make_dense_net_box_test(
        B=B, n_in=n_in, n_out=n_out, weight=W, bias=b,
        lb_in=lb_in, ub_in=ub_in,
        assert_params={
            "kind": "LINEAR_LE",
            "c": torch.tensor([1.0, 1.0], device=device, dtype=dtype),
            "d": 10.0,
        },
    )

    results = verify_once(net)
    assert len(results) == B, f"expected {B} results, got {len(results)}"
    for i, r in enumerate(results):
        assert r.status == VerifyStatus.CERTIFIED, (
            f"sample {i}: expected CERTIFIED, got {r.status}"
        )


def _test_verify_once_b8_mixed_outcomes() -> None:  # pragma: no cover
    # 8 input boxes designed to produce CERT/FALS/UNK mix in one run,
    # proving the cert pass + concrete falsification operate sample-wise
    # rather than collapsing the batch.
    from act.util.device_manager import get_default_device, get_default_dtype
    from act.util.stats import VerifyStatus

    device = get_default_device()
    dtype = get_default_dtype()

    B, n_in, n_out = 8, 2, 2
    W = torch.eye(n_out, device=device, dtype=dtype)
    b = torch.zeros(n_out, device=device, dtype=dtype)
    lb_in = torch.tensor(
        [
            [2.0, -2.0],
            [1.0, -2.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [-1.0, -1.0],
            [-2.0, -1.0],
            [1.0, -1.0],
            [-1.0, 0.0],
        ],
        device=device, dtype=dtype,
    )
    ub_in = torch.tensor(
        [
            [3.0, -1.0],
            [2.0, -1.5],
            [1.0, 2.0],
            [1.0, 2.0],
            [1.0, 0.5],
            [2.0, 0.5],
            [2.0, 0.0],
            [1.0, 1.0],
        ],
        device=device, dtype=dtype,
    )
    net = _make_dense_net_box_test(
        B=B, n_in=n_in, n_out=n_out, weight=W, bias=b,
        lb_in=lb_in, ub_in=ub_in,
        assert_params={
            "kind": "TOP1_ROBUST",
            "y_true": torch.zeros(B, dtype=torch.long, device=device),
        },
    )

    def model_fn(x: torch.Tensor) -> torch.Tensor:
        return x

    results = verify_once(net, model_fn=model_fn)
    assert len(results) == B, f"expected {B} results, got {len(results)}"

    valid = {
        VerifyStatus.CERTIFIED, VerifyStatus.FALSIFIED, VerifyStatus.UNKNOWN,
    }
    statuses = [r.status for r in results]
    assert all(s in valid for s in statuses), (
        f"unexpected status enum value in {statuses}"
    )
    assert any(s == VerifyStatus.CERTIFIED for s in statuses), (
        f"no CERTIFIED lane in {statuses}"
    )
    assert any(s == VerifyStatus.FALSIFIED for s in statuses), (
        f"no FALSIFIED lane in {statuses}"
    )
    assert any(s == VerifyStatus.UNKNOWN for s in statuses), (
        f"no UNKNOWN lane in {statuses}"
    )


def _test_verify_lp_batched_multi_b1() -> None:  # pragma: no cover
    from act.back_end.serialization.serialization import load_net_from_file
    from act.back_end.solver.solver_torchlp import TorchLPSolver
    from act.util.stats import VerifyStatus

    net = load_net_from_file(
        "act/back_end/examples/nets/layer_testing_top1_robust.json",
        target_device="cpu",
    )
    results = verify_lp_batched(net, TorchLPSolver, timelimit=1.0)
    valid = {VerifyStatus.CERTIFIED, VerifyStatus.FALSIFIED, VerifyStatus.UNKNOWN}
    assert len(results) == 1, f"expected one result, got {len(results)}"
    assert results[0].status in valid, f"unexpected status {results[0].status}"


def _test_verify_lp_batched_batch_b4() -> None:  # pragma: no cover
    from act.back_end.solver.solver_torchlp import TorchLPSolver
    from act.util.device_manager import get_default_device, get_default_dtype
    from act.util.stats import VerifyStatus

    device = get_default_device()
    dtype = get_default_dtype()
    B = 4
    weight = torch.ones(1, 1, device=device, dtype=dtype)
    bias = torch.zeros(1, device=device, dtype=dtype)
    lb_in = torch.tensor([[1.0], [1.25], [1.5], [1.75]], device=device, dtype=dtype)
    ub_in = torch.tensor([[2.0], [2.25], [2.5], [2.75]], device=device, dtype=dtype)
    net = _make_dense_net_box_test(
        B=B, n_in=1, n_out=1, weight=weight, bias=bias,
        lb_in=lb_in, ub_in=ub_in,
        assert_params={
            "kind": OutKind.LINEAR_LE,
            "c": torch.ones(1, device=device, dtype=dtype),
            "d": torch.tensor(0.0, device=device, dtype=dtype),
        },
    )

    results = verify_lp_batched(net, TorchLPSolver, timelimit=1.0)
    valid = {VerifyStatus.CERTIFIED, VerifyStatus.FALSIFIED, VerifyStatus.UNKNOWN}
    assert len(results) == B, f"expected {B} results, got {len(results)}"
    for i, result in enumerate(results):
        assert result.status in valid, f"lane {i}: unexpected status {result.status}"


_TESTS = [  # pragma: no cover
    _test_build_top1_robust_drops_y_true_row,
    _test_build_linear_le_threshold_is_d_unchanged,
    _test_build_margin_robust_threshold_is_negated_margin,
    _test_build_range_interleaves_pm_e_rows,
    _test_interval_margin_certification_shape,
    _test_setup_and_solve_batch_b1_smoke,
    _test_setup_and_solve_batch_b_greater_than_1,
    _test_verify_once_b3_all_certified,
    _test_verify_once_b8_mixed_outcomes,
    _test_verify_lp_batched_multi_b1,
    _test_verify_lp_batched_batch_b4,
    _test_att_scores_dual_planar_analyze_soundness,
    _test_att_scores_dual_planar_verify_once_certified,
    _test_att_scores_dual_planar_lp_export_solve,
    _test_att_scores_dual_planar_masked_and_clamp_alpha_soundness,
    _test_mini_transformer_block_analyze_soundness,
    _test_mha_split_edge_cases_and_mask_add,
    _test_new_elementwise_tf_soundness,
    _test_dual_transformer_att_cores,
    _test_dual_transformer_matmul,
    _test_dual_lp_embedding_finite_p,
    _test_dual_smooth_activations,
    _test_dual_mha_split_join_not_implemented,
]


def run_all_tests() -> int:
    passed = failed = 0
    for fn in _TESTS:
        try:
            fn()
            passed += 1
            print(f"  PASS  {fn.__name__}")
        except Exception as e:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


def main() -> int:
    # Pin device/dtype to CPU/float64 so hosts where CUDA is visible but
    # no kernel matches the runtime's compute capability don't raise on
    # the default GPU init path in act.util.device_manager.
    from act.util.device_manager import initialize_device

    initialize_device("cpu", "float64")
    print("Running verifier self-tests (act.back_end.verifier)\n")
    return run_all_tests()


if __name__ == "__main__":
    import sys

    sys.exit(main())
