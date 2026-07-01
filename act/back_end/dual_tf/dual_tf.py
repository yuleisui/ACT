#===- act/back_end/dual_tf/dual_tf.py - Dual Backward Registry Holder ---====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025- ACT Team
# Licensed under AGPLv3+; distributed without warranty.
#===---------------------------------------------------------------------===#
# DualTF: backward-kernel registry holder. Not a TransferFunction — dual is a
# --solver choice, not a --tf-mode. Instantiated internally by
# act.back_end.solver.solver_dual.DualSolver.
#===---------------------------------------------------------------------===#
# pyright: reportImportCycles=false, reportOptionalMemberAccess=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportArgumentType=false
# justification: registry imports are intentionally cyclic with kernel modules; Layer params and LinearBound fields are dynamically validated tensors.


import torch
from typing import Dict, List, Optional, Sequence, Tuple, cast
from act.back_end.core import Bounds, Layer, Net
from act.back_end.layer_schema import LayerKind
from .tf_mlp import (
    backward_dense, backward_relu, backward_bias, backward_scale,
    backward_bn, backward_identity, backward_mean,
    forward_dense, forward_relu, forward_bias, forward_scale,
    forward_bn, forward_lrelu, forward_identity, forward_reshape, forward_mean,
)
from .tf_cnn import (
    backward_conv2d, backward_maxpool2d, backward_avgpool2d,
    forward_conv2d, forward_maxpool2d, forward_avgpool2d,
)
from .tf_smooth import (
    backward_sigmoid, backward_tanh, backward_erf, backward_sqrt, backward_sin, backward_cos, backward_quantize,
    forward_sigmoid, forward_tanh, forward_erf, forward_sqrt, forward_sin, forward_cos, forward_quantize,
)
from .tf_rnn import forward_lstm, backward_lstm, forward_gru, backward_gru
from .tf_transformer import (
    forward_attention, backward_attention,
    forward_matmul, backward_matmul,
    forward_mha, backward_mha,
    forward_layernorm, backward_layernorm,
    forward_softmax, backward_softmax,
    forward_gelu, backward_gelu,
)
from .tf_forward import (
    compute_forward_bounds, LinearBound, Frame,
    _sum_linear_bounds, _sum_interval_bounds, _concretize,
    _reset_forward_box, _align, _int_param,
)


def forward_constant(
    L: Layer, parent_boxes: List[Bounds], parent_lins: List[LinearBound],
    parent_frames: List[Frame], preds: List[int], post_activation: bool,
    device: torch.device, dtype: torch.dtype,
) -> Tuple[Bounds, Bounds, LinearBound, Frame]:
    """Materialize a fixed tensor as a point-valued dual forward box."""
    batch_size = parent_boxes[0].lb.shape[0] if parent_boxes else 1
    value = L.params["value"].reshape(1, -1).to(device=device, dtype=dtype)
    lb = value.expand(batch_size, -1).contiguous()
    ub = lb.clone()
    lin, frame = _reset_forward_box(lb, ub, device, dtype)
    out = Bounds(lb, ub)
    return out, out, lin, frame


def backward_constant(L: Layer, nu: torch.Tensor, bounds_dict: Dict[int, Bounds],
                      preds: List[int], M: int = 1, alpha=None
                      ) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """Backward for a fixed-tensor source: absorb ν·value, route nothing.

    A CONSTANT is data-independent (it has no predecessors after graph build),
    so its only effect on the certified bound is the constant term ν·value —
    exactly a bias on a zero input. Returns ``[]`` (no upstream routes) and that
    once-counted contribution.
    """
    value = L.params["value"].flatten().to(device=nu.device, dtype=nu.dtype)
    v = nu.flatten(start_dim=1)
    n = min(v.shape[-1], value.numel())
    contrib = (v[..., :n] * value[:n]).sum(dim=-1)
    return [torch.zeros_like(nu) for _ in preds], contrib


def forward_expand(
    L: Layer, parent_boxes: List[Bounds], parent_lins: List[LinearBound],
    parent_frames: List[Frame], preds: List[int], post_activation: bool,
    device: torch.device, dtype: torch.dtype,
) -> Tuple[Bounds, Bounds, LinearBound, Frame]:
    """Broadcast a predecessor box and reset the dual frame over the result."""
    parent = parent_boxes[0]
    batch_size = parent.lb.shape[0]
    input_shape_value = L.params.get("input_shape", (parent.lb.shape[1],))
    output_shape_value = L.params.get("output_shape", L.params.get("shape", input_shape_value))
    in_shape = tuple(int(d) for d in cast(Tuple[int, ...], input_shape_value))
    out_shape = tuple(int(d) for d in cast(Tuple[int, ...], output_shape_value))
    lb = parent.lb.reshape(batch_size, *in_shape).broadcast_to(batch_size, *out_shape).reshape(batch_size, -1).clone()
    ub = parent.ub.reshape(batch_size, *in_shape).broadcast_to(batch_size, *out_shape).reshape(batch_size, -1).clone()
    lin, frame = _reset_forward_box(lb, ub, device, dtype)
    out = Bounds(lb, ub)
    return out, out, lin, frame


def forward_gather(
    L: Layer, parent_boxes: List[Bounds], parent_lins: List[LinearBound],
    parent_frames: List[Frame], preds: List[int], post_activation: bool,
    device: torch.device, dtype: torch.dtype,
) -> Tuple[Bounds, Bounds, LinearBound, Frame]:
    """Select indices from a predecessor box and reset the dual frame."""
    parent = parent_boxes[0]
    batch_size = parent.lb.shape[0]
    input_shape = tuple(int(d) for d in cast(Tuple[int, ...], L.params["input_shape"]))
    axis = int(L.params.get("axis", 0))
    raw_idx = L.params["indices"]
    indices = raw_idx.to(device=device, dtype=torch.long) if isinstance(raw_idx, torch.Tensor) else torch.as_tensor(raw_idx, device=device, dtype=torch.long)
    x_lb = parent.lb.reshape(batch_size, *input_shape)
    x_ub = parent.ub.reshape(batch_size, *input_shape)
    lb = torch.index_select(x_lb, dim=axis + 1, index=indices).reshape(batch_size, -1)
    ub = torch.index_select(x_ub, dim=axis + 1, index=indices).reshape(batch_size, -1)
    lin, frame = _reset_forward_box(lb, ub, device, dtype)
    out = Bounds(lb, ub)
    return out, out, lin, frame


def _slice_tuple(input_shape: Tuple[int, ...], L: Layer, batch_offset: int) -> Tuple[slice, ...]:
    """Build the per-axis slice tuple shared by SLICE forward/backward.

    ``batch_offset`` is 1 when indexing a [B, *input_shape] tensor, 0 for a
    per-sample index template. Mirrors interval tf_slice: ends are clamped to
    the axis length so the forward selection and the backward scatter index the
    identical positions.
    """
    starts = cast(Sequence[int], L.params["starts"])
    ends = cast(Sequence[int], L.params["ends"])
    axes = cast(Sequence[int], L.params.get("axes", list(range(len(input_shape)))))
    steps = cast(Sequence[int], L.params.get("steps", [1] * len(axes)))
    slices: List[slice] = [slice(None)] * (len(input_shape) + batch_offset)
    for i, axis in enumerate(axes):
        axis = int(axis)
        end = min(int(ends[i]), input_shape[axis])
        slices[axis + batch_offset] = slice(int(starts[i]), end, int(steps[i]))
    return tuple(slices)


def forward_slice(
    L: Layer, parent_boxes: List[Bounds], parent_lins: List[LinearBound],
    parent_frames: List[Frame], preds: List[int], post_activation: bool,
    device: torch.device, dtype: torch.dtype,
) -> Tuple[Bounds, Bounds, LinearBound, Frame]:
    """Select a sub-tensor from a predecessor box and reset the dual frame."""
    parent = parent_boxes[0]
    batch_size = parent.lb.shape[0]
    input_shape = tuple(int(d) for d in cast(Tuple[int, ...], L.params["input_shape"]))
    slc = _slice_tuple(input_shape, L, batch_offset=1)
    x_lb = parent.lb.reshape(batch_size, *input_shape)
    x_ub = parent.ub.reshape(batch_size, *input_shape)
    lb = x_lb[slc].reshape(batch_size, -1).contiguous()
    ub = x_ub[slc].reshape(batch_size, -1).contiguous()
    lin, frame = _reset_forward_box(lb, ub, device, dtype)
    out = Bounds(lb, ub)
    return out, out, lin, frame


def backward_slice(L: Layer, nu: torch.Tensor, bounds_dict: Dict[int, Bounds],
                   preds: List[int], M: int = 1, alpha=None
                   ) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """Scatter ν from sliced output positions to the full pre-slice space.

    SLICE is the 0/1 selection matrix S; its backward is S^T, which places each
    output row's ν at its source input position and zeros elsewhere. Exact (not
    a relaxation): an arange template sliced identically to the forward gives
    the source-position map.
    """
    assert len(preds) == 1, f"SLICE expects 1 predecessor, got {len(preds)}"
    input_shape = tuple(int(d) for d in cast(Tuple[int, ...], L.params["input_shape"]))
    in_dim = 1
    for d in input_shape:
        in_dim *= d
    template = torch.arange(in_dim, device=nu.device).reshape(input_shape)
    sel = template[_slice_tuple(input_shape, L, batch_offset=0)].reshape(-1)
    pred_nu = torch.zeros(nu.shape[0], in_dim, dtype=nu.dtype, device=nu.device)
    pred_nu[:, sel] = nu
    contrib = torch.zeros(nu.shape[0], dtype=nu.dtype, device=nu.device)
    return [pred_nu], contrib


# ---- ADD ----
def forward_add(
    L: Layer, parent_boxes: List[Bounds], parent_lins: List[LinearBound],
    parent_frames: List[Frame], preds: List[int], post_activation: bool,
    device: torch.device, dtype: torch.dtype,
) -> Tuple[Bounds, Bounds, LinearBound, Frame]:
    """ADD multi-pred forward handler.

    Source: tf_forward.py lines 287-322 (ADD branch of compute_forward_bounds).
    Semantics preserved verbatim — when all predecessor frames share the same
    object identity and A_lb shapes match, sum the dual-track linear bounds and
    concretize over the common frame; otherwise fall back to summing interval
    boxes and reset the dual-track state. Bias (if present) is added on both
    paths via _align. Returns (stored, out, lin, frame) where stored == out.
    """
    assert len(parent_boxes) >= 2, "forward_add: requires >=2 predecessors"
    # Compare b_lb widths (always present), not A_lb: A_lb is None for the lazy
    # identity frame, which _sum_linear_bounds materializes from b_lb. With a
    # shared frame (same input dim) matching output widths is the sound dual-sum
    # precondition; .A_lb.shape would crash on the identity branch.
    can_dual = all(
        parent_frames[i] is parent_frames[0] for i in range(1, len(parent_frames))
    ) and all(
        parent_lins[i].b_lb.shape == parent_lins[0].b_lb.shape
        for i in range(1, len(parent_lins))
    )
    if can_dual:
        lin = _sum_linear_bounds(parent_lins)
        bias_param = L.params.get("bias")
        if bias_param is not None:
            bias_vec = _align(bias_param.flatten(), lin.b_lb.shape[1])
            lin = LinearBound(
                A_lb=lin.A_lb,
                b_lb=lin.b_lb + bias_vec,
                A_ub=lin.A_ub,
                b_ub=lin.b_ub + bias_vec,
            )
        frame = parent_frames[0]
        lb, ub = _concretize(lin, *frame)
    else:
        summed = _sum_interval_bounds(parent_boxes)
        lb, ub = summed.lb, summed.ub
        bias_param = L.params.get("bias")
        if bias_param is not None:
            bias_vec = _align(bias_param.flatten(), lb.shape[1])
            lb = lb + bias_vec
            ub = ub + bias_vec
        lin, frame = _reset_forward_box(lb, ub, device, dtype)
    out = Bounds(lb, ub)
    return out, out, lin, frame


def backward_add(L: Layer, nu: torch.Tensor, bounds_dict: Dict[int, Bounds],
                 preds: List[int], M: int = 1, alpha=None
                 ) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """ADD backward: identity skip — same ν routed to every predecessor.

    Bias contrib uses negative sign to match dual_bias_backward / dual_bn_backward
    / dual_dense_backward conventions (y = x + bias ⇒ contrib = -(ν · bias)).
    """
    B = nu.shape[0]
    contrib = torch.zeros(B, dtype=nu.dtype, device=nu.device)
    if "bias" in L.params and L.params["bias"] is not None:
        b = L.params["bias"].flatten()
        v = nu.flatten(start_dim=1)
        n = min(v.shape[-1], b.numel())
        contrib = -(v[..., :n] * b[:n]).sum(dim=-1)
    return [nu for _ in preds], contrib


# ---- CONCAT ----
def forward_concat(
    L: Layer, parent_boxes: List[Bounds], parent_lins: List[LinearBound],
    parent_frames: List[Frame], preds: List[int], post_activation: bool,
    device: torch.device, dtype: torch.dtype,
) -> Tuple[Bounds, Bounds, LinearBound, Frame]:
    """CONCAT multi-pred forward handler.

    Source: tf_forward.py lines 324-346 (CONCAT branch of compute_forward_bounds).
    Semantics preserved verbatim — when all predecessor frames share the same
    object identity and A_lb batch/input axes match, concatenate dual-track
    linear bounds along dim=1 and concretize; otherwise fall back to torch.cat
    on interval boxes along concat_dim (default 1) and reset dual-track state.
    Returns (stored, out, lin, frame) where stored == out.
    """
    assert len(parent_boxes) >= 2, "forward_concat: requires >=2 predecessors"
    concat_dim = _int_param(L.params.get("concat_dim", 1), 1)
    can_dual = all(
        parent_frames[i] is parent_frames[0] for i in range(1, len(parent_frames))
    ) and all(
        parent_lins[i].A_lb.shape[0] == parent_lins[0].A_lb.shape[0]
        and parent_lins[i].A_lb.shape[2] == parent_lins[0].A_lb.shape[2]
        for i in range(1, len(parent_lins))
    )
    if can_dual:
        lin = LinearBound(
            A_lb=torch.cat([lin.A_lb for lin in parent_lins], dim=1),
            b_lb=torch.cat([lin.b_lb for lin in parent_lins], dim=1),
            A_ub=torch.cat([lin.A_ub for lin in parent_lins], dim=1),
            b_ub=torch.cat([lin.b_ub for lin in parent_lins], dim=1),
        )
        frame = parent_frames[0]
        lb, ub = _concretize(lin, *frame)
    else:
        lb = torch.cat([box.lb for box in parent_boxes], dim=concat_dim)
        ub = torch.cat([box.ub for box in parent_boxes], dim=concat_dim)
        lin, frame = _reset_forward_box(lb, ub, device, dtype)
    out = Bounds(lb, ub)
    return out, out, lin, frame


def backward_concat(L, nu, bounds_dict, preds, M: int = 1, alpha=None):
    """CONCAT backward: split ν into per-predecessor slices along the feature axis.

    The forward concatenates predecessor outputs along ``concat_dim`` (feature
    axis, the only non-batch axis for the flattened dual tensors), so backward
    routes each predecessor the contiguous ν slice it produced. Slice widths come
    from the predecessor output boxes in ``bounds_dict``; ``contrib`` is zero
    because concatenation adds no constant. ``alpha`` is unused (no relaxation).
    """
    nu_flat = nu.flatten(start_dim=1)
    widths = [bounds_dict[pid].lb.flatten(start_dim=1).shape[-1] for pid in preds]
    total = sum(widths)
    if total != nu_flat.shape[-1]:
        raise ValueError(
            f"backward_concat: pred widths {widths} sum to {total}, "
            f"expected ν width {nu_flat.shape[-1]}"
        )
    pred_nus: List[torch.Tensor] = []
    offset = 0
    for width in widths:
        pred_nus.append(nu_flat[:, offset:offset + width].clone())
        offset += width
    contrib = torch.zeros(nu.shape[0], dtype=nu.dtype, device=nu.device)
    return pred_nus, contrib


class DualTF:
    """Backward-kernel registry holder for the dual solver.

    Holder of three registries (forward, backward, unimplemented). Dual
    semantics live in DualSolver's backward pass, not in propagated LP
    constraints, so this class is intentionally NOT a TransferFunction.

      * ``_FORWARD_REGISTRY`` — per-kind forward dispatch consumed by
        ``compute_forward_bounds`` (still a real forward computation, but
        invoked internally by ``DualSolver.evaluate_spec`` rather than via
        the analyze()/TF pipeline).
      * ``_BACKWARD_REGISTRY`` — per-kind backward dispatch consumed by
        ``DualSolver.compute_certified_bound``. Each entry has signature
        ``(L, nu, bounds_dict, preds, M=1, alpha=None) -> (pred_nus, contrib)``.
      * ``_UNIMPLEMENTED_KINDS`` — kinds whose backward is a stub
        (raises ``NotImplementedError``); ``supports_layer`` filters them
        so dual-incompatible nets get cleanly SKIPPED.

    DualSolver instantiates this internally; external code uses ``--solver
    dual`` rather than touching DualTF directly.
    """

    _FORWARD_REGISTRY = {
        LayerKind.INPUT.value:      forward_identity,
        LayerKind.INPUT_SPEC.value: forward_identity,
        LayerKind.ASSERT.value:     forward_identity,
        LayerKind.CONSTANT.value:   forward_constant,
        LayerKind.DENSE.value:      forward_dense,
        LayerKind.BIAS.value:       forward_bias,
        LayerKind.SCALE.value:      forward_scale,
        LayerKind.BN.value:         forward_bn,
        LayerKind.RELU.value:       forward_relu,
        LayerKind.LRELU.value:      forward_lrelu,
        "LEAKY_RELU":               forward_lrelu,   # alias (not a LayerKind member)
        LayerKind.SIGMOID.value:    forward_sigmoid,
        LayerKind.TANH.value:       forward_tanh,
        LayerKind.ERF.value:        forward_erf,
        LayerKind.SQRT.value:       forward_sqrt,
        LayerKind.SIN.value:        forward_sin,
        LayerKind.COS.value:        forward_cos,
        LayerKind.QUANTIZE.value:   forward_quantize,
        LayerKind.CONV2D.value:     forward_conv2d,
        LayerKind.MAXPOOL2D.value:  forward_maxpool2d,
        LayerKind.AVGPOOL2D.value:  forward_avgpool2d,
        LayerKind.FLATTEN.value:    forward_reshape,
        LayerKind.RESHAPE.value:    forward_reshape,
        LayerKind.TRANSPOSE.value:  forward_identity,
        LayerKind.SQUEEZE.value:    forward_identity,
        LayerKind.UNSQUEEZE.value:  forward_identity,
        LayerKind.EXPAND.value:     forward_expand,
        LayerKind.GATHER.value:     forward_gather,
        LayerKind.SLICE.value:      forward_slice,
        LayerKind.MEAN.value:       forward_mean,
        LayerKind.ADD.value:        forward_add,
        LayerKind.CONCAT.value:     forward_concat,
        LayerKind.LSTM.value:       forward_lstm,
        LayerKind.GRU.value:        forward_gru,
        LayerKind.ATT_SCORES.value: forward_attention,
        LayerKind.ATT_MIX.value:    forward_attention,
        LayerKind.MATMUL.value:     forward_matmul,
        LayerKind.MHA_SPLIT.value:  forward_mha,
        LayerKind.MHA_JOIN.value:   forward_mha,
        LayerKind.MASK_ADD.value:   forward_mha,
        LayerKind.LAYERNORM.value:  forward_layernorm,
        LayerKind.SOFTMAX.value:    forward_softmax,
        LayerKind.GELU.value:       forward_gelu,
    }

    # η placement invariant (split-constraint KKT multipliers)
    # ---------------------------------------------------------
    # The η subtraction `nu = nu - eta * signs` executes immediately BEFORE the
    # activation-layer's backward handler runs, inside the reverse-topological
    # iteration. `etas` and `split_signs` are keyed by ACTIVATION layer.id
    # (RELU, LRELU, SIGMOID, TANH), not by the upstream linear layer.id. This
    # yields
    #     nu_pre = slope · (nu_post − η · sign) = slope · nu_post − (slope · η) · sign
    # so the effective multiplier at the pre-activation is (slope · η), which
    # remains ≥ 0 under the η ≥ 0 invariant enforced by the optimizer's projection.
    #
    # Projection invariant (joint α/η optimization)
    # ---------------------------------------------
    # After every `optimizer.step()`:
    #     α.data.clamp_(0.0, 1.0)   # slope variables, α ∈ [0, 1]
    #     η.data.clamp_(min=0)      # KKT multipliers, η ≥ 0

    _BACKWARD_REGISTRY = {
        LayerKind.INPUT.value:      backward_identity,
        LayerKind.INPUT_SPEC.value: backward_identity,
        LayerKind.ASSERT.value:     backward_identity,
        LayerKind.CONSTANT.value:   backward_constant,
        LayerKind.DENSE.value:      backward_dense,
        LayerKind.BIAS.value:       backward_bias,
        LayerKind.SCALE.value:      backward_scale,
        LayerKind.BN.value:         backward_bn,
        LayerKind.RELU.value:       backward_relu,
        LayerKind.LRELU.value:      backward_relu,
        "LEAKY_RELU":               backward_relu,   # alias (not a LayerKind member)
        LayerKind.SIGMOID.value:    backward_sigmoid,
        LayerKind.TANH.value:       backward_tanh,
        LayerKind.ERF.value:        backward_erf,
        LayerKind.SQRT.value:       backward_sqrt,
        LayerKind.SIN.value:        backward_sin,
        LayerKind.COS.value:        backward_cos,
        LayerKind.QUANTIZE.value:   backward_quantize,
        LayerKind.CONV2D.value:     backward_conv2d,
        LayerKind.MAXPOOL2D.value:  backward_maxpool2d,
        LayerKind.AVGPOOL2D.value:  backward_avgpool2d,
        LayerKind.FLATTEN.value:    backward_identity,
        LayerKind.RESHAPE.value:    backward_identity,
        LayerKind.TRANSPOSE.value:  backward_identity,
        LayerKind.SQUEEZE.value:    backward_identity,
        LayerKind.UNSQUEEZE.value:  backward_identity,
        LayerKind.EXPAND.value:     backward_identity,
        LayerKind.GATHER.value:     backward_identity,
        LayerKind.SLICE.value:      backward_slice,
        LayerKind.MEAN.value:       backward_mean,
        LayerKind.ADD.value:        backward_add,
        LayerKind.CONCAT.value:     backward_concat,
        LayerKind.LSTM.value:       backward_lstm,
        LayerKind.GRU.value:        backward_gru,
        LayerKind.ATT_SCORES.value: backward_attention,
        LayerKind.ATT_MIX.value:    backward_attention,
        LayerKind.MATMUL.value:     backward_matmul,
        LayerKind.MHA_SPLIT.value:  backward_mha,
        LayerKind.MHA_JOIN.value:   backward_mha,
        LayerKind.MASK_ADD.value:   backward_mha,
        LayerKind.LAYERNORM.value:  backward_layernorm,
        LayerKind.SOFTMAX.value:    backward_softmax,
        LayerKind.GELU.value:       backward_gelu,
    }

    _UNIMPLEMENTED_KINDS = frozenset({
        LayerKind.LSTM.value,
        LayerKind.GRU.value,
        LayerKind.MHA_SPLIT.value,
        LayerKind.MHA_JOIN.value,
        LayerKind.MASK_ADD.value,
        # Backward kernels for these are stubs that raise NotImplementedError
        # at runtime. Listing them here makes supports_layer return False so
        # upstream callers (validate_verifier) cleanly SKIP affected nets
        # instead of surfacing runtime ERROR. ATT_SCORES / ATT_MIX / CONCAT now
        # have real backward kernels and are intentionally absent.
    })

    def supports_layer(self, layer_kind: str) -> bool:
        k = layer_kind.upper()
        return k in self._BACKWARD_REGISTRY and k not in self._UNIMPLEMENTED_KINDS


# Explicit stub registry: any handler whose semantics are "raise NotImplementedError"
# goes here. Membership is the ground truth for stub detection; net_factory filters
# by identity against these sets.
# To implement a stub: fill its body AND remove it from this set in the same commit.
_FORWARD_STUBS = frozenset({
    forward_lstm, forward_gru, forward_mha,
})
_BACKWARD_STUBS = frozenset({
    backward_lstm, backward_gru, backward_mha,
})

# --- registry invariants (fire once at module import) ---
assert set(DualTF._FORWARD_REGISTRY.keys()) == set(DualTF._BACKWARD_REGISTRY.keys()), (
    f"DualTF registry keyset mismatch: "
    f"forward-only={set(DualTF._FORWARD_REGISTRY) - set(DualTF._BACKWARD_REGISTRY)}, "
    f"backward-only={set(DualTF._BACKWARD_REGISTRY) - set(DualTF._FORWARD_REGISTRY)}"
)

# _UNIMPLEMENTED_KINDS must exactly equal the set of layer kinds whose backward
# handler is a stub (raises NotImplementedError). Drift between these two sets
# is a real risk: implementing a stub without updating _UNIMPLEMENTED_KINDS
# would silently keep skipping a now-working kind; the reverse causes runtime
# NotImplementedError on a kind that supports_layer claims to support.
_stub_kinds_from_registry = frozenset(
    k for k, fn in DualTF._BACKWARD_REGISTRY.items() if fn in _BACKWARD_STUBS
)
assert DualTF._UNIMPLEMENTED_KINDS == _stub_kinds_from_registry, (
    f"DualTF _UNIMPLEMENTED_KINDS drift: "
    f"declared={sorted(DualTF._UNIMPLEMENTED_KINDS)}, "
    f"stub-derived={sorted(_stub_kinds_from_registry)}"
)
