#===- act/back_end/dual_tf/tf_smooth.py - Smooth Activation Dual TF -----====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025- ACT Team
# Licensed under AGPLv3+; distributed without warranty.
#===---------------------------------------------------------------------===#
# Batch-aware smooth (S-shaped) activation backward.
# nu: [B, *shape] -> v_out: [B, *shape], contrib: [B].
#===---------------------------------------------------------------------===#

# Note: Gradient enablement for dual backward helpers is governed by the
# caller's torch.set_grad_enabled() context (see DualSolver.evaluate_spec).
# @torch.no_grad() decorators on these helpers were removed to allow
# gradient flow during robust training; verify_once / verify_bab paths
# remain under no_grad via their own outer guards.

import math

import torch
from typing import Tuple, Callable, Dict, Any, List
from act.back_end.core import Bounds
from .tf_forward import LinearBound, Frame, _reset_forward_box


# ---- Shared primitives ----

def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)

def dsigmoid(x: torch.Tensor) -> torch.Tensor:
    s = torch.sigmoid(x); return s * (1 - s)

def tanh(x: torch.Tensor) -> torch.Tensor:
    return torch.tanh(x)

def dtanh(x: torch.Tensor) -> torch.Tensor:
    return 1 - torch.tanh(x) ** 2

def erf(x: torch.Tensor) -> torch.Tensor:
    return torch.erf(x)

def derf(x: torch.Tensor) -> torch.Tensor:
    return (2.0 / math.sqrt(math.pi)) * torch.exp(-x * x)


def _periodic_critical_exists(lo: torch.Tensor, hi: torch.Tensor, offset: float, period: float) -> torch.Tensor:
    eps = torch.finfo(lo.dtype).eps * 16.0
    a = (lo - offset) / period - eps
    b = (hi - offset) / period + eps
    return torch.floor(b) >= torch.ceil(a)


def _sin_interval(lo: torch.Tensor, hi: torch.Tensor) -> Bounds:
    two_pi = 2.0 * math.pi
    finite = torch.isfinite(lo) & torch.isfinite(hi) & (torch.abs(lo) <= 1.0e12) & (torch.abs(hi) <= 1.0e12)
    hi = torch.maximum(hi, lo)
    width = hi - lo
    endpoint_lo = torch.sin(lo)
    endpoint_hi = torch.sin(hi)
    base_lb = torch.minimum(endpoint_lo, endpoint_hi)
    base_ub = torch.maximum(endpoint_lo, endpoint_hi)
    full_period = width >= two_pi
    has_max = _periodic_critical_exists(lo, hi, 0.5 * math.pi, two_pi)
    has_min = _periodic_critical_exists(lo, hi, -0.5 * math.pi, two_pi)
    lb = torch.where(has_min | full_period, torch.full_like(base_lb, -1.0), base_lb)
    ub = torch.where(has_max | full_period, torch.full_like(base_ub, 1.0), base_ub)
    lb = torch.where(finite, lb, torch.full_like(lb, -1.0))
    ub = torch.where(finite, ub, torch.full_like(ub, 1.0))
    return Bounds(lb, ub)


def _cos_interval(lo: torch.Tensor, hi: torch.Tensor) -> Bounds:
    two_pi = 2.0 * math.pi
    finite = torch.isfinite(lo) & torch.isfinite(hi) & (torch.abs(lo) <= 1.0e12) & (torch.abs(hi) <= 1.0e12)
    hi = torch.maximum(hi, lo)
    width = hi - lo
    endpoint_lo = torch.cos(lo)
    endpoint_hi = torch.cos(hi)
    base_lb = torch.minimum(endpoint_lo, endpoint_hi)
    base_ub = torch.maximum(endpoint_lo, endpoint_hi)
    full_period = width >= two_pi
    has_max = _periodic_critical_exists(lo, hi, 0.0, two_pi)
    has_min = _periodic_critical_exists(lo, hi, math.pi, two_pi)
    lb = torch.where(has_min | full_period, torch.full_like(base_lb, -1.0), base_lb)
    ub = torch.where(has_max | full_period, torch.full_like(base_ub, 1.0), base_ub)
    lb = torch.where(finite, lb, torch.full_like(lb, -1.0))
    ub = torch.where(finite, ub, torch.full_like(ub, 1.0))
    return Bounds(lb, ub)


def _quantize_params_flat(L: Any, n: int, device: torch.device, dtype: torch.dtype):
    scale = L.params["scale"].to(device=device, dtype=dtype).flatten()
    zero_point = L.params["zero_point"].to(device=device, dtype=dtype).flatten()
    if scale.numel() == 1:
        scale = scale.expand(n)
    elif scale.numel() != n:
        raise NotImplementedError(f"quantize:{L.id}: scale with {scale.numel()} entries cannot broadcast to flat size {n}")
    if zero_point.numel() == 1:
        zero_point = zero_point.expand(n)
    elif zero_point.numel() != n:
        raise NotImplementedError(f"quantize:{L.id}: zero_point with {zero_point.numel()} entries cannot broadcast to flat size {n}")
    if torch.any(scale <= 0):
        raise ValueError(f"quantize:{L.id}: scale must be positive")
    qmin = torch.full((n,), float(L.params["qmin"]), device=device, dtype=dtype)
    qmax = torch.full((n,), float(L.params["qmax"]), device=device, dtype=dtype)
    return scale, zero_point, qmin, qmax


def _quantize_qdq_value(x: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, qmin: torch.Tensor, qmax: torch.Tensor) -> torch.Tensor:
    return scale * torch.clamp(torch.round(x / scale), min=qmin - zero_point, max=qmax - zero_point)


def _dual_constant_box_backward(nu: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor, M: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    BM = nu.shape[0]
    assert BM % M == 0, f"constant-box backward: nu batch {BM} not divisible by M={M}"
    B = BM // M
    v_flat = nu.flatten(start_dim=1)
    l_B = lower.flatten(start_dim=1) if lower.dim() >= 2 else lower.flatten().unsqueeze(0).expand(B, -1)
    u_B = upper.flatten(start_dim=1) if upper.dim() >= 2 else upper.flatten().unsqueeze(0).expand(B, -1)
    n = min(v_flat.shape[-1], l_B.shape[-1])
    if v_flat.shape[-1] != l_B.shape[-1]:
        v_flat = v_flat[..., :n]
        l_B = l_B[..., :n]
        u_B = u_B[..., :n]
    l = l_B.unsqueeze(1)
    u = u_B.unsqueeze(1)
    v = v_flat.view(B, M, n)
    b = torch.where(v >= 0, l, u)
    v_out = torch.zeros_like(v)
    contrib = (v * b).sum(dim=-1).view(BM)
    return v_out.view(BM, n), contrib


def compute_smooth_relaxation(
    l: torch.Tensor, u: torch.Tensor,
    func: Callable[[torch.Tensor], torch.Tensor],
    dfunc: Callable[[torch.Tensor], torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Linear relaxation (k_lo, b_lo, k_hi, b_hi) for S-shaped f on [l, u].

    Works element-wise on any broadcastable shape (including batched [B, n]).
    """
    from .tf_mlp import _repair_degenerate_interval
    u = _repair_degenerate_interval(l, u, "smooth_relaxation")
    f_l, f_u = func(l), func(u)
    denom = (u - l).clamp(min=1e-12)
    k_chord = (f_u - f_l) / denom
    k_lower, k_upper = k_chord.clone(), k_chord.clone()
    b_lower = f_l - k_lower * l
    b_upper = f_l - k_upper * l

    mask_pos = l >= 0
    if mask_pos.any():
        k_tan = dfunc(l)
        k_lower = torch.where(mask_pos, k_tan, k_lower)
        b_lower = torch.where(mask_pos, f_l - k_tan * l, b_lower)

    mask_neg = u <= 0
    if mask_neg.any():
        k_tan = dfunc(u)
        k_upper = torch.where(mask_neg, k_tan, k_upper)
        b_upper = torch.where(mask_neg, f_u - k_tan * u, b_upper)

    return k_lower, b_lower, k_upper, b_upper


def dual_smooth_backward(
    nu: torch.Tensor, bounds: Bounds,
    func: Callable[[torch.Tensor], torch.Tensor],
    dfunc: Callable[[torch.Tensor], torch.Tensor],
    M: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batched smooth activation backward (sigmoid/tanh) with lazy M-broadcast.

    Same broadcast pattern as :func:`dual_relu_backward`: relaxation
    coefficients ``(k_lower, b_lower, k_upper, b_upper)`` depend on the
    bounds only (spec-agnostic) and are computed once at ``[B, 1, n]``,
    then broadcast against ``nu`` viewed at ``[B, M, n]``.

    Args:
        nu: dual variable, shape ``[B*M, *shape]``.
        bounds: layer bounds, shape ``[B, *shape]``. NOT M-expanded.
        func/dfunc: activation function and its derivative.
        M: spec-row multiplicity (default 1).
    """
    BM = nu.shape[0]
    assert BM % M == 0, f"dual_smooth_backward: nu batch {BM} not divisible by M={M}"
    B = BM // M

    v_flat = nu.flatten(start_dim=1)                              # [BM, n]
    l_B = bounds.lb.flatten(start_dim=1) if bounds.lb.dim() >= 2 \
          else bounds.lb.flatten().unsqueeze(0).expand(B, -1)
    u_B = bounds.ub.flatten(start_dim=1) if bounds.ub.dim() >= 2 \
          else bounds.ub.flatten().unsqueeze(0).expand(B, -1)
    n = min(v_flat.shape[-1], l_B.shape[-1])
    if v_flat.shape[-1] != l_B.shape[-1]:
        v_flat = v_flat[..., :n]
        l_B = l_B[..., :n]
        u_B = u_B[..., :n]

    l = l_B.unsqueeze(1)                                          # [B, 1, n]
    u = u_B.unsqueeze(1)                                          # [B, 1, n]
    k_lower, b_lower, k_upper, b_upper = compute_smooth_relaxation(l, u, func, dfunc)

    v = v_flat.view(B, M, n)                                      # [B, M, n] view
    v_pos = v >= 0                                                # [B, M, n]
    k = torch.where(v_pos, k_lower, k_upper)                      # broadcast → [B, M, n]
    b = torch.where(v_pos, b_lower, b_upper)

    v_out = v * k                                                 # [B, M, n]
    contrib = (v * b).sum(dim=-1).view(BM)                        # [BM]
    return v_out.view(BM, n), contrib


# ---- SIGMOID ----

@torch.no_grad()
def forward_sigmoid(
    L: Any, parent_boxes: List[Bounds], parent_lins: List[LinearBound],
    parent_frames: List[Frame], preds: List[int], post_activation: bool,
    device: torch.device, dtype: torch.dtype,
) -> Tuple[Bounds, Bounds, LinearBound, Frame]:
    """Forward pass for SIGMOID activation.

    Body copied from tf_forward.py lines 426-430 (SIGMOID branch).
    Returns (stored, out, lin, frame).
    """
    parent_box = parent_boxes[0]
    pre_lb, pre_ub = parent_box.lb, parent_box.ub
    out = Bounds(torch.sigmoid(pre_lb), torch.sigmoid(pre_ub))
    stored = out if post_activation else Bounds(pre_lb, pre_ub)
    lin, frame = _reset_forward_box(out.lb, out.ub, device, dtype)
    return stored, out, lin, frame


def backward_sigmoid(L: Any, nu: torch.Tensor, bounds_dict: Dict[int, Bounds],
                     preds: List[int], M: int = 1, alpha=None
                     ) -> Tuple[List[torch.Tensor], torch.Tensor]:
    bounds = bounds_dict.get(L.id)
    if bounds is None:
        raise ValueError(f"backward_sigmoid: layer {L.id} missing bounds in bounds_dict")
    nu_out, contrib = dual_sigmoid_backward(nu, bounds, M)
    assert len(preds) == 1, f"SIGMOID expects 1 predecessor, got {len(preds)}"
    return [nu_out], contrib


def dual_sigmoid_backward(nu: torch.Tensor, bounds: Bounds, M: int = 1):
    return dual_smooth_backward(nu, bounds, sigmoid, dsigmoid, M)


# ---- TANH ----

@torch.no_grad()
def forward_tanh(
    L: Any, parent_boxes: List[Bounds], parent_lins: List[LinearBound],
    parent_frames: List[Frame], preds: List[int], post_activation: bool,
    device: torch.device, dtype: torch.dtype,
) -> Tuple[Bounds, Bounds, LinearBound, Frame]:
    """Forward pass for TANH activation.

    Body copied from tf_forward.py lines 432-436 (TANH branch).
    Returns (stored, out, lin, frame).
    """
    parent_box = parent_boxes[0]
    pre_lb, pre_ub = parent_box.lb, parent_box.ub
    out = Bounds(torch.tanh(pre_lb), torch.tanh(pre_ub))
    stored = out if post_activation else Bounds(pre_lb, pre_ub)
    lin, frame = _reset_forward_box(out.lb, out.ub, device, dtype)
    return stored, out, lin, frame


def backward_tanh(L: Any, nu: torch.Tensor, bounds_dict: Dict[int, Bounds],
                  preds: List[int], M: int = 1, alpha=None
                  ) -> Tuple[List[torch.Tensor], torch.Tensor]:
    bounds = bounds_dict.get(L.id)
    if bounds is None:
        raise ValueError(f"backward_tanh: layer {L.id} missing bounds in bounds_dict")
    nu_out, contrib = dual_tanh_backward(nu, bounds, M)
    assert len(preds) == 1, f"TANH expects 1 predecessor, got {len(preds)}"
    return [nu_out], contrib


def dual_tanh_backward(nu: torch.Tensor, bounds: Bounds, M: int = 1):
    return dual_smooth_backward(nu, bounds, tanh, dtanh, M)


# ---- ERF ----

@torch.no_grad()
def forward_erf(
    L: Any, parent_boxes: List[Bounds], parent_lins: List[LinearBound],
    parent_frames: List[Frame], preds: List[int], post_activation: bool,
    device: torch.device, dtype: torch.dtype,
) -> Tuple[Bounds, Bounds, LinearBound, Frame]:
    """Forward pass for ERF activation, mirroring TANH."""
    parent_box = parent_boxes[0]
    pre_lb, pre_ub = parent_box.lb, parent_box.ub
    out = Bounds(torch.erf(pre_lb), torch.erf(pre_ub))
    stored = out if post_activation else Bounds(pre_lb, pre_ub)
    lin, frame = _reset_forward_box(out.lb, out.ub, device, dtype)
    return stored, out, lin, frame


def backward_erf(L: Any, nu: torch.Tensor, bounds_dict: Dict[int, Bounds],
                 preds: List[int], M: int = 1, alpha=None
                 ) -> Tuple[List[torch.Tensor], torch.Tensor]:
    bounds = bounds_dict.get(L.id)
    if bounds is None:
        raise ValueError(f"backward_erf: layer {L.id} missing bounds in bounds_dict")
    nu_out, contrib = dual_erf_backward(nu, bounds, M)
    assert len(preds) == 1, f"ERF expects 1 predecessor, got {len(preds)}"
    return [nu_out], contrib


def dual_erf_backward(nu: torch.Tensor, bounds: Bounds, M: int = 1):
    return dual_smooth_backward(nu, bounds, erf, derf, M)


# ---- SQRT ----

@torch.no_grad()
def forward_sqrt(
    L: Any, parent_boxes: List[Bounds], parent_lins: List[LinearBound],
    parent_frames: List[Frame], preds: List[int], post_activation: bool,
    device: torch.device, dtype: torch.dtype,
) -> Tuple[Bounds, Bounds, LinearBound, Frame]:
    parent_box = parent_boxes[0]
    pre_lb, pre_ub = parent_box.lb, parent_box.ub
    lo_e = torch.clamp(pre_lb, min=0.0)
    hi_e = torch.clamp(pre_ub, min=0.0)
    out = Bounds(torch.sqrt(lo_e), torch.sqrt(hi_e))
    stored = out if post_activation else Bounds(pre_lb, pre_ub)
    lin, frame = _reset_forward_box(out.lb, out.ub, device, dtype)
    return stored, out, lin, frame


def backward_sqrt(L: Any, nu: torch.Tensor, bounds_dict: Dict[int, Bounds],
                  preds: List[int], M: int = 1, alpha=None
                  ) -> Tuple[List[torch.Tensor], torch.Tensor]:
    bounds = bounds_dict.get(L.id)
    if bounds is None:
        raise ValueError(f"backward_sqrt: layer {L.id} missing bounds in bounds_dict")
    nu_out, contrib = dual_sqrt_backward(nu, bounds, M)
    assert len(preds) == 1, f"SQRT expects 1 predecessor, got {len(preds)}"
    return [nu_out], contrib


def dual_sqrt_backward(nu: torch.Tensor, bounds: Bounds, M: int = 1):
    lo_e = torch.clamp(bounds.lb, min=0.0)
    hi_e = torch.clamp(bounds.ub, min=0.0)
    return _dual_constant_box_backward(nu, torch.sqrt(lo_e), torch.sqrt(hi_e), M)


# ---- SIN ----

@torch.no_grad()
def forward_sin(
    L: Any, parent_boxes: List[Bounds], parent_lins: List[LinearBound],
    parent_frames: List[Frame], preds: List[int], post_activation: bool,
    device: torch.device, dtype: torch.dtype,
) -> Tuple[Bounds, Bounds, LinearBound, Frame]:
    parent_box = parent_boxes[0]
    pre_lb, pre_ub = parent_box.lb, parent_box.ub
    out = _sin_interval(pre_lb, pre_ub)
    stored = out if post_activation else Bounds(pre_lb, pre_ub)
    lin, frame = _reset_forward_box(out.lb, out.ub, device, dtype)
    return stored, out, lin, frame


def backward_sin(L: Any, nu: torch.Tensor, bounds_dict: Dict[int, Bounds],
                 preds: List[int], M: int = 1, alpha=None
                 ) -> Tuple[List[torch.Tensor], torch.Tensor]:
    bounds = bounds_dict.get(L.id)
    if bounds is None:
        raise ValueError(f"backward_sin: layer {L.id} missing bounds in bounds_dict")
    nu_out, contrib = dual_sin_backward(nu, bounds, M)
    assert len(preds) == 1, f"SIN expects 1 predecessor, got {len(preds)}"
    return [nu_out], contrib


def dual_sin_backward(nu: torch.Tensor, bounds: Bounds, M: int = 1):
    box = _sin_interval(bounds.lb, bounds.ub)
    return _dual_constant_box_backward(nu, box.lb, box.ub, M)


# ---- COS ----

@torch.no_grad()
def forward_cos(
    L: Any, parent_boxes: List[Bounds], parent_lins: List[LinearBound],
    parent_frames: List[Frame], preds: List[int], post_activation: bool,
    device: torch.device, dtype: torch.dtype,
) -> Tuple[Bounds, Bounds, LinearBound, Frame]:
    parent_box = parent_boxes[0]
    pre_lb, pre_ub = parent_box.lb, parent_box.ub
    out = _cos_interval(pre_lb, pre_ub)
    stored = out if post_activation else Bounds(pre_lb, pre_ub)
    lin, frame = _reset_forward_box(out.lb, out.ub, device, dtype)
    return stored, out, lin, frame


def backward_cos(L: Any, nu: torch.Tensor, bounds_dict: Dict[int, Bounds],
                 preds: List[int], M: int = 1, alpha=None
                 ) -> Tuple[List[torch.Tensor], torch.Tensor]:
    bounds = bounds_dict.get(L.id)
    if bounds is None:
        raise ValueError(f"backward_cos: layer {L.id} missing bounds in bounds_dict")
    nu_out, contrib = dual_cos_backward(nu, bounds, M)
    assert len(preds) == 1, f"COS expects 1 predecessor, got {len(preds)}"
    return [nu_out], contrib


def dual_cos_backward(nu: torch.Tensor, bounds: Bounds, M: int = 1):
    box = _cos_interval(bounds.lb, bounds.ub)
    return _dual_constant_box_backward(nu, box.lb, box.ub, M)


# ---- QUANTIZE / QDQ real-valued map ----

@torch.no_grad()
def forward_quantize(
    L: Any, parent_boxes: List[Bounds], parent_lins: List[LinearBound],
    parent_frames: List[Frame], preds: List[int], post_activation: bool,
    device: torch.device, dtype: torch.dtype,
) -> Tuple[Bounds, Bounds, LinearBound, Frame]:
    parent_box = parent_boxes[0]
    pre_lb, pre_ub = parent_box.lb, parent_box.ub
    n = pre_lb.numel()
    scale, zp, qmin, qmax = _quantize_params_flat(L, n, pre_lb.device, pre_lb.dtype)
    lo = pre_lb.reshape(-1)
    hi = torch.maximum(pre_ub.reshape(-1), lo)
    z_lo = _quantize_qdq_value(lo, scale, zp, qmin, qmax)
    z_hi = _quantize_qdq_value(hi, scale, zp, qmin, qmax)
    out = Bounds(torch.minimum(z_lo, z_hi).reshape_as(pre_lb), torch.maximum(z_lo, z_hi).reshape_as(pre_ub))
    stored = out if post_activation else Bounds(pre_lb, pre_ub)
    lin, frame = _reset_forward_box(out.lb, out.ub, device, dtype)
    return stored, out, lin, frame


def backward_quantize(L: Any, nu: torch.Tensor, bounds_dict: Dict[int, Bounds],
                      preds: List[int], M: int = 1, alpha=None
                      ) -> Tuple[List[torch.Tensor], torch.Tensor]:
    bounds = bounds_dict.get(L.id)
    if bounds is None:
        raise ValueError(f"backward_quantize: layer {L.id} missing bounds in bounds_dict")
    nu_out, contrib = dual_quantize_backward(L, nu, bounds, M)
    assert len(preds) == 1, f"QUANTIZE expects 1 predecessor, got {len(preds)}"
    return [nu_out], contrib


def dual_quantize_backward(L: Any, nu: torch.Tensor, bounds: Bounds, M: int = 1):
    n = bounds.lb.numel()
    scale, zp, qmin, qmax = _quantize_params_flat(L, n, bounds.lb.device, bounds.lb.dtype)
    lo = bounds.lb.reshape(-1)
    hi = torch.maximum(bounds.ub.reshape(-1), lo)
    z_lo = _quantize_qdq_value(lo, scale, zp, qmin, qmax).reshape_as(bounds.lb)
    z_hi = _quantize_qdq_value(hi, scale, zp, qmin, qmax).reshape_as(bounds.ub)
    return _dual_constant_box_backward(nu, torch.minimum(z_lo, z_hi), torch.maximum(z_lo, z_hi), M)
