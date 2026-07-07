# ===- act/back_end/hybridz_tf/tf_mlp.py - HybridZ MLP Transfer Functions ====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===---------------------------------------------------------------------===#
#
# Purpose:
#   HybridZ MLP Transfer Functions. Implements HybridZ-based transfer functions
#   for MLP layers including dense, activation, and element-wise operations.
#
# ===---------------------------------------------------------------------===#

import torch
import torch.nn.functional as F
from act.back_end.core import Bounds, Fact
from act.back_end.solver.solver_hz import (
    HZono,
    hz_multiply,
    hz_add_const,
    hz_from_bounds,
    hz_compute_bounds,
    hz_concat,
    hz_sgm_add,
    hz_sub,
)
import act.back_end.interval_tf.tf_mlp as interval
import act.back_end.interval_tf.tf_cnn as interval_cnn


def _hz_fact(fact: Fact, hz: HZono) -> Fact:
    hb = hz_compute_bounds(hz)
    lb = torch.maximum(hb.lb.reshape_as(fact.bounds.lb), fact.bounds.lb)
    ub = torch.minimum(hb.ub.reshape_as(fact.bounds.ub), fact.bounds.ub)
    return Fact(
        bounds=Bounds(lb=lb, ub=ub),
        cons=fact.cons,
    )


def _hz_apply_per_batch_linear(hz: HZono, W: torch.Tensor, B: int) -> HZono:
    in_dim = W.shape[1]
    out_dim = W.shape[0]
    if B == 1:
        return hz_multiply(hz, W)
    ng = hz.Gc.shape[1]
    nb = hz.Gb.shape[1]
    c3 = hz.c.view(B, in_dim, 1)
    new_c = (W @ c3).reshape(B * out_dim, 1)
    if ng:
        new_Gc = (W @ hz.Gc.view(B, in_dim, ng)).reshape(B * out_dim, ng)
    else:
        new_Gc = hz.Gc.new_zeros(B * out_dim, 0)
    if nb:
        new_Gb = (W @ hz.Gb.view(B, in_dim, nb)).reshape(B * out_dim, nb)
    else:
        new_Gb = hz.Gb.new_zeros(B * out_dim, 0)
    return HZono(
        c=new_c, Gc=new_Gc, Gb=new_Gb,
        Ac=hz.Ac.clone(), Ab=hz.Ab.clone(), b=hz.b.clone(),
        eq_mask=None if hz.eq_mask is None else hz.eq_mask.clone(),
        col_ids=None if hz.col_ids is None else hz.col_ids.clone(),
        bcol_ids=None if hz.bcol_ids is None else hz.bcol_ids.clone(),
    )


def _hz_add_per_channel(hz: HZono, v: torch.Tensor, B: int) -> HZono:
    v = v.to(dtype=hz.c.dtype, device=hz.c.device).flatten()
    if B > 1:
        v = v.repeat(B)
    return hz_add_const(hz, v.view(-1, 1))


def _hz_scale_per_channel(hz: HZono, a: torch.Tensor, B: int) -> HZono:
    a = a.to(dtype=hz.c.dtype, device=hz.c.device).flatten()
    if B > 1:
        a = a.repeat(B)
    a_col = a.view(-1, 1)
    return HZono(
        c=a_col * hz.c,
        Gc=a_col * hz.Gc,
        Gb=a_col * hz.Gb,
        Ac=hz.Ac.clone(), Ab=hz.Ab.clone(), b=hz.b.clone(),
        eq_mask=None if hz.eq_mask is None else hz.eq_mask.clone(),
        col_ids=None if hz.col_ids is None else hz.col_ids.clone(),
        bcol_ids=None if hz.bcol_ids is None else hz.bcol_ids.clone(),
    )


def _hz_is_point(hz: HZono) -> bool:
    gc_zero = hz.Gc.numel() == 0 or bool((hz.Gc.abs() <= 1e-12).all())
    gb_zero = hz.Gb.numel() == 0 or bool((hz.Gb.abs() <= 1e-12).all())
    return gc_zero and gb_zero


def _broadcast_flat(v: torch.Tensor, n: int) -> torch.Tensor:
    v = v.flatten()
    if v.numel() == n:
        return v
    if v.numel() == 1:
        return v.expand(n)
    if n % v.numel() == 0:
        return v.repeat(n // v.numel())
    raise ValueError(f"cannot broadcast {v.numel()} values to {n}")


def _hz_scale_elementwise(hz: HZono, a: torch.Tensor) -> HZono:
    a = _broadcast_flat(a.to(dtype=hz.c.dtype, device=hz.c.device), hz.c.shape[0])
    acol = a.view(-1, 1)
    return HZono(
        c=acol * hz.c,
        Gc=acol * hz.Gc,
        Gb=acol * hz.Gb,
        Ac=hz.Ac.clone(),
        Ab=hz.Ab.clone(),
        b=hz.b.clone(),
        eq_mask=None if hz.eq_mask is None else hz.eq_mask.clone(),
        col_ids=None if hz.col_ids is None else hz.col_ids.clone(),
        bcol_ids=None if hz.bcol_ids is None else hz.bcol_ids.clone(),
    )


def _prod(shape) -> int:
    out = 1
    for dim in shape:
        out *= int(dim)
    return out


def _shared_const_block(flat: torch.Tensor, shape, batch: int):
    dim = _prod(shape)
    if dim == 0:
        return None
    if flat.numel() == dim:
        return flat.view(*shape)
    if batch > 0 and flat.numel() == batch * dim:
        blocks = flat.view(batch, dim)
        if bool(torch.allclose(blocks, blocks[:1].expand_as(blocks))):
            return blocks[0].view(*shape)
    return None


def _hz_matmul_const(L, hz: HZono, const: torch.Tensor, *, variable_is_left: bool):
    dtype, device = hz.c.dtype, hz.c.device
    x_shape = tuple(int(d) for d in L.params["x_shape"])
    y_shape = tuple(int(d) for d in L.params["y_shape"])
    in_shape = x_shape if variable_is_left else y_shape
    in_dim = _prod(in_shape)
    if in_dim == 0 or hz.c.shape[0] % in_dim != 0:
        return None
    B = hz.c.shape[0] // in_dim
    C = const.to(dtype=dtype, device=device).flatten()
    if variable_is_left:
        W = _shared_const_block(C, y_shape, B)
        if W is None:
            return None
        eye = torch.eye(in_dim, dtype=dtype, device=device).view(in_dim, *x_shape)
        out = torch.matmul(eye, W).reshape(in_dim, -1)
    else:
        W = _shared_const_block(C, x_shape, B)
        if W is None:
            return None
        eye = torch.eye(in_dim, dtype=dtype, device=device).view(in_dim, *y_shape)
        out = torch.matmul(W, eye).reshape(in_dim, -1)
    return _hz_apply_per_batch_linear(hz, out.t().contiguous(), B)


# ============================================================================
# HZ layer functions: HZono -> Optional[HZono] per layer kind
# Each takes (L, hz_in, tf) and returns the transformed HZono or None.
# ============================================================================


# --- HZ transfer functions (MLP) ---


def tf_dense(L, bounds, tf):
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        W = L.params["weight"].to(hz_in.c)
        in_dim = W.shape[1]
        B = hz_in.c.shape[0] // in_dim
        hz = _hz_apply_per_batch_linear(hz_in, W, B)
        bias = L.params.get("bias")
        if bias is not None:
            hz = _hz_add_per_channel(hz, bias, B)
        tf._hz_cache[L.id] = hz
    fact = interval.tf_dense(L, bounds)
    if hz_in is not None:
        return _hz_fact(fact, tf._hz_cache[L.id])
    return fact


def tf_bias(L, bounds, tf):
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        c = L.params["c"].to(hz_in.c)
        if c.ndim == 1:
            B = hz_in.c.shape[0] // c.numel()
            tf._hz_cache[L.id] = _hz_add_per_channel(hz_in, c, B)
        else:
            tf._hz_cache[L.id] = hz_add_const(hz_in, c)
    fact = interval.tf_bias(L, bounds)
    if hz_in is not None:
        return _hz_fact(fact, tf._hz_cache[L.id])
    return fact


def tf_scale(L, bounds, tf):
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        a = L.params["a"].to(hz_in.c).flatten()
        B = hz_in.c.shape[0] // a.numel()
        tf._hz_cache[L.id] = _hz_scale_per_channel(hz_in, a, B)
    fact = interval.tf_scale(L, bounds)
    if hz_in is not None:
        return _hz_fact(fact, tf._hz_cache[L.id])
    return fact


def tf_relu(L, bounds, tf):
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        tf._hz_cache[L.id] = hz_reduce(hz_apply_relu(hz_in))
    fact = interval.tf_relu(L, bounds)
    if hz_in is not None:
        return _hz_fact(fact, tf._hz_cache[L.id])
    return fact


def tf_lrelu(L, bounds, tf):
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        tf._hz_cache[L.id] = hz_reduce(
            hz_apply_leaky_relu(hz_in, float(L.params.get("negative_slope", 0.01)))
        )
    fact = interval.tf_lrelu(L, bounds)
    if hz_in is not None:
        return _hz_fact(fact, tf._hz_cache[L.id])
    return fact


def tf_tanh(L, bounds, tf):
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        tf._hz_cache[L.id] = hz_reduce(hz_apply_tanh(hz_in, K=tf._tanh_K))
    fact = interval.tf_tanh(L, bounds)
    if hz_in is not None:
        return _hz_fact(fact, tf._hz_cache[L.id])
    return fact


def tf_sigmoid(L, bounds, tf):
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        tf._hz_cache[L.id] = hz_reduce(hz_apply_sigmoid(hz_in, K=tf._sigmoid_K))
    fact = interval.tf_sigmoid(L, bounds)
    if hz_in is not None:
        return _hz_fact(fact, tf._hz_cache[L.id])
    return fact


def tf_erf(L, bounds, tf):
    tf._hz_cache[L.id] = None
    return interval.tf_erf(L, bounds)


def tf_sqrt(L, bounds, tf):
    tf._hz_cache[L.id] = None
    return interval.tf_sqrt(L, bounds)


def tf_sin(L, bounds, tf):
    tf._hz_cache[L.id] = None
    return interval.tf_sin(L, bounds)


def tf_cos(L, bounds, tf):
    tf._hz_cache[L.id] = None
    return interval.tf_cos(L, bounds)


def tf_quantize(L, bounds, tf):
    tf._hz_cache[L.id] = None
    return interval.tf_quantize(L, bounds)


def tf_abs(L, bounds, tf):
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        dtype, device = hz_in.c.dtype, hz_in.c.device
        bds = hz_compute_bounds(hz_in)
        lb_out = torch.where(
            bds.lb >= 0,
            bds.lb,
            torch.where(bds.ub <= 0, -bds.ub, torch.zeros_like(bds.lb)),
        )
        tf._hz_cache[L.id] = hz_from_bounds(
            Bounds(lb=lb_out, ub=torch.maximum(bds.lb.abs(), bds.ub.abs())),
            dtype,
            device,
        )
    fact = interval.tf_abs(L, bounds)
    if hz_in is not None:
        return _hz_fact(fact, tf._hz_cache[L.id])
    return fact


def tf_bn(L, bounds, tf):
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        A, c = L.params["A"], L.params["c"]
        B = hz_in.c.shape[0] // A.numel()
        tf._hz_cache[L.id] = _hz_add_per_channel(
            _hz_scale_per_channel(hz_in, A, B), c, B
        )
    fact = interval.tf_bn(L, bounds)
    if hz_in is not None:
        return _hz_fact(fact, tf._hz_cache[L.id])
    return fact


def tf_add(L, bounds, tf):
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        preds = tf._net.preds.get(L.id, [])
        hz2 = tf._hz_cache.get(preds[1]) if len(preds) > 1 else None
        if hz2 is not None:
            tf._hz_cache[L.id] = hz_sgm_add(hz_in, hz2)
        else:
            hz_in = None
    fact = interval.tf_add(
        L,
        tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 0),
        tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 1),
    )
    if hz_in is not None:
        return _hz_fact(fact, tf._hz_cache[L.id])
    return fact


def tf_mul(L, bounds, tf):
    bx = tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 0)
    by = tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 1)
    fact = interval.tf_mul(L, bx, by)
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        preds = tf._net.preds.get(L.id, [])
        hz2 = tf._hz_cache.get(preds[1]) if len(preds) > 1 else None
        if hz2 is not None:
            if _hz_is_point(hz2):
                tf._hz_cache[L.id] = _hz_scale_elementwise(hz_in, hz2.c)
            elif _hz_is_point(hz_in):
                tf._hz_cache[L.id] = _hz_scale_elementwise(hz2, hz_in.c)
            else:
                tf._hz_cache.pop(L.id, None)
                return fact
        else:
            hz_in = None
    if hz_in is not None:
        return _hz_fact(fact, tf._hz_cache[L.id])
    return fact


def tf_div(L, bounds, tf):
    bx = tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 0)
    by = tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 1)
    fact = interval.tf_div(L, bx, by)
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        preds = tf._net.preds.get(L.id, [])
        hz2 = tf._hz_cache.get(preds[1]) if len(preds) > 1 else None
        if hz2 is not None and _hz_is_point(hz2):
            denom = _broadcast_flat(
                hz2.c.to(dtype=hz_in.c.dtype, device=hz_in.c.device),
                hz_in.c.shape[0],
            )
            if bool((denom.abs() > 1e-12).all()):
                tf._hz_cache[L.id] = _hz_scale_elementwise(hz_in, 1.0 / denom)
                return _hz_fact(fact, tf._hz_cache[L.id])
        tf._hz_cache.pop(L.id, None)
    return fact


def tf_constant(L, bounds, tf):
    val = L.params["value"].flatten()
    n = val.numel()
    if bounds is not None and n > 0:
        in_numel = int(bounds.lb.numel())
        if in_numel > 0 and in_numel % n == 0:
            B = in_numel // n
            if B > 1:
                val = val.repeat(B)
                n = val.numel()
    tf._hz_cache[L.id] = HZono(
        c=val.view(-1, 1),
        Gc=val.new_zeros(n, 0),
        Gb=val.new_zeros(n, 0),
        Ac=val.new_zeros(0, 0),
        Ab=val.new_zeros(0, 0),
        b=val.new_zeros(0, 1),
    )
    return interval.tf_constant(L, bounds)


def tf_sign(L, bounds, tf):
    tf._hz_cache.pop(L.id, None)
    return interval.tf_sign(L, bounds)


def tf_compare(L, bounds, tf):
    tf._hz_cache.pop(L.id, None)
    return interval.tf_compare(
        L,
        tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 0),
        tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 1),
    )


def tf_where(L, bounds, tf):
    tf._hz_cache.pop(L.id, None)
    return interval.tf_where(
        L,
        tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 0),
        tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 1),
        tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 2),
    )


def tf_matmul(L, bounds, tf):
    bx = tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 0)
    by = tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 1)
    fact = interval.tf_matmul(L, bx, by)
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        preds = tf._net.preds.get(L.id, [])
        hz2 = tf._hz_cache.get(preds[1]) if len(preds) > 1 else None
        if hz2 is not None:
            out = None
            if _hz_is_point(hz2):
                out = _hz_matmul_const(L, hz_in, hz2.c, variable_is_left=True)
            elif _hz_is_point(hz_in):
                out = _hz_matmul_const(L, hz2, hz_in.c, variable_is_left=False)
            if out is not None:
                tf._hz_cache[L.id] = out
                return _hz_fact(fact, tf._hz_cache[L.id])
    tf._hz_cache.pop(L.id, None)
    return fact


def tf_arg_extremum(L, bounds, tf):
    tf._hz_cache.pop(L.id, None)
    return interval.tf_arg_extremum(L, bounds)


def _row_indices_upsample_nearest(L, n_in: int, n_out: int):
    mode = str(L.params.get("mode", "nearest")).lower()
    if mode != "nearest":
        return None
    in_shape = L.params.get("input_shape")
    if in_shape is None:
        return None
    in_shape = tuple(int(d) for d in in_shape)
    if _prod(in_shape) != int(n_in) or len(in_shape) < 3:
        return None
    view_shape = (1, *in_shape) if len(in_shape) == 3 else in_shape
    spatial_rank = len(view_shape) - 2
    size = L.params.get("size")
    scale_factor = L.params.get("scale_factor")
    if size is not None and isinstance(size, (list, tuple)):
        size = tuple(int(s) for s in size)
        if len(size) > spatial_rank:
            size = size[-spatial_rank:]
    if scale_factor is not None and isinstance(scale_factor, (list, tuple)):
        scale_factor = tuple(float(s) for s in scale_factor)
        if len(scale_factor) > spatial_rank:
            scale_factor = scale_factor[-spatial_rank:]
    if size is None and scale_factor is None:
        out_shape = L.params.get("output_shape")
        if out_shape is None:
            return None
        out_shape = tuple(int(d) for d in out_shape)
        out_view_shape = (1, *out_shape) if len(out_shape) == 3 else out_shape
        if len(out_view_shape) != len(view_shape):
            return None
        size = out_view_shape[2:]
    base = torch.arange(int(n_in), dtype=torch.float64).view(*view_shape)
    idx = F.interpolate(base, size=size, scale_factor=scale_factor, mode="nearest")
    idx = idx.reshape(-1).long()
    return idx if idx.numel() == int(n_out) else None


def tf_upsample(L, bounds, tf):
    fact = interval_cnn.tf_upsample(L, bounds)
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is None:
        return fact
    rows = _row_indices_upsample_nearest(L, hz_in.c.shape[0], fact.bounds.lb.numel())
    if rows is None:
        tf._hz_cache.pop(L.id, None)
        return fact
    tf._hz_cache[L.id] = _hz_gather_rows(hz_in, rows.to(device=hz_in.c.device))
    return _hz_fact(fact, tf._hz_cache[L.id])


def _row_indices_slice(L, n: int):
    if "input_shape" not in L.params:
        return None
    inp_shape = tuple(int(d) for d in L.params["input_shape"])
    per = _prod(inp_shape)
    if per == 0 or int(n) % per != 0:
        return None
    batch = int(n) // per
    idx = torch.arange(int(n)).view(batch, *inp_shape)
    starts = L.params.get("starts", [])
    ends = L.params.get("ends", [])
    axes = L.params.get("axes", list(range(len(inp_shape))))
    steps = L.params.get("steps", [1] * len(axes))
    slices = [slice(None)] * (len(inp_shape) + 1)
    for i, axis in enumerate(axes):
        axis = int(axis)
        end = ends[i]
        if end > inp_shape[axis]:
            end = inp_shape[axis]
        slices[axis + 1] = slice(starts[i], end, steps[i])
    return idx[tuple(slices)].reshape(-1)


def _row_indices_gather(L, n: int):
    if "input_shape" not in L.params:
        return None
    inp_shape = tuple(int(d) for d in L.params["input_shape"])
    per = _prod(inp_shape)
    if per == 0 or int(n) % per != 0:
        return None
    batch = int(n) // per
    axis = int(L.params.get("axis", 0))
    if axis < 0:
        axis += len(inp_shape)
    raw_idx = L.params["indices"]
    if isinstance(raw_idx, (list, tuple)):
        indices = torch.tensor(raw_idx, dtype=torch.long)
    elif hasattr(raw_idx, "detach"):
        indices = raw_idx.detach().cpu().long()
    else:
        indices = torch.as_tensor(raw_idx, dtype=torch.long)
    idx = torch.arange(int(n)).view(batch, *inp_shape)
    return torch.index_select(idx, dim=axis + 1, index=indices.reshape(-1)).reshape(-1)


def _row_indices_expand(L, n: int):
    in_shape = L.params.get("input_shape")
    out_shape = L.params.get("output_shape") or L.params.get("shape")
    if in_shape is None or out_shape is None:
        return None
    in_shape = tuple(int(d) for d in in_shape)
    out_shape = tuple(int(d) for d in out_shape)
    per = _prod(in_shape)
    if per == 0 or int(n) % per != 0:
        return None
    batch = int(n) // per
    try:
        return torch.arange(int(n)).view(batch, *in_shape).broadcast_to(
            batch, *out_shape
        ).reshape(-1)
    except RuntimeError:
        return None


def _row_indices_reduce_sum(L, n_in: int, n_out: int):
    in_shape = L.params.get("input_shape")
    if in_shape is None:
        return None
    in_shape = tuple(int(d) for d in in_shape)
    per = _prod(in_shape)
    if per == 0 or int(n_in) % per != 0:
        return None
    batch = int(n_in) // per
    axes = L.params.get("axes")
    axes = list(range(len(in_shape))) if not axes else [int(a) for a in axes]
    axes = [(a + len(in_shape)) if a < 0 else a for a in axes]
    keepdims = bool(L.params.get("keepdims", 0))
    out_shape = []
    for i, dim in enumerate(in_shape):
        if i in axes:
            if keepdims:
                out_shape.append(1)
        else:
            out_shape.append(dim)
    if _prod(out_shape) * batch != int(n_out):
        return None
    out_idx = torch.arange(int(n_out)).view(batch, *out_shape)
    view_shape = [batch]
    for i, dim in enumerate(in_shape):
        view_shape.append(1 if i in axes else dim)
    return out_idx.reshape(tuple(view_shape)).broadcast_to(batch, *in_shape).reshape(-1)


def tf_scatter_nd(L, bounds, tf):
    tf._hz_cache.pop(L.id, None)
    return interval.tf_scatter_nd(
        L,
        tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 0),
        tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 1),
        tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 2),
    )


def tf_reduce_sum(L, bounds, tf):
    hz_in = tf._hz_cache.get(L.id)
    fact = interval.tf_reduce_sum(L, bounds)
    if hz_in is not None:
        rows = _row_indices_reduce_sum(
            L, hz_in.c.shape[0], fact.bounds.lb.numel()
        )
        if rows is None:
            tf._hz_cache[L.id] = hz_from_bounds(
                fact.bounds, fact.bounds.lb.dtype, fact.bounds.lb.device
            )
        else:
            rows = rows.to(device=hz_in.c.device)
            out_n = int(fact.bounds.lb.numel())
            c = hz_in.c.new_zeros(out_n, 1)
            Gc = hz_in.Gc.new_zeros(out_n, hz_in.Gc.shape[1])
            Gb = hz_in.Gb.new_zeros(out_n, hz_in.Gb.shape[1])
            c.index_add_(0, rows, hz_in.c)
            if hz_in.Gc.shape[1]:
                Gc.index_add_(0, rows, hz_in.Gc)
            if hz_in.Gb.shape[1]:
                Gb.index_add_(0, rows, hz_in.Gb)
            tf._hz_cache[L.id] = HZono(
                c=c,
                Gc=Gc,
                Gb=Gb,
                Ac=hz_in.Ac,
                Ab=hz_in.Ab,
                b=hz_in.b,
                eq_mask=hz_in.eq_mask,
                col_ids=hz_in.col_ids,
                bcol_ids=hz_in.bcol_ids,
            )
    return fact


def tf_concat(L, bounds, tf):
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        preds = tf._net.preds.get(L.id, [])
        parts = [tf._hz_cache.get(pid) for pid in preds]
        if all(p is not None for p in parts):
            tf._hz_cache[L.id] = hz_concat(parts)
        else:
            hz_in = None
    fact = interval.tf_concat(
        L, tf._net.get_all_predecessor_bounds(L.id, tf._after, tf._before)
    )
    if hz_in is not None:
        return _hz_fact(fact, tf._hz_cache[L.id])
    return fact


def tf_sub(L, bounds, tf):
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        preds = tf._net.preds.get(L.id, [])
        hz2 = tf._hz_cache.get(preds[1]) if len(preds) > 1 else None
        if hz2 is not None:
            tf._hz_cache[L.id] = hz_sub(hz_in, hz2)
        else:
            hz_in = None
    fact = interval.tf_sub(
        L,
        tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 0),
        tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 1),
    )
    if hz_in is not None:
        return _hz_fact(fact, tf._hz_cache[L.id])
    return fact


def tf_flatten(L, bounds, tf):
    fact = interval_cnn.tf_flatten(L, bounds)
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        tf._hz_cache[L.id] = _hz_rebind(hz_in)
        return _hz_fact(fact, tf._hz_cache[L.id])
    return fact


def tf_reshape(L, bounds, tf):
    fact = interval.tf_reshape(L, bounds)
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        tf._hz_cache[L.id] = _hz_rebind(hz_in)
        return _hz_fact(fact, tf._hz_cache[L.id])
    return fact


def _hz_rebind(hz: HZono) -> HZono:
    return HZono(
        c=hz.c,
        Gc=hz.Gc,
        Gb=hz.Gb,
        Ac=hz.Ac,
        Ab=hz.Ab,
        b=hz.b,
        eq_mask=hz.eq_mask,
        col_ids=hz.col_ids,
        bcol_ids=hz.bcol_ids,
    )


def _hz_gather_rows(hz: HZono, row_idx: torch.Tensor) -> HZono:
    ri = row_idx.to(device=hz.c.device, dtype=torch.long)
    return HZono(
        c=hz.c[ri],
        Gc=hz.Gc[ri],
        Gb=hz.Gb[ri],
        Ac=hz.Ac,
        Ab=hz.Ab,
        b=hz.b,
        eq_mask=hz.eq_mask,
        col_ids=hz.col_ids,
        bcol_ids=hz.bcol_ids,
    )


def tf_squeeze(L, bounds, tf):
    fact = interval.tf_squeeze(L, bounds)
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        tf._hz_cache[L.id] = _hz_rebind(hz_in)
        return _hz_fact(fact, tf._hz_cache[L.id])
    return fact


def tf_unsqueeze(L, bounds, tf):
    fact = interval.tf_unsqueeze(L, bounds)
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        tf._hz_cache[L.id] = _hz_rebind(hz_in)
        return _hz_fact(fact, tf._hz_cache[L.id])
    return fact


def tf_transpose(L, bounds, tf):
    fact = interval.tf_transpose(L, bounds)
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        tf._hz_cache[L.id] = _hz_rebind(hz_in)
        return _hz_fact(fact, tf._hz_cache[L.id])
    return fact


def tf_slice(L, bounds, tf):
    fact = interval.tf_slice(L, bounds)
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        rows = _row_indices_slice(L, hz_in.c.shape[0])
        if rows is not None and rows.numel() == fact.bounds.lb.numel():
            tf._hz_cache[L.id] = _hz_gather_rows(hz_in, rows)
            return _hz_fact(fact, tf._hz_cache[L.id])
        tf._hz_cache.pop(L.id, None)
    return fact


def tf_gather(L, bounds, tf):
    fact = interval.tf_gather(L, bounds)
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        rows = _row_indices_gather(L, hz_in.c.shape[0])
        if rows is not None and rows.numel() == fact.bounds.lb.numel():
            tf._hz_cache[L.id] = _hz_gather_rows(hz_in, rows)
            return _hz_fact(fact, tf._hz_cache[L.id])
        tf._hz_cache.pop(L.id, None)
    return fact


def tf_expand(L, bounds, tf):
    fact = interval.tf_expand(L, bounds)
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        rows = _row_indices_expand(L, hz_in.c.shape[0])
        if rows is not None and rows.numel() == fact.bounds.lb.numel():
            tf._hz_cache[L.id] = _hz_gather_rows(hz_in, rows)
            return _hz_fact(fact, tf._hz_cache[L.id])
        tf._hz_cache.pop(L.id, None)
    return fact


# --- HZ activation encodings (zonotope domain) ---


def hz_apply_relu(hz: HZono) -> HZono:
    """Exact ReLU via equality constraints + linking equality.

    Per unstable neuron i with bounds [alpha, beta] (alpha < 0 < beta):
      ng += 4 (xi1, xi2, xi3, xi4)
      nb += 1 (z)
      nc += 3 equalities
    """
    dtype, device = hz.c.dtype, hz.c.device
    n = hz.c.shape[0]
    ng = hz.Gc.shape[1]
    nb = hz.Gb.shape[1]
    nc = hz.Ac.shape[0]

    bounds = hz_compute_bounds(hz)
    lb = bounds.lb.flatten()
    ub = bounds.ub.flatten()

    active = lb >= 0
    inactive = ub <= 0
    unstable = ~active & ~inactive
    unstable_idx = torch.where(unstable)[0]
    k = len(unstable_idx)

    out_Gc = hz.c.new_zeros(n, ng + 4 * k)
    out_Gb = hz.c.new_zeros(n, nb + k)
    out_c = hz.c.new_zeros(n, 1)

    if active.any():
        out_c[active] = hz.c[active]
        out_Gc[active, :ng] = hz.Gc[active]
        out_Gb[active, :nb] = hz.Gb[active]

    if k == 0:
        return HZono(
            c=out_c,
            Gc=out_Gc[:, :ng],
            Gb=out_Gb[:, :nb],
            Ac=hz.Ac.clone(),
            Ab=hz.Ab.clone(),
            b=hz.b.clone(),
        )

    alpha = lb[unstable_idx]
    beta = ub[unstable_idx]
    t = torch.arange(k, device=device)

    col_xi1 = ng + t
    col_xi2 = ng + k + t
    col_xi3 = ng + 2 * k + t
    col_xi4 = ng + 3 * k + t
    col_z = nb + t

    out_c[unstable_idx, 0] = beta / 2.0
    out_Gc[unstable_idx, col_xi2] = -beta / 2.0

    ng_new = ng + 4 * k
    nb_new = nb + k

    eq_Ac = hz.c.new_zeros(3 * k, ng_new)
    eq_Ab = hz.c.new_zeros(3 * k, nb_new)
    eq_b = hz.c.new_zeros(3 * k, 1)

    r1 = 3 * t
    r2 = 3 * t + 1

    eq_Ac[r1, col_xi1] = 1.0
    eq_Ac[r1, col_xi3] = 1.0
    eq_Ab[r1, col_z] = 1.0
    eq_b[r1, 0] = 1.0

    eq_Ac[r2, col_xi2] = 1.0
    eq_Ac[r2, col_xi4] = 1.0
    eq_Ab[r2, col_z] = -1.0
    eq_b[r2, 0] = 1.0

    r3 = 3 * t + 2
    eq_Ac[r3, col_xi1] = alpha / 2.0
    eq_Ac[r3, col_xi2] = -beta / 2.0
    eq_Ac[r3, :ng] = -hz.Gc[unstable_idx]
    eq_Ab[r3, :nb] = -hz.Gb[unstable_idx]
    eq_Ab[r3, col_z] = alpha / 2.0
    eq_b[r3, 0] = hz.c[unstable_idx, 0] - beta / 2.0

    old_Ac_ext = torch.cat(
        [hz.Ac, hz.c.new_zeros(nc, 4 * k)], dim=1
    )
    old_Ab_ext = torch.cat(
        [hz.Ab, hz.c.new_zeros(nc, k)], dim=1
    )

    return HZono(
        c=out_c,
        Gc=out_Gc,
        Gb=out_Gb,
        Ac=torch.cat([old_Ac_ext, eq_Ac], dim=0),
        Ab=torch.cat([old_Ab_ext, eq_Ab], dim=0),
        b=torch.cat([hz.b, eq_b], dim=0),
    )


def hz_apply_leaky_relu(hz: HZono, alpha_arg: float) -> HZono:
    """Exact LeakyReLU via the same encoding as ReLU.

    Per unstable neuron: ng += 4 (xi1, xi2, xi3, xi4), nb += 1 (z), nc += 3
    (graph eq 1, graph eq 2, linking eq) -- identical to hz_apply_relu.

    Decomposition: y = max(s*x, x) where s = alpha_arg. On the unstable
    branch, using the same switching mechanism as ReLU (z=+1 -> inactive
    with xi2 forced to 1; z=-1 -> active with xi1 forced to 1), we set
    the output as::

        y_h = beta/2 + (s*alpha/2) xi1 - (beta/2) xi2 + (s*alpha/2) z

    which degenerates exactly to ReLU's ``y_h = (beta/2)(1 - xi2)`` when
    s = 0. The graph equalities (xi1+xi3+z=1, xi2+xi4-z=1) and the linking
    equality (that ties x_h to xi1, xi2, z) are identical to ReLU.
    """
    dtype, device = hz.c.dtype, hz.c.device
    n = hz.c.shape[0]
    ng = hz.Gc.shape[1]
    nb = hz.Gb.shape[1]
    nc = hz.Ac.shape[0]
    s = alpha_arg
    assert 0.0 <= s <= 1.0, f"hz_apply_leaky_relu: slope must be in [0, 1], got {s}"

    bounds = hz_compute_bounds(hz)
    lb = bounds.lb.flatten()
    ub = bounds.ub.flatten()

    active = lb >= 0
    inactive = ub <= 0
    unstable = ~active & ~inactive
    unstable_idx = torch.where(unstable)[0]
    k = len(unstable_idx)

    out_Gc = hz.c.new_zeros(n, ng + 4 * k)
    out_Gb = hz.c.new_zeros(n, nb + k)
    out_c = hz.c.new_zeros(n, 1)

    if active.any():
        out_c[active] = hz.c[active]
        out_Gc[active, :ng] = hz.Gc[active]
        out_Gb[active, :nb] = hz.Gb[active]

    if inactive.any():
        out_c[inactive] = s * hz.c[inactive]
        out_Gc[inactive, :ng] = s * hz.Gc[inactive]
        out_Gb[inactive, :nb] = s * hz.Gb[inactive]

    if k == 0:
        return HZono(
            c=out_c,
            Gc=out_Gc[:, :ng],
            Gb=out_Gb[:, :nb],
            Ac=hz.Ac.clone(),
            Ab=hz.Ab.clone(),
            b=hz.b.clone(),
        )

    alpha = lb[unstable_idx]
    beta = ub[unstable_idx]
    t = torch.arange(k, device=device)

    col_xi1 = ng + t
    col_xi2 = ng + k + t
    col_xi3 = ng + 2 * k + t
    col_xi4 = ng + 3 * k + t
    col_z = nb + t

    # Output encoding: y_h = beta/2 + (s*alpha/2) xi1 - (beta/2) xi2 + (s*alpha/2) z
    out_c[unstable_idx, 0] = beta / 2.0
    out_Gc[unstable_idx, col_xi1] = s * alpha / 2.0
    out_Gc[unstable_idx, col_xi2] = -beta / 2.0
    out_Gb[unstable_idx, col_z] = s * alpha / 2.0

    ng_new = ng + 4 * k
    nb_new = nb + k

    eq_Ac = hz.c.new_zeros(3 * k, ng_new)
    eq_Ab = hz.c.new_zeros(3 * k, nb_new)
    eq_b = hz.c.new_zeros(3 * k, 1)

    r1 = 3 * t
    r2 = 3 * t + 1

    # Graph equality 1: xi1 + xi3 + z = 1
    eq_Ac[r1, col_xi1] = 1.0
    eq_Ac[r1, col_xi3] = 1.0
    eq_Ab[r1, col_z] = 1.0
    eq_b[r1, 0] = 1.0

    # Graph equality 2: xi2 + xi4 - z = 1
    eq_Ac[r2, col_xi2] = 1.0
    eq_Ac[r2, col_xi4] = 1.0
    eq_Ab[r2, col_z] = -1.0
    eq_b[r2, 0] = 1.0

    # Linking equality: ties x_h to (xi1, xi2, z)
    # Same form as ReLU; x_h has the same input expression.
    r3 = 3 * t + 2
    eq_Ac[r3, col_xi1] = alpha / 2.0
    eq_Ac[r3, col_xi2] = -beta / 2.0
    eq_Ac[r3, :ng] = -hz.Gc[unstable_idx]
    eq_Ab[r3, :nb] = -hz.Gb[unstable_idx]
    eq_Ab[r3, col_z] = alpha / 2.0
    eq_b[r3, 0] = hz.c[unstable_idx, 0] - beta / 2.0

    old_Ac_ext = torch.cat(
        [hz.Ac, hz.c.new_zeros(nc, 4 * k)], dim=1
    )
    old_Ab_ext = torch.cat(
        [hz.Ab, hz.c.new_zeros(nc, k)], dim=1
    )

    return HZono(
        c=out_c,
        Gc=out_Gc,
        Gb=out_Gb,
        Ac=torch.cat([old_Ac_ext, eq_Ac], dim=0),
        Ab=torch.cat([old_Ab_ext, eq_Ab], dim=0),
        b=torch.cat([hz.b, eq_b], dim=0),
    )


def hz_apply_piecewise(hz: HZono, func, dfunc, K: int = 2) -> HZono:
    """Piecewise linear approximation for monotone activations (tangent parallelogram)."""
    dtype, device = hz.c.dtype, hz.c.device
    n = hz.c.shape[0]
    ng = hz.Gc.shape[1]
    nb = hz.Gb.shape[1]
    nc = hz.Ac.shape[0]

    bounds = hz_compute_bounds(hz)
    lb = bounds.lb.flatten()
    ub = bounds.ub.flatten()

    wide = (ub - lb) > 1e-12
    narrow = ~wide
    wide_idx = torch.where(wide)[0]
    m = int(wide_idx.sum() if wide_idx.ndim == 0 else wide_idx.shape[0])

    new_c = hz.c.clone()
    new_c[narrow] = func(hz.c[narrow])
    new_Gc_base = hz.Gc.clone()
    new_Gc_base[narrow] = 0.0
    new_Gb_base = hz.Gb.clone()
    new_Gb_base[narrow] = 0.0

    if m == 0:
        return HZono(
            c=new_c,
            Gc=new_Gc_base,
            Gb=new_Gb_base,
            Ac=hz.Ac.clone(),
            Ab=hz.Ab.clone(),
            b=hz.b.clone(),
        )

    lb_w, ub_w = lb[wide_idx], ub[wide_idx]
    segment_ids = torch.arange(K, dtype=dtype, device=device).unsqueeze(1)
    segment_width = (ub_w - lb_w).unsqueeze(0) / K
    a = lb_w.unsqueeze(0) + segment_ids * segment_width
    b_seg = a + segment_width
    fa, fb = func(a), func(b_seg)
    la, lb_slope = dfunc(a), dfunc(b_seg)
    centers_x = (a + b_seg) / 2.0
    centers_y = (fa + fb) / 2.0
    nearly_linear = (la - lb_slope).abs() < 1e-10

    denom = lb_slope - la
    safe_denom = torch.where(nearly_linear, torch.ones_like(denom), denom)
    p1 = (fb - fa + lb_slope * a - la * b_seg) / safe_denom
    p2 = a + b_seg - p1
    g1x_tang = (p1 - a) / 2.0
    g1y_tang = lb_slope * (p1 - a) / 2.0
    g2x_tang = (p2 - a) / 2.0
    g2y_tang = la * (p2 - a) / 2.0

    hw = (b_seg - a) / 2.0
    slope = (fb - fa) / (b_seg - a + 1e-30)
    t_pts = torch.linspace(0.0, 1.0, 50, dtype=dtype, device=device).view(50, 1, 1)
    pts = a.unsqueeze(0) + t_pts * (b_seg - a).unsqueeze(0)
    f_pts = func(pts)
    resid = f_pts - (
        slope.unsqueeze(0) * pts + (fa - slope * a).unsqueeze(0)
    )
    max_err = resid.abs().max(dim=0).values
    g1x_lin, g1y_lin = hw, slope * hw
    g2x_lin, g2y_lin = torch.zeros_like(hw), max_err

    g1_x = torch.where(nearly_linear, g1x_lin, g1x_tang)
    g1_y = torch.where(nearly_linear, g1y_lin, g1y_tang)
    g2_x = torch.where(nearly_linear, g2x_lin, g2x_tang)
    g2_y = torch.where(nearly_linear, g2y_lin, g2y_tang)

    dx = pts - centers_x.unsqueeze(0)
    dy = f_pts - centers_y.unsqueeze(0)
    det = g1_y * g2_x - g1_x * g2_y
    safe_det = torch.where(det.abs() < 1e-30, torch.ones_like(det), det)
    xi1 = (dy * g2_x.unsqueeze(0) - dx * g2_y.unsqueeze(0)) / safe_det.unsqueeze(0)
    xi2 = (dy * g1_x.unsqueeze(0) - dx * g1_y.unsqueeze(0)) / (-safe_det.unsqueeze(0))
    max_xi = torch.maximum(xi1.abs().amax(dim=0), xi2.abs().amax(dim=0))
    scale_factor = torch.where(max_xi > 1.0, max_xi * 1.01, torch.ones_like(max_xi))
    scale_factor = torch.where(det.abs() < 1e-30, torch.ones_like(scale_factor), scale_factor)
    g1_x = g1_x * scale_factor
    g1_y = g1_y * scale_factor
    g2_x = g2_x * scale_factor
    g2_y = g2_y * scale_factor

    cy_sum = centers_y.sum(dim=0)
    new_c[wide_idx] = (cy_sum / 2.0).unsqueeze(1)
    new_Gc_base[wide_idx] = 0.0
    new_Gb_base[wide_idx] = 0.0

    n_real = 2 * K * m
    n_slack = 4 * K * m
    Gc_new = hz.c.new_zeros(n, n_real + n_slack)
    g1_cols = torch.arange(K * m, device=device).reshape(K, m)
    g2_cols = (K * m + torch.arange(K * m, device=device)).reshape(K, m)
    wide_rows = wide_idx.unsqueeze(0).expand(K, -1)
    Gc_new[wide_rows, g1_cols] = g1_y
    Gc_new[wide_rows, g2_cols] = g2_y

    Gb_new = hz.c.new_zeros(n, K * m)
    z_cols = torch.arange(K * m, device=device).reshape(K, m)
    Gb_new[wide_rows, z_cols] = -centers_y / 2.0

    out_Gc = torch.cat([new_Gc_base, Gc_new], dim=1)
    out_Gb = torch.cat([new_Gb_base, Gb_new], dim=1)
    ng_total = ng + n_real + n_slack
    nb_total = nb + K * m

    n_box = 4 * K * m
    n_eq_total = n_box + m + m
    eq_Ac = hz.c.new_zeros(n_eq_total, ng_total)
    eq_Ab = hz.c.new_zeros(n_eq_total, nb_total)
    eq_b = hz.c.new_zeros(n_eq_total, 1)

    segment_grid = torch.arange(K * m, device=device).reshape(K, m)
    g1_col_grid = ng + segment_grid
    g2_col_grid = ng + K * m + segment_grid
    z_col_grid = nb + segment_grid
    slack_base_grid = ng + n_real + 4 * segment_grid
    row_grid = 4 * segment_grid

    flat_rows = row_grid.reshape(-1)
    flat_g1_cols = g1_col_grid.reshape(-1)
    flat_g2_cols = g2_col_grid.reshape(-1)
    flat_z_cols = z_col_grid.reshape(-1)
    flat_slack_bases = slack_base_grid.reshape(-1)

    eq_Ac[flat_rows, flat_g1_cols] = 1.0
    eq_Ac[flat_rows, flat_slack_bases] = 1.0
    eq_Ab[flat_rows, flat_z_cols] = -0.5
    eq_b[flat_rows, 0] = 0.5

    eq_Ac[flat_rows + 1, flat_g1_cols] = -1.0
    eq_Ac[flat_rows + 1, flat_slack_bases + 1] = 1.0
    eq_Ab[flat_rows + 1, flat_z_cols] = -0.5
    eq_b[flat_rows + 1, 0] = 0.5

    eq_Ac[flat_rows + 2, flat_g2_cols] = 1.0
    eq_Ac[flat_rows + 2, flat_slack_bases + 2] = 1.0
    eq_Ab[flat_rows + 2, flat_z_cols] = -0.5
    eq_b[flat_rows + 2, 0] = 0.5

    eq_Ac[flat_rows + 3, flat_g2_cols] = -1.0
    eq_Ac[flat_rows + 3, flat_slack_bases + 3] = 1.0
    eq_Ab[flat_rows + 3, flat_z_cols] = -0.5
    eq_b[flat_rows + 3, 0] = 0.5

    link_rows = n_box + torch.arange(m, device=device)
    link_row_grid = link_rows.unsqueeze(1).expand(-1, K)
    eq_Ac[link_row_grid, g1_col_grid.transpose(0, 1)] = -g1_x.transpose(0, 1)
    eq_Ac[link_row_grid, g2_col_grid.transpose(0, 1)] = -g2_x.transpose(0, 1)
    eq_Ab[link_row_grid, z_col_grid.transpose(0, 1)] = centers_x.transpose(0, 1) / 2.0
    eq_Ac[link_rows, :ng] = hz.Gc[wide_idx]
    eq_Ab[link_rows, :nb] = hz.Gb[wide_idx]
    eq_b[link_rows, 0] = centers_x.sum(dim=0) / 2.0 - hz.c[wide_idx, 0]

    sum_rows = n_box + m + torch.arange(m, device=device)
    sum_row_grid = sum_rows.unsqueeze(1).expand(-1, K)
    eq_Ab[sum_row_grid, z_col_grid.transpose(0, 1)] = 1.0
    eq_b[sum_rows, 0] = hz.c.new_full((m,), float(K - 2))

    old_Ac_ext = torch.cat(
        [hz.Ac, hz.c.new_zeros(nc, n_real + n_slack)], dim=1
    )
    old_Ab_ext = torch.cat(
        [hz.Ab, hz.c.new_zeros(nc, K * m)], dim=1
    )

    return HZono(
        c=new_c,
        Gc=out_Gc,
        Gb=out_Gb,
        Ac=torch.cat([old_Ac_ext, eq_Ac], dim=0),
        Ab=torch.cat([old_Ab_ext, eq_Ab], dim=0),
        b=torch.cat([hz.b, eq_b], dim=0),
    )


def hz_apply_sigmoid(hz: HZono, K: int = 2) -> HZono:
    """Piecewise linear sigmoid via tangent parallelogram encoding."""
    return hz_apply_piecewise(
        hz, torch.sigmoid, lambda x: torch.sigmoid(x) * (1 - torch.sigmoid(x)), K
    )


def hz_apply_tanh(hz: HZono, K: int = 2) -> HZono:
    """Piecewise linear tanh via tangent parallelogram encoding."""
    return hz_apply_piecewise(hz, torch.tanh, lambda x: 1 - torch.tanh(x) ** 2, K)


# --- HZ order reduction ---


def hz_reduce(hz: HZono, max_order: float = 3.0) -> HZono:
    """Reduce HZ complexity via Girard's method (sound over-approximation)."""
    n = hz.c.shape[0]
    ng = hz.Gc.shape[1]
    nb = hz.Gb.shape[1]
    nc = hz.Ac.shape[0]

    if n == 0:
        return hz

    max_ng = max(int(max_order * n), n + 1)
    max_nb = max(2 * n, 1)

    # Step 1: Relax excess binary generators to continuous
    if nb > max_nb:
        col_norms = hz.Gb.abs().sum(dim=0)
        _, sorted_idx = col_norms.sort()
        n_relax = nb - max_nb
        relax_idx = sorted_idx[:n_relax]
        keep_idx = sorted_idx[n_relax:]
        extra_Gc = hz.Gb[:, relax_idx]
        extra_Ac = (
            hz.Ab[:, relax_idx]
            if nc > 0
            else hz.c.new_zeros(0, n_relax)
        )
        hz = HZono(
            c=hz.c,
            Gc=torch.cat([hz.Gc, extra_Gc], dim=1),
            Gb=hz.Gb[:, keep_idx],
            Ac=torch.cat([hz.Ac, extra_Ac], dim=1)
            if nc > 0
            else hz.c.new_zeros(0, ng + n_relax),
            Ab=hz.Ab[:, keep_idx]
            if nc > 0
            else hz.c.new_zeros(0, max_nb),
            b=hz.b.clone(),
        )
        ng = hz.Gc.shape[1]
        nb = hz.Gb.shape[1]

    # Step 2: Reduce continuous generators
    if ng > max_ng:
        col_norms = hz.Gc.abs().sum(dim=0)
        _, sorted_idx = col_norms.sort(descending=True)
        keep_idx = sorted_idx[: max_ng - n]
        drop_idx = sorted_idx[max_ng - n :]
        Gc_keep = hz.Gc[:, keep_idx]
        new_Gc = torch.cat(
            [Gc_keep, torch.diag(hz.Gc[:, drop_idx].abs().sum(dim=1))], dim=1
        )

        if nc > 0:
            has_dropped = hz.Ac[:, drop_idx].abs().max(dim=1).values > 1e-15
            keep_mask = ~has_dropped
            krt = torch.where(keep_mask)[0]
            if krt.numel() > 0:
                new_Ac = torch.cat(
                    [
                        hz.Ac[krt][:, keep_idx],
                        hz.c.new_zeros(krt.numel(), n),
                    ],
                    dim=1,
                )
                new_Ab = hz.Ab[krt]
                new_b = hz.b[krt]
            else:
                new_Ac = hz.c.new_zeros(0, new_Gc.shape[1])
                new_Ab = hz.c.new_zeros(0, nb)
                new_b = hz.c.new_zeros(0, 1)
        else:
            new_Ac = hz.c.new_zeros(0, new_Gc.shape[1])
            new_Ab = hz.c.new_zeros(0, nb)
            new_b = hz.c.new_zeros(0, 1)

        hz = HZono(c=hz.c, Gc=new_Gc, Gb=hz.Gb, Ac=new_Ac, Ab=new_Ab, b=new_b)

    return hz
