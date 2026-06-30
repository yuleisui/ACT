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
    hz_minkowski_sum,
    hz_from_bounds,
    hz_compute_bounds,
)
import act.back_end.interval_tf.tf_mlp as interval
import act.back_end.interval_tf.tf_cnn as interval_cnn


def _hz_fact(fact: Fact, hz: HZono) -> Fact:
    """Combine HZ-refined bounds (flat ``(n, 1)`` shape) with interval's
    batch-aware fact: reshape HZ bounds to match ``fact.bounds`` and keep
    interval's constraint set. Use everywhere a hybridz handler returns
    after refining the HZ cache.
    """
    hb = hz_compute_bounds(hz)
    return Fact(
        bounds=Bounds(
            lb=hb.lb.reshape_as(fact.bounds.lb),
            ub=hb.ub.reshape_as(fact.bounds.ub),
        ),
        cons=fact.cons,
    )


# ============================================================================
# Batch-native HZ helpers
# ----------------------------------------------------------------------------
# HZono stores ``c: (n, 1)``, ``Gc: (n, ng)``, ``Gb: (n, nb)`` where the
# leading dimension ``n`` is the *flattened* output size of the encoded
# layer including any leading batch axis ``B``. For per-channel ops
# (DENSE, BIAS, SCALE) we recover ``B`` from ``n // per_channel`` and
# operate via broadcasted 3D matmul / per-row scaling so that no
# block-diagonal weight is materialised.
# ============================================================================


def _hz_apply_per_batch_linear(hz: HZono, W: torch.Tensor, B: int) -> HZono:
    """Apply ``y = W x`` independently to each of ``B`` instances stacked
    along the leading axis of ``hz``. Equivalent to
    ``hz_multiply(hz, block_diag(W, ...))`` without materialising the
    block-diagonal matrix.
    """
    in_dim = W.shape[1]
    out_dim = W.shape[0]
    if B == 1:
        return hz_multiply(hz, W)
    ng = hz.Gc.shape[1]
    nb = hz.Gb.shape[1]
    # (out, in) @ (B, in, *) broadcasts → (B, out, *)
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
    )


def _hz_add_per_channel(hz: HZono, v: torch.Tensor, B: int) -> HZono:
    """Add per-channel constant ``v: (out,)`` to each of ``B`` stacked
    instances in ``hz.c``. ``hz.c`` has shape ``(B*out, 1)``.
    """
    v = v.to(dtype=hz.c.dtype, device=hz.c.device).flatten()
    if B > 1:
        v = v.repeat(B)
    return hz_add_const(hz, v.view(-1, 1))


def _hz_scale_per_channel(hz: HZono, a: torch.Tensor, B: int) -> HZono:
    """Multiply hz fields by per-channel ``a: (out,)``. ``hz.c`` shape
    is ``(B*out, 1)``; we broadcast ``a`` once per batch via repeat.
    Equivalent to ``hz_multiply(hz, diag(a_repeated))`` without building
    the diagonal matrix.
    """
    a = a.to(dtype=hz.c.dtype, device=hz.c.device).flatten()
    if B > 1:
        a = a.repeat(B)
    a_col = a.view(-1, 1)
    return HZono(
        c=a_col * hz.c,
        Gc=a_col * hz.Gc,
        Gb=a_col * hz.Gb,
        Ac=hz.Ac.clone(), Ab=hz.Ab.clone(), b=hz.b.clone(),
    )


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
    # BN's HZ refinement built a per-sample diag(A) of shape (n, n) and tried
    # to multiply with hz_in.c of shape (B*n, 1). That only works for B=1;
    # at B>1 it raises "mat1 and mat2 shapes cannot be multiplied" because
    # the HZ row layout flattens batch×feature into a single leading dim.
    # A batch-aware fix would require block-diag(A) replicated B times, which
    # is the same scope as a proper HZ batchify rewrite. Until that lands,
    # fall back to interval (sound, works at any B).
    tf._hz_cache[L.id] = None
    return interval.tf_bn(L, bounds)


def tf_add(L, bounds, tf):
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        preds = tf._net.preds.get(L.id, [])
        hz2 = tf._hz_cache.get(preds[1]) if len(preds) > 1 else None
        if hz2 is not None:
            tf._hz_cache[L.id] = hz_minkowski_sum(hz_in, hz2)
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
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        dtype, device = hz_in.c.dtype, hz_in.c.device
        preds = tf._net.preds.get(L.id, [])
        hz2 = tf._hz_cache.get(preds[1]) if len(preds) > 1 else None
        if hz2 is not None:
            b1, b2 = hz_compute_bounds(hz_in), hz_compute_bounds(hz2)
            corners = torch.stack(
                [b1.lb * b2.lb, b1.lb * b2.ub, b1.ub * b2.lb, b1.ub * b2.ub]
            )
            tf._hz_cache[L.id] = hz_from_bounds(
                Bounds(lb=corners.min(0)[0], ub=corners.max(0)[0]), dtype, device
            )
        else:
            hz_in = None
    fact = interval.tf_mul(
        L,
        tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 0),
        tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 1),
    )
    if hz_in is not None:
        return _hz_fact(fact, tf._hz_cache[L.id])
    return fact


def tf_constant(L, bounds, tf):
    val = L.params["value"].flatten()
    n = val.numel()
    # When the surrounding net is batched (e.g., upstream ADD sibling is
    # ``[B, *shape]``), replicate the constant per batch element so the
    # downstream HZ Minkowski-sum / element-wise ops see matching sizes.
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
    tf._hz_cache.pop(L.id, None)
    return interval.tf_matmul(
        L,
        tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 0),
        tf._net.get_predecessor_bounds(L.id, tf._after, tf._before, 1),
    )


def tf_arg_extremum(L, bounds, tf):
    tf._hz_cache.pop(L.id, None)
    return interval.tf_arg_extremum(L, bounds)


def tf_upsample(L, bounds, tf):
    tf._hz_cache.pop(L.id, None)
    return interval_cnn.tf_upsample(L, bounds)


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
        dtype, device = hz_in.c.dtype, hz_in.c.device
        tf._hz_cache[L.id] = hz_from_bounds(fact.bounds, dtype, device)
    return fact


def tf_concat(L, bounds, tf):
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        preds = tf._net.preds.get(L.id, [])
        parts = [tf._hz_cache.get(pid) for pid in preds]
        if all(p is not None for p in parts):
            result = parts[0]
            for p in parts[1:]:
                result = hz_minkowski_sum(result, p)
            tf._hz_cache[L.id] = result
        else:
            hz_in = None
    fact = interval.tf_concat(
        L, tf._net.get_all_predecessor_bounds(L.id, tf._after, tf._before)
    )
    if hz_in is not None:
        return _hz_fact(fact, tf._hz_cache[L.id])
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
