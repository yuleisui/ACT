#===- act/back_end/hybridz_tf/tf_cnn.py - HybridZ CNN Transfer Functions ====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   HybridZ CNN Transfer Functions. Implements HybridZ-based transfer functions
#   for CNN layers including convolution, pooling, and tensor reshaping
#   operations.
#
#===---------------------------------------------------------------------===#

import torch
import torch.nn.functional as F
from act.back_end.core import Bounds, Fact
from act.back_end.solver.solver_hz import HZono
from act.back_end.hybridz_tf.tf_mlp import _hz_fact
import act.back_end.interval_tf.tf_cnn as interval


# --- HZ transfer functions (CNN) ---

def tf_conv2d(L, bounds, tf):
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        input_shape = L.params.get("input_shape")
        if input_shape is not None:
            tf._hz_cache[L.id] = hz_conv2d(
                hz_in, L.params["weight"], L.params.get("bias"),
                L.params.get("stride", 1), L.params.get("padding", 0),
                L.params.get("dilation", 1), L.params.get("groups", 1), input_shape,
            )
        else:
            hz_in = None
    fact = interval.tf_conv2d(L, bounds)
    if hz_in is not None:
        return _hz_fact(fact, tf._hz_cache[L.id])
    return fact


def tf_maxpool2d(L, bounds, tf):
    # MaxPool is not affine in the HZ rows; keep interval bounds for soundness.
    tf._hz_cache[L.id] = None
    return interval.tf_maxpool2d(L, bounds)


# --- HZ conv2d (zonotope domain) ---

def _conv2d_generators(
    G, weight, B, C, H, W, stride, padding, dilation, groups, n_out_per_sample
):
    """Apply conv2d to a generator matrix ``(B*C*H*W, ng)`` and return
    a generator matrix ``(B*n_out_per_sample, ng)``. Each generator
    column is convolved independently per batch element by stacking
    ``ng * B`` images into conv2d's leading "batch" axis.
    """
    if G.shape[1] == 0:
        return G.new_zeros(B * n_out_per_sample, 0)
    ng = G.shape[1]
    imgs = G.t().contiguous().view(ng, B, C, H, W).reshape(ng * B, C, H, W)
    out = F.conv2d(
        imgs,
        weight,
        bias=None,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
    _, Cp, Hp, Wp = out.shape
    return (
        out.view(ng, B, Cp, Hp, Wp)
        .permute(1, 2, 3, 4, 0)
        .contiguous()
        .reshape(B * Cp * Hp * Wp, ng)
    )


def hz_conv2d(
    hz: HZono, weight, bias, stride, padding, dilation, groups, input_shape
) -> HZono:
    if len(input_shape) == 4:
        _, C, H, W = input_shape
    elif len(input_shape) == 3:
        C, H, W = input_shape
    else:
        raise ValueError(f"Unexpected input_shape={input_shape}, expected 3D or 4D")
    weight = weight.to(hz.c)

    spatial_in = C * H * W
    B = hz.c.numel() // spatial_in
    c_img = hz.c.view(B, C, H, W)
    out_c = F.conv2d(
        c_img,
        weight,
        bias=bias.to(hz.c) if bias is not None else None,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
    _, Cp, Hp, Wp = out_c.shape
    new_c = out_c.reshape(-1, 1)
    n_out_per_sample = Cp * Hp * Wp

    new_Gc = _conv2d_generators(
        hz.Gc, weight, B, C, H, W, stride, padding, dilation, groups, n_out_per_sample
    )
    new_Gb = _conv2d_generators(
        hz.Gb, weight, B, C, H, W, stride, padding, dilation, groups, n_out_per_sample
    )

    return HZono(
        c=new_c,
        Gc=new_Gc,
        Gb=new_Gb,
        Ac=hz.Ac.clone(),
        Ab=hz.Ab.clone(),
        b=hz.b.clone(),
        eq_mask=None if hz.eq_mask is None else hz.eq_mask.clone(),
        col_ids=None if hz.col_ids is None else hz.col_ids.clone(),
        bcol_ids=None if hz.bcol_ids is None else hz.bcol_ids.clone(),
    )


def _spatial_op_generators(G, op_fn, B, C, H, W, n_out_per_sample):
    """Apply a linear BCHW spatial operator to each HZ generator column.

    If y = S(x) + b and x = c + G xi, then each output generator is
    G'[:, j] = vec(S(unvec(G[:, j]))), without materializing the matrix for S.
    """
    if G.shape[1] == 0:
        return G.new_zeros(B * n_out_per_sample, 0)
    ng = G.shape[1]
    imgs = G.t().contiguous().view(ng, B, C, H, W).reshape(ng * B, C, H, W)
    out = op_fn(imgs)
    _, Cp, Hp, Wp = out.shape
    return (
        out.view(ng, B, Cp, Hp, Wp)
        .permute(1, 2, 3, 4, 0)
        .contiguous()
        .reshape(B * Cp * Hp * Wp, ng)
    )


def _hz_spatial_affine(hz: HZono, op_fn, input_shape, bias=None) -> HZono:
    if len(input_shape) == 4:
        _, C, H, W = input_shape
    elif len(input_shape) == 3:
        C, H, W = input_shape
    else:
        raise ValueError(f"Unexpected input_shape={input_shape}")
    spatial_in = C * H * W
    B = hz.c.numel() // spatial_in
    out_c = op_fn(hz.c.view(B, C, H, W))
    _, Cp, Hp, Wp = out_c.shape
    if bias is not None:
        out_c = out_c + bias.to(hz.c).view(1, -1, 1, 1)
    n_out = Cp * Hp * Wp
    return HZono(
        c=out_c.reshape(-1, 1),
        Gc=_spatial_op_generators(hz.Gc, op_fn, B, C, H, W, n_out),
        Gb=_spatial_op_generators(hz.Gb, op_fn, B, C, H, W, n_out),
        Ac=hz.Ac.clone(),
        Ab=hz.Ab.clone(),
        b=hz.b.clone(),
        eq_mask=None if hz.eq_mask is None else hz.eq_mask.clone(),
        col_ids=None if hz.col_ids is None else hz.col_ids.clone(),
        bcol_ids=None if hz.bcol_ids is None else hz.bcol_ids.clone(),
    )


def hz_avgpool2d(hz, kernel_size, stride, padding, input_shape) -> HZono:
    op = lambda x: F.avg_pool2d(
        x,
        kernel_size=kernel_size,
        stride=stride if stride is not None else kernel_size,
        padding=padding,
    )
    return _hz_spatial_affine(hz, op, input_shape)


def tf_avgpool2d(L, bounds, tf):
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        ishape = L.params.get("input_shape")
        if ishape is not None:
            tf._hz_cache[L.id] = hz_avgpool2d(
                hz_in,
                L.params.get("kernel_size"),
                L.params.get("stride"),
                L.params.get("padding", 0),
                ishape,
            )
        else:
            hz_in = None
    fact = interval.tf_avgpool2d(L, bounds)
    if hz_in is not None:
        return _hz_fact(fact, tf._hz_cache[L.id])
    return fact


def hz_convtranspose2d(
    hz, weight, bias, stride, padding, output_padding, dilation, groups, input_shape
) -> HZono:
    weight = weight.to(hz.c)
    op = lambda x: F.conv_transpose2d(
        x,
        weight,
        bias=None,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
        groups=groups,
    )
    return _hz_spatial_affine(hz, op, input_shape, bias=bias)


def tf_convtranspose2d(L, bounds, tf):
    hz_in = tf._hz_cache.get(L.id)
    if hz_in is not None:
        ishape = L.params.get("input_shape")
        if ishape is not None:
            tf._hz_cache[L.id] = hz_convtranspose2d(
                hz_in,
                L.params["weight"],
                L.params.get("bias"),
                L.params.get("stride", 1),
                L.params.get("padding", 0),
                L.params.get("output_padding", 0),
                L.params.get("dilation", 1),
                L.params.get("groups", 1),
                ishape,
            )
        else:
            hz_in = None
    fact = interval.tf_convtranspose2d(L, bounds)
    if hz_in is not None:
        return _hz_fact(fact, tf._hz_cache[L.id])
    return fact
