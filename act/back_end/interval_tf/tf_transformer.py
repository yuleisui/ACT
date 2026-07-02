#===- act/back_end/interval_tf/tf_transformer.py - Transformer Interval TF ====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Transformer Interval Transfer Functions. Provides interval-based transfer
#   functions for transformer components including attention mechanisms.
#
#===---------------------------------------------------------------------===#

import torch
from typing import List, Optional, Tuple, cast
from act.back_end.core import Bounds, Con, ConSet, Fact, Layer
from act.back_end.utils import pwl_meta, scale_interval
from act.back_end.interval_tf.tf_mlp import tf_concat
from act.back_end.interval_tf.tf_attention import LinearBounds, att_scores_dual_planar

# tf_embedding is provided by act.back_end.interval_tf.tf_rnn (signature
# (L, Bin) -> Fact). The previous transformer-local definition had a wrong
# signature and would shadow the rnn one via `from tf_transformer import *`
# in interval_tf.py — both EMBEDDING and EMBEDDING_TF would have raised
# TypeError at runtime. Single source of truth lives in tf_rnn.

def _round_nan_outward(lb: torch.Tensor, ub: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Round 0*inf / inf-inf NaNs outward to a sound extended-real interval.

    Interval TFs are seeded with +/-inf placeholders on the first worklist
    visit, before a predecessor has converged. The LayerNorm std-normalisation
    product cx*inv and the GELU polynomial x*(1+tanh(...)) then evaluate the
    indeterminate 0*inf at an unbounded endpoint, which IEEE returns as NaN.
    NaN is unsound as a bound and, worse, invisible to the fixpoint change test
    (every NaN comparison is false), so it would stick forever and never refine.
    Widening NaN to -inf (lb) / +inf (ub) keeps the interval a sound
    over-approximation and lets a later visit with finite predecessor bounds
    recompute a finite result.
    """
    lb = torch.where(torch.isnan(lb), lb.new_full((), float("-inf")), lb)
    ub = torch.where(torch.isnan(ub), ub.new_full((), float("inf")), ub)
    return lb, ub

def tf_posenc(L: Layer, Bin: Bounds) -> Fact:
    P = cast(torch.Tensor, L.params["pos_vec"]); B=Bounds(Bin.lb+P, Bin.ub+P); C=ConSet()
    C.replace(Con("EQ", tuple(L.out_vars+L.in_vars), {"tag":f"posenc:{L.id}"})); C.add_box(L.id,L.out_vars,B); return Fact(B,C)

def tf_layernorm(L: Layer, Bin: Bounds) -> Fact:
    if Bin.lb.dim() < 2:
        raise ValueError(f"LAYERNORM expects batched bounds [B, *], got shape {tuple(Bin.lb.shape)}")
    variant = L.params.get("variant", L.params.get("layer_norm", "standard"))
    norm_dims = tuple(range(1, Bin.lb.dim()))
    mu_lb = torch.mean(Bin.lb, dim=norm_dims, keepdim=True)
    mu_ub = torch.mean(Bin.ub, dim=norm_dims, keepdim=True)
    cx_lb, cx_ub = Bin.lb - mu_ub, Bin.ub - mu_lb
    if variant == "no_var":
        # The no-variance form omits the std division, so the centered
        # interval already is the normalized output (tighter, no relaxation).
        sh_lb, sh_ub = cx_lb, cx_ub
    else:
        radius = 0.5 * (Bin.ub - Bin.lb)
        v_lo = torch.zeros_like(mu_lb)
        v_hi = torch.mean((2 * radius) ** 2, dim=norm_dims, keepdim=True)
        eps=float(cast(float, L.params.get("eps",1e-5)))
        eps_t = Bin.lb.new_tensor(eps)
        inv_lb = torch.rsqrt(v_hi + eps_t)
        inv_ub = torch.rsqrt(torch.clamp_min(v_lo, 0.0) + eps_t)
        sh_lb, sh_ub = scale_interval(cx_lb, cx_ub, inv_lb, inv_ub)
    gamma = cast(torch.Tensor, L.params["gamma"])
    beta = cast(torch.Tensor, L.params["beta"])
    if gamma.numel() != sh_lb.shape[-1] and sh_lb.shape[-1] % gamma.numel() == 0:
        repeat = sh_lb.shape[-1] // gamma.numel()
        gamma = gamma.repeat(repeat)
        beta = beta.repeat(repeat)
    lb=torch.where(gamma>=0, gamma*sh_lb+beta, gamma*sh_ub+beta)
    ub=torch.where(gamma>=0, gamma*sh_ub+beta, gamma*sh_lb+beta)
    lb, ub = _round_nan_outward(lb, ub)
    B=Bounds(lb,ub); C=ConSet(); C.replace(Con("INEQ", tuple(L.out_vars+L.in_vars), {"tag":f"layernorm:{L.id}"}))
    C.add_box(L.id,L.out_vars,B); return Fact(B,C)

def tf_gelu(L: Layer, Bin: Bounds) -> Fact:
    GELU_MIN_X = -0.7517916
    GELU_MIN_Y = -0.17004
    f = lambda x: 0.5*x*(1+torch.tanh(torch.sqrt(torch.tensor(2.0/torch.pi))*(x+0.044715*(x**3))))
    f_lb, f_ub = f(Bin.lb), f(Bin.ub)
    contains_min = (Bin.lb <= GELU_MIN_X) & (Bin.ub >= GELU_MIN_X)
    lb = torch.where(contains_min, torch.full_like(f_lb, GELU_MIN_Y), torch.minimum(f_lb, f_ub))
    ub = torch.maximum(f_lb, f_ub)
    lb, ub = _round_nan_outward(lb, ub)
    B = Bounds(lb, ub); C = ConSet()
    C.replace(Con("INEQ", tuple(L.out_vars+L.in_vars), {"tag":f"gelu:{L.id}","segs":pwl_meta(Bin.lb,Bin.ub,3)}))
    C.add_box(L.id, L.out_vars, B); return Fact(B, C)

def tf_att_scores(L: Layer, Bq: Bounds, Bk: Bounds) -> Fact:
    if L.params.get("attn_mode") == "dual_planar":
        return _tf_att_scores_dual_planar(L)
    batch_size = Bq.lb.shape[0]
    if Bk.lb.shape[0] != batch_size:
        raise ValueError(f"ATT_SCORES expects matching batch dims, got {batch_size} and {Bk.lb.shape[0]}")
    s=Bq.lb.new_tensor(1.0/float(cast(float, cast(object, L.params["dk"]))))
    lo=torch.minimum(torch.minimum(Bq.lb*Bk.lb, Bq.lb*Bk.ub), torch.minimum(Bq.ub*Bk.lb, Bq.ub*Bk.ub))
    hi=torch.maximum(torch.maximum(Bq.lb*Bk.lb, Bq.lb*Bk.ub), torch.maximum(Bq.ub*Bk.lb, Bq.ub*Bk.ub))
    lb=s*lo.sum(dim=-1, keepdim=True); ub=s*hi.sum(dim=-1, keepdim=True)
    mask = L.params.get("mask")
    if isinstance(mask, torch.Tensor): lb=lb+mask; ub=ub+mask
    B=Bounds(lb,ub); C=ConSet()
    q_vars = cast(List[int], cast(object, L.params["q_vars"]))
    k_vars = cast(List[int], cast(object, L.params["k_vars"]))
    C.replace(Con("INEQ", tuple(L.out_vars + q_vars + k_vars), {"tag":f"att_scores:{L.id}","scale":float(s.item()),"mcc":True}))
    C.add_box(L.id,L.out_vars,B); return Fact(B,C)

def _tf_att_scores_dual_planar(L: Layer) -> Fact:
    """ReLU-catalyzed dual-planar QKt scores as a tighter interval box.

    Reads per-head query/key ``LinearBounds`` frames from the layer params,
    fuses the two attention planes (rule slope when ``clamp_alpha`` is unset,
    optimized-alpha init when set), and concretizes to a sound box. The McCormick
    ``att_scores`` path stays the default; this mode is the tighter dual-path
    bound and emits its own ``att_dual_planar`` tag for the solver export.
    """
    fused = att_scores_dual_planar(
        cast(LinearBounds, cast(object, L.params["q_lb"])),
        cast(LinearBounds, cast(object, L.params["k_lb"])),
        head_size=cast(int, L.params["head_size"]),
        k=float(cast(float, cast(object, L.params.get("k_thresh", 1.0)))),
        clamp_alpha=bool(L.params.get("clamp_alpha", False)),
        mask=cast(Optional[torch.Tensor], L.params.get("mask")),
    )
    lo, hi = fused.concretize()
    bs = lo.shape[0]
    B = Bounds(lo.reshape(bs, -1), hi.reshape(bs, -1))
    C = ConSet()
    C.replace(Con("INEQ", tuple(L.out_vars), {"tag": f"att_dual_planar:{L.id}"}))
    C.add_box(L.id, L.out_vars, B)
    return Fact(B, C)

def tf_softmax(L: Layer, Bin: Bounds) -> Fact:
    B=Bounds(torch.zeros_like(Bin.lb), torch.ones_like(Bin.ub))
    rowsize=int(Bin.lb.shape[-1]); mode=L.params.get("mode","simplex"); tag=f"softmax:{mode}:{L.id}"
    C=ConSet()
    if mode=="simplex": C.replace(Con("INEQ", tuple(L.out_vars), {"tag":tag,"rowsize":rowsize}))
    elif mode=="pwl":  C.replace(Con("INEQ", tuple(L.out_vars+L.in_vars), {"tag":tag,"rowsize":rowsize,"segs":{"K":3}}))
    else:              C.replace(Con("BIN",  tuple(L.out_vars+L.in_vars), {"tag":tag,"rowsize":rowsize,"K":4,"sos2":True}))
    C.add_box(L.id,L.out_vars,B); return Fact(B,C)

def tf_att_mix(L: Layer, Bw: Bounds, Bv: Bounds) -> Fact:
    batch_size = Bw.lb.shape[0]
    if Bv.lb.shape[0] != batch_size:
        raise ValueError(f"ATT_MIX expects matching batch dims, got {batch_size} and {Bv.lb.shape[0]}")
    lo=torch.minimum(torch.minimum(Bw.lb*Bv.lb, Bw.lb*Bv.ub), torch.minimum(Bw.ub*Bv.lb, Bw.ub*Bv.ub)).sum(dim=-1, keepdim=True)
    hi=torch.maximum(torch.maximum(Bw.lb*Bv.lb, Bw.lb*Bv.ub), torch.maximum(Bw.ub*Bv.lb, Bw.ub*Bv.ub)).sum(dim=-1, keepdim=True)
    B=Bounds(lo,hi); C=ConSet()
    C.replace(Con("INEQ", tuple(L.out_vars + cast(List[int], cast(object, L.params["w_vars"])) + cast(List[int], cast(object, L.params["v_vars"]))), {"tag":f"att_mix:{L.id}","mcc":True,"rowsize":L.params["rowsize"]}))
    C.add_box(L.id,L.out_vars,B); return Fact(B,C)

def tf_mha_split(L: Layer, Bin: Bounds) -> Fact:
    """Project and select query/key/value slices for explicit attention graphs."""
    weight_value = L.params.get("weight")
    if not isinstance(weight_value, torch.Tensor):
        return Fact(Bin.copy(), ConSet())
    weight = weight_value
    batch_size = Bin.lb.shape[0]
    input_shape_value = cast(Tuple[int, ...], cast(object, L.params.get("input_shape", (1, Bin.lb.shape[1]))))
    input_shape = tuple(int(d) for d in input_shape_value)
    hidden_size = int(cast(int, cast(object, L.params.get("hidden_size", weight.shape[1]))))
    seq_len = int(input_shape[-2]) if len(input_shape) >= 3 else max(Bin.lb.shape[1] // max(hidden_size, 1), 1)
    x_lb = Bin.lb.reshape(batch_size, seq_len, hidden_size)
    x_ub = Bin.ub.reshape(batch_size, seq_len, hidden_size)
    W = weight.to(device=Bin.lb.device, dtype=Bin.lb.dtype)
    W_pos = torch.clamp(W, min=0)
    W_neg = torch.clamp(W, max=0)
    bias = L.params.get("bias")
    b = bias.to(device=Bin.lb.device, dtype=Bin.lb.dtype) if isinstance(bias, torch.Tensor) else Bin.lb.new_zeros(W.shape[0])
    proj_lb = x_lb @ W_pos.T + x_ub @ W_neg.T + b
    proj_ub = x_ub @ W_pos.T + x_lb @ W_neg.T + b
    role = str(L.params.get("role", ""))
    if role in {"query", "key"}:
        position = int(cast(int, cast(object, L.params.get("position", 0))))
        out_lb = proj_lb[:, position, :]
        out_ub = proj_ub[:, position, :]
    elif role == "value":
        feature = int(cast(int, cast(object, L.params.get("feature", 0))))
        out_lb = proj_lb[:, :, feature]
        out_ub = proj_ub[:, :, feature]
    else:
        out_lb = proj_lb.reshape(batch_size, -1)
        out_ub = proj_ub.reshape(batch_size, -1)
    B = Bounds(out_lb.reshape(batch_size, -1), out_ub.reshape(batch_size, -1))
    C = ConSet(); C.add_box(L.id, L.out_vars, B)
    return Fact(B, C)
def tf_mha_join(L: Layer, Bs: List[Bounds]) -> Fact:
    shaped = [Bounds(b.lb.unsqueeze(-1), b.ub.unsqueeze(-1)) if b.lb.dim() == 1 else b for b in Bs]
    return tf_concat(L, shaped)
def tf_mask_add(L: Layer, Bin: Bounds) -> Fact:
    M = cast(torch.Tensor, L.params["M"]); B=Bounds(Bin.lb+M, Bin.ub+M); C=ConSet()
    C.replace(Con("EQ", tuple(L.out_vars+L.in_vars), {"tag":f"mask:{L.id}"})); C.add_box(L.id,L.out_vars,B); return Fact(B,C)
