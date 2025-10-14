# analyze.pseudo
import torch
from collections import deque
from typing import Dict, Tuple
from act.back_end.core import Bounds, Fact, Net, ConSet
from act.back_end.utils import box_join, changed_or_maskdiff, update_cache
from act.back_end.transfer_funs.tf_mlp import *
from act.back_end.transfer_funs.tf_transformer import *
from act.back_end.transfer_funs.tf_cnn import *
from act.back_end.transfer_funs.tf_rnn import *

@torch.no_grad()
def dispatch_tf(L, before, after, net):
    k = L.kind.upper()
    # MLP basics
    if k=="DENSE": return tf_dense(L, before[L.id].bounds)
    if k=="BIAS":  return tf_bias(L, before[L.id].bounds)
    if k=="SCALE": return tf_scale(L, before[L.id].bounds)
    if k=="RELU":  return tf_relu(L, before[L.id].bounds)
    if k=="LRELU": return tf_lrelu(L, before[L.id].bounds)
    if k=="ABS":   return tf_abs(L, before[L.id].bounds)
    if k=="CLIP":  return tf_clip(L, before[L.id].bounds)
    if k=="ADD":   return tf_add(L, before[net.preds[L.id][0]].bounds, before[net.preds[L.id][1]].bounds)
    if k=="MUL":   return tf_mul(L, before[net.preds[L.id][0]].bounds, before[net.preds[L.id][1]].bounds)
    if k=="CONCAT":return tf_concat(L, [after[p].bounds for p in net.preds[L.id]])
    if k=="BN":    return tf_bn(L, before[L.id].bounds)
    # CNN layers
    if k=="CONV2D": return tf_conv2d(L, before[L.id].bounds)
    if k=="MAXPOOL2D": return tf_maxpool2d(L, before[L.id].bounds)
    if k=="AVGPOOL2D": return tf_avgpool2d(L, before[L.id].bounds)
    if k=="FLATTEN": return tf_flatten(L, before[L.id].bounds)
    # RNN layers
    if k=="LSTM": return tf_lstm(L, before[L.id].bounds)
    if k=="GRU": return tf_gru(L, before[L.id].bounds)
    if k=="RNN": return tf_rnn(L, before[L.id].bounds)
    if k=="EMBEDDING": return tf_embedding(L, before[L.id].bounds)
    # less-common
    if k=="SIGMOID": return tf_sigmoid(L, before[L.id].bounds)
    if k=="TANH":    return tf_tanh(L, before[L.id].bounds)
    if k=="SOFTPLUS":return tf_softplus(L, before[L.id].bounds)
    if k=="SILU":    return tf_silu(L, before[L.id].bounds)
    if k=="MAX":     return tf_max(L, [before[p].bounds for p in net.preds[L.id]])
    if k=="MIN":     return tf_min(L, [before[p].bounds for p in net.preds[L.id]])
    if k=="SQUARE":  return tf_square(L, before[L.id].bounds)
    if k=="POWER":   return tf_power(L, before[L.id].bounds)
    # transformer (keeping original embedding for backward compatibility)
    if k=="EMBEDDING_TF":  return tf_embedding(L)
    if k=="POSENC":     return tf_posenc(L, before[L.id].bounds)
    if k=="LAYERNORM":  return tf_layernorm(L, before[L.id].bounds)
    if k=="GELU":       return tf_gelu(L, before[L.id].bounds)
    if k=="ATT_SCORES": return tf_att_scores(L, before[L.meta["q_src"]].bounds, before[L.meta["k_src"]].bounds)
    if k=="SOFTMAX":    return tf_softmax(L, before[L.id].bounds)
    if k=="ATT_MIX":    return tf_att_mix(L, before[L.meta["w_src"]].bounds, before[L.meta["v_src"]].bounds)
    if k=="MHA_SPLIT":  return tf_mha_split(L, before[L.id].bounds)
    if k=="MHA_JOIN":   return tf_mha_join(L, [after[p].bounds for p in net.preds[L.id]])
    if k=="MASK_ADD":   return tf_mask_add(L, before[L.id].bounds)
    raise NotImplementedError(k)

@torch.no_grad()
def analyze(net: Net, entry_id: int, entry_bounds: Bounds, eps: float=1e-9) -> Tuple[Dict[int, Fact], Dict[int, Fact], ConSet]:
    before: Dict[int, Fact] = {}
    after:  Dict[int, Fact] = {}
    globalC = ConSet()

    # init with +/- inf boxes (vector length per layer's out_vars)
    for L in net.layers:
        n = len(L.out_vars)
        hi = torch.full((n,), -float("inf"), device=entry_bounds.lb.device, dtype=entry_bounds.lb.dtype)
        lo = torch.full((n,), +float("inf"), device=entry_bounds.lb.device, dtype=entry_bounds.lb.dtype)
        before[L.id] = Fact(bounds=Bounds(lo.clone(), hi.clone()), cons=ConSet())
        after[L.id]  = Fact(bounds=Bounds(lo.clone(), hi.clone()), cons=ConSet())
        L.cache.clear()

    # seed entry
    before[entry_id].bounds = entry_bounds
    before[entry_id].cons.add_box(entry_id, net.by_id[entry_id].in_vars, entry_bounds)

    WL = deque([entry_id])
    while WL:
        lid = WL.popleft(); L = net.by_id[lid]

        # merge predecessors into before[lid]
        if net.preds.get(lid):
            ref = after[net.preds[lid][0]].bounds
            Bjoin = Bounds(lb=torch.full_like(ref.lb, +float("inf")), ub=torch.full_like(ref.ub, -float("inf")))
            Cjoin = ConSet()
            for pid in net.preds[lid]:
                Bjoin = box_join(Bjoin, after[pid].bounds)
                for con in after[pid].cons.S.values(): Cjoin.replace(con)
            before[lid] = Fact(Bjoin, Cjoin)

        out_fact = dispatch_tf(L, before, after, net)

        if changed_or_maskdiff(L, out_fact.bounds, None, eps):
            after[lid] = out_fact
            update_cache(L, out_fact.bounds, None)
            for con in out_fact.cons.S.values(): globalC.replace(con)
            for sid in net.succs.get(lid, []): WL.append(sid)

    return before, after, globalC
