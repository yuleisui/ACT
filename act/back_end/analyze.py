# analyze.py
import torch
from collections import deque
from typing import Dict, Tuple
from act.back_end.core import Bounds, Fact, Net, ConSet
from act.back_end.utils import box_join, changed_or_maskdiff, update_cache
from act.back_end.transfer_function import dispatch_tf, set_transfer_function_mode

# Initialize default transfer function mode
def initialize_tf_mode(mode: str = "interval"):
    """Initialize transfer function mode. Call this before using analyze()."""
    set_transfer_function_mode(mode)

@torch.no_grad()
def analyze(net: Net, entry_id: int, entry_bounds: Bounds, eps: float=1e-9) -> Tuple[Dict[int, Fact], Dict[int, Fact], ConSet]:
    # Auto-initialize transfer function mode if not set
    try:
        from act.back_end.transfer_function import get_transfer_function
        get_transfer_function()  # Check if already initialized
    except RuntimeError:
        initialize_tf_mode("interval")  # Default to interval mode
        
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
