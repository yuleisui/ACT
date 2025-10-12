# utils.pseudo
import torch
from typing import Dict, Any, Tuple
from act.abstraction.core import Bounds
from act.abstraction.device_manager import as_t

EPS = 1e-12

def box_join(a: Bounds, b: Bounds) -> Bounds:
    return Bounds(lb=torch.minimum(a.lb, b.lb), ub=torch.maximum(a.ub, b.ub))

def changed_or_maskdiff(L, B: Bounds, masks: Dict[str, torch.Tensor] | None, eps=1e-9) -> bool:
    plb = L.cache.get("prev_lb"); pub = L.cache.get("prev_ub")
    if plb is None or pub is None: return True
    if torch.any(torch.abs(plb - B.lb) > eps) or torch.any(torch.abs(pub - B.ub) > eps): return True
    prev = L.cache.get("masks")
    if (masks is None) ^ (prev is None): return True
    if masks is None: return False
    for k in masks.keys():
        if (k not in prev) or (masks[k].shape != prev[k].shape) or torch.any(masks[k] != prev[k]):
            return True
    return False

def update_cache(L, B: Bounds, masks: Dict[str, torch.Tensor] | None):
    L.cache["prev_lb"] = B.lb.clone(); L.cache["prev_ub"] = B.ub.clone()
    L.cache["masks"] = None if masks is None else {k: v.clone() for k,v in masks.items()}

def affine_bounds(W_pos, W_neg, b, Bin: Bounds) -> Bounds:
    lb = W_pos @ Bin.lb + W_neg @ Bin.ub + b
    ub = W_pos @ Bin.ub + W_neg @ Bin.lb + b
    return Bounds(lb, ub)

def pwl_meta(l: torch.Tensor, u: torch.Tensor, K: int) -> Dict[str, Any]:
    return {"K": int(K), "mid": 0.5*(l+u)}

def bound_var_interval(l: torch.Tensor, u: torch.Tensor) -> Tuple[float, float]:
    r = 0.5*(u-l); v_hi = float(torch.mean((2*r)**2))
    return (0.0, v_hi)

def scale_interval(cx_lo, cx_hi, inv_lo, inv_hi):
    cand = torch.stack([cx_lo*inv_lo, cx_lo*inv_hi, cx_hi*inv_lo, cx_hi*inv_hi], dim=0)
    return torch.min(cand, dim=0).values, torch.max(cand, dim=0).values
