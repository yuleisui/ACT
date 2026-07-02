#===- act/back_end/utils.py - Backend Utility Functions ----------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Backend utility functions for ACT verification framework.
#   Provides common utilities for bounds manipulation and tensor operations.
#
#===---------------------------------------------------------------------===#

import torch
from typing import Dict, Any, Tuple, Optional
from act.back_end.core import Bounds, ConSet
from act.util.options import PerformanceOptions

EPS = 1e-12

def box_join(a: Bounds, b: Bounds) -> Bounds:
    return Bounds(lb=torch.minimum(a.lb, b.lb), ub=torch.maximum(a.ub, b.ub))

def changed_or_maskdiff(L, B: Bounds, masks: Optional[Dict[str, torch.Tensor]], eps=1e-9) -> bool:
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

def update_cache(L, B: Bounds, masks: Optional[Dict[str, torch.Tensor]]):
    L.cache["prev_lb"] = B.lb.clone(); L.cache["prev_ub"] = B.ub.clone()
    L.cache["masks"] = None if masks is None else {k: v.clone() for k,v in masks.items()}

def affine_bounds(W_pos, W_neg, b, Bin: Bounds) -> Bounds:
    """Batched affine-transform bounds via interval arithmetic.

    Maps [B, n_in] input bounds to [B, n_out] output bounds for any B>=1.

    Args:
        W_pos: clamp(W, min=0), shape [out_features, in_features]
        W_neg: clamp(W, max=0), shape [out_features, in_features]
        b: bias vector, shape [out_features]
        Bin: input bounds, lb/ub shape [B, in_features]

    Returns:
        Bounds with lb/ub shape [B, out_features].
    """
    assert Bin.lb.dim() == 2, f"affine_bounds expects 2D [B, n_in] bounds, got shape {tuple(Bin.lb.shape)}"
    lo, hi = Bin.lb, Bin.ub
    if not (torch.isinf(lo).any() or torch.isinf(hi).any()):
        lb = lo @ W_pos.T + hi @ W_neg.T + b
        ub = hi @ W_pos.T + lo @ W_neg.T + b
        return Bounds(lb, ub)
    return _affine_bounds_inf_safe(W_pos, W_neg, b, lo, hi)


def _affine_bounds_inf_safe(W_pos, W_neg, b, lo, hi) -> Bounds:
    # Interval-arithmetic convention 0 * (±inf) = 0: a zero weight drops its
    # input coordinate from the affine map, so an unbounded input range times a
    # zero weight must contribute 0. IEEE evaluates 0*inf as NaN, which the fused
    # matmul then spreads across the whole output row (an unsound NaN "bound").
    # Compute the finite contribution with infinities zeroed, then re-inject the
    # true infinities from the sign pattern of the already-split nonnegative
    # (W_pos) / nonpositive (W_neg) weights, which never multiplies a zero weight
    # by an infinity. NaN inputs are left intact so genuine upstream NaN surfaces.
    dt = lo.dtype
    lo0 = torch.where(torch.isinf(lo), torch.zeros_like(lo), lo)
    hi0 = torch.where(torch.isinf(hi), torch.zeros_like(hi), hi)
    lb = lo0 @ W_pos.T + hi0 @ W_neg.T + b
    ub = hi0 @ W_pos.T + lo0 @ W_neg.T + b

    wp = (W_pos > 0).to(dt)   # strictly positive weights (W > 0)
    wn = (W_neg < 0).to(dt)   # strictly negative weights (W < 0)
    lo_neg = (lo == float("-inf")).to(dt)
    hi_pos = (hi == float("inf")).to(dt)
    # min_x (Wx+b) is -inf iff some coord has (lo=-inf & W>0) or (hi=+inf & W<0)
    lb_neg = (lo_neg @ wp.T + hi_pos @ wn.T) > 0
    # max_x (Wx+b) is +inf iff some coord has (hi=+inf & W>0) or (lo=-inf & W<0)
    ub_pos = (hi_pos @ wp.T + lo_neg @ wn.T) > 0
    lb = torch.where(lb_neg, torch.full_like(lb, float("-inf")), lb)
    ub = torch.where(ub_pos, torch.full_like(ub, float("inf")), ub)
    return Bounds(lb, ub)

def pwl_meta(l: torch.Tensor, u: torch.Tensor, K: int) -> Dict[str, Any]:
    return {"K": int(K), "mid": 0.5*(l+u)}

def bound_var_interval(l: torch.Tensor, u: torch.Tensor) -> Tuple[float, float]:
    r = 0.5*(u-l); v_hi = float(torch.mean((2*r)**2))
    return (0.0, v_hi)

def scale_interval(cx_lo, cx_hi, inv_lo, inv_hi):
    cand = torch.stack([cx_lo*inv_lo, cx_lo*inv_hi, cx_hi*inv_lo, cx_hi*inv_hi], dim=0)
    return torch.min(cand, dim=0).values, torch.max(cand, dim=0).values


def validate_constraints(globalC, after: Dict, net) -> bool:
    """Validate constraint set for common errors (targeted validation).
    
    This function performs targeted validation by:
    1. Collecting only the variable IDs referenced by constraints in globalC
    2. Building var_bounds dict for only those variables from the 'after' facts
    3. Validating constraint dimensions and bounds existence
    
    Checks (when enabled):
    - All variable IDs referenced in constraints have bounds in 'after' facts
    - LIN_POLY dimensions match variable count
    - No NaN/Inf in constraint parameters
    
    Args:
        globalC: Constraint set to validate (ConSet)
        after: Dictionary mapping layer_id -> Fact (from analyze())
        net: ACT network with layer definitions
    
    Returns:
        True if all checks pass, False otherwise
    """
    if not PerformanceOptions.validate_constraints:
        return True  # Skip validation when disabled
    
    # Step 1: Collect all variable IDs referenced by constraints
    var_ids_used = set()
    for con in globalC:
        var_ids_used.update(con.var_ids)
    
    # Step 2: Build var_bounds dict for only the variables referenced by constraints
    var_bounds = {}
    for layer_id, fact in after.items():
        layer = net.by_id[layer_id]
        for i, var_id in enumerate(layer.out_vars):
            if var_id in var_ids_used:
                # Extract individual bounds for this variable
                var_bounds[var_id] = Bounds(
                    lb=fact.bounds.lb[i:i+1],  # Keep as 1D tensor with single element
                    ub=fact.bounds.ub[i:i+1]
                )
    
    # Step 3: Validate constraints
    all_valid = True
    issues = []
    
    for i, con in enumerate(globalC):
        # Check variable IDs exist
        for var_id in con.var_ids:
            if var_id not in var_bounds:
                issues.append(f"Constraint {i}: Variable ID {var_id} not in var_bounds")
                all_valid = False
        
        # Check LIN_POLY dimensions
        if con.kind == 'LIN_POLY':
            expected_vars = con.A.shape[1]
            actual_vars = len(con.var_ids)
            if expected_vars != actual_vars:
                issues.append(
                    f"Constraint {i}: A.shape[1]={expected_vars} != len(var_ids)={actual_vars}"
                )
                all_valid = False
            
            # Check for NaN/Inf
            if torch.isnan(con.A).any() or torch.isinf(con.A).any():
                issues.append(f"Constraint {i}: A matrix contains NaN/Inf")
                all_valid = False
            if torch.isnan(con.b).any() or torch.isinf(con.b).any():
                issues.append(f"Constraint {i}: b vector contains NaN/Inf")
                all_valid = False
    
    # Write to debug file (GUARDED - only if debug_tf is also enabled)
    if PerformanceOptions.debug_tf:
        with open(PerformanceOptions.debug_output_file, 'a') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"CONSTRAINT VALIDATION (Targeted)\n")
            f.write(f"{'='*80}\n")
            f.write(f"Total constraints: {len(globalC)}\n")
            f.write(f"Unique variables referenced: {len(var_ids_used)}\n")
            f.write(f"Variables with bounds found: {len(var_bounds)}\n")
            f.write(f"Status: {'✅ VALID' if all_valid else '❌ INVALID'}\n")
            if issues:
                f.write(f"\nIssues found:\n")
                for issue in issues:
                    f.write(f"  - {issue}\n")
            f.write("\n")
    
    return all_valid


def _test_affine_bounds_inf_safe_soundness() -> None:  # pragma: no cover
    torch.manual_seed(0)
    n_in, n_out = 6, 5
    W = torch.randn(n_out, n_in, dtype=torch.float64)
    W[0, :] = 0.0            # all-zero row: pure 0*inf annihilation
    b = torch.randn(n_out, dtype=torch.float64)
    W_pos, W_neg = torch.clamp(W, min=0), torch.clamp(W, max=0)

    lo = torch.tensor([[-1., -2., -0.5, 0., -3., 1.]], dtype=torch.float64)
    hi = torch.tensor([[1., 0., 0.5, 2., 1., 2.]], dtype=torch.float64)
    Bf = affine_bounds(W_pos, W_neg, b, Bounds(lo, hi))
    assert not torch.isnan(Bf.lb).any() and not torch.isnan(Bf.ub).any()
    assert torch.all(Bf.lb <= Bf.ub)
    for _ in range(300):
        x = lo + (hi - lo) * torch.rand_like(lo)
        y = x @ W.T + b
        assert torch.all(y >= Bf.lb - 1e-9), "finite-box lb not sound"
        assert torch.all(y <= Bf.ub + 1e-9), "finite-box ub not sound"

    inf = float("inf")
    lo_i = torch.tensor([[-inf, -inf, -1., -inf, -inf, -inf]], dtype=torch.float64)
    hi_i = torch.tensor([[inf, inf, 1., inf, inf, inf]], dtype=torch.float64)
    Bi = affine_bounds(W_pos, W_neg, b, Bounds(lo_i, hi_i))
    assert not torch.isnan(Bi.lb).any(), "affine_bounds produced NaN lb on +/-inf input"
    assert not torch.isnan(Bi.ub).any(), "affine_bounds produced NaN ub on +/-inf input"
    assert torch.all(Bi.lb <= Bi.ub)
    assert torch.isfinite(Bi.lb[0, 0]) and torch.isfinite(Bi.ub[0, 0])
    assert torch.allclose(Bi.lb[0, 0], b[0]) and torch.allclose(Bi.ub[0, 0], b[0])
    for j in range(1, n_out):
        assert torch.isneginf(Bi.lb[0, j]) or torch.isposinf(Bi.ub[0, j]), (
            f"row {j} has a nonzero weight onto an unbounded coord but was finitised"
        )
    for _ in range(300):
        x = torch.randn(1, n_in, dtype=torch.float64) * 1e3
        y = x @ W.T + b
        assert torch.all(y >= Bi.lb - 1e-6), "inf-box lb not sound"
        assert torch.all(y <= Bi.ub + 1e-6), "inf-box ub not sound"


_TESTS = [_test_affine_bounds_inf_safe_soundness]  # pragma: no cover


def main() -> int:  # pragma: no cover
    from act.util.device_manager import initialize_device

    initialize_device("cpu", "float64")
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


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(main())
