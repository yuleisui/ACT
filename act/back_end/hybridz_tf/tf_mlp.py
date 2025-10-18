#===- act/back_end/hybridz_tf/tf_mlp.py - HybridZ MLP Transfer Functions ====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   HybridZ MLP Transfer Functions. Implements HybridZ-based transfer functions
#   for MLP layers including dense, activation, and basic arithmetic operations.
#
#===---------------------------------------------------------------------===#


import torch
from typing import Optional
from act.back_end.core import Bounds, Fact, Layer, ConSet


@torch.no_grad()
def hybridz_tf_dense(L: Layer, Bin: Bounds) -> Fact:
    """HybridZ transfer function for dense/linear layers with zonotope precision."""
    # Extract parameters
    W = L.params["weight"]  # (out_features, in_features)
    b = L.params.get("bias", None)
    
    # Apply linear transformation with HybridZ operations
    # For now, use interval arithmetic as base implementation
    # TODO: Integrate actual HybridZ zonotope operations
    
    # Compute bounds: y = W @ x + b
    if W.shape[1] != Bin.lb.shape[0]:
        raise ValueError(f"Dense layer input mismatch: W expects {W.shape[1]}, got {Bin.lb.shape[0]}")
    
    # Interval multiplication: [a,b] * [c,d] with mixed signs
    W_pos = torch.clamp(W, min=0)
    W_neg = torch.clamp(W, max=0)
    
    # Compute output bounds
    lb = W_pos @ Bin.lb + W_neg @ Bin.ub
    ub = W_pos @ Bin.ub + W_neg @ Bin.lb
    
    if b is not None:
        lb = lb + b
        ub = ub + b
    
    Bout = Bounds(lb=lb, ub=ub)
    
    # Generate constraint for dense layer
    cons = ConSet()
    cons.add_dense(L.id, L.in_vars, L.out_vars, W, b)
    
    return Fact(bounds=Bout, cons=cons)


@torch.no_grad()
def hybridz_tf_bias(L: Layer, Bin: Bounds) -> Fact:
    """HybridZ transfer function for bias addition."""
    c = L.params["bias"]
    
    # Simple translation
    lb = Bin.lb + c
    ub = Bin.ub + c
    Bout = Bounds(lb=lb, ub=ub)
    
    cons = ConSet()
    cons.add_bias(L.id, L.in_vars, L.out_vars, c)
    
    return Fact(bounds=Bout, cons=cons)


@torch.no_grad()
def hybridz_tf_scale(L: Layer, Bin: Bounds) -> Fact:
    """HybridZ transfer function for element-wise scaling."""
    a = L.params["scale"]
    
    # Handle positive/negative scaling
    a_pos = torch.clamp(a, min=0)
    a_neg = torch.clamp(a, max=0)
    
    lb = a_pos * Bin.lb + a_neg * Bin.ub
    ub = a_pos * Bin.ub + a_neg * Bin.lb
    Bout = Bounds(lb=lb, ub=ub)
    
    cons = ConSet()
    cons.add_scale(L.id, L.in_vars, L.out_vars, a)
    
    return Fact(bounds=Bout, cons=cons)


@torch.no_grad()
def hybridz_tf_relu(L: Layer, Bin: Bounds) -> Fact:
    """HybridZ transfer function for ReLU activation with precise constraint handling."""
    # Determine ReLU phases
    idx_on = torch.where(Bin.lb >= 0)[0]  # Always active
    idx_off = torch.where(Bin.ub <= 0)[0]  # Always inactive
    idx_amb = torch.where((Bin.lb < 0) & (Bin.ub > 0))[0]  # Ambiguous
    
    # Compute output bounds
    lb = torch.clamp(Bin.lb, min=0)
    ub = torch.clamp(Bin.ub, min=0)
    Bout = Bounds(lb=lb, ub=ub)
    
    # HybridZ-specific ReLU constraint generation
    cons = ConSet()
    
    # For ambiguous neurons, use HybridZ slope computation
    slope = torch.zeros_like(Bin.lb)
    shift = torch.zeros_like(Bin.lb)
    
    if len(idx_amb) > 0:
        # HybridZ: More precise slope computation
        denom = Bin.ub[idx_amb] - Bin.lb[idx_amb]
        slope[idx_amb] = torch.where(denom > 1e-8, Bin.ub[idx_amb] / denom, torch.ones_like(denom))
        shift[idx_amb] = torch.zeros_like(idx_amb, dtype=slope.dtype)
    
    cons.add_relu(L.id, L.in_vars, L.out_vars, idx_on, idx_off, idx_amb, slope, shift)
    
    return Fact(bounds=Bout, cons=cons)


@torch.no_grad()
def hybridz_tf_lrelu(L: Layer, Bin: Bounds) -> Fact:
    """HybridZ transfer function for LeakyReLU."""
    alpha = float(L.params.get("alpha", 0.01))
    
    # Determine phases
    idx_on = torch.where(Bin.lb >= 0)[0]
    idx_off = torch.where(Bin.ub <= 0)[0]
    idx_amb = torch.where((Bin.lb < 0) & (Bin.ub > 0))[0]
    
    # Output bounds
    lb = torch.where(Bin.lb >= 0, Bin.lb, alpha * Bin.lb)
    ub = torch.where(Bin.ub <= 0, alpha * Bin.ub, Bin.ub)
    Bout = Bounds(lb=lb, ub=ub)
    
    # HybridZ slope computation for ambiguous region
    slope = torch.zeros_like(Bin.lb)
    shift = torch.zeros_like(Bin.lb)
    
    if len(idx_amb) > 0:
        y_at_ub = Bin.ub[idx_amb]
        y_at_lb = alpha * Bin.lb[idx_amb]
        denom = Bin.ub[idx_amb] - Bin.lb[idx_amb]
        slope[idx_amb] = torch.where(denom > 1e-8, (y_at_ub - y_at_lb) / denom, torch.ones_like(denom))
        shift[idx_amb] = y_at_lb - slope[idx_amb] * Bin.lb[idx_amb]
    
    cons = ConSet()
    cons.add_lrelu(L.id, L.in_vars, L.out_vars, alpha, idx_on, idx_off, idx_amb, slope, shift)
    
    return Fact(bounds=Bout, cons=cons)


@torch.no_grad()
def hybridz_tf_abs(L: Layer, Bin: Bounds) -> Fact:
    """HybridZ transfer function for absolute value."""
    # Determine phases
    idx_pos = torch.where(Bin.lb >= 0)[0]  # Always positive
    idx_neg = torch.where(Bin.ub <= 0)[0]  # Always negative
    idx_amb = torch.where((Bin.lb < 0) & (Bin.ub > 0))[0]  # Crosses zero
    
    # Output bounds
    lb = torch.where(idx_amb[:, None] == torch.arange(len(Bin.lb))[None, :], 
                     torch.zeros_like(Bin.lb), 
                     torch.where(Bin.lb >= 0, Bin.lb, -Bin.ub))
    ub = torch.maximum(torch.abs(Bin.lb), torch.abs(Bin.ub))
    Bout = Bounds(lb=lb, ub=ub)
    
    cons = ConSet()
    cons.add_abs(L.id, L.in_vars, L.out_vars, idx_pos, idx_neg, idx_amb)
    
    return Fact(bounds=Bout, cons=cons)


@torch.no_grad()
def hybridz_tf_add(L: Layer, Bin1: Bounds, Bin2: Bounds) -> Fact:
    """HybridZ transfer function for element-wise addition."""
    # Simple interval addition
    lb = Bin1.lb + Bin2.lb
    ub = Bin1.ub + Bin2.ub
    Bout = Bounds(lb=lb, ub=ub)
    
    cons = ConSet()
    cons.add_add(L.id, L.in_vars, L.out_vars)
    
    return Fact(bounds=Bout, cons=cons)


@torch.no_grad()  
def hybridz_tf_mul(L: Layer, Bin1: Bounds, Bin2: Bounds) -> Fact:
    """HybridZ transfer function for element-wise multiplication with McCormick relaxation."""
    # McCormick envelope for bilinear terms
    # z = x * y, with x ∈ [lx, ux], y ∈ [ly, uy]
    lx, ux = Bin1.lb, Bin1.ub
    ly, uy = Bin2.lb, Bin2.ub
    
    # Four corner points
    corners = torch.stack([
        lx * ly,  # lower-left
        lx * uy,  # lower-right  
        ux * ly,  # upper-left
        ux * uy   # upper-right
    ])
    
    lb = torch.min(corners, dim=0)[0]
    ub = torch.max(corners, dim=0)[0]
    Bout = Bounds(lb=lb, ub=ub)
    
    # McCormick constraints
    cons = ConSet()
    cons.add_mcc(L.id, L.in_vars, L.out_vars, lx, ux, ly, uy)
    
    return Fact(bounds=Bout, cons=cons)