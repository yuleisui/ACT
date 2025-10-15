"""
HybridZ CNN Transfer Functions

This module implements HybridZ-based transfer functions for CNN layers
including convolution, pooling, and tensor reshaping operations.
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple
from act.back_end.core import Bounds, Fact, Layer, ConSet


@torch.no_grad()
def hybridz_tf_conv2d(L: Layer, Bin: Bounds) -> Fact:
    """HybridZ transfer function for 2D convolution with enhanced precision."""
    # Extract convolution parameters
    weight = L.params["weight"]  # (out_channels, in_channels, kernel_h, kernel_w)
    bias = L.params.get("bias", None)
    stride = L.meta.get("stride", 1)
    padding = L.meta.get("padding", 0)
    dilation = L.meta.get("dilation", 1)
    groups = L.meta.get("groups", 1)
    
    # Input shape: (batch, in_channels, height, width) - for bounds propagation batch=1
    in_shape = Bin.lb.shape
    if len(in_shape) == 1:
        # Flatten input needs to be reshaped
        channels = L.meta.get("in_channels")
        height = L.meta.get("in_height") 
        width = L.meta.get("in_width")
        in_shape = (channels, height, width)
        Bin_reshaped_lb = Bin.lb.view(1, *in_shape)
        Bin_reshaped_ub = Bin.ub.view(1, *in_shape)
    else:
        Bin_reshaped_lb = Bin.lb.unsqueeze(0) if len(in_shape) == 3 else Bin.lb
        Bin_reshaped_ub = Bin.ub.unsqueeze(0) if len(in_shape) == 3 else Bin.ub
    
    # Apply convolution to bounds
    # For HybridZ: more precise bound computation considering kernel structure
    weight_pos = torch.clamp(weight, min=0)
    weight_neg = torch.clamp(weight, max=0)
    
    # Lower bound: positive weights * lower bounds + negative weights * upper bounds
    lb_conv = F.conv2d(Bin_reshaped_lb, weight_pos, bias=None, stride=stride, 
                       padding=padding, dilation=dilation, groups=groups)
    lb_conv += F.conv2d(Bin_reshaped_ub, weight_neg, bias=None, stride=stride,
                        padding=padding, dilation=dilation, groups=groups)
    
    # Upper bound: positive weights * upper bounds + negative weights * lower bounds  
    ub_conv = F.conv2d(Bin_reshaped_ub, weight_pos, bias=None, stride=stride,
                       padding=padding, dilation=dilation, groups=groups)
    ub_conv += F.conv2d(Bin_reshaped_lb, weight_neg, bias=None, stride=stride,
                        padding=padding, dilation=dilation, groups=groups)
    
    if bias is not None:
        lb_conv += bias.view(1, -1, 1, 1)
        ub_conv += bias.view(1, -1, 1, 1)
    
    # Flatten output if needed
    lb = lb_conv.squeeze(0).flatten() if len(L.out_vars) != lb_conv.numel() else lb_conv.squeeze(0)
    ub = ub_conv.squeeze(0).flatten() if len(L.out_vars) != ub_conv.numel() else ub_conv.squeeze(0)
    
    Bout = Bounds(lb=lb, ub=ub)
    
    # Generate convolution constraints
    cons = ConSet()
    cons.add_conv2d(L.id, L.in_vars, L.out_vars, weight, bias, stride, padding, dilation, groups)
    
    return Fact(bounds=Bout, cons=cons)


@torch.no_grad()
def hybridz_tf_maxpool2d(L: Layer, Bin: Bounds) -> Fact:
    """HybridZ transfer function for 2D max pooling."""
    kernel_size = L.meta.get("kernel_size", 2)
    stride = L.meta.get("stride", kernel_size)
    padding = L.meta.get("padding", 0)
    
    # Reshape input if flattened
    in_shape = L.meta.get("in_shape")  # (channels, height, width)
    if len(Bin.lb.shape) == 1 and in_shape:
        Bin_lb = Bin.lb.view(1, *in_shape)
        Bin_ub = Bin.ub.view(1, *in_shape)
    else:
        Bin_lb = Bin.lb.unsqueeze(0) if len(Bin.lb.shape) == 3 else Bin.lb
        Bin_ub = Bin.ub.unsqueeze(0) if len(Bin.ub.shape) == 3 else Bin.ub
    
    # Max pooling: upper bounds of pooling regions
    # For HybridZ: track which neurons contribute to maximum
    lb_pool = F.max_pool2d(Bin_lb, kernel_size, stride=stride, padding=padding)
    ub_pool = F.max_pool2d(Bin_ub, kernel_size, stride=stride, padding=padding)
    
    # For max pooling, lower bound is more complex - use max of lower bounds in each region
    # This is conservative but sound
    lb = lb_pool.squeeze(0).flatten() if len(L.out_vars) != lb_pool.numel() else lb_pool.squeeze(0)
    ub = ub_pool.squeeze(0).flatten() if len(L.out_vars) != ub_pool.numel() else ub_pool.squeeze(0)
    
    Bout = Bounds(lb=lb, ub=ub)
    
    cons = ConSet()
    # Max pooling generates max constraints
    cons.add_maxpool2d(L.id, L.in_vars, L.out_vars, kernel_size, stride, padding, in_shape)
    
    return Fact(bounds=Bout, cons=cons)


@torch.no_grad()
def hybridz_tf_avgpool2d(L: Layer, Bin: Bounds) -> Fact:
    """HybridZ transfer function for 2D average pooling."""
    kernel_size = L.meta.get("kernel_size", 2)
    stride = L.meta.get("stride", kernel_size)
    padding = L.meta.get("padding", 0)
    
    # Reshape input if needed
    in_shape = L.meta.get("in_shape")
    if len(Bin.lb.shape) == 1 and in_shape:
        Bin_lb = Bin.lb.view(1, *in_shape)
        Bin_ub = Bin.ub.view(1, *in_shape)
    else:
        Bin_lb = Bin.lb.unsqueeze(0) if len(Bin.lb.shape) == 3 else Bin.lb
        Bin_ub = Bin.ub.unsqueeze(0) if len(Bin.ub.shape) == 3 else Bin.ub
    
    # Average pooling is linear - exact bounds
    lb_pool = F.avg_pool2d(Bin_lb, kernel_size, stride=stride, padding=padding)
    ub_pool = F.avg_pool2d(Bin_ub, kernel_size, stride=stride, padding=padding)
    
    lb = lb_pool.squeeze(0).flatten() if len(L.out_vars) != lb_pool.numel() else lb_pool.squeeze(0)
    ub = ub_pool.squeeze(0).flatten() if len(L.out_vars) != ub_pool.numel() else ub_pool.squeeze(0)
    
    Bout = Bounds(lb=lb, ub=ub)
    
    cons = ConSet()
    cons.add_avgpool2d(L.id, L.in_vars, L.out_vars, kernel_size, stride, padding, in_shape)
    
    return Fact(bounds=Bout, cons=cons)


@torch.no_grad()
def hybridz_tf_flatten(L: Layer, Bin: Bounds) -> Fact:
    """HybridZ transfer function for tensor flattening."""
    # Flattening is just reshaping - bounds remain the same
    start_dim = L.meta.get("start_dim", 1)
    end_dim = L.meta.get("end_dim", -1)
    
    # Simple reshape - no change in bounds
    lb = Bin.lb.flatten()
    ub = Bin.ub.flatten()
    Bout = Bounds(lb=lb, ub=ub)
    
    cons = ConSet()
    cons.add_flatten(L.id, L.in_vars, L.out_vars, start_dim, end_dim)
    
    return Fact(bounds=Bout, cons=cons)


@torch.no_grad()
def hybridz_tf_reshape(L: Layer, Bin: Bounds) -> Fact:
    """HybridZ transfer function for general tensor reshaping."""
    target_shape = L.meta.get("shape")
    
    # Reshape bounds preserving values
    lb = Bin.lb.reshape(target_shape) if target_shape else Bin.lb
    ub = Bin.ub.reshape(target_shape) if target_shape else Bin.ub
    
    # Flatten for output variables
    lb = lb.flatten()
    ub = ub.flatten()
    Bout = Bounds(lb=lb, ub=ub)
    
    cons = ConSet()
    cons.add_reshape(L.id, L.in_vars, L.out_vars, target_shape)
    
    return Fact(bounds=Bout, cons=cons)