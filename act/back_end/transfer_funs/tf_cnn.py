# transfer_cnn.py
"""
CNN Transfer Functions for ACT Abstraction Framework

This module provides transfer functions for CNN layers to enable
the abstraction framework to handle convolutional neural networks.
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple
from act.back_end.core import Bounds, Con, ConSet, Fact, Layer
from act.back_end.utils import affine_bounds, pwl_meta, bound_var_interval, scale_interval


def tf_conv2d(L: Layer, Bin: Bounds) -> Fact:
    """
    Transfer function for Conv2d layer.
    
    Linearizes the convolution operation using im2col transformation.
    """
    # Extract convolution parameters
    weight = L.params["weight"]  # [out_channels, in_channels, kernel_h, kernel_w]
    bias = L.params.get("bias", None)
    stride = L.meta.get("stride", 1)
    padding = L.meta.get("padding", 0)
    dilation = L.meta.get("dilation", 1)
    groups = L.meta.get("groups", 1)
    
    # Input shape information
    input_shape = L.meta["input_shape"]  # [batch, channels, height, width]
    output_shape = L.meta["output_shape"]  # [batch, out_channels, out_h, out_w]
    
    batch_size, in_channels, in_h, in_w = input_shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    _, _, out_h, out_w = output_shape
    
    # Flatten input bounds for processing
    input_flat_size = in_channels * in_h * in_w
    output_flat_size = out_channels * out_h * out_w
    
    # Create equivalent linear transformation matrix using im2col
    # This converts the convolution to matrix multiplication
    W_equiv = _conv2d_to_linear_matrix(
        weight, input_shape, output_shape, stride, padding, dilation, groups
    )
    
    # Apply affine transformation
    if bias is not None:
        b_equiv = bias.repeat(out_h * out_w)  # Broadcast bias across spatial dimensions
    else:
        b_equiv = torch.zeros(output_flat_size, dtype=weight.dtype, device=weight.device)
    
    # Compute bounds using affine transformation
    W_pos = torch.clamp(W_equiv, min=0)
    W_neg = torch.clamp(W_equiv, max=0)
    
    # Reshape input bounds to flat format
    input_bounds_flat = Bounds(
        Bin.lb.view(-1),  # [input_flat_size]
        Bin.ub.view(-1)   # [input_flat_size]
    )
    
    # Apply linear transformation
    B_output = affine_bounds(W_pos, W_neg, b_equiv, input_bounds_flat)
    
    # Create constraints
    C = ConSet()
    C.replace(Con("EQ", tuple(L.out_vars + L.in_vars), {
        "tag": f"conv2d:{L.id}",
        "W": W_equiv,
        "b": b_equiv,
        "input_shape": input_shape,
        "output_shape": output_shape,
        "conv_params": {
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "groups": groups
        }
    }))
    
    C.add_box(L.id, L.out_vars, B_output)
    return Fact(B_output, C)


def tf_maxpool2d(L: Layer, Bin: Bounds) -> Fact:
    """
    Transfer function for MaxPool2d layer.
    
    Uses interval arithmetic to bound the max pooling operation.
    """
    # Extract pooling parameters
    kernel_size = L.meta["kernel_size"]
    stride = L.meta.get("stride", kernel_size)
    padding = L.meta.get("padding", 0)
    dilation = L.meta.get("dilation", 1)
    
    # Shape information
    input_shape = L.meta["input_shape"]  # [batch, channels, height, width]
    output_shape = L.meta["output_shape"]  # [batch, channels, out_h, out_w]
    
    batch_size, channels, in_h, in_w = input_shape
    _, _, out_h, out_w = output_shape
    
    # For max pooling, we need to consider all possible inputs in each pool window
    # The output bounds are the max of upper bounds and max of lower bounds in each window
    
    # Reshape bounds for pooling operation
    input_lb = Bin.lb.view(batch_size, channels, in_h, in_w)
    input_ub = Bin.ub.view(batch_size, channels, in_h, in_w)
    
    # Apply max pooling to bounds
    # For lower bound: take max of lower bounds in each window
    # For upper bound: take max of upper bounds in each window
    output_lb = F.max_pool2d(input_lb, kernel_size, stride, padding, dilation)
    output_ub = F.max_pool2d(input_ub, kernel_size, stride, padding, dilation)
    
    # Flatten output bounds
    B_output = Bounds(output_lb.view(-1), output_ub.view(-1))
    
    # Create constraints for max pooling
    C = ConSet()
    C.replace(Con("INEQ", tuple(L.out_vars + L.in_vars), {
        "tag": f"maxpool2d:{L.id}",
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "input_shape": input_shape,
        "output_shape": output_shape
    }))
    
    C.add_box(L.id, L.out_vars, B_output)
    return Fact(B_output, C)


def tf_flatten(L: Layer, Bin: Bounds) -> Fact:
    """
    Transfer function for Flatten layer.
    
    Simply reshapes the bounds without changing values.
    """
    # Extract shape information
    input_shape = L.meta["input_shape"]
    output_shape = L.meta["output_shape"]
    
    # Flatten is just a reshape operation - bounds remain the same
    input_size = torch.prod(torch.tensor(input_shape)).item()
    output_size = torch.prod(torch.tensor(output_shape)).item()
    
    if input_size != output_size:
        raise ValueError(f"Flatten: input size {input_size} != output size {output_size}")
    
    # Bounds are preserved, just reshaped
    B_output = Bounds(Bin.lb.view(-1), Bin.ub.view(-1))
    
    # Create identity constraint
    C = ConSet()
    C.replace(Con("EQ", tuple(L.out_vars + L.in_vars), {
        "tag": f"flatten:{L.id}",
        "input_shape": input_shape,
        "output_shape": output_shape
    }))
    
    C.add_box(L.id, L.out_vars, B_output)
    return Fact(B_output, C)


def _conv2d_to_linear_matrix(
    weight: torch.Tensor,
    input_shape: Tuple[int, ...],
    output_shape: Tuple[int, ...],
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1
) -> torch.Tensor:
    """
    Convert Conv2d operation to equivalent linear transformation matrix.
    
    This uses the im2col algorithm to unfold the convolution into matrix multiplication.
    """
    batch_size, in_channels, in_h, in_w = input_shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    _, _, out_h, out_w = output_shape
    
    # Create input and output flat sizes
    input_flat_size = in_channels * in_h * in_w
    output_flat_size = out_channels * out_h * out_w
    
    # Initialize the equivalent weight matrix
    W_equiv = torch.zeros(output_flat_size, input_flat_size, dtype=weight.dtype, device=weight.device)
    
    # Convert stride, padding, dilation to tuples if they're integers
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    
    # For each output position, find corresponding input positions
    for out_c in range(out_channels):
        for out_y in range(out_h):
            for out_x in range(out_w):
                # Calculate output linear index
                out_idx = out_c * (out_h * out_w) + out_y * out_w + out_x
                
                # For each kernel position
                for in_c in range(in_channels):
                    for k_y in range(kernel_h):
                        for k_x in range(kernel_w):
                            # Calculate input position
                            in_y = out_y * stride[0] - padding[0] + k_y * dilation[0]
                            in_x = out_x * stride[1] - padding[1] + k_x * dilation[1]
                            
                            # Check bounds
                            if 0 <= in_y < in_h and 0 <= in_x < in_w:
                                # Calculate input linear index
                                in_idx = in_c * (in_h * in_w) + in_y * in_w + in_x
                                
                                # Set weight in equivalent matrix
                                W_equiv[out_idx, in_idx] = weight[out_c, in_c, k_y, k_x]
    
    return W_equiv


def tf_avgpool2d(L: Layer, Bin: Bounds) -> Fact:
    """
    Transfer function for AvgPool2d layer.
    
    Uses linear transformation to handle average pooling.
    """
    # Extract pooling parameters
    kernel_size = L.meta["kernel_size"]
    stride = L.meta.get("stride", kernel_size)
    padding = L.meta.get("padding", 0)
    
    # Input/output shape information
    input_shape = L.meta["input_shape"]
    output_shape = L.meta["output_shape"]
    
    batch_size, channels, in_h, in_w = input_shape
    _, _, out_h, out_w = output_shape
    
    # Create equivalent linear transformation for average pooling
    input_flat_size = channels * in_h * in_w
    output_flat_size = channels * out_h * out_w
    
    W_equiv = _avgpool2d_to_linear_matrix(
        input_shape, output_shape, kernel_size, stride, padding
    )
    
    # No bias for average pooling
    b_equiv = torch.zeros(output_flat_size, dtype=Bin.lb.dtype, device=Bin.lb.device)
    
    # Apply linear transformation
    W_pos = torch.clamp(W_equiv, min=0)
    W_neg = torch.clamp(W_equiv, max=0)
    
    input_bounds_flat = Bounds(Bin.lb.view(-1), Bin.ub.view(-1))
    B_output = affine_bounds(W_pos, W_neg, b_equiv, input_bounds_flat)
    
    # Create constraints
    C = ConSet()
    C.replace(Con("EQ", tuple(L.out_vars + L.in_vars), {
        "tag": f"avgpool2d:{L.id}",
        "W": W_equiv,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "input_shape": input_shape,
        "output_shape": output_shape
    }))
    
    C.add_box(L.id, L.out_vars, B_output)
    return Fact(B_output, C)


def _avgpool2d_to_linear_matrix(
    input_shape: Tuple[int, ...],
    output_shape: Tuple[int, ...],
    kernel_size: int,
    stride: int,
    padding: int
) -> torch.Tensor:
    """Convert AvgPool2d to equivalent linear transformation matrix."""
    batch_size, channels, in_h, in_w = input_shape
    _, _, out_h, out_w = output_shape
    
    input_flat_size = channels * in_h * in_w
    output_flat_size = channels * out_h * out_w
    
    W_equiv = torch.zeros(output_flat_size, input_flat_size)
    
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    
    kernel_h, kernel_w = kernel_size
    
    for c in range(channels):
        for out_y in range(out_h):
            for out_x in range(out_w):
                out_idx = c * (out_h * out_w) + out_y * out_w + out_x
                
                # Count valid kernel positions
                valid_count = 0
                
                for k_y in range(kernel_h):
                    for k_x in range(kernel_w):
                        in_y = out_y * stride[0] - padding[0] + k_y
                        in_x = out_x * stride[1] - padding[1] + k_x
                        
                        if 0 <= in_y < in_h and 0 <= in_x < in_w:
                            in_idx = c * (in_h * in_w) + in_y * in_w + in_x
                            valid_count += 1
                
                # Set weights for average (1/count for each valid position)
                if valid_count > 0:
                    weight_val = 1.0 / valid_count
                    
                    for k_y in range(kernel_h):
                        for k_x in range(kernel_w):
                            in_y = out_y * stride[0] - padding[0] + k_y
                            in_x = out_x * stride[1] - padding[1] + k_x
                            
                            if 0 <= in_y < in_h and 0 <= in_x < in_w:
                                in_idx = c * (in_h * in_w) + in_y * in_w + in_x
                                W_equiv[out_idx, in_idx] = weight_val
    
    return W_equiv