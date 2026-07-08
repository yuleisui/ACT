#===- act/back_end/interval_tf/tf_rnn.py - RNN Interval Transfer Func ---====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   RNN Interval Transfer Functions. Provides transfer functions for RNN layers
#   to enable the abstraction framework to handle recurrent neural networks.


import torch
from act.back_end.core import Bounds, Con, ConSet, Fact, Layer
from act.back_end.utils import affine_bounds, split_weight, four_corner_envelope


def tf_lstm(L: Layer, Bin: Bounds) -> Fact:
    """
    Transfer function for LSTM layer.
    
    Handles LSTM cell computation with interval bounds propagation.
    """
    # Extract LSTM parameters
    weight_ih = _get_tensor_param(L, "weight_ih_l0")  # input-to-hidden weights
    weight_hh = _get_tensor_param(L, "weight_hh_l0")  # hidden-to-hidden weights
    bias_ih = _get_optional_tensor_param(L, "bias_ih_l0")
    bias_hh = _get_optional_tensor_param(L, "bias_hh_l0")
    
    # LSTM dimensions
    input_size = L.get_int("input_size")
    hidden_size = L.get_int("hidden_size")
    num_layers = L.get_int("num_layers", 1)
    batch_first = bool(L.get_scalar("batch_first", False))
    bidirectional = bool(L.get_scalar("bidirectional", False))
    
    # Shape information
    input_shape = _get_shape_param(L, "input_shape")  # [batch, seq_len, input_size] or [seq_len, batch, input_size]
    output_shape = _get_shape_param(L, "output_shape")
    
    seq_bounds = _reshape_sequence_bounds(Bin, input_shape, batch_first)
    batch_size, seq_len, _ = seq_bounds.lb.shape

    # For verification, we approximate LSTM with its linearized form
    # This is a conservative approximation using the worst-case bounds

    # Initialize hidden and cell states bounds (typically zeros)
    h_bounds = Bounds(
        Bin.lb.new_zeros(batch_size, hidden_size),
        Bin.lb.new_zeros(batch_size, hidden_size),
    )
    c_bounds = Bounds(
        Bin.lb.new_zeros(batch_size, hidden_size),
        Bin.lb.new_zeros(batch_size, hidden_size),
    )
    
    # Process each time step
    output_bounds_list = []
    
    for t in range(seq_len):
        # Extract input at time step t
        x_t_bounds = Bounds(seq_bounds.lb[:, t, :], seq_bounds.ub[:, t, :])
        
        # LSTM cell computation with bounds
        h_bounds, c_bounds = _lstm_cell_bounds(
            x_t_bounds, h_bounds, c_bounds,
            weight_ih, weight_hh, bias_ih, bias_hh
        )
        
        output_bounds_list.append(h_bounds)
    
    output_lb, output_ub = _stack_step_bounds(output_bounds_list)
    
    if bidirectional:
        raise NotImplementedError(
            "tf_lstm: bidirectional=True not supported (reverse-direction "
            "sweep over weight_*_l0_reverse not implemented)."
        )
    
    B_output = Bounds(output_lb, output_ub)
    
    # Create constraints
    C = ConSet()
    C.replace(Con("INEQ", tuple(L.out_vars + L.in_vars), {
        "tag": f"lstm:{L.id}",
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "bidirectional": bidirectional,
        "batch_first": batch_first,
        "input_shape": input_shape,
        "output_shape": output_shape
    }))
    
    C.add_box(L.id, L.out_vars, B_output)
    return Fact(B_output, C)


def tf_gru(L: Layer, Bin: Bounds) -> Fact:
    """
    Transfer function for GRU layer.
    
    Handles GRU cell computation with interval bounds propagation.
    """
    # Extract GRU parameters
    weight_ih = _get_tensor_param(L, "weight_ih_l0")
    weight_hh = _get_tensor_param(L, "weight_hh_l0")
    bias_ih = _get_optional_tensor_param(L, "bias_ih_l0")
    bias_hh = _get_optional_tensor_param(L, "bias_hh_l0")
    
    # GRU dimensions
    input_size = L.get_int("input_size")
    hidden_size = L.get_int("hidden_size")
    num_layers = L.get_int("num_layers", 1)
    batch_first = bool(L.get_scalar("batch_first", False))
    bidirectional = bool(L.get_scalar("bidirectional", False))
    
    # Input shape information
    input_shape = _get_shape_param(L, "input_shape")
    output_shape = _get_shape_param(L, "output_shape")
    
    seq_bounds = _reshape_sequence_bounds(Bin, input_shape, batch_first)
    batch_size, seq_len, _ = seq_bounds.lb.shape

    # Initialize hidden state bounds
    h_bounds = Bounds(
        Bin.lb.new_zeros(batch_size, hidden_size),
        Bin.lb.new_zeros(batch_size, hidden_size),
    )

    # Process each time step
    output_bounds_list = []

    for t in range(seq_len):
        # Extract input at time step t
        x_t_bounds = Bounds(seq_bounds.lb[:, t, :], seq_bounds.ub[:, t, :])

        # GRU cell computation with bounds
        h_bounds = _gru_cell_bounds(
            x_t_bounds, h_bounds,
            weight_ih, weight_hh, bias_ih, bias_hh
        )
        
        output_bounds_list.append(h_bounds)
    
    output_lb, output_ub = _stack_step_bounds(output_bounds_list)
    
    if bidirectional:
        # Same soundness issue as tf_lstm — see that function for details.
        raise NotImplementedError(
            "tf_gru: bidirectional=True not supported (reverse-direction "
            "sweep over weight_*_l0_reverse not implemented)."
        )
    
    B_output = Bounds(output_lb, output_ub)
    
    # Create constraints
    C = ConSet()
    C.replace(Con("INEQ", tuple(L.out_vars + L.in_vars), {
        "tag": f"gru:{L.id}",
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "bidirectional": bidirectional,
        "batch_first": batch_first,
        "input_shape": input_shape,
        "output_shape": output_shape
    }))
    
    C.add_box(L.id, L.out_vars, B_output)
    return Fact(B_output, C)


def tf_rnn(L: Layer, Bin: Bounds) -> Fact:
    """
    Transfer function for vanilla RNN layer.
    
    Handles simple RNN cell computation with interval bounds propagation.
    """
    # Extract RNN parameters
    weight_ih = _get_tensor_param(L, "weight_ih_l0")
    weight_hh = _get_tensor_param(L, "weight_hh_l0")
    bias_ih = _get_optional_tensor_param(L, "bias_ih_l0")
    bias_hh = _get_optional_tensor_param(L, "bias_hh_l0")
    
    # RNN dimensions
    input_size = L.get_int("input_size")
    hidden_size = L.get_int("hidden_size")
    nonlinearity = str(L.get_scalar("nonlinearity", "tanh"))  # 'tanh' or 'relu'
    batch_first = bool(L.get_scalar("batch_first", False))
    bidirectional = bool(L.get_scalar("bidirectional", False))
    
    # Input shape information
    input_shape = _get_shape_param(L, "input_shape")
    output_shape = _get_shape_param(L, "output_shape")
    
    seq_bounds = _reshape_sequence_bounds(Bin, input_shape, batch_first)
    batch_size, seq_len, _ = seq_bounds.lb.shape

    # Initialize hidden state bounds
    h_bounds = Bounds(
        Bin.lb.new_zeros(batch_size, hidden_size),
        Bin.lb.new_zeros(batch_size, hidden_size),
    )
    
    # Process each time step
    output_bounds_list = []
    
    for t in range(seq_len):
        # Extract input at time step t
        x_t_bounds = Bounds(seq_bounds.lb[:, t, :], seq_bounds.ub[:, t, :])
        
        # RNN cell computation: h_t = tanh(W_ih * x_t + W_hh * h_{t-1} + b)
        # Linear transformation
        ih_transform = _apply_linear_bounds(x_t_bounds, weight_ih, bias_ih)
        hh_transform = _apply_linear_bounds(h_bounds, weight_hh, bias_hh)
        
        # Add transformations
        combined_bounds = Bounds(
            ih_transform.lb + hh_transform.lb,
            ih_transform.ub + hh_transform.ub
        )
        
        # Apply nonlinearity
        if nonlinearity == "tanh":
            h_bounds = _apply_tanh_bounds(combined_bounds)
        elif nonlinearity == "relu":
            h_bounds = _apply_relu_bounds(combined_bounds)
        else:
            raise ValueError(f"Unsupported RNN nonlinearity: {nonlinearity}")
        
        output_bounds_list.append(h_bounds)
    
    output_lb, output_ub = _stack_step_bounds(output_bounds_list)
    
    if bidirectional:
        # Same soundness issue as tf_lstm — see that function for details.
        raise NotImplementedError(
            "tf_rnn: bidirectional=True not supported (reverse-direction "
            "sweep over weight_*_l0_reverse not implemented)."
        )
    
    B_output = Bounds(output_lb, output_ub)
    
    # Create constraints
    C = ConSet()
    C.replace(Con("INEQ", tuple(L.out_vars + L.in_vars), {
        "tag": f"rnn:{L.id}",
        "input_size": input_size,
        "hidden_size": hidden_size,
        "nonlinearity": nonlinearity,
        "bidirectional": bidirectional,
        "batch_first": batch_first,
        "input_shape": input_shape,
        "output_shape": output_shape
    }))
    
    C.add_box(L.id, L.out_vars, B_output)
    return Fact(B_output, C)


def tf_embedding(L: Layer, Bin: Bounds) -> Fact:
    """
    Transfer function for Embedding layer.
    
    Handles embedding lookup with interval bounds.
    """
    # Extract embedding parameters
    weight = _get_tensor_param(L, "weight")  # [num_embeddings, embedding_dim]
    num_embeddings = L.get_int("num_embeddings")
    embedding_dim = L.get_int("embedding_dim")
    padding_idx = L.params.get("padding_idx", None)
    
    # Input shape information (indices)
    input_shape = _get_shape_param(L, "input_shape")
    output_shape = _get_shape_param(L, "output_shape")
    
    # For embedding, the bounds depend on the range of possible embeddings
    # Since we don't know which embeddings will be selected, we use worst-case bounds
    weight_min = torch.min(weight, dim=0)[0]  # [embedding_dim]
    weight_max = torch.max(weight, dim=0)[0]  # [embedding_dim]
    
    # Broadcast to output shape
    output_size = 1
    for dim in output_shape:
        output_size *= dim
    embedding_elements = output_size // embedding_dim
    
    output_lb = weight_min.repeat(embedding_elements)
    output_ub = weight_max.repeat(embedding_elements)
    
    B_output = Bounds(output_lb, output_ub)
    
    # Create constraints
    C = ConSet()
    C.replace(Con("INEQ", tuple(L.out_vars + L.in_vars), {
        "tag": f"embedding:{L.id}",
        "num_embeddings": num_embeddings,
        "embedding_dim": embedding_dim,
        "padding_idx": padding_idx,
        "input_shape": input_shape,
        "output_shape": output_shape
    }))
    
    C.add_box(L.id, L.out_vars, B_output)
    return Fact(B_output, C)


# Helper functions for RNN computations

def _get_tensor_param(L: Layer, key: str) -> torch.Tensor:
    value = L.get_tensor(key)
    if value is None:
        raise TypeError(f"Expected tensor param '{key}' on layer {L.id}")
    return value


def _get_optional_tensor_param(L: Layer, key: str):
    return L.get_tensor(key)


def _get_shape_param(L: Layer, key: str) -> tuple[int, ...]:
    return tuple(int(dim) for dim in L.get_tuple(key))

def _reshape_sequence_bounds(Bin: Bounds, input_shape, batch_first: bool) -> Bounds:
    """Reshape sequence inputs and normalize to [B, T, F] for the time loop.

    ``input_shape`` records the JSON's native batch dim (typically B=1); the
    actual bounds may have been batchified to B>1 by validate_verifier or
    test fixtures. We infer the runtime B from the bounds' numel and the
    per-batch shape recorded in ``input_shape``.
    """
    shape = tuple(int(d) for d in input_shape)
    t_dim = shape[1] if batch_first else shape[0]
    f_dim = shape[2]
    per_batch_numel = t_dim * f_dim
    B_runtime = Bin.lb.numel() // per_batch_numel
    target = (B_runtime, t_dim, f_dim) if batch_first else (t_dim, B_runtime, f_dim)
    seq_lb = Bin.lb.reshape(target)
    seq_ub = Bin.ub.reshape(target)
    if not batch_first:
        seq_lb = seq_lb.permute(1, 0, 2).contiguous()
        seq_ub = seq_ub.permute(1, 0, 2).contiguous()
    return Bounds(seq_lb, seq_ub)


def _stack_step_bounds(output_bounds_list):
    """Stack per-step hidden-state bounds with batch preserved as dim 0."""
    return (
        torch.stack([b.lb for b in output_bounds_list], dim=1),
        torch.stack([b.ub for b in output_bounds_list], dim=1),
    )

def _lstm_cell_bounds(x_bounds, h_bounds, c_bounds, weight_ih, weight_hh, bias_ih, bias_hh):
    """Compute LSTM cell bounds for one time step."""
    # LSTM gates: i, f, g, o = input, forget, cell, output gates
    
    # Linear transformations
    ih_all = _apply_linear_bounds(x_bounds, weight_ih, bias_ih)
    hh_all = _apply_linear_bounds(h_bounds, weight_hh, bias_hh)
    
    # Split into 4 gates
    hidden_size = h_bounds.lb.shape[-1]
    
    # Input gate
    i_bounds = Bounds(
        ih_all.lb[:, :hidden_size] + hh_all.lb[:, :hidden_size],
        ih_all.ub[:, :hidden_size] + hh_all.ub[:, :hidden_size]
    )
    i_bounds = _apply_sigmoid_bounds(i_bounds)
    
    # Forget gate
    f_bounds = Bounds(
        ih_all.lb[:, hidden_size:2*hidden_size] + hh_all.lb[:, hidden_size:2*hidden_size],
        ih_all.ub[:, hidden_size:2*hidden_size] + hh_all.ub[:, hidden_size:2*hidden_size]
    )
    f_bounds = _apply_sigmoid_bounds(f_bounds)
    
    # Cell gate
    g_bounds = Bounds(
        ih_all.lb[:, 2*hidden_size:3*hidden_size] + hh_all.lb[:, 2*hidden_size:3*hidden_size],
        ih_all.ub[:, 2*hidden_size:3*hidden_size] + hh_all.ub[:, 2*hidden_size:3*hidden_size]
    )
    g_bounds = _apply_tanh_bounds(g_bounds)
    
    # Output gate
    o_bounds = Bounds(
        ih_all.lb[:, 3*hidden_size:] + hh_all.lb[:, 3*hidden_size:],
        ih_all.ub[:, 3*hidden_size:] + hh_all.ub[:, 3*hidden_size:]
    )
    o_bounds = _apply_sigmoid_bounds(o_bounds)
    
    # Cell state update: c_t = f * c_{t-1} + i * g
    fc_bounds = _multiply_bounds(f_bounds, c_bounds)
    ig_bounds = _multiply_bounds(i_bounds, g_bounds)
    new_c_bounds = Bounds(
        fc_bounds.lb + ig_bounds.lb,
        fc_bounds.ub + ig_bounds.ub
    )
    
    # Hidden state update: h_t = o * tanh(c_t)
    tanh_c_bounds = _apply_tanh_bounds(new_c_bounds)
    new_h_bounds = _multiply_bounds(o_bounds, tanh_c_bounds)
    
    return new_h_bounds, new_c_bounds


def _gru_cell_bounds(x_bounds, h_bounds, weight_ih, weight_hh, bias_ih, bias_hh):
    """Compute GRU cell bounds for one time step."""
    # GRU gates: r, z, n = reset, update, new gates
    
    # Linear transformations
    ih_all = _apply_linear_bounds(x_bounds, weight_ih, bias_ih)
    hh_all = _apply_linear_bounds(h_bounds, weight_hh, bias_hh)
    
    hidden_size = h_bounds.lb.shape[-1]
    
    # Reset gate
    r_bounds = Bounds(
        ih_all.lb[:, :hidden_size] + hh_all.lb[:, :hidden_size],
        ih_all.ub[:, :hidden_size] + hh_all.ub[:, :hidden_size]
    )
    r_bounds = _apply_sigmoid_bounds(r_bounds)
    
    # Update gate
    z_bounds = Bounds(
        ih_all.lb[:, hidden_size:2*hidden_size] + hh_all.lb[:, hidden_size:2*hidden_size],
        ih_all.ub[:, hidden_size:2*hidden_size] + hh_all.ub[:, hidden_size:2*hidden_size]
    )
    z_bounds = _apply_sigmoid_bounds(z_bounds)
    
    # New gate computation: n = tanh(W_ih_n * x + W_hh_n * (r * h))
    rh_bounds = _multiply_bounds(r_bounds, h_bounds)
    # weight_hh has shape (3*H, H): rows index gates, cols index hidden.
    # The "n" gate's hh sub-matrix is weight_hh[2H:3H, :] (rows), not [:, 2H:].
    hh_n = _apply_linear_bounds(rh_bounds, weight_hh[2*hidden_size:3*hidden_size, :], bias_hh[2*hidden_size:3*hidden_size] if bias_hh is not None else None)
    
    n_bounds = Bounds(
        ih_all.lb[:, 2*hidden_size:] + hh_n.lb,
        ih_all.ub[:, 2*hidden_size:] + hh_n.ub
    )
    n_bounds = _apply_tanh_bounds(n_bounds)
    
    # Hidden state update: h_t = (1 - z) * n + z * h
    one_minus_z_bounds = Bounds(1 - z_bounds.ub, 1 - z_bounds.lb)
    term1 = _multiply_bounds(one_minus_z_bounds, n_bounds)
    term2 = _multiply_bounds(z_bounds, h_bounds)
    
    new_h_bounds = Bounds(
        term1.lb + term2.lb,
        term1.ub + term2.ub
    )
    
    return new_h_bounds


def _apply_linear_bounds(input_bounds, weight, bias=None):
    """Apply linear transformation to bounds."""
    W_pos, W_neg = split_weight(weight)
    if bias is None:
        bias = weight.new_zeros(weight.shape[0])
    # input_bounds.lb shape (..., in_features); weight shape (out, in).
    out = affine_bounds(W_pos, W_neg, bias, Bounds(input_bounds.lb, input_bounds.ub))
    return Bounds(out.lb, out.ub)


def _apply_sigmoid_bounds(bounds):
    """Apply sigmoid function to bounds."""
    # Sigmoid is monotonic, so we can apply it directly
    return Bounds(torch.sigmoid(bounds.lb), torch.sigmoid(bounds.ub))


def _apply_tanh_bounds(bounds):
    """Apply tanh function to bounds."""
    # Tanh is monotonic, so we can apply it directly
    return Bounds(torch.tanh(bounds.lb), torch.tanh(bounds.ub))


def _apply_relu_bounds(bounds):
    """Apply ReLU function to bounds."""
    # ReLU: max(0, x)
    return Bounds(
        torch.clamp(bounds.lb, min=0),
        torch.clamp(bounds.ub, min=0)
    )


def _multiply_bounds(bounds1, bounds2):
    """Multiply two interval bounds."""
    # For interval [a,b] * [c,d], result is [min(ac,ad,bc,bd), max(ac,ad,bc,bd)]
    lb, ub = four_corner_envelope(bounds1.lb, bounds1.ub, bounds2.lb, bounds2.ub)
    return Bounds(lb, ub)
