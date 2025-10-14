# core.py
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any

@dataclass(eq=True, frozen=True)
class Bounds:
    lb: torch.Tensor
    ub: torch.Tensor
    def copy(self) -> "Bounds": return Bounds(self.lb.clone(), self.ub.clone())

@dataclass(eq=False)
class Con:
    kind: str                      # 'EQ' | 'INEQ' | 'BIN'
    var_ids: Tuple[int, ...]
    meta: Dict[str, Any] = field(default_factory=dict)
    # Optional numeric payloads (unused internally; only for compatibility)
    A: Any=None; b: Any=None; C: Any=None; d: Any=None
    def signature(self) -> Tuple[str, Tuple[int, ...], str]:
        return (self.kind, self.var_ids, self.meta.get("tag",""))

@dataclass
class ConSet:
    S: Dict[Tuple[str, Tuple[int, ...], str], Con] = field(default_factory=dict)
    def replace(self, c: Con): self.S[c.signature()] = c
    def add_box(self, layer_id: int, var_ids: List[int], B: Bounds):
        self.replace(Con("INEQ", tuple(var_ids), {"tag": f"box:{layer_id}", "lb": B.lb.clone(), "ub": B.ub.clone()}))

@dataclass
class Fact:
    bounds: Bounds
    cons: ConSet

# Supported layer types:
# MLP: DENSE, BIAS, SCALE, RELU, LRELU, ABS, CLIP, ADD, MUL, CONCAT, BN
# CNN: CONV2D, MAXPOOL2D, AVGPOOL2D, FLATTEN  
# RNN: LSTM, GRU, RNN, EMBEDDING
# Activations: SIGMOID, TANH, SOFTPLUS, SILU, GELU
# Transformer: POSENC, LAYERNORM, ATT_SCORES, SOFTMAX, ATT_MIX, etc.

# ============================================================================
# Layer Architecture: Unified Neural Network Layer Representation
# ============================================================================
# 
# - params: ONLY PyTorch tensors (weights, biases) for GPU operations
# - meta: Non-tensor metadata (shapes, strides, hyperparameters) for CPU access
#
# layer.params["weight"], layer.meta["stride"]
#
# ============================================================================
# Parameter Classification by Layer Type
# ============================================================================
#
# DENSE layer:
#   params: {"W": weight_tensor, "W_pos": pos_weights, "W_neg": neg_weights, "b": bias}  
#   meta:   {"input_shape": shape, "output_shape": shape, "activation": func}
#
# CONV2D layer:
#   params: {"weight": weight_tensor, "bias": bias_tensor}
#   meta:   {"stride": stride, "padding": padding, "input_shape": shape, "output_shape": shape}
#
# RELU/LRELU layer:
#   params: {} (no tensors)
#   meta:   {"negative_slope": float} (for LRELU only)
#
# BATCH_NORM layer:
#   params: {"weight": gamma, "bias": beta, "running_mean": mean, "running_var": var}
#   meta:   {"eps": epsilon, "momentum": momentum, "track_running_stats": bool}
#
# MAXPOOL2D/AVGPOOL2D layer:
#   params: {} (no tensors)
#   meta:   {"kernel_size": size, "stride": stride, "padding": padding}
#
# LSTM/GRU layer:
#   params: {"weight_ih": input_weights, "weight_hh": hidden_weights, "bias_ih": input_bias, "bias_hh": hidden_bias}
#   meta:   {"input_size": int, "hidden_size": int, "num_layers": int, "bidirectional": bool}
#
# EMBEDDING layer:
#   params: {"weight": embedding_matrix}
#   meta:   {"num_embeddings": int, "embedding_dim": int, "padding_idx": int}
#
# ============================================================================
# Tensor Parameters (params dict) - GPU-bound tensors
# ============================================================================
# Core Neural Network Parameters:
# - weight, bias: Standard layer weights and biases
# - W, b: Dense layer weight matrix and bias vector  
# - W_pos, W_neg: Positive and negative weight decomposition (for interval arithmetic)
#
# Batch Normalization Parameters:
# - running_mean, running_var: Batch norm running statistics
# - gamma, beta: Batch norm scale and shift parameters
#
# RNN/LSTM Parameters:
# - weight_ih, weight_hh: Input-to-hidden and hidden-to-hidden weights
# - bias_ih, bias_hh: Input and hidden biases
#
# Embedding Parameters:
# - weight: Embedding lookup table matrix
#
# ============================================================================
# Metadata Parameters (meta dict) - CPU-bound configuration
# ============================================================================
# Convolution Settings:
# - stride, padding, dilation: Convolution operation parameters
# - kernel_size, groups: Convolution kernel configuration
#
# Layer Dimensions:
# - input_shape, output_shape: Layer input/output tensor shapes
# - input_size, hidden_size, output_size: Layer size specifications
#
# Activation & Normalization:
# - eps, momentum: Batch normalization hyperparameters
# - negative_slope: LeakyReLU slope parameter
# - activation: Activation function specification
#
# Pooling Configuration:
# - kernel_size, stride, padding: Pooling operation parameters
# - ceil_mode, count_include_pad: Pooling behavior flags
#
# RNN Configuration:
# - num_layers, bidirectional: RNN architecture parameters
# - dropout, batch_first: RNN training parameters
#
# Embedding Configuration:
# - num_embeddings, embedding_dim: Embedding table dimensions  
# - padding_idx, max_norm: Embedding behavior parameters
#
# General Configuration:
# - bias_enabled: Whether bias is used in the layer
# - track_running_stats: Whether to track batch norm statistics
# - Any other non-tensor configuration data
#
# ============================================================================
# Device Management Strategy
# ============================================================================
# - All tensors in 'params' are automatically moved to appropriate device (CPU/GPU)
# - All data in 'meta' remains on CPU for efficient access
# - Cache tensors are managed separately for runtime computations

@dataclass
class Layer:
    id: int                                     # Unique layer identifier
    kind: str                                   # UPPER name (e.g., "DENSE", "CONV2D", "RELU")
    params: Dict[str, torch.Tensor]            # Numeric tensors (weights, biases) on device
    meta: Dict[str, Any]                       # Non-numeric metadata (shapes, strides, etc.)
    in_vars: List[int]                         # Input variable indices 
    out_vars: List[int]                        # Output variable indices
    cache: Dict[str, torch.Tensor] = field(default_factory=dict)  # Runtime cache tensors

@dataclass
class Net:
    layers: List[Layer]
    preds: Dict[int, List[int]]
    succs: Dict[int, List[int]]
    by_id: Dict[int, Layer] = field(init=False)
    def __post_init__(self):
        self.by_id = {L.id: L for L in self.layers}
