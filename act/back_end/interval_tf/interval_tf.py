"""
Interval Transfer Function Implementation

This module implements the IntervalTF class that provides interval-based 
transfer functions for standard bounds propagation analysis.
"""

import torch
from typing import Dict, List
from act.back_end.core import Bounds, Fact, Layer, Net, ConSet
from act.back_end.transfer_function import TransferFunction, AnalysisContext
from act.back_end.interval_tf.tf_mlp import *
from act.back_end.interval_tf.tf_cnn import *
from act.back_end.interval_tf.tf_rnn import *
from act.back_end.interval_tf.tf_transformer import *


class IntervalTF(TransferFunction):
    """Interval-based transfer functions for standard bounds propagation."""
    
    # Layer kind to function mapping
    _LAYER_REGISTRY = {
        # Identity/constraint layers
        "INPUT": lambda L, bounds, ctx: Fact(bounds=bounds, cons=ConSet()),
        "INPUT_SPEC": lambda L, bounds, ctx: Fact(bounds=bounds, cons=ConSet()),
        "ASSERT": lambda L, bounds, ctx: Fact(bounds=bounds, cons=ConSet()),
        
        # MLP operations
        "DENSE": lambda L, bounds, ctx: tf_dense(L, bounds),
        "BIAS": lambda L, bounds, ctx: tf_bias(L, bounds),
        "SCALE": lambda L, bounds, ctx: tf_scale(L, bounds),
        "RELU": lambda L, bounds, ctx: tf_relu(L, bounds),
        "LRELU": lambda L, bounds, ctx: tf_lrelu(L, bounds),
        "ABS": lambda L, bounds, ctx: tf_abs(L, bounds),
        "CLIP": lambda L, bounds, ctx: tf_clip(L, bounds),
        
        # Multi-input operations  
        "ADD": lambda L, bounds, ctx: tf_add(L, 
            ctx.get_predecessor_bounds(L.id, 0), 
            ctx.get_predecessor_bounds(L.id, 1)),
        "MUL": lambda L, bounds, ctx: tf_mul(L,
            ctx.get_predecessor_bounds(L.id, 0),
            ctx.get_predecessor_bounds(L.id, 1)),
        "CONCAT": lambda L, bounds, ctx: tf_concat(L, ctx.get_all_predecessor_bounds(L.id)),
        "BN": lambda L, bounds, ctx: tf_bn(L, bounds),
        
        # CNN operations
        "CONV2D": lambda L, bounds, ctx: tf_conv2d(L, bounds),
        "CONV1D": lambda L, bounds, ctx: tf_conv1d(L, bounds),
        "CONV3D": lambda L, bounds, ctx: tf_conv3d(L, bounds),
        "CONVTRANSPOSE2D": lambda L, bounds, ctx: tf_convtranspose2d(L, bounds),
        "MAXPOOL2D": lambda L, bounds, ctx: tf_maxpool2d(L, bounds),
        "AVGPOOL2D": lambda L, bounds, ctx: tf_avgpool2d(L, bounds),
        "FLATTEN": lambda L, bounds, ctx: tf_flatten(L, bounds),
        
        # RNN operations
        "LSTM": lambda L, bounds, ctx: tf_lstm(L, bounds),
        "GRU": lambda L, bounds, ctx: tf_gru(L, bounds),
        "RNN": lambda L, bounds, ctx: tf_rnn(L, bounds),
        "EMBEDDING": lambda L, bounds, ctx: tf_embedding(L, bounds),
        
        # Activation functions
        "SIGMOID": lambda L, bounds, ctx: tf_sigmoid(L, bounds),
        "TANH": lambda L, bounds, ctx: tf_tanh(L, bounds),
        "SOFTPLUS": lambda L, bounds, ctx: tf_softplus(L, bounds),
        "SILU": lambda L, bounds, ctx: tf_silu(L, bounds),
        "RELU6": lambda L, bounds, ctx: tf_relu6(L, bounds),
        "HARDTANH": lambda L, bounds, ctx: tf_hardtanh(L, bounds),
        "HARDSIGMOID": lambda L, bounds, ctx: tf_hardsigmoid(L, bounds),
        "HARDSWISH": lambda L, bounds, ctx: tf_hardswish(L, bounds),
        "MISH": lambda L, bounds, ctx: tf_mish(L, bounds),
        "SOFTSIGN": lambda L, bounds, ctx: tf_softsign(L, bounds),
        
        # Element-wise operations
        "MAX": lambda L, bounds, ctx: tf_max(L, ctx.get_all_predecessor_bounds(L.id)),
        "MIN": lambda L, bounds, ctx: tf_min(L, ctx.get_all_predecessor_bounds(L.id)),
        "SQUARE": lambda L, bounds, ctx: tf_square(L, bounds),
        "POWER": lambda L, bounds, ctx: tf_power(L, bounds),
        
        # Tensor operations
        "RESHAPE": lambda L, bounds, ctx: tf_reshape(L, bounds),
        "TRANSPOSE": lambda L, bounds, ctx: tf_transpose(L, bounds),
        "SQUEEZE": lambda L, bounds, ctx: tf_squeeze(L, bounds),
        "UNSQUEEZE": lambda L, bounds, ctx: tf_unsqueeze(L, bounds),
        "TILE": lambda L, bounds, ctx: tf_tile(L, bounds),
        "EXPAND": lambda L, bounds, ctx: tf_expand(L, bounds),
        
        # Transformer operations
        "EMBEDDING_TF": lambda L, bounds, ctx: tf_embedding(L),
        "POSENC": lambda L, bounds, ctx: tf_posenc(L, bounds),
        "LAYERNORM": lambda L, bounds, ctx: tf_layernorm(L, bounds),
        "GELU": lambda L, bounds, ctx: tf_gelu(L, bounds),
        "ATT_SCORES": lambda L, bounds, ctx: tf_att_scores(L,
            ctx.before[L.meta["q_src"]].bounds,
            ctx.before[L.meta["k_src"]].bounds),
        "SOFTMAX": lambda L, bounds, ctx: tf_softmax(L, bounds),
        "ATT_MIX": lambda L, bounds, ctx: tf_att_mix(L,
            ctx.before[L.meta["w_src"]].bounds, 
            ctx.before[L.meta["v_src"]].bounds),
        "MHA_SPLIT": lambda L, bounds, ctx: tf_mha_split(L, bounds),
        "MHA_JOIN": lambda L, bounds, ctx: tf_mha_join(L, ctx.get_all_predecessor_bounds(L.id)),
        "MASK_ADD": lambda L, bounds, ctx: tf_mask_add(L, bounds),
    }
    
    @property
    def name(self) -> str:
        return "IntervalTF"
        
    def supports_layer(self, layer_kind: str) -> bool:
        """Check if this transfer function supports the given layer kind."""
        return layer_kind.upper() in self._LAYER_REGISTRY
        
    def apply(self, L: Layer, input_bounds: Bounds, net: Net,
              before: Dict[int, Fact], after: Dict[int, Fact]) -> Fact:
        """Apply interval transfer function to layer L."""
        k = L.kind.upper()
        if k not in self._LAYER_REGISTRY:
            raise NotImplementedError(f"IntervalTF: Unsupported layer kind '{k}'")
            
        ctx = AnalysisContext(net, before, after)
        transfer_fn = self._LAYER_REGISTRY[k]
        return transfer_fn(L, input_bounds, ctx)