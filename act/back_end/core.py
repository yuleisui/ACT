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

# DENSE layer params
# {"W": weight_tensor, "W_pos": pos_weights, "W_neg": neg_weights, "b": bias}

# CONV2D layer params
# {"weight": weight, "bias": bias, "stride": stride, "padding": padding,
#  "input_shape": shape, "output_shape": shape}

@dataclass
class Layer:
    id: int                        # Unique layer identifier
    kind: str                      # UPPER name (e.g., "DENSE", "CONV2D", "RELU")
    params: Dict[str, Any]         # Layer-specific parameters (weights, biases, etc.)
    in_vars: List[int]             # Input variable indices 
    out_vars: List[int]            # Output variable indices
    cache: Dict[str, Any] = field(default_factory=dict)         # Runtime cache (prev_lb/prev_ub/masks)

@dataclass
class Net:
    layers: List[Layer]
    preds: Dict[int, List[int]]
    succs: Dict[int, List[int]]
    by_id: Dict[int, Layer] = field(init=False)
    def __post_init__(self):
        self.by_id = {L.id: L for L in self.layers}
