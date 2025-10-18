#===- act/back_end/core.py - ACT Core Data Structures ------------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Core data structures for ACT verification framework including Layer,
#   Net, Bounds, and constraint set definitions.
#
#===---------------------------------------------------------------------===#

# core.py
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any

# Import validation functions
from act.back_end.layer_util import validate_layer, validate_graph, validate_wrapper_graph

# Supported layer types: Please see them in act/back_end/layer_schema.py
@dataclass
class Layer:
    id: int                                     # Unique layer identifier
    kind: str                                   # UPPER name (e.g., "DENSE", "CONV2D", "RELU")
    params: Dict[str, torch.Tensor]            # Numeric tensors (weights, biases) on device
    meta: Dict[str, Any]                       # Non-numeric metadata (shapes, strides, etc.)
    in_vars: List[int]                         # Input variable indices 
    out_vars: List[int]                        # Output variable indices
    cache: Dict[str, torch.Tensor] = field(default_factory=dict)  # Runtime cache tensors

    def __post_init__(self):
        validate_layer(self)

    def is_validation(self) -> bool:
        return self.kind == "ASSERT"
    

@dataclass
class Net:
    layers: List[Layer]
    preds: Dict[int, List[int]]
    succs: Dict[int, List[int]]
    by_id: Dict[int, Layer] = field(init=False)
    
    def __post_init__(self):
        self.by_id = {L.id: L for L in self.layers}
        # Validate the graph structure
        validate_graph(self.layers)
        validate_wrapper_graph(self.layers)

    # helpers
    def last_validation(self) -> Optional[Layer]:
        for L in reversed(self.layers):
            if L.is_validation(): return L
        return None

    def assert_last_is_validation(self) -> None:
        if not self.layers or not self.layers[-1].is_validation():
            raise ValueError(f"Expected last layer to be ASSERT, got {self.layers[-1].kind if self.layers else 'EMPTY'}")
        
        
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