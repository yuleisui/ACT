# core.pseudo
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from act.abstraction.device_manager import get_device, get_dtype, as_t

DEFAULT_DEVICE = get_device()
DEFAULT_DTYPE  = get_dtype()

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

@dataclass
class Layer:
    id: int
    kind: str                      # UPPER name
    params: Dict[str, Any]
    in_vars: List[int]
    out_vars: List[int]
    cache: Dict[str, Any] = field(default_factory=dict)  # prev_lb/prev_ub/masks

@dataclass
class Net:
    layers: List[Layer]
    preds: Dict[int, List[int]]
    succs: Dict[int, List[int]]
    by_id: Dict[int, Layer] = field(init=False)
    def __post_init__(self):
        self.by_id = {L.id: L for L in self.layers}
