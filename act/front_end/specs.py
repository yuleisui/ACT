from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import torch

class InKind:
    BOX = "BOX"
    LINF_BALL = "LINF_BALL"
    LIN_POLY = "LIN_POLY"

@dataclass
class InputSpec:
    kind: str
    lb: Optional[torch.Tensor] = None
    ub: Optional[torch.Tensor] = None
    center: Optional[torch.Tensor] = None
    eps: Optional[float] = None
    A: Optional[torch.Tensor] = None
    b: Optional[torch.Tensor] = None

class OutKind:
    LINEAR_LE   = "LINEAR_LE"
    TOP1_ROBUST = "TOP1_ROBUST"
    MARGIN_ROBUST = "MARGIN_ROBUST"
    RANGE = "RANGE"

@dataclass
class OutputSpec:
    kind: str
    c: Optional[torch.Tensor] = None
    d: Optional[float] = None
    y_true: Optional[int] = None
    margin: float = 0.0
    lb: Optional[torch.Tensor] = None
    ub: Optional[torch.Tensor] = None
    meta: Dict[str, Any] = field(default_factory=dict)
