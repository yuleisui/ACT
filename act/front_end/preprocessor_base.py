
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, List, Tuple
import torch
import numpy as np
from pathlib import Path
from act.front_end.device_manager import get_default_device, get_default_dtype
from act.front_end.specs import InputSpec, OutputSpec, InKind, OutKind

@dataclass
class ModelSignature:
    modality: str            # 'image', 'text', 'tabular', ...
    layout: str              # 'NCHW', '[seq]', ...
    input_shape: tuple       # per-sample shape, e.g., (C,H,W)
    meta: Dict[str, Any]     # e.g., mean/std, tokenizer, vocab

class Preprocessor:
    """Base preprocessor: raw <-> model tensors <-> verifier flat vector."""
    def __init__(self, signature: ModelSignature):
        self.signature = signature
        self.device = get_default_device()
        self.dtype = get_default_dtype()

    # ---------- RAW -> MODEL ----------
    def prepare_sample(self, sample) -> torch.Tensor:
        raise NotImplementedError

    def prepare_label(self, label):
        return label

    # ---------- LOAD RAW DATA ----------
    def load_raw_sample_label_pairs(self, file_path: str) -> Tuple[torch.Tensor, Any]:
        """Load raw sample and label from file path.
        
        Args:
            file_path: Path to the raw data file
            
        Returns:
            Tuple of (prepared_sample_tensor, prepared_label)
        """
        raise NotImplementedError

    # ---------- SPECS (RAW/MODEL) -> VERIFIER ----------
    def canonicalize_input_spec(self, input_spec_raw: InputSpec, *, center=None, eps: Optional[float]=None) -> InputSpec:
        raise NotImplementedError

    def canonicalize_output_spec(self, output_spec_raw: OutputSpec, *, label=None) -> OutputSpec:
        return output_spec_raw

    # ---------- FLATTEN / UNFLATTEN ----------
    def flatten_model_input(self, x: torch.Tensor) -> np.ndarray:
        return x.contiguous().view(-1).detach().cpu().numpy()

    def unflatten_to_model_input(self, flat: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(flat).view(*self.signature.input_shape).to(self.device, self.dtype)

    def inverse_to_raw_space(self, x_model: torch.Tensor):
        return x_model.detach().cpu()
