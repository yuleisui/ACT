
from __future__ import annotations
import numpy as np, torch
from act.front_end.specs import InputSpec, OutputSpec, InKind, OutKind

def mock_image_sample(C=3, H=32, W=32, seed=0):
    rng = np.random.default_rng(seed)
    img = (rng.random((H,W,C))*255).astype(np.uint8)
    label = int(rng.integers(0,10))
    return img, label

def mock_image_specs(x0_t: torch.Tensor, eps=0.03, y_true=0):
    I = InputSpec(kind=InKind.LINF_BALL, center=x0_t, eps=eps)
    O = OutputSpec(kind=OutKind.MARGIN_ROBUST, y_true=y_true, margin=0.0)
    return I, O

def mock_text_sample(seq_len=16, vocab_size=100, seed=0):
    rng = np.random.default_rng(seed)
    ids = rng.integers(2, vocab_size, size=(seq_len,), endpoint=False).astype(np.int64)
    label = int(rng.integers(0, 5))
    return ids.tolist(), label

def mock_text_specs(center_tokens, y_true=0):
    lb = torch.tensor(center_tokens)
    ub = torch.tensor(center_tokens)
    I = InputSpec(kind=InKind.BOX, lb=lb, ub=ub)
    O = OutputSpec(kind=OutKind.MARGIN_ROBUST, y_true=y_true, margin=0.0)
    return I, O
