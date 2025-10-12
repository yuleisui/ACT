
from __future__ import annotations
from typing import Optional, List, Dict
import torch, numpy as np

from act.front_end.preprocessor_base import Preprocessor, ModelSignature
from act.front_end.specs import InputSpec, OutputSpec, InKind, OutKind

class SimpleTokenizer:
    def __init__(self, vocab: Optional[Dict[str,int]]=None, unk_token="<unk>", pad_token="<pad>"):
        self.unk_token = unk_token; self.pad_token = pad_token
        if vocab is None:
            self.vocab = {pad_token:0, unk_token:1}
        else:
            self.vocab = dict(vocab)
            if pad_token not in self.vocab: self.vocab[pad_token] = 0
            if unk_token not in self.vocab: self.vocab[unk_token] = 1

    def build_vocab(self, texts: List[str], min_freq: int = 1):
        from collections import Counter
        cnt = Counter()
        for t in texts:
            for tok in t.lower().split():
                cnt[tok] += 1
        idx = max(self.vocab.values()) + 1
        for w, c in cnt.items():
            if c >= min_freq and w not in self.vocab:
                self.vocab[w] = idx; idx += 1

    def encode(self, text: str, seq_len: int):
        toks = [t for t in text.lower().split() if t]
        ids = [self.vocab.get(t, self.vocab[self.unk_token]) for t in toks]
        ids = ids[:seq_len]
        att = [1]*len(ids)
        while len(ids) < seq_len: ids.append(self.vocab[self.pad_token]); att.append(0)
        return {"input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.tensor(att, dtype=torch.long)}

class TextPre(Preprocessor):
    def __init__(self, seq_len: int, vocab: Optional[Dict[str,int]] = None,
                 device="cpu", dtype=torch.float32):
        sig = ModelSignature(modality="text", layout="[seq]", input_shape=(seq_len,), meta={})
        super().__init__(sig, device=device, dtype=dtype)
        self.seq_len = seq_len
        self.tok = SimpleTokenizer(vocab=vocab)

    def prepare_sample(self, sample) -> torch.Tensor:
        if isinstance(sample, str):
            enc = self.tok.encode(sample, self.seq_len)
            return enc["input_ids"].to(self.device)
        if isinstance(sample, (list, tuple)):
            return torch.tensor(sample[:self.seq_len], device=self.device)
        raise TypeError("TextPre expects str or list[int]")

    def prepare_label(self, label):
        if isinstance(label, int): return int(label)
        if isinstance(label, str) and label.isdigit(): return int(label)
        return 0

    def canonicalize_input_spec(self, input_spec_raw: InputSpec, *, center=None, eps: Optional[float]=None) -> InputSpec:
        if input_spec_raw.kind == InKind.BOX:
            lb = (input_spec_raw.lb if isinstance(input_spec_raw.lb, torch.Tensor)
                  else torch.tensor(input_spec_raw.lb, device=self.device))
            ub = (input_spec_raw.ub if isinstance(input_spec_raw.ub, torch.Tensor)
                  else torch.tensor(input_spec_raw.ub, device=self.device))
            return InputSpec(kind=InKind.BOX, lb=lb.to(self.device), ub=ub.to(self.device))
        if input_spec_raw.kind == InKind.LINF_BALL:
            raise NotImplementedError("LINF_BALL not meaningful for token ids; supply BOX or custom set.")
        if input_spec_raw.kind == InKind.LIN_POLY:
            A = input_spec_raw.A.to(self.device) if input_spec_raw.A is not None else None
            b = input_spec_raw.b.to(self.device) if input_spec_raw.b is not None else None
            return InputSpec(kind=InKind.LIN_POLY, A=A, b=b)
        raise NotImplementedError

    def canonicalize_output_spec(self, output_spec_raw: OutputSpec, *, label=None) -> OutputSpec:
        if output_spec_raw.kind in (OutKind.TOP1_ROBUST, OutKind.MARGIN_ROBUST):
            y_true = self.prepare_label(label if label is not None else output_spec_raw.y_true)
            return OutputSpec(kind=OutKind.MARGIN_ROBUST, y_true=int(y_true),
                              margin=float(output_spec_raw.margin if output_spec_raw.margin is not None else 0.0))
        if output_spec_raw.kind == OutKind.LINEAR_LE:
            return output_spec_raw
        return output_spec_raw

    def flatten_model_input(self, x: torch.Tensor):
        return x.detach().cpu().numpy().astype(np.int64)

    def unflatten_to_model_input(self, flat):
        return torch.from_numpy(flat).to(self.device)
