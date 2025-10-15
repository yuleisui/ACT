
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional
import numpy as np
import torch

from act.front_end.preprocessor_base import Preprocessor
from act.front_end.specs import InputSpec, OutputSpec, InKind, OutKind
from act.front_end.preprocessor_image import ImgPre
from act.front_end.mocks import mock_image_sample, mock_image_specs
from act.back_end.verify_status import VerifResult

@dataclass
class SampleRecord:
    idx: int
    sample_raw: Any
    label_raw: Any
    in_spec_raw: InputSpec
    out_spec_raw: OutputSpec

@dataclass
class ItemResult:
    idx: int
    status: str
    ce_x: Optional[np.ndarray] = None
    ce_y: Optional[np.ndarray] = None
    model_stats: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

@dataclass
class BatchConfig:
    time_budget_s: Optional[float] = None
    maximize_violation: bool = False
    input_ids: Optional[List[int]] = None
    output_ids: Optional[List[int]] = None
    entry_id: int = 0

def _default_input_ids(pre: Preprocessor, x_model: torch.Tensor) -> List[int]:
    return list(range(int(x_model.numel())))

def run_batch(
    items: Iterable[SampleRecord],
    pre: Preprocessor,
    net: Any,
    solver: Any,
    verify_once_fn: Callable[..., Any],
    output_dim: int,
    cfg: Optional[BatchConfig] = None,
    seed_from_input_spec_fn: Optional[Callable[[InputSpec], Any]] = None,
) -> List[ItemResult]:
    """Batch pipeline (front-end only specs).
    - seed_from_input_spec_fn: optional callback provided by your verifier to build a seed Bounds.
    """
    results: List[ItemResult] = []
    cfg = cfg or BatchConfig()

    for it in items:
        try:
            x_model = pre.prepare_sample(it.sample_raw)
            y_label = pre.prepare_label(it.label_raw)

            in_spec = pre.canonicalize_input_spec(it.in_spec_raw, center=it.sample_raw)
            out_spec = pre.canonicalize_output_spec(it.out_spec_raw, label=y_label)

            x_flat = pre.flatten_model_input(x_model)
            input_ids = cfg.input_ids or _default_input_ids(pre, x_model)
            output_ids = cfg.output_ids or list(range(output_dim))

            seed_bounds = seed_from_input_spec_fn(in_spec) if seed_from_input_spec_fn is not None else None

            res = verify_once_fn(
                net, cfg.entry_id, input_ids, output_ids,
                in_spec, out_spec, seed_bounds, solver,
                timelimit=cfg.time_budget_s, maximize_violation=cfg.maximize_violation
            )
            results.append(ItemResult(idx=it.idx, status=res.status, ce_x=res.ce_x, ce_y=res.ce_y, model_stats=getattr(res, "model_stats", {})))
        except Exception as ex:
            results.append(ItemResult(idx=it.idx, status="ERROR", error=str(ex)))
    return results


def verify_once_stub(net, entry_id, input_ids, output_ids, input_spec, output_spec, seed_bounds, solver, timelimit=None, maximize_violation=False) -> VerifResult:
    """A silly stub: always return UNKNOWN with no CE"""
    return VerifResult(status="UNKNOWN", model_stats={"note":"stub"})


def demo_batch_image():
    """Demo function for batch image processing"""
    pre = ImgPre(H=32, W=32, C=3)
    # create a few records
    items = []
    for i in range(4):
        img, y = mock_image_sample(seed=100+i)
        x_t = pre.prepare_sample(img)
        I_raw, O_raw = mock_image_specs(x_t, eps=0.03, y_true=y)
        items.append(SampleRecord(idx=i, sample_raw=img, label_raw=y, in_spec_raw=I_raw, out_spec_raw=O_raw))

    # dummy net/solver
    net = object()
    solver = object()

    cfg = BatchConfig(time_budget_s=2.0, maximize_violation=True, entry_id=0, output_ids=list(range(10)))
    res = run_batch(items, pre, net, solver, verify_once_stub, output_dim=10, cfg=cfg)
    for r in res:
        print(f"idx={r.idx} status={r.status} err={r.error}")


if __name__ == "__main__":
    demo_batch_image()
