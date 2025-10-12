
from __future__ import annotations
import numpy as np, torch
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

# Import front-end
from act.front_end.preprocessor_image import ImgPre
from act.front_end.mocks import mock_image_sample, mock_image_specs
from act.front_end.specs import InputSpec, OutputSpec, InKind, OutKind
from act.front_end.batch import run_batch, SampleRecord, BatchConfig
from act.back_end.verify_status import VerifResult

def verify_once_stub(net, entry_id, input_ids, output_ids, input_spec, output_spec, seed_bounds, solver, timelimit=None, maximize_violation=False) -> VerifResult:
    # A silly stub: always return UNKNOWN with no CE
    return VerifResult(status="UNKNOWN", model_stats={"note":"stub"})

def demo_batch_image():
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
