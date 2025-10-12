
# 🧩 Front-End Preprocessing (Self-Contained Specs)

This front-end prepares **samples, labels, and specs** for DNN verification using its **own spec types**.

## Files
- `specs.py` — `InputSpec`/`OutputSpec`, and enums `InKind`/`OutKind` (Torch-native)
- `utils_image.py` — PIL/numpy/torch → CHW torch, resize/crop, CHW↔HWC
- `preprocessor_base.py` — `Preprocessor` base + `ModelSignature`
- `preprocessor_image.py` — `ImgPre` for images (normalize + canonicalize)
- `preprocessor_text.py` — `TextPre` for token ids (simple tokenizer)
- `mocks.py` — mock samples/specs for testing
- `batch.py` — `run_batch` (optional `seed_from_input_spec_fn` callback)
- `demo_driver.py` — demo with stub verifier

## Batch usage
```python
from front_end.preprocessor_image import ImgPre
from front_end.mocks import mock_image_sample, mock_image_specs
from front_end.batch import run_batch, SampleRecord, BatchConfig
from front_end.specs import InputSpec, OutputSpec

from my_verifier.verify import verify_once, seed_from_input_spec  # your own

pre = ImgPre(H=32, W=32, C=3, device="cuda:0")

items = []
for i in range(8):
    img, y = mock_image_sample(seed=100+i)
    x_t = pre.prepare_sample(img)
    I_raw, O_raw = mock_image_specs(x_t, eps=0.03, y_true=y)
    items.append(SampleRecord(idx=i, sample_raw=img, label_raw=y, in_spec_raw=I_raw, out_spec_raw=O_raw))

net = ...; solver = ...
cfg = BatchConfig(time_budget_s=30.0, maximize_violation=True, entry_id=0, output_ids=list(range(num_outputs)))

results = run_batch(items, pre, net, solver, verify_once_fn=verify_once, output_dim=num_outputs, cfg=cfg, seed_from_input_spec_fn=seed_from_input_spec)
for r in results:
    print(r.idx, r.status, r.error)
```

## Notes
- No dependency on external spec definitions — everything uses `front_end.specs`.
- To compute a seed box (`Bounds`) for your verifier, pass your function as `seed_from_input_spec_fn`.
- Counterexamples can be mapped back via `pre.unflatten_to_model_input()` and `pre.inverse_to_raw_space()`.
