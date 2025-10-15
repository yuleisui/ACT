# 🧩 Spec-Free, Input-Free Torch→ACT + Verification — Two-File Design

This document specifies a compact, production-ready pattern to convert **wrapped PyTorch models** to ACT
and run verification with **no external inputs or specs** passed at runtime.

**Exactly two files to implement**:
- `../pipeline/torch2act.py` — Torch→ACT converter (reads embedded specs from wrapper; no input_shape needed)
- `verifier.py` — Single-shot (**verify_once**) and Branch-and-Bound (**verify_bab**) verification using only the ACT `Net`

> All input/output specifications are embedded in the wrapper via `InputSpecLayer` and `OutputSpecLayer`.
> The converter and verifier read those layers directly; **no external spec or input tensors** are required.

---

## ✅ Wrapper Contract

Your wrapped model is an `nn.Sequential` that includes:

```
InputLayer → InputAdapterLayer → InputSpecLayer → [optional Flatten] → Model → OutputSpecLayer
```

- `InputLayer(shape=(1,...), center=?)` — declares the input variable block (symbolic).
- `InputAdapterLayer(...)` — deterministic preprocessing: permute/reorder/slice/pad/affine/linear-proj.
- `InputSpecLayer(spec=InputSpec(...))` — **already in post-adapter space** (BOX, L∞ as BOX, or LIN_POLY).
- `[optional nn.Flatten]` — reshaping only.
- `Model` — learned layers (e.g., `nn.Linear`, `nn.ReLU`).
- `OutputSpecLayer(spec=OutputSpec(...))` — final property (`ASSERT`) over outputs.

---

## 📦 File 1: `../pipeline/torch2act.py` (Converter)

### Responsibilities
- Convert the wrapper into an ACT `Net` of `Layer` objects.
- Put **numeric tensors** (weights, bounds) in `Layer.params` and **JSON-able** flags/shapes in `Layer.meta`.
- Enforce a verification-ready wrapper via **hard assertions**:
  - exactly one `InputLayer` (no `input_shape` arg needed),
  - at least one `InputSpecLayer`,
  - last module is `OutputSpecLayer` (ACT last layer is `ASSERT`).

### Module → ACT Layer Mapping
| Torch module | ACT kind | Notes |
|--------------|----------|-------|
| `InputLayer` | `INPUT` | allocates initial variable block; `params['shape']`[, `center`] |
| `InputAdapterLayer` | `PERMUTE`, `REORDER`, `SLICE`, `PAD`, `SCALE_SHIFT`, `LINEAR_PROJ` | emitted in order |
| `InputSpecLayer` | `INPUT_SPEC` | **constraint-only**, `out_vars == in_vars` |
| `nn.Flatten` | `FLATTEN` | reshape only |
| `nn.Linear` | `DENSE` | `params['W']`, `params['b']` |
| `nn.ReLU` | `RELU` | same-width block |
| `OutputSpecLayer` | `ASSERT` | **constraint-only**, `out_vars == in_vars` |

### Public API
```python
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import torch, torch.nn as nn

@dataclass
class Layer:
    id: int
    kind: str
    params: Dict[str, torch.Tensor]
    meta: Dict[str, Any]
    in_vars: List[int]
    out_vars: List[int]
    cache: Dict[str, torch.Tensor] = field(default_factory=dict)
    def is_validation(self) -> bool: return self.kind == "ASSERT"

@dataclass
class Net:
    layers: List[Layer]
    preds: Dict[int, List[int]]
    succs: Dict[int, List[int]]
    by_id: Dict[int, Layer] = field(init=False)
    def __post_init__(self): self.by_id = {L.id: L for L in self.layers}
    def last_validation(self): ...
    def assert_last_is_validation(self): ...

class TorchToACT:
    def __init__(self, wrapped: nn.Sequential): ...
    def run(self) -> Net: ...
```

---

## 📁 File 2: `verifier.py` (Verification)

This module is **spec-free, input-free**. All constraints are extracted from the ACT `Net`.
The public entry points **do not accept** input shapes, var ids, or external spec objects.

### Public API
```python
from dataclasses import dataclass, field
from typing import Optional, List, Callable, Dict, Any
import numpy as np
import torch
from act.back_end.solver.solver_base import Solver

class VerifStatus: CERTIFIED="CERTIFIED"; COUNTEREXAMPLE="COUNTEREXAMPLE"; UNKNOWN="UNKNOWN"

@dataclass
class VerifResult:
    status: str
    ce_x: Optional[np.ndarray]=None
    ce_y: Optional[np.ndarray]=None
    model_stats: Dict[str, Any]=field(default_factory=dict)

@torch.no_grad()
def verify_once(net, solver: Solver, timelimit: Optional[float]=None, maximize_violation: bool=False) -> VerifResult: ...

def verify_bab(net, solver: Solver, model_fn: Callable[[torch.Tensor], torch.Tensor],
               max_depth: int=20, max_nodes: int=2000, time_budget_s: float=300.0) -> VerifResult: ...
```

### How it works
- Extract from `net`:
  - `entry_id` from `INPUT` layer,
  - `input_ids` from `INPUT.out_vars`,
  - `output_ids` from `ASSERT.in_vars`,
  - list of `INPUT_SPEC` layers,
  - final `ASSERT` layer.
- Build **seed box** from `INPUT_SPEC` layers (`BOX` or `L∞`; `LIN_POLY` alone requires a seed policy → error).
- Call `analyze(net, entry_id, seed)` → `(before, after, globalC)`.
- Add all input specs to `globalC` (`BOX`/`L∞` as boxes; `LIN_POLY` tagged as inequalities).
- `export_to_solver(globalC, solver, ...)` and `materialise_input_poly(...)` to push linear rows.
- Add **negated** ASSERT to the solver (LINEAR_LE, TOP1_ROBUST, MARGIN_ROBUST, RANGE policy).
- Set a linear objective (max violation for robust kinds if requested).
- Solve and interpret:
  - `INFEASIBLE` → `CERTIFIED`
  - `FEASIBLE/OPTIMAL` + solution → `COUNTEREXAMPLE`
  - else → `UNKNOWN`

### Branch-and-Bound
- Root node uses the seed box.
- Each node calls the same solve path with `node.box`.
- If SAT → obtain `x_ce` and numerically check against `ASSERT` (TRUE_CE/FALSE_CE).
- If INFEASIBLE → node is certified and pruned.
- Otherwise branch on widest box dimension and continue.

---

## ⚠️ Edge Cases & Policies

- **Multiple INPUT_SPEC layers** supported — all added to constraints. First BOX/L∞ is used as the seed.
- **LIN_POLY-only** inputs: require a seed box or raise `ValueError` (unchanged policy).
- **RANGE** negation is disjunctive; the default encodes a one-sided violation (≥ ub + ε). If needed, add a second pass or branching for the ≤ lb − ε side.
- **Unsupported Torch modules** should raise `NotImplementedError` in `../pipeline/torch2act.py`.

---

## 🧪 Minimal Example

```python
# Convert (spec-free)
from act.pipeline.torch2act import TorchToACT
net = TorchToACT(wrapped).run()

# Solve once
from verifier import verify_once, verify_bab
res = verify_once(net, solver=my_solver, timelimit=30.0)
print(res.status, res.model_stats)

# Branch & bound
res_bab = verify_bab(net, solver=my_solver, model_fn=lambda x: wrapped(x), max_depth=18, time_budget_s=120.0)
print(res_bab.status, res_bab.model_stats)
```

---

## ✅ Checklist

- [ ] Implement `../pipeline/torch2act.py` with strong assertions and mapping table.
- [ ] Use `verifier.py` from this repo (spec-free, input-free).
- [ ] Ensure `INPUT_SPEC` and `ASSERT` are constraint-only (`out_vars == in_vars`).
- [ ] Keep numerics in `params` (Torch tensors) and flags/shapes in `meta`.
- [ ] Decide RANGE negation policy (one-sided or both via branching).

---

## 📜 License

Add your project license here.
