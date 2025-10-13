
# dnnverif_torch — Torch‑Native DNN Verification (Bounds + MILP/LP + BaB)

A concise, modular verification toolkit that does **all analysis in PyTorch** (for ergonomics and optional GPU),
and converts to **NumPy only at the solver boundary**. It supports MLPs and key Transformer components, with
worklist-based bounds propagation, constraint extraction, MILP/LP export, and Branch‑and‑Bound (BaB) with
counterexample validation.

---

## Why this design?

- **Torch inside**: easy interop with models, layers, and CE validation; one tensor world.
- **Single export boundary**: solver backends remain NumPy‑facing; plug in Gurobi/HiGHS/CBC, etc.
- **DAG‑aware worklist**: efficient, incremental propagation with caching and change detection.
- **Layer coverage**: MLP + less‑common activations + Transformer blocks with sound relaxations.
- **Refinement loop**: BaB + CE validation to distinguish true vs. false counterexamples.

---

## Core Data Structures

```python
@dataclass(eq=True, frozen=True)
class Bounds:      # Box bounds for a contiguous vector of variables
    lb: torch.Tensor
    ub: torch.Tensor

@dataclass(eq=False)
class Con:         # Canonical constraint template (solver-agnostic)
    kind: str                  # 'EQ' | 'INEQ' | 'BIN'
    var_ids: Tuple[int, ...]   # variable ids referenced by this constraint
    meta: Dict[str, Any]       # parameters (Torch tensors allowed) + {'tag': str}
    # Optional numeric payloads (unused in-core; only for compatibility)
    A: Any=None; b: Any=None; C: Any=None; d: Any=None
    def signature(self) -> Tuple[str, Tuple[int, ...], str]: ...

@dataclass
class ConSet:       # Replace-by-signature semantics
    S: Dict[Tuple[str, Tuple[int, ...], str], Con]
    def replace(self, c: Con): ...
    def add_box(self, layer_id: int, var_ids: List[int], B: Bounds): ...

@dataclass
class Fact:         # Dataflow fact per layer: (bounds, constraints)
    bounds: Bounds
    cons: ConSet

@dataclass
class Layer:
    id: int
    kind: str       # e.g. 'DENSE', 'RELU', 'SOFTMAX', ...
    params: Dict[str, Any]
    in_vars: List[int]
    out_vars: List[int]
    cache: Dict[str, Any]      # prev_lb/prev_ub/masks (for change detection)

@dataclass
class Net:          # Topo-ordered DAG
    layers: List[Layer]
    preds: Dict[int, List[int]]
    succs: Dict[int, List[int]]
    by_id: Dict[int, Layer] = field(init=False)
```

- **`meta['tag']`** drives export logic (e.g. `relu:{id}`, `dense:{id}`, `softmax:simplex:{id}`, …).
- Constraints are **templates** kept compact during analysis, then materialized at export time.

---

## Device & Dtype Policy

- Analysis uses **Torch** everywhere with `DEFAULT_DEVICE` (CPU/GPU) and `DEFAULT_DTYPE` (e.g. `float32`).
- Helpers:
  - `as_t(x, device, dtype)` converts inputs and parameters consistently.
  - `@torch.no_grad()` guards analysis & CE validation paths.
- Export converts tensors to **NumPy float64** in one place: `exporter.to_numpy(x)`.

---

## Supported Layers

### MLP Basics
`DENSE, BIAS, SCALE, RELU, LRELU, ABS, CLIP, MUL, ADD, CONCAT, BN`

### Less‑Common MLP‑ish
`SIGMOID, TANH, SOFTPLUS, SILU, MAX, MIN, SQUARE, POWER`

### Transformer Components
`EMBEDDING, POSENC, LAYERNORM, GELU, ATT_SCORES, SOFTMAX, ATT_MIX, MHA_SPLIT, MHA_JOIN, MASK_ADD`

Each layer has a **transfer function** that:
1) computes new **interval bounds** (sound over-approximation);
2) adds a minimal **template constraint** (with `meta['tag']`) for later solver export.

---

## Bounds Propagation (Worklist, DAG‑aware)

- Start from the **entry layer**: seed bounds with the input spec (box / ℓ∞ ball).
- Use a **worklist** (queue) of layer ids.
- For a layer `L`:
  1. **Join** predecessors’ `after` bounds into `before[L]`.
  2. Run `dispatch_tf(L, before, after, net)` ⇒ `out_fact`.
  3. If changed (bounds or masks), update caches, merge constraints, push successors.
- Termination when the worklist empties (monotone joins + finite precision).

**Performance**: vectorized Torch ops, caching, and dedup constraints (replace-by-signature).

---

## Constraint Export (Solver‑Agnostic)

- `export_to_solver(globalC, solver)` performs the **only** Torch→NumPy conversion.
- Merges all `box:*` constraints into global variable bounds.
- Materializes per‑layer templates:
  - `dense:` → linear equalities (y = Wx + b)
  - `relu:` → phase splits (on/off) and convex relaxation for ambiguous
  - `mcc:`  → McCormick envelopes for bilinear terms (e.g., `MUL`, attention mixes)
  - `softmax:simplex:` → probability simplex per row (≥0, sum=1)
  - etc.
- Backends implement `Solver` (see next section).

---

## Solver Interface & Gurobi Backend

```python
class Solver:
    def begin(...); def add_vars(n); def set_bounds(idxs, lb, ub)
    def add_lin_eq(...); def add_lin_le(...); def add_lin_ge(...)
    def add_sum_eq(...); def add_ge_zero(...); def add_sos2(...)
    def set_objective_linear(..., sense="min"); def optimize(tlim=None)
    def status() -> str; def has_solution() -> bool; def get_values(ids) -> np.ndarray
    @property def n(self) -> int
```

- **Backends** only see NumPy arrays.
- Provided example: **`GurobiSolver`** (others can implement the same API).

---

## Input / Output Specs

**InputSpec**: `BOX`, `LINF_BALL`, or extra **linear polytope** constraints (`A x ≤ b`).  
**OutputSpec** (negated for searching counterexamples):
- `LINEAR_LE`: find `c^T y ≥ d + ε`
- `TOP1_ROBUST`: find a class with `y_j ≥ y_true`
- `MARGIN_ROBUST`: find `y_j − y_true ≥ δ`

`verify_once(...)`:
1. Run `analyse()` to collect bounds + constraint templates.
2. Export to solver, add input spec and **negated** output spec.
3. Optimize (optionally maximizing violation) →
   - **INFEASIBLE** ⇒ `CERTIFIED`
   - **FEASIBLE** ⇒ return **counterexample** (input/output witness).

---

## Branch‑and‑Bound (BaB) + CE Validation

- Priority queue by box “width” score (sum of side lengths).
- For each node:
  1. `verify_once` on the sub‑box.
  2. If **INFEASIBLE** ⇒ pruned (certified).
  3. If **CE found** ⇒ **validate** with Torch model forward:
     - If **violates** spec → **True CE** (stop).
     - If **does not** violate → **False CE** (likely relaxation gap).
  4. **Branch** on the widest dimension, enqueue children.
- Terminates with **CERTIFIED** if no true CE is found within limits.

This forms a **refinement loop**: false CEs drive further splitting → tighter local bounds.

---

## Extending the System

- **New layers**: add transfer in `transfer.py` + export case in `exporter.py`.
- **New solvers**: subclass `Solver` and implement the abstract methods.
- **Advanced relaxations**: you can store extra parameters in `Con.meta` (Torch tensors),
  then convert in the exporter when materializing constraints.

---

## Performance Tips

- Set a **global device** (CPU/GPU) and dtype up front.
- Use `@torch.no_grad()` in analysis & CE validation.
- Batch BaB nodes for large networks (optional extension).
- Keep constraints minimal; rely on `box:*` and a few tight relaxations.
- Watch BLAS threading: Torch vs. solver (set `OMP_NUM_THREADS`, `MKL_NUM_THREADS` if needed).

---

## Minimal Usage Sketch

```python
import torch, numpy as np
from dnnverif_torch.core import Layer, Net, Bounds, as_t
from act.back_end import InputSpec, OutputSpec, InKind, OutKind, seed_from_input_spec
from dnnverif_torch.bab import verify_bab
from dnnverif_torch.solver_gurobi import GurobiSolver

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.set_default_dtype(torch.float32)

# Build a tiny MLP: x -> Dense -> ReLU -> y
n_in, n_out = 3, 4
W = torch.randn(n_out, n_in, device=device); b = torch.randn(n_out, device=device)
W_pos, W_neg = torch.clamp(W, min=0), torch.clamp(W, max=0)
x_ids = list(range(n_in)); y_ids = list(range(n_in, n_in+n_out))

L0 = Layer(id=0, kind="DENSE", params={"W":W, "W_pos":W_pos, "W_neg":W_neg, "b":b}, in_vars=x_ids, out_vars=y_ids)
L1 = Layer(id=1, kind="RELU",  params={}, in_vars=y_ids, out_vars=y_ids)
net = Net(layers=[L0, L1], preds={0:[], 1:[0]}, succs={0:[1], 1:[]})

I = InputSpec(kind=InKind.BOX, lb=as_t(torch.full((n_in,), -1.0, device=device)),
                          ub=as_t(torch.full((n_in,), +1.0, device=device)))
root_box = seed_from_input_spec(I)
O = OutputSpec(kind=OutKind.MARGIN_ROBUST, y_true=2, margin=0.0)

@torch.no_grad()
def forward_fn(x: torch.Tensor) -> torch.Tensor:
    return torch.maximum(W @ x + b, torch.zeros_like(b))

solver = GurobiSolver()
res = verify_bab(net, entry_id=0, input_ids=x_ids, output_ids=y_ids,
                 input_spec=I, output_spec=O, root_box=root_box,
                 solver=solver, model_fn=forward_fn,
                 max_depth=10, max_nodes=200, time_budget_s=10.0)

print(res.status, res.model_stats)
if res.ce_x is not None:
    print("CE x*:", res.ce_x); print("CE y*:", res.ce_y)
```

---

## File Layout (suggested)

```
dnnverif_torch/
  core.py
  utils.py
  tr_mlp.py
  tf_transformer.py
  analyze.py
  solver_base.py
  cons_exporter.py
  verif_status.py
  bab.py
  solver_gurobi.py
  solver_torch.py
  driver.py
  # device_manager.py (moved to act/front_end/)
```

---

## License & Notes

- This blueprint focuses on clarity, extensibility, and solver portability.
- Replace/extend relaxations as needed for tighter bounds (CROWN, triangle relaxations, etc.).
- Backends other than Gurobi are straightforward once the `Solver` interface is implemented.
