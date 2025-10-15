# layer_validation.py
"""
Validation functions for ACT layers and networks.
Separated from layer_schema.py to avoid circular import issues.
"""

from __future__ import annotations
from typing import Dict, Any, List
import difflib

# Import validation components
try:
    # Try relative import first (when used as module)
    from .layer_schema import REGISTRY, LayerKind
except ImportError:
    # Fallback to absolute import (when run standalone)
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from act.back_end.layer_schema import REGISTRY, LayerKind

# Import Layer from core to avoid circular import issues
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    try:
        from .core import Layer
    except ImportError:
        # Will import at runtime when needed to avoid circular import
        pass

# Wrapper-only constraints (moved from layer_schema.py to avoid circular imports)
ADAPTER_KINDS = {
    LayerKind.PERMUTE.value, LayerKind.REORDER.value, LayerKind.SLICE.value, LayerKind.PAD.value,
    LayerKind.SCALE_SHIFT.value, LayerKind.LINEAR_PROJ.value,
}


try:
    import torch
    Tensor = torch.Tensor
except Exception:  # typing only
    Tensor = "torch.Tensor"  # type: ignore

# ------------------------------
# Strict validation & helpers
# ------------------------------
def _missing(required: List[str], got: Dict[str, Any]) -> List[str]:
    return [k for k in required if k not in got]

def _unknown(allowed: List[str], got: Dict[str, Any]) -> List[str]:
    return [k for k in got.keys() if k not in allowed]

def _suggest(key: str, candidates: List[str]) -> List[str]:
    return difflib.get_close_matches(key, candidates, n=3, cutoff=0.6)

def _format_unknown(kind: str, category: str, unknowns: List[str], allowed: List[str]) -> str:
    parts = []
    for u in unknowns:
        sugg = _suggest(u, allowed)
        if sugg:
            parts.append(f"'{u}' (did you mean {', '.join(sugg)}?)")
        else:
            parts.append(f"'{u}' (no close match)")
    hint = f"Add to REGISTRY['{kind}']['{category}'] in layer_schema.py if intentional."
    return f"Unknown {category}: " + ", ".join(parts) + f". {hint}"

def validate_layer(layer: "Layer") -> None:
    """Strict validation against REGISTRY with friendly messages."""
    kind = layer.kind
    if kind not in REGISTRY:
        raise ValueError(f"Kind '{kind}' not in REGISTRY. Add it to REGISTRY in act_layers.py.")

    spec = REGISTRY[kind]

    # Type check for params
    for name, val in layer.params.items():
        # If torch isn't available, skip the runtime type check.
        try:
            import torch  # noqa
            if not isinstance(val, Tensor):  # type: ignore[arg-type]
                raise TypeError(f"{kind}.params['{name}'] must be torch.Tensor, got {type(val)}.")
        except Exception:
            pass

    miss_p = _missing(spec['params_required'], layer.params)
    miss_m = _missing(spec['meta_required'], layer.meta)

    allowed_p = spec['params_required'] + spec['params_optional']
    allowed_m = spec['meta_required'] + spec['meta_optional']

    unk_p = _unknown(allowed_p, layer.params)
    unk_m = _unknown(allowed_m, layer.meta)

    errs: List[str] = []
    if miss_p:
        errs.append(f"Missing required PARAMS: {miss_p}. Add them or relax schema in REGISTRY['{kind}']['params_required'].")
    if miss_m:
        errs.append(f"Missing required META: {miss_m}. Add them or relax schema in REGISTRY['{kind}']['meta_required'].")
    if unk_p:
        errs.append(_format_unknown(kind, "params_optional/params_required", unk_p, allowed_p))
    if unk_m:
        errs.append(_format_unknown(kind, "meta_optional/meta_required", unk_m, allowed_m))

    # Critical op sanity
    if kind == LayerKind.CONCAT.value and not isinstance(layer.meta.get("concat_dim", None), int):
        errs.append("CONCAT.meta['concat_dim'] must be int.")
    if kind == LayerKind.SOFTMAX.value and not isinstance(layer.meta.get("axis", None), int):
        errs.append("SOFTMAX.meta['axis'] must be int.")
    if kind == LayerKind.MHA.value:
        has_any = any(k in layer.params for k in ("in_proj_weight","q_proj.weight","k_proj.weight","v_proj.weight","out_proj.weight"))
        if not has_any:
            errs.append("MHA requires in_proj_* or split {q,k,v}_proj.* or out_proj.weight.")

    if errs:
        raise ValueError(f"Layer(id={layer.id}, kind={kind}) schema violation:\n- " + "\n- ".join(errs))

def validate_graph(layers: List["Layer"]) -> None:
    seen = set()
    for ly in layers:
        if ly.id in seen:
            raise ValueError(f"Duplicate layer id {ly.id}")
        seen.add(ly.id)
        validate_layer(ly)
    for ly in layers:
        for v in ly.in_vars + ly.out_vars:
            if not isinstance(v, int) or v < 0:
                raise ValueError(f"Invalid var id {v} in layer {ly.id}")

def validate_wrapper_graph(layers: List["Layer"]) -> None:
    """Hard assertions for the wrapper layout."""
    if not layers:
        raise ValueError("Empty graph")

    kinds = [ly.kind for ly in layers]
    input_count = kinds.count(LayerKind.INPUT.value)
    input_spec_count = kinds.count(LayerKind.INPUT_SPEC.value)
    if input_count != 1:
        raise ValueError(f"Wrapper must have exactly one INPUT layer, found {input_count}.")
    if input_spec_count < 1:
        raise ValueError("Wrapper must include at least one INPUT_SPEC layer.")
    if kinds[-1] != LayerKind.ASSERT.value:
        raise ValueError(f"Last layer must be ASSERT, found {kinds[-1]}.")

    first_spec_idx = kinds.index(LayerKind.INPUT_SPEC.value)
    input_idx = kinds.index(LayerKind.INPUT.value)

    # Adapters only between INPUT..first INPUT_SPEC
    for i in range(input_idx + 1, first_spec_idx):
        if kinds[i] not in ADAPTER_KINDS:
            raise ValueError(f"Only adapters {sorted(ADAPTER_KINDS)} allowed between INPUT and INPUT_SPEC; got {kinds[i]} at pos {i}.")

    # No INPUT/INPUT_SPEC after first spec (except final ASSERT at end)
    for i, k in enumerate(kinds[first_spec_idx+1:-1], start=first_spec_idx+1):
        if k in (LayerKind.INPUT.value, LayerKind.INPUT_SPEC.value):
            raise ValueError(f"Unexpected {k} after the first INPUT_SPEC at index {i}.")

def create_layer(id: int, kind: str, params: Dict[str, "Tensor"], meta: Dict[str, Any],
                 in_vars: List[int], out_vars: List[int]) -> "Layer":
    """Create and validate a layer."""
    try:
        from .core import Layer
    except ImportError:
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
        from act.back_end.core import Layer
    
    ly = Layer(id=id, kind=kind, params=params, meta=meta, in_vars=in_vars, out_vars=out_vars)
    return ly

# ---------------------
# Tiny example (run file)
# ---------------------
if __name__ == "__main__":
    try:
        import torch  # type: ignore
        import sys
        import os
        # Add parent directory to path for absolute imports
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
        
        from act.back_end.core import Layer
        from act.back_end.layer_schema import LayerKind
        from typing import List
        layers: List[Layer] = []

        # INPUT
        layers.append(create_layer(
            id=0, kind=LayerKind.INPUT.value,
            params={}, meta={"shape": (1,3,32,32)},
            in_vars=[0], out_vars=[0],
        ))
        # Adapter
        layers.append(create_layer(
            id=1, kind=LayerKind.PERMUTE.value,
            params={}, meta={"perm": (0,2,3,1)},
            in_vars=[0], out_vars=[0],
        ))
        # SPEC
        layers.append(create_layer(
            id=2, kind=LayerKind.INPUT_SPEC.value,
            params={}, meta={"kind": "BOX", "lb": -1.0, "ub": 1.0},
            in_vars=[0], out_vars=[0],
        ))
        # Model toy
        layers.append(create_layer(
            id=3, kind=LayerKind.FLATTEN.value,
            params={}, meta={"start_dim": 1, "end_dim": -1},
            in_vars=[0], out_vars=[1],
        ))
        W, b = torch.randn(10, 3072), torch.randn(10)
        layers.append(create_layer(
            id=4, kind=LayerKind.DENSE.value,
            params={"W": W, "b": b}, meta={},
            in_vars=[1], out_vars=[2],
        ))
        layers.append(create_layer(
            id=5, kind=LayerKind.ASSERT.value,
            params={}, meta={"kind": "TOP1_ROBUST", "y_true": 3},
            in_vars=[2], out_vars=[2],
        ))

        validate_graph(layers)
        validate_wrapper_graph(layers)
        print("OK — wrapper model passes with", len(layers), "layers.")
    except Exception as e:
        print("Example failed:\n", e)