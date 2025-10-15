# act_schema.py
"""
ACT Layers: Single-File Schema + Strict Registry + Wrapper Checks + Tiny Example

WHY WERE THERE EXTRA CLASSES BEFORE?
- Enums like DataFormat/PaddingMode and dataclasses like ConvMeta/PoolMeta/NormMeta are *convenience types*.
  They provide defaults and IDE/type hints, but they are **not required** for validation. In a verification
  toolchain where you want a slim, explicit surface, they can be replaced by plain strings/tuples stored in
  `Layer.meta` and validated by a central registry.

WHAT THIS FILE PROVIDES (concise):
1) LayerKind enum + Layer dataclass (the only structured types you need).
2) A single REGISTRY that lists **all allowed params/meta keys** per kind.
3) Strict validators in layer_validation.py: `validate_layer`, `validate_graph`, and `validate_wrapper_graph` (for wrapper layout).
4) `create_layer(...)` helper that validates on creation.
5) Clear header on how to add new kinds/keys.
6) A tiny usage example runnable via `python layer_validation.py`.

HOW TO ADD NEW STUFF (READ THIS):
- Add a new LAYER KIND:
    1. Append a value to LayerKind (e.g., MYOP = "MYOP").
    2. Add REGISTRY[LayerKind.MYOP.value] = {
           "params_required": [...],
           "params_optional": [...],
           "meta_required":   [...],
           "meta_optional":   [...],
       }
    3. Done. The validator will enforce that only those keys are used.

- Add a NEW PARAM or META KEY to an existing kind:
    * If every instance MUST have it, put it in the corresponding `..._required` list.
    * Otherwise, add it to `..._optional`.
    * Re-run; unknown keys will fail with a message that suggests the closest valid key
      or tells you to add the key to REGISTRY.

WRAPPER LAYOUT (validated by `validate_wrapper_graph` in layer_validation.py):
InputLayer → InputAdapterLayer(s) → InputSpecLayer → Model → OutputSpecLayer
- Exactly one `INPUT`
- ≥1 `INPUT_SPEC`
- Final layer must be `ASSERT`
- Adapters are deterministic ops: PERMUTE, REORDER, SLICE, PAD, SCALE_SHIFT, LINEAR_PROJ
  (ADAPTER_KINDS defined in layer_validation.py)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List
import enum

# Import Layer from core to avoid circular import issues
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .core import Layer

try:
    import torch
    Tensor = torch.Tensor
except Exception:  # typing only
    Tensor = "torch.Tensor"  # type: ignore

# -------------------------------
# Minimal enum of operation kinds
# -------------------------------
class LayerKind(str, enum.Enum):
    # Wrapper & specs
    INPUT = "INPUT"           # meta: shape (required), center (optional)
    INPUT_SPEC = "INPUT_SPEC" # meta: kind ('BOX'|'LINF_BALL'|'LIN_POLY'), constraints in meta
    ASSERT = "ASSERT"         # meta: kind ('LINEAR_LE'|'TOP1_ROBUST'|'MARGIN_ROBUST'|'RANGE'), fields in meta

    # Input adapters
    PERMUTE = "PERMUTE"
    REORDER = "REORDER"
    SLICE = "SLICE"
    PAD = "PAD"
    SCALE_SHIFT = "SCALE_SHIFT"   # params: scale, shift
    LINEAR_PROJ = "LINEAR_PROJ"   # params: A[, b]

    # Core MLP/CNN ops (subset can be extended easily)
    DENSE = "DENSE"
    CONV1D = "CONV1D"
    CONV2D = "CONV2D"
    CONV3D = "CONV3D"
    CONVTRANSPOSE2D = "CONVTRANSPOSE2D"

    # Pooling
    MAXPOOL1D = "MAXPOOL1D"
    MAXPOOL2D = "MAXPOOL2D"
    MAXPOOL3D = "MAXPOOL3D"
    AVGPOOL1D = "AVGPOOL1D"
    AVGPOOL2D = "AVGPOOL2D"
    AVGPOOL3D = "AVGPOOL3D"
    ADAPTIVEAVGPOOL2D = "ADAPTIVEAVGPOOL2D"

    # Activations / elementwise
    RELU = "RELU"
    LRELU = "LRELU"
    PRELU = "PRELU"
    SIGMOID = "SIGMOID"
    TANH = "TANH"
    SOFTPLUS = "SOFTPLUS"
    SILU = "SILU"
    GELU = "GELU"
    RELU6 = "RELU6"
    HARDTANH = "HARDTANH"
    HARDSIGMOID = "HARDSIGMOID"
    HARDSWISH = "HARDSWISH"
    MISH = "MISH"
    SOFTSIGN = "SOFTSIGN"
    ABS = "ABS"
    CLIP = "CLIP"
    ADD = "ADD"
    SUB = "SUB"
    MUL = "MUL"
    DIV = "DIV"
    POW = "POW"
    MIN = "MIN"
    MAX = "MAX"

    # Tensor plumbing
    CONCAT = "CONCAT"
    STACK = "STACK"
    RESHAPE = "RESHAPE"
    FLATTEN = "FLATTEN"
    TRANSPOSE = "TRANSPOSE"
    SQUEEZE = "SQUEEZE"
    UNSQUEEZE = "UNSQUEEZE"
    TILE = "TILE"
    EXPAND = "EXPAND"
    UPSAMPLE = "UPSAMPLE"

    # Sequences & attention
    EMBEDDING = "EMBEDDING"
    RNN = "RNN"
    GRU = "GRU"
    LSTM = "LSTM"
    SOFTMAX = "SOFTMAX"
    MHA = "MHA"
    POSENC = "POSENC"

# -------------------------------------------
# Strict schema: flat registry (easy to edit)
# -------------------------------------------
REGISTRY: Dict[str, Dict[str, List[str]]] = {
    # Wrapper & specs
    LayerKind.INPUT.value:       {"params_required": [], "params_optional": [], "meta_required": ["shape"], "meta_optional": ["center"]},
    LayerKind.INPUT_SPEC.value:  {"params_required": [], "params_optional": [], "meta_required": ["kind"], "meta_optional": ["lb","ub","center","eps","A","b"]},
    LayerKind.ASSERT.value:      {"params_required": [], "params_optional": [], "meta_required": ["kind"], "meta_optional": ["c","d","y_true","range"]},

    # Adapters
    LayerKind.PERMUTE.value:     {"params_required": [], "params_optional": [], "meta_required": [], "meta_optional": ["perm"]},
    LayerKind.REORDER.value:     {"params_required": [], "params_optional": [], "meta_required": [], "meta_optional": ["order","mapping"]},
    LayerKind.SLICE.value:       {"params_required": [], "params_optional": [], "meta_required": [], "meta_optional": ["starts","ends","axes","steps"]},
    LayerKind.PAD.value:         {"params_required": [], "params_optional": [], "meta_required": [], "meta_optional": ["pads","mode","value"]},
    LayerKind.SCALE_SHIFT.value: {"params_required": ["scale","shift"], "params_optional": [], "meta_required": [], "meta_optional": ["broadcast","axis"]},
    LayerKind.LINEAR_PROJ.value: {"params_required": ["A"], "params_optional": ["b"], "meta_required": [], "meta_optional": ["input_shape","output_shape"]},

    # Dense/CNN
    LayerKind.DENSE.value:       {"params_required": ["W"], "params_optional": ["b","W_pos","W_neg"], "meta_required": [], "meta_optional": ["activation","input_shape","output_shape","bias_enabled"]},
    LayerKind.CONV1D.value:      {"params_required": ["weight"], "params_optional": ["bias","weight_pos","weight_neg"], "meta_required": ["stride","padding","dilation","groups"], "meta_optional": ["transposed","output_padding","padding_mode","input_shape","output_shape","data_format"]},
    LayerKind.CONV2D.value:      {"params_required": ["weight"], "params_optional": ["bias","weight_pos","weight_neg"], "meta_required": ["stride","padding","dilation","groups"], "meta_optional": ["transposed","output_padding","padding_mode","input_shape","output_shape","data_format"]},
    LayerKind.CONV3D.value:      {"params_required": ["weight"], "params_optional": ["bias","weight_pos","weight_neg"], "meta_required": ["stride","padding","dilation","groups"], "meta_optional": ["transposed","output_padding","padding_mode","input_shape","output_shape","data_format"]},
    LayerKind.CONVTRANSPOSE2D.value: {"params_required": ["weight"], "params_optional": ["bias"], "meta_required": ["stride","padding","dilation","groups"], "meta_optional": ["transposed","output_padding","padding_mode","input_shape","output_shape","data_format"]},

    # Pooling
    LayerKind.MAXPOOL1D.value:   {"params_required": [], "params_optional": [], "meta_required": ["kernel_size"], "meta_optional": ["stride","padding","dilation","ceil_mode","count_include_pad","output_size"]},
    LayerKind.MAXPOOL2D.value:   {"params_required": [], "params_optional": [], "meta_required": ["kernel_size"], "meta_optional": ["stride","padding","dilation","ceil_mode","count_include_pad","output_size"]},
    LayerKind.MAXPOOL3D.value:   {"params_required": [], "params_optional": [], "meta_required": ["kernel_size"], "meta_optional": ["stride","padding","dilation","ceil_mode","count_include_pad","output_size"]},
    LayerKind.AVGPOOL1D.value:   {"params_required": [], "params_optional": [], "meta_required": ["kernel_size"], "meta_optional": ["stride","padding","dilation","ceil_mode","count_include_pad","output_size"]},
    LayerKind.AVGPOOL2D.value:   {"params_required": [], "params_optional": [], "meta_required": ["kernel_size"], "meta_optional": ["stride","padding","dilation","ceil_mode","count_include_pad","output_size"]},
    LayerKind.AVGPOOL3D.value:   {"params_required": [], "params_optional": [], "meta_required": ["kernel_size"], "meta_optional": ["stride","padding","dilation","ceil_mode","count_include_pad","output_size"]},
    LayerKind.ADAPTIVEAVGPOOL2D.value: {"params_required": [], "params_optional": [], "meta_required": [], "meta_optional": ["output_size"]},

    # Activations / elementwise
    LayerKind.RELU.value:        {"params_required": [], "params_optional": [], "meta_required": [], "meta_optional": []},
    LayerKind.LRELU.value:       {"params_required": [], "params_optional": [], "meta_required": [], "meta_optional": ["negative_slope"]},
    LayerKind.PRELU.value:       {"params_required": ["weight"], "params_optional": [], "meta_required": [], "meta_optional": []},
    LayerKind.SIGMOID.value:     {"params_required": [], "params_optional": [], "meta_required": [], "meta_optional": []},
    LayerKind.TANH.value:        {"params_required": [], "params_optional": [], "meta_required": [], "meta_optional": []},
    LayerKind.SOFTPLUS.value:    {"params_required": [], "params_optional": [], "meta_required": [], "meta_optional": []},
    LayerKind.SILU.value:        {"params_required": [], "params_optional": [], "meta_required": [], "meta_optional": []},
    LayerKind.GELU.value:        {"params_required": [], "params_optional": [], "meta_required": [], "meta_optional": ["approximate"]},
    LayerKind.RELU6.value:       {"params_required": [], "params_optional": [], "meta_required": [], "meta_optional": []},
    LayerKind.HARDTANH.value:    {"params_required": [], "params_optional": [], "meta_required": [], "meta_optional": ["min_val","max_val"]},
    LayerKind.HARDSIGMOID.value: {"params_required": [], "params_optional": [], "meta_required": [], "meta_optional": ["alpha","beta"]},
    LayerKind.HARDSWISH.value:   {"params_required": [], "params_optional": [], "meta_required": [], "meta_optional": []},
    LayerKind.SOFTSIGN.value:    {"params_required": [], "params_optional": [], "meta_required": [], "meta_optional": []},
    LayerKind.ABS.value:         {"params_required": [], "params_optional": [], "meta_required": [], "meta_optional": []},
    LayerKind.CLIP.value:        {"params_required": [], "params_optional": [], "meta_required": [], "meta_optional": ["min","max"]},
    LayerKind.ADD.value:         {"params_required": [], "params_optional": [], "meta_required": [], "meta_optional": ["broadcast","axis"]},
    LayerKind.SUB.value:         {"params_required": [], "params_optional": [], "meta_required": [], "meta_optional": ["broadcast","axis"]},
    LayerKind.MUL.value:         {"params_required": [], "params_optional": [], "meta_required": [], "meta_optional": ["broadcast","axis"]},
    LayerKind.DIV.value:         {"params_required": [], "params_optional": [], "meta_required": [], "meta_optional": ["broadcast","axis"]},
    LayerKind.POW.value:         {"params_required": [], "params_optional": [], "meta_required": [], "meta_optional": ["broadcast","axis"]},
    LayerKind.MIN.value:         {"params_required": [], "params_optional": [], "meta_required": [], "meta_optional": ["broadcast","axis"]},
    LayerKind.MAX.value:         {"params_required": [], "params_optional": [], "meta_required": [], "meta_optional": ["broadcast","axis"]},

    # Tensor plumbing
    LayerKind.CONCAT.value:      {"params_required": [], "params_optional": [], "meta_required": ["concat_dim"], "meta_optional": []},
    LayerKind.STACK.value:       {"params_required": [], "params_optional": [], "meta_required": ["axis"], "meta_optional": []},
    LayerKind.RESHAPE.value:     {"params_required": [], "params_optional": [], "meta_required": [], "meta_optional": ["target_shape"]},
    LayerKind.FLATTEN.value:     {"params_required": [], "params_optional": [], "meta_required": [], "meta_optional": ["start_dim","end_dim"]},
    LayerKind.TRANSPOSE.value:   {"params_required": [], "params_optional": [], "meta_required": [], "meta_optional": ["perm"]},
    LayerKind.SQUEEZE.value:     {"params_required": [], "params_optional": [], "meta_required": [], "meta_optional": ["dims"]},
    LayerKind.UNSQUEEZE.value:   {"params_required": [], "params_optional": [], "meta_required": [], "meta_optional": ["dims"]},
    LayerKind.TILE.value:        {"params_required": [], "params_optional": [], "meta_required": [], "meta_optional": ["repeats"]},
    LayerKind.EXPAND.value:      {"params_required": [], "params_optional": [], "meta_required": [], "meta_optional": ["shape"]},
    LayerKind.UPSAMPLE.value:    {"params_required": [], "params_optional": [], "meta_required": [], "meta_optional": ["mode","align_corners","scale_factor","size"]},

    # Sequences / attention
    LayerKind.EMBEDDING.value:   {"params_required": ["weight"], "params_optional": [], "meta_required": ["num_embeddings","embedding_dim"], "meta_optional": ["padding_idx","max_norm","norm_type","scale_grad_by_freq","sparse"]},
    LayerKind.RNN.value:         {"params_required": [], "params_optional": ["weight_ih_l0","weight_hh_l0","bias_ih_l0","bias_hh_l0"], "meta_required": ["input_size","hidden_size","num_layers","bidirectional"], "meta_optional": ["dropout","batch_first","nonlinearity","proj_size","gate_order","packed_sequence"]},
    LayerKind.GRU.value:         {"params_required": [], "params_optional": ["weight_ih_l0","weight_hh_l0","bias_ih_l0","bias_hh_l0"], "meta_required": ["input_size","hidden_size","num_layers","bidirectional"], "meta_optional": ["dropout","batch_first","nonlinearity","proj_size","gate_order","packed_sequence"]},
    LayerKind.LSTM.value:        {"params_required": [], "params_optional": ["weight_ih_l0","weight_hh_l0","bias_ih_l0","bias_hh_l0"], "meta_required": ["input_size","hidden_size","num_layers","bidirectional"], "meta_optional": ["dropout","batch_first","nonlinearity","proj_size","gate_order","packed_sequence"]},
    LayerKind.SOFTMAX.value:     {"params_required": [], "params_optional": [], "meta_required": ["axis"], "meta_optional": []},
    LayerKind.MHA.value:         {"params_required": [], "params_optional": ["in_proj_weight","in_proj_bias","q_proj.weight","q_proj.bias","k_proj.weight","k_proj.bias","v_proj.weight","v_proj.bias","out_proj.weight","out_proj.bias","bias_k","bias_v","rel_pos_bias"], "meta_required": ["num_heads"], "meta_optional": ["head_dim","scale","dropout","add_zero_attn","batch_first","causal","mask_kind","mask_format","axis","qkv_layout","posenc_kind","rope_theta"]},
    LayerKind.POSENC.value:      {"params_required": [], "params_optional": ["weight","slopes"], "meta_required": [], "meta_optional": ["kind","seq_len","embedding_dim","theta"]},
}
