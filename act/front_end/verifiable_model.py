#===- act/front_end/verifiable_model.py - PyTorch Wrapper Layers -------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Batch-native PyTorch wrapper layers for spec-free verification.
#   Embeds constraints directly into models, enabling efficient batched
#   verification where N samples are checked in a single forward pass.
#
# Key Features:
#   - Batch-native: Processes (N, ...) tensors, verifies N samples 
#   - Spec-free: Constraints embedded in model, not external files
#   - PyTorch-native: Full nn.Module compatibility
#   - Automatic verification: Returns per-sample constraint satisfaction
#
# Core Layers:
#
#   InputLayer(labeled_input, shape=(N, C, H, W), dtype=...)
#     Declares batched input with attributes. No-op at inference.
#
#   InputSpecLayer(spec=InputSpec(lb=(N,...), ub=(N,...)))
#     Checks N input constraints. Returns (x, satisfied(N,), explanation).
#
#   OutputSpecLayer(spec=OutputSpec(y_true=(N,), margin=(N,)))
#     Checks N output properties. Returns (y, satisfied(N,), explanation).
#
# Batched Workflow:
#   # 1. Build model with batched specs (N=3 samples)
#   class TinyMLP(nn.Module):
#       def __init__(self):
#           super().__init__()
#           self.flatten = nn.Flatten(start_dim=1)
#           self.fc = nn.Linear(784, 10)
#       def forward(self, x):
#           return self.fc(self.flatten(x))
#
#   labeled_input = LabeledInputTensor(
#       tensor=torch.zeros(3, 1, 28, 28), label=torch.tensor([5, 3, 7])
#   )
#   model = VerifiableModel(
#       input_layer=InputLayer(labeled_input, shape=(3, 1, 28, 28), dtype=torch.float32),
#       input_spec=InputSpecLayer(InputSpec(
#           kind=InKind.BOX,
#           lb=torch.zeros(3, 1, 28, 28),
#           ub=torch.ones(3, 1, 28, 28),
#       )),
#       model=TinyMLP(),
#       output_spec=OutputSpecLayer(OutputSpec(
#           kind=OutKind.TOP1_ROBUST, y_true=torch.tensor([5, 3, 7])
#       )),
#   )
#
#   # 2. Run: 3 samples verified in one forward pass
#   result = model(batch_input)  # batch_input: (3, 1, 28, 28)
#   # Returns: {output: (3,10), input_satisfied: bool, output_satisfied: bool}
#
#   # 3. Convert to ACT for formal verification
#   from act.pipeline.torch2act import TorchToACT
#   act_net = TorchToACT(model).run()
#
# Supported Constraints:
#   InKind: BOX, LINF_BALL, LIN_POLY, LP_EMBEDDING
#     LP_EMBEDDING: Lp-norm ball in embedding space for verify-from-embeddings
#       text models. Perturbs only selected token positions (perturbed_positions)
#       by eps under p_norm, holding all other positions fixed at center.
#       See act.front_end.specs.InKind for the full definition.
#   OutKind: TOP1_ROBUST, MARGIN_ROBUST, LINEAR_LE, RANGE, UNSAFE_LINEAR
#
#===---------------------------------------------------------------------===#

from __future__ import annotations
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple, Union

# Import ACT components
from act.front_end.specs import InputSpec, OutputSpec, InKind, OutKind
from act.front_end.spec_creator_base import LabeledInputTensor
from act.back_end.layer_schema import LayerKind, REGISTRY
from act.back_end.layer_util import create_layer


def prod(seq: Tuple[int, ...]) -> int:
    """Helper function to compute product of shape dimensions."""
    p = 1
    for s in seq:
        p *= s
    return p


class VerifiableModel(nn.Module):
    """
    verification wrapper for PyTorch models.
    
    Processes batched tensors (N, ...) where N samples are verified independently
    in a single forward pass. Collects constraint results from spec layers.
    
    Batched Architecture:
        Input: (N, C, H, W), (N, features), or embedding block (N, L, D)
        → InputLayer: declares symbolic input block (pass-through at inference)
        → InputSpecLayer: checks each sample's input constraints
        → model: inner network (any nn.Module — Sequential, ResNet, Transformer, ...)
        → OutputSpecLayer: checks each sample's output properties
        Output: Aggregated bool (all samples satisfied)
    
    Args (keyword-only):
        input_layer: InputLayer declaring the symbolic input block.
        input_spec: InputSpecLayer carrying input constraints.
        model: inner nn.Module. Can be any graph (Sequential, DAG, skip-connection net).
        output_spec: OutputSpecLayer carrying output property (TOP1/MARGIN/LINEAR_LE/RANGE).
    
    Returns:
        Dict with:
        - output: (N, ...) tensor
        - input_satisfied: bool (all N samples satisfy input constraints)
        - output_satisfied: bool (all N samples satisfy output properties)
        - input_explanation/output_explanation: Human-readable results
    """
    
    # Class-level strict mode setting (shared across all instances)
    _strict_mode: bool = False
    
    def __init__(self, *, input_layer, input_spec, model, output_spec):
        super().__init__()
        self.input_layer = input_layer
        self.input_spec = input_spec
        self.model = model
        self.output_spec = output_spec
    
    @classmethod
    def set_strict_mode(cls, enabled: bool) -> None:
        """
        When enabled, forward() raises ValueError on input/output violations.
        """
        cls._strict_mode = enabled
    
    @classmethod
    def get_strict_mode(cls) -> bool:
        return cls._strict_mode
    
    def forward(self, x):
        input_satisfied = True
        input_explanation = "No INPUT_SPEC layer"
        output_satisfied = True
        output_explanation = "No OUTPUT_SPEC layer"

        x = self.input_layer(x)

        result = self.input_spec(x)
        if isinstance(result, tuple) and len(result) == 3:
            x, input_satisfied, input_explanation = result
        else:
            x = result

        x = self.model(x)

        result = self.output_spec(x)
        if isinstance(result, tuple) and len(result) == 3:
            x, output_satisfied, output_explanation = result
        else:
            x = result
        
        # Spec layers always return tensors; collapse them to a single
        # bool for the dict-shaped output expected by callers.
        if isinstance(input_satisfied, torch.Tensor):
            input_satisfied = bool(input_satisfied.all().item())
        if isinstance(output_satisfied, torch.Tensor):
            output_satisfied = bool(output_satisfied.all().item())
        
        # Strict mode: raise on constraint violations
        if self._strict_mode:
            if not input_satisfied:
                print(f"[STRICT MODE] {input_explanation}")
                raise ValueError(
                    f"Input constraint violated in strict mode: {input_explanation}"
                )
            if not output_satisfied:
                print(f"[STRICT MODE] {output_explanation}")
                raise ValueError(
                    f"Output constraint violated in strict mode: {output_explanation}"
                )
        
        # Return comprehensive verification result
        return {
            'output': x,
            'input_satisfied': input_satisfied,
            'input_explanation': input_explanation,
            'output_satisfied': output_satisfied,
            'output_explanation': output_explanation
        }


class InputLayer(nn.Module):
    """
    Declares batched input with attributes. No-op during forward pass.
    
    Batch Format:
        shape: (N, C, H, W) or (N, features) where N is batch size
        Allocates N * per_sample_vars for ACT verification
        Stores labeled_input: (tensor, label) with N samples
    
    Args:
        labeled_input: LabeledInputTensor with (N, ...) tensor and labels
        shape: Input shape including batch dimension
        dtype: REQUIRED - affects verification precision/range
        dataset_name, layout, etc.: Optional attributes
    
    Forward:
        x → x (pass-through, attributes only)
    """
    def __init__(
        self,
        labeled_input: "LabeledInputTensor",  # Complete input sample with label
        shape: Tuple[int, ...],
        dtype: torch.dtype,  # REQUIRED: Critical for verification soundness
        desc: str = "input",
        # Tier 1: Essential attributes (strongly recommended)
        layout: Optional[str] = None,
        dataset_name: Optional[str] = None,
        # Tier 2: Important attributes
        num_classes: Optional[int] = None,
        value_range: Optional[Tuple[float, float]] = None,
        scale_hint: Optional[str] = None,
        distribution: Optional[str] = None,  # "uniform", "normal", "normalized", "unknown", or custom
        # Tier 3: Optional attributes
        sample_id: Optional[Union[int, str]] = None,
        domain: Optional[str] = None,
        channels: Optional[int] = None,
    ):
        super().__init__()
        # Allow any batch size >= 1 for verification
        if shape[0] < 1:
            raise ValueError(f"Batch size must be >= 1, got {shape[0]}")
        
        # Core attributes (dtype is required by callers downstream).
        self.shape = tuple(shape)
        self._batched = shape[0] > 1  # Track if batched for to_act_layers
        self.dtype = dtype  # REQUIRED
        self.desc = desc
        
        # Tier 1: Essential attributes
        self.layout = layout
        self.dataset_name = dataset_name
        
        # Tier 2: Important attributes
        self.num_classes = num_classes
        self.value_range = tuple(value_range) if value_range else None
        self.scale_hint = scale_hint
        self.distribution = distribution
        
        # Tier 3: Optional attributes
        self.sample_id = sample_id
        self.domain = domain
        self.channels = channels
        
        # Store labeled input as PyTorch buffers
        # Assumes device/dtype already initialized by caller (e.g., via initialize_device)
        self.register_buffer("_input_tensor", labeled_input.tensor)
        
        # Store label (tensor or None)
        # Note: Labels use int64 dtype
        if labeled_input.label is not None:
            if not isinstance(labeled_input.label, torch.Tensor):
                raise TypeError(
                    f"labeled_input.label must be torch.Tensor or None, "
                    f"got {type(labeled_input.label).__name__}"
                )
            self.register_buffer("_label_tensor", labeled_input.label.reshape(-1))
        else:
            # No label provided
            self.register_buffer("_label_tensor", torch.tensor([], dtype=torch.int64))
        
        self._validate_schema()
    
    @property
    def labeled_input(self) -> "LabeledInputTensor":
        """Get the complete labeled input (tensor + tensor label)."""
        from act.front_end.spec_creator_base import LabeledInputTensor
        # Return tensor label directly
        label = self._label_tensor if self._label_tensor.numel() > 0 else None
        return LabeledInputTensor(tensor=self._input_tensor, label=label)
    
    @property
    def input_tensor(self) -> torch.Tensor:
        """Get input tensor (convenience accessor)."""
        return self._input_tensor
    
    @property
    def label(self) -> torch.Tensor:
        """Get label tensor (always (N,) shape)."""
        return self._label_tensor

    def _validate_schema(self):
        """Validate parameters against INPUT layer schema"""
        schema = REGISTRY[LayerKind.INPUT.value]
        
        # Build unified params dict
        params = {
            "labeled_input": self.labeled_input,
            "shape": self.shape,
            "dtype": str(self.dtype),  # REQUIRED
        }
        
        # Add non-default desc
        if self.desc != "input":
            params["desc"] = self.desc
        
        # Add optional fields (only if not None)
        if self.layout is not None:
            params["layout"] = self.layout
        if self.dataset_name is not None:
            params["dataset_name"] = self.dataset_name
        if self.num_classes is not None:
            params["num_classes"] = self.num_classes
        if self.value_range is not None:
            params["value_range"] = self.value_range
        if self.scale_hint is not None:
            params["scale_hint"] = self.scale_hint
        if self.distribution is not None:
            params["distribution"] = self.distribution
        if self.sample_id is not None:
            params["sample_id"] = self.sample_id
        if self.domain is not None:
            params["domain"] = self.domain
        if self.channels is not None:
            params["channels"] = self.channels
        
        # Check required/optional params
        for key in schema["params_required"]:
            if key not in params:
                raise ValueError(f"InputLayer missing required param: {key}")
        allowed_params = schema["params_required"] + schema["params_optional"]
        for key in params:
            if key not in allowed_params:
                raise ValueError(f"InputLayer has unknown param: {key}")

    def to_act_layers(self, layer_id_start: int, in_vars: List[int]) -> Tuple[List, List[int]]:
        """Convert to ACT Layer(s) and return (layers, out_vars)
        
        For batched inputs (shape[0] > 1), allocates batch_size * per_sample_vars.
        """
        batch_size = self.shape[0]
        per_sample = prod(self.shape[1:])
        N = batch_size * per_sample  # Total vars for all samples in batch
        out_vars = list(range(len(in_vars), len(in_vars) + N))
        
        # Build unified params dict
        params = {
            "labeled_input": self.labeled_input,
            "shape": self.shape,
            "dtype": str(self.dtype),  # REQUIRED
            "batch_size": batch_size,  # Track batch size for batched verification
        }
        
        if self.desc != "input":
            params["desc"] = self.desc
        
        # Add all optional fields (only if not None)
        if self.layout is not None:
            params["layout"] = self.layout
        if self.dataset_name is not None:
            params["dataset_name"] = self.dataset_name
        if self.num_classes is not None:
            params["num_classes"] = self.num_classes
        if self.value_range is not None:
            params["value_range"] = self.value_range
        if self.scale_hint is not None:
            params["scale_hint"] = self.scale_hint
        if self.distribution is not None:
            params["distribution"] = self.distribution
        if self.sample_id is not None:
            params["sample_id"] = self.sample_id
        if self.domain is not None:
            params["domain"] = self.domain
        if self.channels is not None:
            params["channels"] = self.channels
        
        layer = create_layer(
            id=layer_id_start,
            kind=LayerKind.INPUT.value,
            params=params,
            in_vars=in_vars,
            out_vars=out_vars
        )
        return [layer], out_vars

    def get_attributes_summary(self) -> Dict[str, Any]:
        """Return a summary of all attributes for debugging/logging."""
        return {
            "shape": self.shape,
            "desc": self.desc,
            "dtype": str(self.dtype),
            "layout": self.layout,
            "dataset_name": self.dataset_name,
            "num_classes": self.num_classes,
            "value_range": self.value_range,
            "scale_hint": self.scale_hint,
            "distribution": self.distribution,
            "label": self.label.tolist(),
            "sample_id": self.sample_id,
            "domain": self.domain,
            "channels": self.channels,
            "input_tensor_shape": tuple(self.input_tensor.shape),
        }

    def __repr__(self) -> str:
        """Enhanced string representation with key attributes."""
        meta_str = f"shape={self.shape}"
        if self.dataset_name:
            meta_str += f", dataset={self.dataset_name}"
        # Display label in readable format
        label_display = self.label.tolist() if self.label.numel() > 1 else self.label.item()
        meta_str += f", label={label_display}"
        if self.layout:
            meta_str += f", layout={self.layout}"
        return f"InputLayer({meta_str})"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class InputSpecLayer(nn.Module):
    """
    Batched input constraint checker (BOX, L∞, LIN_POLY, LP_EMBEDDING).

    This layer serves two perturbation regimes carried by the wrapped
    ``InputSpec``:

    * **Image/input perturbation** (``BOX``, ``LINF_BALL``): the constraint
      acts on pixel or raw-feature space (x in R^n).

      - LINF_BALL:  { x : ||x - center||_inf <= eps }
                    <=>  x_i in [center_i - eps, center_i + eps]  for every i
      - BOX:        { x : lb_i <= x_i <= ub_i }  (explicit per-coordinate box)

      ONE ball (or box) over ALL input coordinates — perturbs pixels/features.
      ``x`` is compared directly against ``lb``/``ub`` (BOX) or
      ``center``/``eps`` (LINF_BALL).

    * **Embedding/position perturbation** (``LP_EMBEDDING``): the constraint
      acts in the embedding space of a transformer (BERT, ViT token
      embeddings).  For embeddings e in R^{L x d} with ``spec.center`` c and
      P = ``spec.perturbed_positions``, the admissible set is:

          { e :  ||e_t - c_t||_{p_norm} <= eps   for t in P,
                 e_t = c_t                         for t not in P }

      A SEPARATE per-token Lp ball of radius ``eps`` in the d-dimensional
      embedding space at each SELECTED token position t in P; all OTHER
      positions are PINNED to ``spec.center``.  The materialized ``lb``/``ub``
      box is pre-computed at construction time via
      ``InputSpec.materialize_box_seed``.

      Contrast: image = ONE ball over all input coordinates (perturbs pixels);
      LP_EMBEDDING = localized per-token balls at chosen sequence positions in
      EMBEDDING space (rest fixed) — perturbs token embeddings, not pixels.

    Batch Processing:
        Input: x with shape (N, C, H, W) or (N, L, D) for embeddings
        Constraints: lb/ub/center with shape (N, ...) - per-sample bounds
        Checks: Each sample x[i] verified independently against its bounds
        Output: (x, satisfied, explanation)
            - satisfied: (N,) bool tensor, one per sample
            - explanation: "✅ INPUT BOX: {n_ok}/{N} satisfied"

    Args:
        spec: InputSpec with batched constraint tensors

    Forward:
        x (N, ...) → (x, satisfied (N,), explanation str)
    """
    def __init__(self, spec: Optional[InputSpec] = None, **kwargs):
        super().__init__()
        self.spec = spec or InputSpec(**kwargs)
        self.kind = self.spec.kind

        if self.kind == InKind.LP_EMBEDDING:
            self.spec.lb, self.spec.ub = self.spec.materialize_box_seed()
        elif self.kind == InKind.LINF_BALL and self.spec.center is not None and self.spec.eps is not None:
            eps = self.spec.eps
            if isinstance(eps, torch.Tensor):
                eps_t = eps.to(device=self.spec.center.device, dtype=self.spec.center.dtype)
            else:
                eps_t = self.spec.center.new_tensor(eps)
            self.spec.lb = self.spec.center - eps_t
            self.spec.ub = self.spec.center + eps_t
        
        # Register eps as buffer for device mobility
        if isinstance(self.spec.eps, torch.Tensor):
            self.register_buffer("eps", self.spec.eps)
        else:
            self.eps = None

        if isinstance(self.spec.p_norm, torch.Tensor):
            self.register_buffer("p_norm", self.spec.p_norm)
        else:
            self.p_norm = None

        if isinstance(self.spec.perturbed_positions, torch.Tensor):
            self.register_buffer("perturbed_positions", self.spec.perturbed_positions)
        else:
            self.perturbed_positions = None

        # Register tensor fields as buffers so .to(device) works
        for name in ("lb", "ub", "center", "A", "b"):
            val = getattr(self.spec, name, None)
            if isinstance(val, torch.Tensor):
                self.register_buffer(name, val)
            else:
                setattr(self, name, None)
        self._validate_schema()

    def _validate_schema(self):
        """Validate parameters against INPUT_SPEC layer schema"""
        schema = REGISTRY[LayerKind.INPUT_SPEC.value]
        params = {"kind": self.kind}
        for name in ("lb", "ub", "center", "A", "b"):
            val = getattr(self, name, None)
            if val is not None:
                params[name] = val
        if self.eps is not None:
            params["eps"] = self.eps
        if self.kind == InKind.LP_EMBEDDING:
            if self.p_norm is not None:
                params["p_norm"] = self.p_norm
            if self.perturbed_positions is not None:
                params["perturbed_positions"] = self.perturbed_positions
        if self.kind == InKind.LP_EMBEDDING:
            if self.p_norm is None:
                raise ValueError("LP_EMBEDDING requires p_norm")
            if self.perturbed_positions is None:
                raise ValueError("LP_EMBEDDING requires perturbed_positions")
        
        # Check schema compliance
        for key in schema["params_required"]:
            if key not in params:
                raise ValueError(f"InputSpecLayer missing required param: {key}")
        for key in params:
            if key not in schema["params_required"] + schema["params_optional"]:
                raise ValueError(f"InputSpecLayer has unknown param: {key}")

    def to_act_layers(
        self,
        layer_id_start: int,
        in_vars: List[int],
        B: int = 1,
    ) -> Tuple[List, List[int]]:
        """Convert to ACT Layer(s) - INPUT_SPEC doesn't create new vars.

        ``B`` is accepted for signature parity with ``OutputSpecLayer`` so
        callers can pass the batch dim uniformly; currently unused because
        ``InputSpec`` tensors are already batched at construction time
        (lb/ub/center carry the leading B dim from the wrapped tensors).
        """
        del B
        params = {"kind": self.kind}
        for name in ("lb", "ub", "center", "A", "b"):
            val = getattr(self, name, None)
            if val is not None:
                params[name] = val
        if self.eps is not None:
            params["eps"] = self.eps
        if self.kind == InKind.LP_EMBEDDING:
            if self.p_norm is not None:
                params["p_norm"] = self.p_norm
            if self.perturbed_positions is not None:
                params["perturbed_positions"] = self.perturbed_positions
        layer = create_layer(
            id=layer_id_start,
            kind=LayerKind.INPUT_SPEC.value,
            params=params,
            in_vars=in_vars,
            out_vars=in_vars,
        )
        if self.kind == InKind.LP_EMBEDDING:
            if self.p_norm is not None:
                layer.cache["p_norm"] = self.p_norm
            if self.perturbed_positions is not None:
                layer.cache["perturbed_positions"] = self.perturbed_positions
        return [layer], in_vars

    def forward(self, x: torch.Tensor):
        """
        Forward pass with unified constraint checking for single or batched input.
        
        Returns:
            Tuple of (tensor, satisfied, explanation)
            - For batch=1 andbatch>1: satisfied is (batch,) bool tensor
        """
        if self.spec is None:
            return (x, True, "✅ INPUT: No constraints")
        
        batch_size = x.shape[0]
        
        if self.kind == InKind.BOX:
            if self.lb is None or self.ub is None:
                return (x, True, "⚠️ INPUT BOX: Missing lb/ub")
            
            lb, ub = self.lb, self.ub
            # model_synthesis.py ensures lb.shape[0] == batch_size

            lb_ok = (x >= lb).reshape(batch_size, -1).all(dim=1)  # (batch,)
            ub_ok = (x <= ub).reshape(batch_size, -1).all(dim=1)  # (batch,)
            satisfied = lb_ok & ub_ok  # (batch,) bool tensor
            n_ok = satisfied.sum().item()
            explanation = f"✅ INPUT BOX: {n_ok}/{batch_size} satisfied"
            
            # Always return tensor (unified output format)
            return (x, satisfied, explanation)
        
        elif self.kind == InKind.LINF_BALL:
            if self.center is None or self.eps is None:
                return (x, True, "⚠️ INPUT L∞: Missing center/eps")
            
            center = self.center
            
            # Per-sample L∞ distance
            linf = (x - center).abs().reshape(batch_size, -1).max(dim=1)[0]  # (batch,)
            
            # Compare against per-sample eps threshold.
            eps = self.eps.reshape(self.eps.shape[0])
            satisfied = linf <= eps
            
            n_ok = satisfied.sum().item()
            explanation = f"✅ INPUT L∞: {n_ok}/{batch_size} satisfied"
            
            # Always return tensor (unified output format)
            return (x, satisfied, explanation)
        
        elif self.kind == InKind.LP_EMBEDDING:
            if self.lb is None or self.ub is None:
                return (x, True, "⚠️ INPUT LP_EMBEDDING: Missing lb/ub")

            lb_ok = (x >= self.lb).reshape(batch_size, -1).all(dim=1)
            ub_ok = (x <= self.ub).reshape(batch_size, -1).all(dim=1)
            satisfied = lb_ok & ub_ok
            n_ok = satisfied.sum().item()
            explanation = f"✅ INPUT LP_EMBEDDING: {n_ok}/{batch_size} satisfied"
            return (x, satisfied, explanation)

        elif self.kind == InKind.LIN_POLY:
            if self.A is None or self.b is None:
                return (x, True, "⚠️ INPUT LIN_POLY: Missing A/b")
            
            # LIN_POLY typically single sample; batched requires per-sample A,b
            x_flat = x.reshape(batch_size, -1)  # (batch, n_vars)
            # For now, apply same A,b to each sample
            residuals = x_flat @ self.A.T - self.b  # (batch, n_constraints)
            max_viol = residuals.max(dim=1)[0]  # (batch,)
            satisfied = max_viol <= 0
            
            # Unified explanation format (consistent across all batch sizes)
            n_ok = satisfied.sum().item()
            explanation = f"✅ INPUT LIN_POLY: {n_ok}/{batch_size} satisfied"
            
            # Always return tensor (unified output format)
            return (x, satisfied, explanation)
        
        else:
            return (x, True, f"⚠️ INPUT: Unknown kind {self.kind}")


class OutputSpecLayer(nn.Module):
    """
    Batched output property checker (TOP1_ROBUST, MARGIN_ROBUST, etc.).
    
    Batch Processing:
        Input: y with shape (N, n_classes) or (N, features)
        Properties: y_true (N,), margin (N,) - per-sample targets
        Checks: Each sample y[i] verified independently against its property
        Output: (y, satisfied, explanation)
            - satisfied: (N,) bool tensor, one per sample
            - explanation: "✅ OUTPUT TOP1: {n_ok}/{N} robust"
    
    Args:
        spec: OutputSpec with batched property tensors
    
    Forward:
        y (N, ...) → (y, satisfied (N,), explanation str)
    """
    def __init__(self, spec: Optional[OutputSpec] = None, **kwargs):
        super().__init__()
        self.spec = spec or OutputSpec(**kwargs)
        self.kind = self.spec.kind
        self.d = self.spec.d
        
        # register as buffer for device mobility
        if self.spec.y_true is not None:
            self.register_buffer("y_true", self.spec.y_true)
        else:
            self.y_true = None
        
        if self.spec.margin is not None:
            self.register_buffer("margin", self.spec.margin)
        else:
            self.margin = None

        for name in ("c", "lb", "ub"):
            val = getattr(self.spec, name, None)
            if isinstance(val, torch.Tensor):
                self.register_buffer(name, val)
            else:
                setattr(self, name, None)
        self._validate_schema()

    def _validate_schema(self):
        """Pre-flight kind validation; full ASSERT-layer schema (including
        pre-encoded ``C`` / ``thresholds`` / ``M``) is enforced at
        ``to_act_layers`` time via ``create_layer``.
        """
        supported = (
            OutKind.LINEAR_LE, OutKind.UNSAFE_LINEAR,
            OutKind.TOP1_ROBUST, OutKind.MARGIN_ROBUST, OutKind.RANGE,
        )
        if self.kind not in supported:
            raise ValueError(
                f"OutputSpecLayer: unsupported kind {self.kind!r}; "
                f"expected one of {supported}"
            )

    def to_act_layers(
        self,
        layer_id_start: int,
        in_vars: List[int],
        B: int = 1,
    ) -> Tuple[List, List[int]]:
        """Build the ACT ASSERT Layer for this output spec.

        Encodes the spec via ``OutputSpec.encode_linear``, producing a
        params dict with both high-level fields (for the BaB MILP path)
        and pre-encoded ``C`` / ``thresholds`` / ``M`` (for ``verify_once``).

        Args:
            layer_id_start: numeric id to assign to the new ASSERT layer.
            in_vars: variable ids carrying the network output values that
                this assert constrains (``out_vars == in_vars`` for ASSERT).
            B: batch size of the surrounding net; flows from the InputLayer's
                ``shape[0]``. Defaults to 1 when the caller has no batch context.
        """
        from act.util.device_manager import (
            get_default_device, get_default_dtype,
        )
        n_out = len(in_vars)
        device = get_default_device()
        dtype = get_default_dtype()
        params = self.spec.encode_linear(
            B=B, n_out=n_out, device=device, dtype=dtype,
        )
        layer = create_layer(
            id=layer_id_start,
            kind=LayerKind.ASSERT.value,
            params=params,
            in_vars=in_vars,
            out_vars=in_vars,
        )
        return [layer], in_vars

    def forward(self, y: torch.Tensor):
        """
        Forward pass with unified constraint checking for single or batched output.
        
        Returns:
            Tuple of (tensor, satisfied, explanation)
            For batch=1 and batch>1: satisfied is (batch,) bool tensor
        """
        if self.spec is None:
            return (y, True, "✅ OUTPUT: No constraints")
        
        batch_size = y.shape[0]
        
        if self.kind == OutKind.TOP1_ROBUST:
            if self.y_true is None:
                return (y, True, "⚠️ OUTPUT TOP1: Missing y_true")
            
            preds = y.argmax(dim=1)  # (batch,)
            satisfied = preds == self.y_true  # (batch,) bool tensor
            
            # Unified explanation format (consistent across all batch sizes)
            n_ok = satisfied.sum().item()
            explanation = f"✅ OUTPUT TOP1: {n_ok}/{batch_size} robust"
            
            # Always return tensor (unified output format)
            return (y, satisfied, explanation)
        
        elif self.kind == OutKind.MARGIN_ROBUST:
            if self.y_true is None:
                return (y, True, "⚠️ OUTPUT MARGIN: Missing y_true")
            
            # y is (batch, n_classes)
            n_classes = y.shape[1]
            true_scores = y[torch.arange(batch_size, device=y.device), self.y_true]  # (batch,)
            
            # Mask out true class to get max of others
            mask = torch.ones_like(y, dtype=torch.bool)
            mask[torch.arange(batch_size, device=y.device), self.y_true] = False
            other_scores = y.masked_fill(~mask, float('-inf'))
            max_other = other_scores.max(dim=1)[0]  # (batch,)
            
            actual_margin = true_scores - max_other  # (batch,)
            margin_threshold = self.margin  # always a tensor, including the n=1 case
            satisfied = actual_margin >= margin_threshold  # (batch,) bool
            
            # Unified explanation format (consistent across all batch sizes)
            n_ok = satisfied.sum().item()
            explanation = f"✅ OUTPUT MARGIN: {n_ok}/{batch_size} satisfied"
            
            # Always return tensor (unified output format)
            return (y, satisfied, explanation)
        
        elif self.kind == OutKind.LINEAR_LE:
            if self.c is None or self.d is None:
                return (y, True, "⚠️ OUTPUT LINEAR_LE: Missing c/d")

            y_2d = y.reshape(batch_size, -1)  # (batch, n_vars)
            # c is [n_out] (single-sample) or [B, n_out] (batch-native).
            # Element-wise multiply + sum gives a per-lane lhs in both cases
            # (broadcasts when c is 1-D); plain matmul would mismatch on the
            # batch-native shape.
            lhs = (y_2d * self.c).sum(dim=1)  # (batch,)
            satisfied = lhs <= self.d.reshape(-1)  # (batch,) bool
            n_ok = satisfied.sum().item()
            explanation = f"✅ OUTPUT LINEAR_LE: {n_ok}/{batch_size} satisfied"
            return (y, satisfied, explanation)
        
        elif self.kind == OutKind.RANGE:
            if self.lb is None or self.ub is None:
                return (y, True, "⚠️ OUTPUT RANGE: Missing lb/ub")
            
            lb, ub = self.lb, self.ub
            # ensures lb.shape[0] == batch_size
            # No broadcast needed - shapes always match
            
            lb_ok = (y >= lb).reshape(batch_size, -1).all(dim=1)  # (batch,)
            ub_ok = (y <= ub).reshape(batch_size, -1).all(dim=1)  # (batch,)
            satisfied = lb_ok & ub_ok  # (batch,) bool
            
            # Unified explanation format (consistent across all batch sizes)
            n_ok = satisfied.sum().item()
            explanation = f"✅ OUTPUT RANGE: {n_ok}/{batch_size} satisfied"
            
            # Always return tensor (unified output format)
            return (y, satisfied, explanation)
        
        elif self.kind == OutKind.UNSAFE_LINEAR:
            if self.c is None or self.d is None:
                return (y, True, "⚠️ OUTPUT UNSAFE_LINEAR: Missing c/d")
            y_2d = y.reshape(batch_size, -1)
            # c shape branches:
            #   [n_out]          single-row legacy → promote to [1, n_out]
            #   [N, n_out]       single-sample, N rows
            #   [B, N, n_out]    batch-native, per-lane rows
            # d shape pairs naturally: [1] / [N] / [B, N]. y is in {Cy <= d}
            # iff ALL rows hold; safe iff NOT in the polytope.
            if self.c.dim() == 3:
                Cy = torch.einsum("bmn,bn->bm", self.c, y_2d)  # [B, N]
                d_per_lane = self.d  # [B, N]
            else:
                C = self.c if self.c.dim() == 2 else self.c.unsqueeze(0)
                Cy = y_2d @ C.T  # [B, N]
                d_per_lane = self.d.reshape(-1)  # [N]
            unsafe = (Cy <= d_per_lane).all(dim=1)
            satisfied = ~unsafe
            n_ok = satisfied.sum().item()
            explanation = f"✅ OUTPUT UNSAFE_LINEAR: {n_ok}/{batch_size} safe (UNSAFE={batch_size - n_ok})"
            return (y, satisfied, explanation)
        
        else:
            return (y, True, f"⚠️ OUTPUT: Unknown kind {self.kind}")
