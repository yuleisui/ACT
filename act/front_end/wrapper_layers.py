"""
🎭 Wrapper Layers for ACT-PyTorch Integration

PyTorch nn.Module wrappers that bridge PyTorch models with ACT verification framework.
These layers provide the essential PyTorch compatibility while enabling conversion to ACT format.
"""

from __future__ import annotations
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple, Union

# Import ACT components
from act.front_end.specs import InputSpec, OutputSpec, InKind, OutKind
from act.back_end.layer_schema import LayerKind, REGISTRY
from act.back_end.layer_validation import create_layer
from act.util.device_manager import get_default_device, get_default_dtype


def prod(seq: Tuple[int, ...]) -> int:
    """Helper function to compute product of shape dimensions."""
    p = 1
    for s in seq:
        p *= s
    return p


class InputLayer(nn.Module):
    """Declares the symbolic input block (shape/optional center). No-op at inference."""
    def __init__(self, shape: Tuple[int, ...], center: Optional[torch.Tensor] = None, desc: str = "input"):
        super().__init__()
        assert shape[0] == 1, "Verification wrapper assumes batch=1."
        self.shape = tuple(shape)
        self.desc = desc
        
        # GPU-first device management
        if center is not None:
            center = center.to(device=get_default_device(), dtype=get_default_dtype())
            self.register_buffer("center", center.reshape(-1))
        else:
            self.center = None
        self._validate_schema()

    def _validate_schema(self):
        """Validate parameters against INPUT layer schema"""
        schema = REGISTRY[LayerKind.INPUT.value]
        params = {}
        if self.center is not None:
            params["center"] = self.center
        meta = {"shape": self.shape}
        if self.desc != "input":
            meta["desc"] = self.desc
        
        # Check required/optional params and meta
        for key in schema["params_required"]:
            if key not in params:
                raise ValueError(f"InputLayer missing required param: {key}")
        for key in params:
            if key not in schema["params_required"] + schema["params_optional"]:
                raise ValueError(f"InputLayer has unknown param: {key}")
        for key in schema["meta_required"]:
            if key not in meta:
                raise ValueError(f"InputLayer missing required meta: {key}")
        for key in meta:
            if key not in schema["meta_required"] + schema["meta_optional"]:
                raise ValueError(f"InputLayer has unknown meta: {key}")

    def to_act_layers(self, layer_id_start: int, in_vars: List[int]) -> Tuple[List, List[int]]:
        """Convert to ACT Layer(s) and return (layers, out_vars)"""
        N = prod(self.shape[1:])
        out_vars = list(range(len(in_vars), len(in_vars) + N))
        
        params = {}
        if self.center is not None:
            params["center"] = self.center
        meta = {"shape": self.shape}
        if self.desc != "input":
            meta["desc"] = self.desc
        
        layer = create_layer(
            id=layer_id_start,
            kind=LayerKind.INPUT.value,
            params=params,
            meta=meta,
            in_vars=in_vars,
            out_vars=out_vars
        )
        return [layer], out_vars

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class InputAdapterLayer(nn.Module):
    """
    General, config-driven input adapter. Applies any subset of:
      - permute over non-batch dims (e.g., HWC->CHW)
      - reorder indices (gather)
      - slice indices (subset)
      - pad constants (append features)
      - per-element affine: z = a ⊙ x + c (a,c scalar or per-element)
      - linear projection: z = A x + b
    """
    def __init__(
        self,
        permute_axes: Optional[Tuple[int, ...]] = None,
        reorder_idx: Optional[torch.Tensor] = None,
        slice_idx: Optional[torch.Tensor] = None,
        pad_values: Optional[torch.Tensor] = None,
        affine_a: Optional[torch.Tensor | float] = None,
        affine_c: Optional[torch.Tensor | float] = None,
        linproj_A: Optional[torch.Tensor] = None,
        linproj_b: Optional[torch.Tensor] = None,
        reshape_to: Optional[Tuple[int, ...]] = None,  # New: target shape for model input (excluding batch)
        adapt_channels: Optional[str] = None,  # New: channel adaptation strategy
    ):
        super().__init__()
        self.permute_axes = permute_axes
        self.reorder_idx = reorder_idx
        self.slice_idx = slice_idx
        self.pad_values = pad_values
        self.affine_a = affine_a
        self.affine_c = affine_c
        self.linproj_A = linproj_A
        self.linproj_b = linproj_b
        self.reshape_to = reshape_to
        self.adapt_channels = adapt_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x
        B = t.shape[0]

        # 1) Permute over non-batch dims
        if self.permute_axes is not None and t.dim() >= 2:
            axes = (0,) + tuple(a + 1 for a in self.permute_axes)
            t = t.permute(*axes)

        # 2) Flatten to features for index ops
        t = t.reshape(B, -1)

        # 3) Reorder / slice / pad
        if self.reorder_idx is not None:
            t = t.index_select(1, self.reorder_idx.to(t.device))
        if self.slice_idx is not None:
            t = t.index_select(1, self.slice_idx.to(t.device))
        if self.pad_values is not None:
            pad = self.pad_values.to(t.device, t.dtype).reshape(1, -1).expand(B, -1)
            t = torch.cat([t, pad], dim=1)

        # 4) Elementwise affine
        if self.affine_a is not None:
            a = torch.as_tensor(self.affine_a, device=t.device, dtype=t.dtype)
            if a.numel() == 1:
                a = a.expand_as(t)
            t = t * a
        if self.affine_c is not None:
            c = torch.as_tensor(self.affine_c, device=t.device, dtype=t.dtype)
            if c.numel() == 1:
                c = c.expand_as(t)
            t = t + c

        # 5) Linear projection
        if self.linproj_A is not None:
            A = self.linproj_A.to(t.device, t.dtype)  # [M, N]
            t = t @ A.t()
            if self.linproj_b is not None:
                t = t + self.linproj_b.to(t.device, t.dtype)

        # 6) Final reshaping for model compatibility (e.g., flatten -> image shape for CNNs)
        if self.reshape_to is not None:
            target_shape = (B,) + self.reshape_to
            t = t.reshape(target_shape)

        # 7) Channel adaptation for model compatibility
        if self.adapt_channels is not None and t.dim() == 4:  # Only for image tensors (B, C, H, W)
            B, C, H, W = t.shape
            if self.adapt_channels == "1to3" and C == 1:
                # Convert 1-channel (grayscale) to 3-channel (RGB) by replicating
                t = t.repeat(1, 3, 1, 1)
            elif self.adapt_channels == "3to1" and C == 3:
                # Convert 3-channel (RGB) to 1-channel (grayscale) by averaging
                t = t.mean(dim=1, keepdim=True)
            elif self.adapt_channels == "1to3_pad" and C == 1:
                # Convert 1-channel to 3-channel by padding with zeros
                zeros = torch.zeros(B, 2, H, W, device=t.device, dtype=t.dtype)
                t = torch.cat([t, zeros], dim=1)
            elif self.adapt_channels == "resize" and (C != self.reshape_to[0] if self.reshape_to else False):
                # Generic resize by interpolation (more advanced)
                target_channels = self.reshape_to[0] if self.reshape_to else C
                if C < target_channels:
                    # Repeat channels to match target
                    repeat_factor = target_channels // C
                    remainder = target_channels % C
                    t_repeated = t.repeat(1, repeat_factor, 1, 1)
                    if remainder > 0:
                        t_extra = t[:, :remainder, :, :]
                        t = torch.cat([t_repeated, t_extra], dim=1)
                elif C > target_channels:
                    # Take subset of channels
                    t = t[:, :target_channels, :, :]

        return t

    def to_act_layers(self, layer_id_start: int, in_vars: List[int]) -> Tuple[List, List[int]]:
        """Convert InputAdapterLayer to multiple ACT layers"""
        from act.back_end.core import Layer
        from act.back_end.layer_schema import LayerKind
        from act.back_end.layer_validation import create_layer
        
        layers = []
        current_vars = in_vars
        current_id = layer_id_start
        
        # Convert each adapter operation to corresponding ACT layer
        if self.permute_axes is not None:
            params = {}
            meta = {"perm": self.permute_axes}
            layer = create_layer(
                id=current_id,
                kind=LayerKind.PERMUTE.value,
                params=params,
                meta=meta,
                in_vars=current_vars,
                out_vars=current_vars.copy()  # Same size
            )
            layers.append(layer)
            current_id += 1
        
        if self.reorder_idx is not None:
            params = {"idx": self.reorder_idx}
            meta = {}
            layer = create_layer(
                id=current_id,
                kind=LayerKind.REORDER.value,
                params=params,
                meta=meta,
                in_vars=current_vars,
                out_vars=current_vars.copy()  # Same size for reorder
            )
            layers.append(layer)
            current_id += 1
        
        if self.slice_idx is not None:
            params = {"idx": self.slice_idx}
            meta = {}
            new_vars = list(range(len(current_vars), len(current_vars) + len(self.slice_idx)))
            layer = create_layer(
                id=current_id,
                kind=LayerKind.SLICE.value,
                params=params,
                meta=meta,
                in_vars=current_vars,
                out_vars=new_vars
            )
            layers.append(layer)
            current_vars = new_vars
            current_id += 1
        
        if self.pad_values is not None:
            params = {"values": self.pad_values}
            meta = {}
            new_size = len(current_vars) + len(self.pad_values)
            new_vars = list(range(len(current_vars), len(current_vars) + new_size))
            layer = create_layer(
                id=current_id,
                kind=LayerKind.PAD.value,
                params=params,
                meta=meta,
                in_vars=current_vars,
                out_vars=new_vars
            )
            layers.append(layer)
            current_vars = new_vars
            current_id += 1
        
        if self.affine_a is not None or self.affine_c is not None:
            params = {}
            if self.affine_a is not None:
                params["scale"] = torch.as_tensor(self.affine_a)
            if self.affine_c is not None:
                params["shift"] = torch.as_tensor(self.affine_c)
            meta = {}
            layer = create_layer(
                id=current_id,
                kind=LayerKind.SCALE_SHIFT.value,
                params=params,
                meta=meta,
                in_vars=current_vars,
                out_vars=current_vars.copy()  # Same size
            )
            layers.append(layer)
            current_id += 1
        
        if self.linproj_A is not None:
            params = {"A": self.linproj_A}
            if self.linproj_b is not None:
                params["b"] = self.linproj_b
            meta = {}
            new_size = self.linproj_A.shape[0]
            new_vars = list(range(len(current_vars), len(current_vars) + new_size))
            layer = create_layer(
                id=current_id,
                kind=LayerKind.LINEAR_PROJ.value,
                params=params,
                meta=meta,
                in_vars=current_vars,
                out_vars=new_vars
            )
            layers.append(layer)
            current_vars = new_vars
            current_id += 1
        
        return layers, current_vars


class InputSpecLayer(nn.Module):
    """
    Wraps ACT's InputSpec AND is an nn.Module. No-op in forward; used by converters.
    The spec it carries should already be EXPRESSED IN POST-ADAPTER SPACE.
    """
    def __init__(self, spec: Optional[InputSpec] = None, **kwargs):
        super().__init__()
        self.spec = spec or InputSpec(**kwargs)
        self.kind = self.spec.kind
        self.eps = float(self.spec.eps) if self.spec.eps is not None else None

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
        params = {}
        for name in ("lb", "ub", "center", "A", "b"):
            val = getattr(self, name, None)
            if val is not None:
                params[name] = val
        meta = {"kind": self.kind}
        if self.eps is not None:
            meta["eps"] = self.eps
        
        # Check schema compliance
        for key in schema["meta_required"]:
            if key not in meta:
                raise ValueError(f"InputSpecLayer missing required meta: {key}")
        for key in meta:
            if key not in schema["meta_required"] + schema["meta_optional"]:
                raise ValueError(f"InputSpecLayer has unknown meta: {key}")

    def to_act_layers(self, layer_id_start: int, in_vars: List[int]) -> Tuple[List, List[int]]:
        """Convert to ACT Layer(s) - INPUT_SPEC doesn't create new vars"""
        params = {}
        for name in ("lb", "ub", "center", "A", "b"):
            val = getattr(self, name, None)
            if val is not None:
                params[name] = val
        meta = {"kind": self.kind}
        if self.eps is not None:
            meta["eps"] = self.eps
        
        layer = create_layer(
            id=layer_id_start,
            kind=LayerKind.INPUT_SPEC.value,
            params=params,
            meta=meta,
            in_vars=in_vars,
            out_vars=in_vars  # INPUT_SPEC doesn't change variables
        )
        return [layer], in_vars

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class OutputSpecLayer(nn.Module):
    """
    Wraps ACT's OutputSpec AND is an nn.Module. No-op in forward; used by converters.
    """
    def __init__(self, spec: Optional[OutputSpec] = None, **kwargs):
        super().__init__()
        self.spec = spec or OutputSpec(**kwargs)
        self.kind = self.spec.kind
        self.y_true = self.spec.y_true
        self.margin = float(self.spec.margin)
        self.d = None if self.spec.d is None else float(self.spec.d)
        self.meta = dict(self.spec.meta)

        for name in ("c", "lb", "ub"):
            val = getattr(self.spec, name, None)
            if isinstance(val, torch.Tensor):
                self.register_buffer(name, val)
            else:
                setattr(self, name, None)
        self._validate_schema()

    def _validate_schema(self):
        """Validate parameters against ASSERT layer schema"""
        schema = REGISTRY[LayerKind.ASSERT.value]
        params = {}
        for name in ("c", "lb", "ub"):
            val = getattr(self, name, None)
            if val is not None:
                params[name] = val
        meta = {"kind": self.kind}
        if self.y_true is not None:
            meta["y_true"] = self.y_true
        if self.margin is not None:
            meta["margin"] = self.margin
        if self.d is not None:
            meta["d"] = self.d
        
        # Check schema compliance
        for key in schema["meta_required"]:
            if key not in meta:
                raise ValueError(f"OutputSpecLayer missing required meta: {key}")

    def to_act_layers(self, layer_id_start: int, in_vars: List[int]) -> Tuple[List, List[int]]:
        """Convert to ACT Layer(s) - ASSERT doesn't create new vars"""
        params = {}
        for name in ("c", "lb", "ub"):
            val = getattr(self, name, None)
            if val is not None:
                params[name] = val
        meta = {"kind": self.kind}
        if self.y_true is not None:
            meta["y_true"] = self.y_true
        if self.margin is not None:
            meta["margin"] = self.margin
        if self.d is not None:
            meta["d"] = self.d
        
        layer = create_layer(
            id=layer_id_start,
            kind=LayerKind.ASSERT.value,
            params=params,
            meta=meta,
            in_vars=in_vars,
            out_vars=in_vars  # ASSERT doesn't change variables
        )
        return [layer], in_vars

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return y