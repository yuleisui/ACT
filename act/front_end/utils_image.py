
from __future__ import annotations
from typing import Tuple
import numpy as np
import torch

try:
    from PIL import Image
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False

def _to_numpy(img) -> np.ndarray:
    if isinstance(img, np.ndarray):
        return img
    if _HAS_PIL and isinstance(img, Image.Image):
        return np.array(img)
    if isinstance(img, torch.Tensor):
        t = img.detach().cpu()
        if t.ndim == 3 and t.shape[0] in (1,3):  # CHW -> HWC
            t = t.permute(1,2,0)
        return t.numpy()
    raise TypeError(f"Unsupported image type: {type(img)} (need numpy/PIL/torch)")

def to_torch_image(img, *, value_range=(0.0,1.0), device=None, dtype=torch.float32) -> torch.Tensor:
    arr = _to_numpy(img)
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype not in (np.float32, np.float64):
        arr = arr.astype(np.float32) / 255.0
    t = torch.from_numpy(arr)
    if t.ndim != 3:
        raise ValueError(f"Expected 3D image, got shape {t.shape}")
    t = t.permute(2,0,1).contiguous()  # HWC -> CHW
    t = t.to(device=device, dtype=dtype)
    lo, hi = value_range
    if (lo,hi) != (0.0,1.0):
        t = t*(hi-1.0) + lo
    return t

def resize_center_crop_chw(x: torch.Tensor, shape: Tuple[int,int,int]) -> torch.Tensor:
    import torch.nn.functional as F
    C,H,W = shape
    if x.ndim != 3:
        raise ValueError("x must be CHW")
    if x.shape[0] != C and x.shape[0] in (1,3):
        if C == 3 and x.shape[0] == 1:
            x = x.repeat(3,1,1)
        elif C == 1 and x.shape[0] == 3:
            x = x.mean(dim=0, keepdim=True)
        else:
            raise ValueError(f"Channel mismatch: got {x.shape[0]}, want {C}")
    x4 = x.unsqueeze(0)  # NCHW
    x4 = F.interpolate(x4, size=(H,W), mode="bilinear", align_corners=False)
    return x4.squeeze(0)

def chw_to_hwc_uint8(x: torch.Tensor) -> np.ndarray:
    t = x.detach().cpu().clamp(0,1)
    t = t.permute(1,2,0).contiguous().numpy()
    t = (t*255.0 + 0.5).astype(np.uint8)
    return t
