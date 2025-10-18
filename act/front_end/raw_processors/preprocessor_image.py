
#===- act/front_end/raw_processors/preprocessor_image.py - Image Processor ====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Image Preprocessor. Provides image preprocessing capabilities including
#   normalization, resizing, and format conversion for image-based verification
#   tasks in the ACT framework.
#
#===---------------------------------------------------------------------===#

from __future__ import annotations
from typing import Optional, Tuple, Any
import torch
from pathlib import Path
from act.util.device_manager import get_default_device, get_default_dtype
from act.front_end.raw_processors.preprocessor_base import Preprocessor, ModelSignature
from act.front_end.specs import InputSpec, OutputSpec, InKind, OutKind
from act.front_end.utils_image import to_torch_image, resize_center_crop_chw, chw_to_hwc_uint8

class ImgPre(Preprocessor):
    def __init__(self, H: int, W: int, C: int = 3,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        device = get_default_device()
        dtype = get_default_dtype()
        # Handle mean/std for different channel counts
        if C == 1:
            mean = mean[0] if isinstance(mean, (tuple, list)) else mean
            std = std[0] if isinstance(std, (tuple, list)) else std
            mean, std = (mean,), (std,)
        elif C == 3:
            if len(mean) == 1:
                mean = (mean[0], mean[0], mean[0])
            if len(std) == 1:
                std = (std[0], std[0], std[0])
        
        sig = ModelSignature(modality="image", layout="NCHW", input_shape=(C, H, W),
                             meta={"mean": mean, "std": std})
        super().__init__(sig)
        self.C, self.H, self.W = C, H, W
        self.mean = torch.tensor(mean, device=self.device, dtype=dtype).view(C,1,1)
        self.std  = torch.tensor(std,  device=self.device, dtype=dtype).view(C,1,1)

    def prepare_sample(self, sample) -> torch.Tensor:
        x = to_torch_image(sample, device=self.device, dtype=self.dtype)  # CHW in [0,1]
        x = resize_center_crop_chw(x, (self.C, self.H, self.W))
        x = (x - self.mean) / self.std
        return x

    def prepare_label(self, label) -> int:
        if isinstance(label, int):
            return int(label)
        if isinstance(label, str):
            return int(label) if label.isdigit() else 0
        raise TypeError(f"Unsupported label type: {type(label)}")

    def load_raw_sample_label_pairs(self, file_path: str) -> Tuple[torch.Tensor, int]:
        """Load image and extract label from filename.
        
        Args:
            file_path: Path to image file or directory containing images
            
        Returns:
            Tuple of (prepared_image_tensor, prepared_label)
        """
        file_path = Path(file_path)
        
        # If it's a directory, look for images in data/raw/images
        if file_path.is_dir() or not file_path.exists():
            image_dir = Path("data/raw/images")
            if not image_dir.exists():
                raise FileNotFoundError(f"Image directory {image_dir} does not exist")
            
            # Find image files with supported extensions
            image_extensions = ['.png', '.jpg', '.jpeg']
            image_files = []
            for ext in image_extensions:
                image_files.extend(image_dir.glob(f"*{ext}"))
                image_files.extend(image_dir.glob(f"*{ext.upper()}"))
            
            if not image_files:
                raise FileNotFoundError(f"No image files found in {image_dir}")
            
            # Use the first image found
            file_path = image_files[0]
        
        # Check if file exists and has valid extension
        if not file_path.exists():
            raise FileNotFoundError(f"Image file {file_path} does not exist")
        
        image_extensions = ['.png', '.jpg', '.jpeg']
        if file_path.suffix.lower() not in image_extensions:
            raise ValueError(f"Unsupported image extension: {file_path.suffix}")
        
        # Load image using PIL
        try:
            from PIL import Image
            with Image.open(file_path) as img:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Prepare sample using existing method
                sample_tensor = self.prepare_sample(img)
                
                # Extract label from filename (without extension)
                label_str = file_path.stem
                prepared_label = self.prepare_label(label_str)
                
                return sample_tensor, prepared_label
                
        except ImportError:
            raise ImportError("PIL (Pillow) is required for image loading. Install with: pip install Pillow")
        except Exception as e:
            raise RuntimeError(f"Failed to load image {file_path}: {e}")

    def _apply_affine(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def _prep_componentwise(self, arr) -> torch.Tensor:
        t = to_torch_image(arr, device=self.device, dtype=self.dtype)
        t = resize_center_crop_chw(t, (self.C, self.H, self.W))
        return self._apply_affine(t)

    def canonicalize_input_spec(self, input_spec_raw: InputSpec, *, center=None, eps: Optional[float]=None) -> InputSpec:
        if input_spec_raw.kind == InKind.BOX:
            lb = self._prep_componentwise(input_spec_raw.lb)
            ub = self._prep_componentwise(input_spec_raw.ub)
            return InputSpec(kind=InKind.BOX, lb=lb, ub=ub)
        if input_spec_raw.kind == InKind.LINF_BALL:
            if center is None:
                raise ValueError("center sample is required for LINF_BALL canonicalization.")
            c = self.prepare_sample(center)
            e = torch.as_tensor(input_spec_raw.eps if input_spec_raw.eps is not None else eps,
                                device=self.device, dtype=self.dtype)
            eprime = e / self.std
            eps_scalar = float(torch.max(torch.abs(eprime)))
            return InputSpec(kind=InKind.LINF_BALL, center=c, eps=eps_scalar)
        if input_spec_raw.kind == InKind.LIN_POLY:
            if input_spec_raw.A is None or input_spec_raw.b is None:
                raise ValueError("LIN_POLY requires A and b.")
            A = input_spec_raw.A.to(self.device, self.dtype)
            b = input_spec_raw.b.to(self.device, self.dtype)
            return InputSpec(kind=InKind.LIN_POLY, A=A, b=b)
        raise NotImplementedError(input_spec_raw.kind)

    def canonicalize_output_spec(self, output_spec_raw: OutputSpec, *, label=None) -> OutputSpec:
        if output_spec_raw.kind in (OutKind.TOP1_ROBUST, OutKind.MARGIN_ROBUST):
            y_true = self.prepare_label(label if label is not None else output_spec_raw.y_true)
            return OutputSpec(kind=OutKind.MARGIN_ROBUST, y_true=int(y_true),
                              margin=float(output_spec_raw.margin if output_spec_raw.margin is not None else 0.0))
        if output_spec_raw.kind == OutKind.LINEAR_LE:
            c = output_spec_raw.c.to(self.device, self.dtype) if output_spec_raw.c is not None else None
            return OutputSpec(kind=OutKind.LINEAR_LE, c=c, d=output_spec_raw.d)
        if output_spec_raw.kind == OutKind.RANGE:
            lb = output_spec_raw.lb.to(self.device, self.dtype) if output_spec_raw.lb is not None else None
            ub = output_spec_raw.ub.to(self.device, self.dtype) if output_spec_raw.ub is not None else None
            return OutputSpec(kind=OutKind.RANGE, lb=lb, ub=ub)
        return output_spec_raw

    def flatten_model_input(self, x: torch.Tensor):
        return x.contiguous().view(-1).detach().cpu().numpy()

    def unflatten_to_model_input(self, flat):
        return torch.from_numpy(flat).view(self.C,self.H,self.W).to(self.device, self.dtype)

    def inverse_to_raw_space(self, x_model: torch.Tensor):
        x = x_model * self.std + self.mean
        return chw_to_hwc_uint8(x)
