"""
🧩 ACT Front-End Preprocessing Module (Self-Contained)

This front-end prepares samples, labels, and specs for DNN verification using its own spec types.
No dependencies on external verification frameworks - everything self-contained.

Key Features:
- Unified specification system (InputSpec/OutputSpec)
- Modality-specific preprocessors (ImgPre, TextPre)
- Device-aware tensor management
- Bidirectional mapping (raw ↔ model ↔ verification spaces)

Usage:
    >>> from act.front_end import ImgPre, InputSpec, OutputSpec, InKind, OutKind
    >>> 
    >>> # Create preprocessor
    >>> pre = ImgPre(H=32, W=32, C=3, device="cuda:0")
    >>> 
    >>> # Process samples and create specifications
    >>> input_spec = InputSpec(kind=InKind.LINF_BALL, center=data, eps=0.1)
"""

# Core specification system
from act.front_end.specs import InputSpec, OutputSpec, InKind, OutKind

# Preprocessors
from act.front_end.preprocessor_base import Preprocessor, ModelSignature  
from act.front_end.preprocessor_image import ImgPre
from act.front_end.preprocessor_text import TextPre

# Model and data loading
from act.front_end.loaders import ModelLoader, ModelMetadata, DatasetLoader, SpecLoader

# Device management
from act.front_end.device_manager import get_default_device, get_default_dtype, get_current_settings

# Utilities and mocks
from act.front_end.mocks import (
    mock_wrapped_mlp_mnist, mock_wrapped_cnn_mnist, mock_wrapped_mlp_cifar,
    mock_wrapped_models_collection
)
from act.front_end.utils_image import to_torch_image, resize_center_crop_chw, chw_to_hwc_uint8

__all__ = [
    # Specifications
    'InputSpec', 'OutputSpec', 'InKind', 'OutKind',
    
    # Preprocessors
    'Preprocessor', 'ModelSignature', 'ImgPre', 'TextPre',
    
    # Model and data loading
    'ModelLoader', 'ModelMetadata', 'DatasetLoader', 'SpecLoader',
    
    # Device management
    'get_default_device', 'get_default_dtype', 'get_current_settings',
    
    # Utilities
    'to_torch_image', 'resize_center_crop_chw', 'chw_to_hwc_uint8',
       
    # Mocks - Wrapped Models for torch2act
    'mock_wrapped_mlp_mnist', 'mock_wrapped_cnn_mnist', 'mock_wrapped_mlp_cifar',
    'mock_wrapped_models_collection'
]
