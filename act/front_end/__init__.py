"""
🧩 ACT Front-End Preprocessing Module (Self-Contained)

This front-end prepares samples, labels, and specs for DNN verification using its own spec types.
No dependencies on external verification frameworks - everything self-contained.

Key Features:
- Unified specification system (InputSpec/OutputSpec)
- Modality-specific preprocessors (ImgPre, TextPre)
- Batch processing pipeline with verification interface
- Device-aware tensor management
- Bidirectional mapping (raw ↔ model ↔ verification spaces)

Usage:
    >>> from act.front_end import ImgPre, InputSpec, OutputSpec, InKind, OutKind
    >>> from act.front_end import run_batch, SampleRecord, BatchConfig
    >>> 
    >>> # Create preprocessor
    >>> pre = ImgPre(H=32, W=32, C=3, device="cuda:0")
    >>> 
    >>> # Process samples and create verification batches
    >>> items = []  # List of SampleRecord
    >>> results = run_batch(items, pre, net, solver, verify_fn, output_dim=10)
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

# Batch processing
from act.front_end.batch import run_batch, SampleRecord, BatchConfig, ItemResult

# Utilities and mocks
from act.front_end.mocks import mock_image_sample, mock_image_specs, mock_text_sample, mock_text_specs
from act.front_end.utils_image import to_torch_image, resize_center_crop_chw, chw_to_hwc_uint8

# Demo functions
from act.front_end.demo_driver import main as run_comprehensive_demo

__all__ = [
    # Specifications
    'InputSpec', 'OutputSpec', 'InKind', 'OutKind',
    
    # Preprocessors
    'Preprocessor', 'ModelSignature', 'ImgPre', 'TextPre',
    
    # Model and data loading
    'ModelLoader', 'ModelMetadata', 'DatasetLoader', 'SpecLoader',
    
    # Device management
    'get_default_device', 'get_default_dtype', 'get_current_settings',
    
    # Batch processing
    'run_batch', 'SampleRecord', 'BatchConfig', 'ItemResult',
    
    # Utilities
    'to_torch_image', 'resize_center_crop_chw', 'chw_to_hwc_uint8',
    
    # Mocks and demos
    'mock_image_sample', 'mock_image_specs', 'mock_text_sample', 'mock_text_specs',
    'run_comprehensive_demo'
]
