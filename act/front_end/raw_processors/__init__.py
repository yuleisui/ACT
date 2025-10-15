"""
Raw Data Processors

This module contains preprocessors for different data modalities:
- Image preprocessing (ImgPre) 
- Text preprocessing (TextPre)
- Base preprocessor interface (Preprocessor, ModelSignature)

These processors handle the conversion from raw data to model-ready tensors
and provide canonicalization of input/output specifications for verification.
"""

from .preprocessor_base import Preprocessor, ModelSignature
from .preprocessor_image import ImgPre
from .preprocessor_text import TextPre, SimpleTokenizer

__all__ = [
    'Preprocessor',
    'ModelSignature', 
    'ImgPre',
    'TextPre',
    'SimpleTokenizer'
]