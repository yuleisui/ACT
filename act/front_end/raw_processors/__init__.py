#===- act/front_end/raw_processors/__init__.py - Raw Data Processors ---====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Raw Data Processors. Contains preprocessors for different data modalities
#   including image preprocessing (ImgPre).
#
#===---------------------------------------------------------------------===#

""" 
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