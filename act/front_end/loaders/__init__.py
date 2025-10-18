#===- act/front_end/loaders/__init__.py - Front-End Loaders Module -----====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Front-End Loaders Module. Contains the three core loaders for the
#   front-end integration: ModelLoader for ONNX model loading and
#   preprocessor auto-creation.
#
#===---------------------------------------------------------------------===#

"""
- DatasetLoader: Pure data loading (CSV, images, VNNLIB anchors)
- SpecLoader: Specification generation and data+spec combination

These loaders provide automated setup for verification pipelines while
maintaining compatibility with existing front-end preprocessors and batch processing.
"""

from act.front_end.loaders.model_loader import ModelLoader, ModelMetadata
from act.front_end.loaders.data_loader import DatasetLoader
from act.front_end.loaders.spec_loader import SpecLoader

__all__ = [
    'ModelLoader', 'ModelMetadata',
    'DatasetLoader', 
    'SpecLoader'
]