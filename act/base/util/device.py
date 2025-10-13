#===- util.device.py - Device Management Utilities ----#
#
#                 ACT: Abstract Constraints Transformer
#
# Copyright (C) <2025->  ACT Team
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#
# Purpose:
# Device consistency management utilities for PyTorch tensors and models,
# providing automatic device detection and tensor movement functionality.
#
#===----------------------------------------------------------------------===#

import torch
from typing import Tuple

from act.base.util.stats import ACTLog


class DeviceConsistencyError(Exception):
    """Raised when device consistency issues are detected."""
    pass


class DeviceManager:
    """Device consistency management for tensor operations."""
    
    @staticmethod
    def ensure_device_consistency(model: torch.nn.Module, 
                                *tensors: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Ensure all tensors are on the same device as the model.
        
        Args:
            model: PyTorch model to get device from
            *tensors: Variable number of tensors to move to model device
            
        Returns:
            Tuple of tensors moved to the correct device
            
        Raises:
            DeviceConsistencyError: If device movement fails
        """
        try:
            device = next(model.parameters()).device
            moved_tensors = []
            
            for tensor in tensors:
                if tensor.device != device:
                    ACTLog.log_verification_info(f"Moving tensor from {tensor.device} to {device}")
                    moved_tensors.append(tensor.to(device))
                else:
                    moved_tensors.append(tensor)
            
            return tuple(moved_tensors) if len(moved_tensors) > 1 else moved_tensors[0]
            
        except Exception as e:
            raise DeviceConsistencyError(f"Failed to ensure device consistency: {e}") from e