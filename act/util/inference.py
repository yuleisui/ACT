#===- util.inference.py neural network model inference utilities -----------#
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
#   Utility functions for neural network model inference and prediction validation.
#   Provides common inference operations used across different verifiers.
#
#===----------------------------------------------------------------------===#

from typing import Dict
import torch

from util.stats import ACTLog


def perform_model_inference(
    model: torch.nn.Module,
    sample_tensor: torch.Tensor, 
    ground_truth_label: int, 
    input_adaptor,
    prediction_stats: Dict[str, int],
    sample_index: int = 0,
    verbose: bool = True
) -> bool:
    """
    Perform model inference on a sample tensor and validate prediction accuracy.
    
    Executes forward pass on a concrete sample tensor and compares prediction with ground truth. 
    Essential validation before verification attempts. Expects sample tensor to be already normalized.
    
    Args:
        model: PyTorch model for inference
        sample_tensor: Sample tensor (must be normalized with proper batch dimension)
        ground_truth_label: Expected correct classification label
        input_adaptor: Input adaptor for validation checks
        prediction_stats: Dictionary to update with prediction statistics
        sample_index: Sample index for logging (default: 0)
        verbose: Whether to enable verbose logging (default: True)
        
    Returns:
        True if model prediction matches ground truth, False otherwise
    """
    prediction_stats['total_samples'] += 1
    
    try:
        # Assert that sample tensor is properly normalized
        assert input_adaptor.is_sample_tensor_properly_normalized(sample_tensor), \
            "Sample tensor must be properly normalized for model inference"
            
        try:
            with torch.no_grad():
                model.eval()
                outputs = model(sample_tensor)
                predicted = torch.argmax(outputs, dim=1).item()
        except Exception as e:
            raise ValueError(f"Model inference failed: {e}") from e
        
        # Evaluate prediction accuracy
        is_correct = (predicted == ground_truth_label)
        
        if is_correct:
            prediction_stats['clean_correct'] += 1
            ACTLog.log_correct_prediction(predicted, ground_truth_label, sample_index, verbose)
        else:
            prediction_stats['clean_incorrect'] += 1
            ACTLog.log_incorrect_prediction(predicted, ground_truth_label, sample_index, verbose)
            
        return is_correct
        
    except (RuntimeError, ValueError) as inference_error:
        ACTLog.log_prediction_failure(inference_error)
        prediction_stats['clean_incorrect'] += 1
        return False