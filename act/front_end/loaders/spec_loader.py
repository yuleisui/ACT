"""
📋 Specification Loading for Front-End Integration

Specification generation and loading - separate from data loading.
Handles input/output specification creation and VNNLIB constraint parsing.
"""

from __future__ import annotations
import os
import numpy as np
import torch
from typing import List, Tuple, Dict, Any, Optional, Union
from pathlib import Path

from act.front_end.specs import InputSpec, OutputSpec, InKind, OutKind
from act.front_end.batch import SampleRecord
from act.front_end.preprocessor_base import Preprocessor


class SpecLoader:
    """Specification generation and loading"""
    
    def __init__(self, preprocessor: Optional[Preprocessor] = None):
        """
        Initialize SpecLoader
        
        Args:
            preprocessor: Optional preprocessor for model-space spec generation
        """
        self.preprocessor = preprocessor
        
    def create_input_specs(self, samples: List[Any], 
                          spec_config: Dict[str, Any]) -> List[InputSpec]:
        """
        Generate input specifications for raw samples
        
        Args:
            samples: List of raw samples (numpy arrays, PIL Images, etc.)
            spec_config: Configuration dictionary with spec parameters
                - type: "linf_ball", "l2_ball", "box", "custom"
                - epsilon: perturbation radius (for ball specs)
                - bounds: custom bounds (for box specs)
                - norm_type: normalization before applying constraints
                
        Returns:
            List of InputSpec objects
        """
        spec_type = spec_config.get("type", "linf_ball")
        epsilon = spec_config.get("epsilon", 0.1)
        
        input_specs = []
        
        for i, sample in enumerate(samples):
            # Convert sample to tensor if needed
            if self.preprocessor:
                # Use preprocessor to convert to model space
                if hasattr(sample, 'shape'):  # numpy array
                    sample_tensor = torch.from_numpy(sample).float()
                else:  # PIL Image or other format
                    sample_tensor = self.preprocessor.prepare_sample(sample)
            else:
                # Raw sample processing
                if not isinstance(sample, torch.Tensor):
                    if hasattr(sample, 'shape'):  # numpy array
                        sample_tensor = torch.from_numpy(sample).float()
                    else:
                        raise ValueError(f"Cannot process sample type {type(sample)} without preprocessor")
                else:
                    sample_tensor = sample
                    
            # Generate specification based on type
            if spec_type == "linf_ball":
                input_spec = self._create_linf_ball_spec(sample_tensor, epsilon)
            elif spec_type == "l2_ball":
                radius = spec_config.get("radius", epsilon)
                input_spec = self._create_l2_ball_spec(sample_tensor, radius)
            elif spec_type == "box":
                bounds = spec_config.get("bounds", None)
                input_spec = self._create_box_spec(sample_tensor, bounds, epsilon)
            elif spec_type == "custom":
                # Custom specification from provided bounds
                lb = spec_config.get("lower_bound")
                ub = spec_config.get("upper_bound")
                if lb is None or ub is None:
                    raise ValueError("Custom spec requires 'lower_bound' and 'upper_bound'")
                input_spec = InputSpec(kind=InKind.BOX, 
                                     lb=torch.tensor(lb), 
                                     ub=torch.tensor(ub))
            else:
                raise ValueError(f"Unsupported input spec type: {spec_type}")
                
            input_specs.append(input_spec)
            
        print(f"📋 Generated {len(input_specs)} input specs of type {spec_type}")
        return input_specs
        
    def _create_linf_ball_spec(self, center: torch.Tensor, epsilon: float) -> InputSpec:
        """Create L∞ ball specification around center point"""
        return InputSpec(
            kind=InKind.LINF_BALL,
            center=center.clone(),
            eps=epsilon
        )
        
    def _create_l2_ball_spec(self, center: torch.Tensor, radius: float) -> InputSpec:
        """Create L2 ball specification around center point"""
        return InputSpec(
            kind=InKind.L2_BALL,
            center=center.clone(),
            eps=radius
        )
        
    def _create_box_spec(self, center: torch.Tensor, 
                        bounds: Optional[Tuple[float, float]] = None,
                        epsilon: float = 0.1) -> InputSpec:
        """Create box specification around center point"""
        if bounds:
            lb_val, ub_val = bounds
            lb = torch.full_like(center, lb_val)
            ub = torch.full_like(center, ub_val)
        else:
            # Create epsilon-box around center
            lb = center - epsilon
            ub = center + epsilon
            
        return InputSpec(kind=InKind.BOX, lb=lb, ub=ub)
        
    def create_output_specs(self, labels: List[int], 
                           spec_config: Dict[str, Any]) -> List[OutputSpec]:
        """
        Generate output specifications for labels
        
        Args:
            labels: List of ground truth labels
            spec_config: Configuration dictionary with spec parameters
                - output_type: "margin_robust", "top1_robust", "linear_le", "range"
                - margin: robustness margin (for margin_robust)
                - target_class: target class for adversarial specs
                
        Returns:
            List of OutputSpec objects
        """
        output_type = spec_config.get("output_type", "margin_robust")
        margin = spec_config.get("margin", 0.0)
        
        output_specs = []
        
        for label in labels:
            if output_type == "margin_robust":
                output_spec = OutputSpec(
                    kind=OutKind.MARGIN_ROBUST,
                    y_true=label,
                    margin=margin
                )
            elif output_type == "top1_robust":
                # Top-1 robustness is equivalent to margin robustness with margin=0
                output_spec = OutputSpec(
                    kind=OutKind.MARGIN_ROBUST,
                    y_true=label,
                    margin=0.0
                )
            elif output_type == "linear_le":
                # Linear constraint: coeffs^T * output <= bound
                coeffs = spec_config.get("coeffs")
                bound = spec_config.get("bound", 0.0)
                if coeffs is None:
                    raise ValueError("linear_le spec requires 'coeffs' parameter")
                output_spec = OutputSpec(
                    kind=OutKind.LINEAR_LE,
                    coeffs=torch.tensor(coeffs),
                    bound=bound
                )
            elif output_type == "range":
                # Output range constraints
                lower = spec_config.get("lower", -float('inf'))
                upper = spec_config.get("upper", float('inf'))
                output_spec = OutputSpec(
                    kind=OutKind.RANGE,
                    lower=lower,
                    upper=upper
                )
            else:
                raise ValueError(f"Unsupported output spec type: {output_type}")
                
            output_specs.append(output_spec)
            
        print(f"📋 Generated {len(output_specs)} output specs of type {output_type}")
        return output_specs
        
    def load_vnnlib_specs(self, vnnlib_path: str) -> List[Tuple[InputSpec, OutputSpec]]:
        """
        Load specifications from VNNLIB file
        
        Args:
            vnnlib_path: Path to .vnnlib specification file
            
        Returns:
            List of (InputSpec, OutputSpec) pairs
            
        Note:
            This integrates with existing VNNLIB parsing from the verifier module.
            For now, this is a placeholder that returns basic specs.
        """
        if not os.path.exists(vnnlib_path):
            raise FileNotFoundError(f"VNNLIB file not found: {vnnlib_path}")
            
        print(f"📋 Loading VNNLIB specifications from {vnnlib_path}")
        
        try:
            # TODO: Integrate with existing vnnlib_parser from verifier module
            # For now, create placeholder specs
            # Real implementation would:
            # 1. Parse VNNLIB file using existing parser
            # 2. Extract input constraints (variable bounds, linear combinations)
            # 3. Extract output properties (safety conditions)
            # 4. Convert to InputSpec/OutputSpec format
            
            # Placeholder: create a simple spec
            # This should be replaced with actual VNNLIB parsing
            print("⚠️  VNNLIB parsing not yet fully implemented")
            print("📋 Creating placeholder specifications")
            
            # Create dummy spec for demo
            dummy_input_spec = InputSpec(
                kind=InKind.BOX,
                lb=torch.zeros(784),  # Placeholder dimensions
                ub=torch.ones(784)
            )
            
            dummy_output_spec = OutputSpec(
                kind=OutKind.LINEAR_LE,
                coeffs=torch.tensor([1.0, -1.0]),  # Placeholder coefficients
                bound=0.0
            )
            
            specs = [(dummy_input_spec, dummy_output_spec)]
            
            print(f"📋 Loaded {len(specs)} VNNLIB specification pairs")
            return specs
            
        except Exception as e:
            raise ValueError(f"Failed to load VNNLIB specs from {vnnlib_path}: {e}")
            
    def combine_data_and_specs(self, data_pairs: List[Tuple[Any, int]], 
                              input_specs: List[InputSpec], 
                              output_specs: List[OutputSpec]) -> List[SampleRecord]:
        """
        Combine data with specifications into SampleRecord objects
        
        Args:
            data_pairs: List of (raw_sample, label) tuples
            input_specs: List of InputSpec objects (same length as data_pairs)
            output_specs: List of OutputSpec objects (same length as data_pairs)
            
        Returns:
            List of complete SampleRecord objects ready for verification
        """
        if not (len(data_pairs) == len(input_specs) == len(output_specs)):
            raise ValueError(f"Length mismatch: data={len(data_pairs)}, "
                           f"input_specs={len(input_specs)}, output_specs={len(output_specs)}")
            
        sample_records = []
        
        for i, ((sample_raw, label_raw), input_spec, output_spec) in enumerate(
            zip(data_pairs, input_specs, output_specs)
        ):
            sample_record = SampleRecord(
                idx=i,
                sample_raw=sample_raw,
                label_raw=label_raw,
                in_spec_raw=input_spec,
                out_spec_raw=output_spec
            )
            sample_records.append(sample_record)
            
        print(f"📦 Created {len(sample_records)} SampleRecord objects")
        return sample_records
        
    def validate_specs(self, input_specs: List[InputSpec], 
                      output_specs: List[OutputSpec]) -> Dict[str, Any]:
        """
        Validate specification lists for consistency and correctness
        
        Returns:
            Dictionary with validation results and statistics
        """
        validation = {
            "input_spec_count": len(input_specs),
            "output_spec_count": len(output_specs),
            "input_spec_types": {},
            "output_spec_types": {},
            "issues": []
        }
        
        # Count specification types
        for spec in input_specs:
            spec_type = spec.kind.name if hasattr(spec.kind, 'name') else str(spec.kind)
            validation["input_spec_types"][spec_type] = validation["input_spec_types"].get(spec_type, 0) + 1
            
        for spec in output_specs:
            spec_type = spec.kind.name if hasattr(spec.kind, 'name') else str(spec.kind)
            validation["output_spec_types"][spec_type] = validation["output_spec_types"].get(spec_type, 0) + 1
            
        # Check for issues
        if len(input_specs) != len(output_specs):
            validation["issues"].append(f"Mismatched spec counts: {len(input_specs)} input vs {len(output_specs)} output")
            
        # Validate individual specs
        for i, spec in enumerate(input_specs):
            if spec.kind == InKind.BOX:
                if hasattr(spec, 'lb') and hasattr(spec, 'ub') and spec.lb is not None and spec.ub is not None:
                    if torch.any(spec.lb > spec.ub):
                        validation["issues"].append(f"Input spec {i}: lower bound > upper bound")
                        
        validation["status"] = "✅ Valid" if not validation["issues"] else f"⚠️  {len(validation['issues'])} issues"
        return validation


def demo_spec_loader():
    """Demo function to test SpecLoader functionality"""
    print("📋 SpecLoader Demo")
    print("=" * 50)
    
    # Create mock data
    mock_samples = [np.random.randn(784) for _ in range(3)]
    mock_labels = [0, 1, 2]
    
    spec_loader = SpecLoader()
    
    # Test input spec generation
    print("\n📊 Testing input specification generation:")
    try:
        input_config = {"type": "linf_ball", "epsilon": 0.03}
        input_specs = spec_loader.create_input_specs(mock_samples, input_config)
        print(f"✅ Generated {len(input_specs)} input specs")
        
        # Test output spec generation
        print("\n📊 Testing output specification generation:")
        output_config = {"output_type": "margin_robust", "margin": 0.0}
        output_specs = spec_loader.create_output_specs(mock_labels, output_config)
        print(f"✅ Generated {len(output_specs)} output specs")
        
        # Test combination
        print("\n📦 Testing data and spec combination:")
        data_pairs = list(zip(mock_samples, mock_labels))
        sample_records = spec_loader.combine_data_and_specs(data_pairs, input_specs, output_specs)
        print(f"✅ Created {len(sample_records)} SampleRecord objects")
        
        # Test validation
        print("\n🔍 Testing specification validation:")
        validation = spec_loader.validate_specs(input_specs, output_specs)
        print(f"Validation: {validation['status']}")
        print(f"Input types: {validation['input_spec_types']}")
        print(f"Output types: {validation['output_spec_types']}")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        
    # Test VNNLIB loading (placeholder)
    print("\n📋 Testing VNNLIB spec loading (placeholder):")
    vnnlib_path = "../data/vnnlib/local_vnnlib_example.vnnlib"
    if os.path.exists(vnnlib_path):
        try:
            vnnlib_specs = spec_loader.load_vnnlib_specs(vnnlib_path)
            print(f"✅ Loaded {len(vnnlib_specs)} VNNLIB spec pairs")
        except Exception as e:
            print(f"❌ VNNLIB loading failed: {e}")
    else:
        print(f"⚠️  VNNLIB file not found: {vnnlib_path}")


if __name__ == "__main__":
    demo_spec_loader()