#!/usr/bin/env python3
"""
Custom inputs example for ACT pipeline testing framework.

This demonstrates how to add new test inputs by editing YAML configuration
without changing any code.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import yaml
from act.pipeline import MockInputFactory, ConfigManager, create_test_scenario


def demonstrate_custom_inputs():
    """Show how to add custom inputs via configuration."""
    print("🔧 Adding Custom Test Inputs - No Code Changes Required!")
    print("=" * 65)
    
    # Create a custom configuration with new input types
    custom_config = {
        "sample_data": {
            "custom_tiny_images": {
                "type": "image",
                "shape": [3, 16, 16],  # Tiny RGB images
                "distribution": "normal",
                "mean": 0.5,
                "std": 0.1,
                "range": [0, 1],
                "batch_size": 5,
                "num_classes": 3
            },
            "time_series_data": {
                "type": "custom",
                "shape": [1, 100],  # Time series data
                "distribution": "gaussian_noise",
                "base_value": 0.0,
                "noise_level": 0.2,
                "batch_size": 8,
                "num_classes": 2
            }
        },
        "input_specs": {
            "tiny_l_inf": {
                "spec_type": "LOCAL_LP",
                "norm_type": "LINF",
                "epsilon": 0.01,  # Very small perturbation
                "center_point": "auto"
            },
            "custom_box_constraints": {
                "spec_type": "SET_BOX",
                "lower_bounds": [-1.0, -1.0, -0.5],
                "upper_bounds": [1.0, 1.0, 0.5]
            }
        },
        "output_specs": {
            "binary_classification": {
                "type": "classification",
                "num_classes": 2,
                "target_label": 1
            },
            "strict_robustness": {
                "type": "robustness",
                "property": "margin >= 0.5",
                "margin_threshold": 0.5
            }
        },
        "models": {
            "tiny_net": {
                "architecture": "feedforward",
                "layers": [48, 20, 3],  # 3*16*16 = 768 inputs
                "activations": ["relu", "linear"]
            },
            "micro_cnn": {
                "architecture": "cnn",
                "conv_layers": [[3, 8, 3], [8, 16, 3]],
                "fc_layers": [16*2*2, 10, 3]
            }
        }
    }
    
    print("\n1. Creating custom configuration...")
    
    # Save custom config to temporary file
    config_path = "/tmp/custom_mock_inputs.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(custom_config, f, default_flow_style=False, indent=2)
    
    print(f"   ✅ Saved custom config to: {config_path}")
    
    print("\n2. Using custom inputs (no code changes needed)...")
    
    # Initialize factory with custom config
    factory = MockInputFactory(config_path)
    
    # List available configurations
    available = factory.list_available_configs()
    print("   Available custom configurations:")
    for section, configs in available.items():
        print(f"     {section}: {configs}")
    
    print("\n3. Generating custom test inputs...")
    
    # Generate tiny images
    try:
        tiny_data, tiny_labels = factory.create_sample_data("custom_tiny_images")
        print(f"   ✅ Tiny images: shape={tiny_data.shape}, range=[{tiny_data.min():.3f}, {tiny_data.max():.3f}]")
    except Exception as e:
        print(f"   ❌ Tiny images failed: {e}")
    
    # Generate time series data
    try:
        ts_data, ts_labels = factory.create_sample_data("time_series_data")
        print(f"   ✅ Time series: shape={ts_data.shape}, range=[{ts_data.min():.3f}, {ts_data.max():.3f}]")
    except Exception as e:
        print(f"   ❌ Time series failed: {e}")
    
    # Generate custom input specs
    try:
        tiny_spec = factory.create_input_spec("tiny_l_inf")
        print(f"   ✅ Tiny L∞ spec: epsilon={tiny_spec.get('epsilon')}")
        
        box_spec = factory.create_input_spec("custom_box_constraints")
        print(f"   ✅ Box constraints: bounds={len(box_spec.get('lower_bounds', []))}D")
    except Exception as e:
        print(f"   ❌ Input specs failed: {e}")
    
    # Generate custom models
    try:
        tiny_model = factory.create_model("tiny_net")
        param_count = sum(p.numel() for p in tiny_model.parameters())
        print(f"   ✅ Tiny model: {param_count} parameters")
        
        micro_cnn = factory.create_model("micro_cnn")
        cnn_param_count = sum(p.numel() for p in micro_cnn.parameters())
        print(f"   ✅ Micro CNN: {cnn_param_count} parameters")
    except Exception as e:
        print(f"   ❌ Models failed: {e}")
    
    print("\n4. Creating custom test scenarios...")
    
    # Create test scenarios using the new inputs
    scenario1 = create_test_scenario(
        sample_data="custom_tiny_images",
        input_spec="tiny_l_inf",
        output_spec="binary_classification",
        model="tiny_net",
        expected_result="UNSAT",
        timeout=15.0
    )
    
    scenario2 = create_test_scenario(
        sample_data="time_series_data", 
        input_spec="custom_box_constraints",
        output_spec="strict_robustness",
        model="micro_cnn",
        timeout=30.0
    )
    
    print("   ✅ Created custom test scenarios:")
    print(f"     Scenario 1: {scenario1['sample_data']} + {scenario1['model']}")
    print(f"     Scenario 2: {scenario2['sample_data']} + {scenario2['model']}")


def demonstrate_custom_generator():
    """Show how to register a custom generator function."""
    print("\n" + "=" * 65)
    print("🔧 Custom Generator Functions")
    print("=" * 65)
    
    # Define a custom generator function
    def generate_checkerboard_pattern(config):
        """Generate checkerboard pattern images."""
        import torch
        
        batch_size = config.get("batch_size", 5)
        size = config.get("size", 8)
        
        # Create checkerboard pattern
        pattern = torch.zeros(batch_size, 1, size, size)
        for i in range(size):
            for j in range(size):
                if (i + j) % 2 == 0:
                    pattern[:, :, i, j] = 1.0
        
        # Generate random labels
        labels = torch.randint(0, 2, (batch_size,))
        
        return pattern, labels
    
    print("\n1. Registering custom generator function...")
    
    # Create factory and register custom generator
    factory = MockInputFactory()
    factory.register_custom_generator("checkerboard_generator", generate_checkerboard_pattern)
    
    print("   ✅ Registered 'checkerboard_generator'")
    
    # Create config that uses custom generator
    custom_config_with_generator = {
        "sample_data": {
            "checkerboard_images": {
                "generator": "checkerboard_generator",
                "size": 16,
                "batch_size": 3
            }
        }
    }
    
    print("\n2. Using custom generator...")
    
    # This would require modifying the factory to load this config
    # For demonstration, we'll call the generator directly
    try:
        checkerboard_data, checkerboard_labels = generate_checkerboard_pattern({
            "size": 16,
            "batch_size": 3
        })
        print(f"   ✅ Generated checkerboard images: shape={checkerboard_data.shape}")
        
        import torch
        print(f"   ✅ Unique values in pattern: {torch.unique(checkerboard_data)}")
    except Exception as e:
        print(f"   ❌ Custom generator failed: {e}")


def main():
    """Run custom inputs demonstration."""
    demonstrate_custom_inputs()
    demonstrate_custom_generator()
    
    print("\n" + "=" * 65)
    print("Custom inputs demo completed! Key takeaways:")
    print("• Add new test inputs by editing YAML files only")
    print("• No code changes required for new input types")
    print("• Register custom generator functions for complex data")
    print("• Create test scenarios combining any inputs")
    print("• Easily extend testing coverage without development")


if __name__ == "__main__":
    main()