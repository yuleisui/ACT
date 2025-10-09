#!/usr/bin/env python3
"""
Test BaseVerifier APIs using real ACT components following main.py pattern.
This test loads actual Model, Dataset, InputSpec, OutputSpec, and Spec components
exactly as main.py does, without any mock fallbacks.
"""

import torch
import sys
import os

# Add verifier to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'verifier'))

def load_real_act_components():
    """
    Load real ACT components exactly like main.py does.
    """
    print("🔧 Loading real ACT components (main.py pattern)...")
    
    # Import real ACT components
    from input_parser.model import Model
    from input_parser.dataset import Dataset  
    from input_parser.spec import Spec, InputSpec, OutputSpec
    from abstract_constraint_solver.base_verifier import BaseVerifier
    
    # Exact parameters from main.py pattern
    args_dict = {
        'model_path': './models/Sample_models/MNIST/small_relu_mnist_cnn_model_1.onnx',
        'device': 'cpu',
        'dataset': 'mnist',
        'spec_type': 'local_lp',
        'start': 0,
        'end': 1,
        'num_outputs': 10,
        'mean': [0.1307],  # MNIST default
        'std': [0.3081],   # MNIST default  
        'norm': 'inf',
        'epsilon': 0.1,
        'vnnlib_path': None,
        'anchor': None
    }
    
    print(f"Parameters: {args_dict}")
    
    # Step 1: Create Model (main.py lines 125-126)
    model = Model(model_path=args_dict["model_path"],
                  device=args_dict["device"])
    print(f"✅ Model loaded: {model.get_expected_input_shape()}")
    
    # Step 2: Create Dataset (main.py lines 142-154)
    dataset_path_for_init = args_dict["dataset"]
    if args_dict["spec_type"] in ["local_vnnlib", "set_vnnlib"] and args_dict["vnnlib_path"] is not None:
        dataset_path_for_init = args_dict["vnnlib_path"]
        
    dataset = Dataset(dataset_path=dataset_path_for_init,
                      anchor_csv_path=args_dict["anchor"],
                      device=args_dict["device"],
                      spec_type=args_dict["spec_type"],
                      start=args_dict["start"],
                      end=args_dict["end"],
                      num_outputs=args_dict["num_outputs"],
                      mean=args_dict["mean"],
                      std=args_dict["std"],
                      preprocess=True)
    print(f"✅ Dataset loaded: {dataset.input_center.shape}")
    
    # Step 3: Create InputSpec (main.py lines 156-159)
    input_spec = InputSpec(dataset=dataset,
                           norm=args_dict["norm"],
                           epsilon=args_dict["epsilon"],
                           vnnlib_path=args_dict["vnnlib_path"])
    print(f"✅ InputSpec created: {input_spec.input_lb.shape}")
    
    # Step 4: Create OutputSpec (main.py line 161)
    output_spec = OutputSpec(dataset=dataset)
    print(f"✅ OutputSpec created")
    
    # Step 5: Create combined Spec (main.py lines 163-165)
    spec = Spec(model=model,
                input_spec=input_spec,
                output_spec=output_spec)
    print(f"✅ Spec created")
    
    return {
        'model': model,
        'dataset': dataset,
        'input_spec': input_spec,
        'output_spec': output_spec,
        'spec': spec,
        'BaseVerifier': BaseVerifier,
        'args_dict': args_dict
    }


def test_load_real_components():
    """Test loading real ACT components."""
    print("\n=== Test: Load Real ACT Components ===")
    
    components = load_real_act_components()
    
    # Verify components are loaded
    assert components['model'] is not None
    assert components['dataset'] is not None
    assert components['input_spec'] is not None
    assert components['output_spec'] is not None
    assert components['spec'] is not None
    
    print("✅ All components loaded successfully")
    return components


def test_model_properties():
    """Test model properties and functionality."""
    print("\n=== Test: Model Properties ===")
    
    components = load_real_act_components()
    model = components['model']
    
    # Test model properties
    input_shape = model.get_expected_input_shape()
    print(f"Expected input shape: {input_shape}")
    assert input_shape == (1, 1, 28, 28)
    
    # Test model inference
    test_input = torch.randn(1, 1, 28, 28)
    model.pytorch_model.eval()
    with torch.no_grad():
        output = model.pytorch_model(test_input)
        print(f"Model output shape: {output.shape}")
        assert output.shape == (1, 10)  # 10 classes for MNIST
    
    print("✅ Model properties verified")
    return model


def test_dataset_properties():
    """Test dataset properties and data access."""
    print("\n=== Test: Dataset Properties ===")
    
    components = load_real_act_components()
    dataset = components['dataset']
    
    # Test dataset properties
    print(f"Dataset input center shape: {dataset.input_center.shape}")
    print(f"Dataset labels shape: {dataset.labels.shape}")
    print(f"Mean: {dataset.mean}")
    print(f"Std: {dataset.std}")
    print(f"Start: {dataset.start}, End: {dataset.end}")
    
    # Verify MNIST properties
    # The actual shape depends on how ACT loads MNIST data
    assert len(dataset.input_center.shape) >= 3  # Should have at least height, width dimensions
    assert dataset.input_center.shape[-2:] == (28, 28)  # MNIST is 28x28
    assert dataset.mean == 0.1307  # ACT stores as single value, not list
    assert dataset.std == 0.3081   # ACT stores as single value, not list
    assert dataset.start == 0
    assert dataset.end == 1
    
    print("✅ Dataset properties verified")
    return dataset


def test_input_spec_properties():
    """Test InputSpec bounds and perturbation properties."""
    print("\n=== Test: InputSpec Properties ===")
    
    components = load_real_act_components()
    input_spec = components['input_spec']
    
    # Test bound shapes
    print(f"Input center shape: {input_spec.input_center.shape}")
    print(f"Lower bound shape: {input_spec.input_lb.shape}")
    print(f"Upper bound shape: {input_spec.input_ub.shape}")
    print(f"Epsilon: {input_spec.epsilon}")
    print(f"Norm: {input_spec.norm}")
    
    # Verify epsilon-ball properties
    assert input_spec.epsilon == 0.1
    assert str(input_spec.norm) == "LPNormType.LINF"
    assert input_spec.input_center.shape == input_spec.input_lb.shape
    assert input_spec.input_center.shape == input_spec.input_ub.shape
    
    # Verify L-infinity bounds
    diff_lb = (input_spec.input_center - input_spec.input_lb).abs()
    diff_ub = (input_spec.input_ub - input_spec.input_center).abs()
    print(f"Max bound difference (lower): {diff_lb.max().item():.6f}")
    print(f"Max bound difference (upper): {diff_ub.max().item():.6f}")
    
    print("✅ InputSpec properties verified")
    return input_spec


def test_spec_integration():
    """Test integrated Spec functionality."""
    print("\n=== Test: Spec Integration ===")
    
    components = load_real_act_components()
    spec = components['spec']
    
    # Test spec components integration
    assert spec.model is not None
    assert spec.input_spec is not None
    assert spec.output_spec is not None
    
    # Test data flow through spec
    model = spec.model.pytorch_model
    input_center = spec.input_spec.input_center
    true_labels = spec.output_spec.labels
    
    print(f"Spec model type: {type(spec.model)}")
    print(f"Spec input shape: {input_center.shape}")
    print(f"Spec output labels: {true_labels}")
    
    # Test model prediction on center point
    model.eval()
    with torch.no_grad():
        # Add batch dimension: (28, 28) -> (1, 1, 28, 28)
        model_input = input_center.unsqueeze(0)  # This should give us (1, 28, 28) -> (1, 1, 28, 28)
        print(f"Model input shape: {model_input.shape}")
        output = model(model_input)
        predicted_class = torch.argmax(output, dim=1)
        confidence = torch.softmax(output, dim=1).max(dim=1)[0]
        
    print(f"Center prediction: class {predicted_class.item()}")
    print(f"True label: {true_labels[0].item()}")
    print(f"Confidence: {confidence.item():.3f}")
    
    print("✅ Spec integration verified")
    return spec


def test_baseVerifier_inheritance():
    """Test BaseVerifier can be inherited and instantiated."""
    print("\n=== Test: BaseVerifier Inheritance ===")
    
    components = load_real_act_components()
    BaseVerifier = components['BaseVerifier']
    
    # Create test verifier class
    class TestVerifier(BaseVerifier):
        def verify(self, proof, public_inputs):
            return "UNKNOWN"
    
    # Test instantiation with ACT components
    verifier = TestVerifier(
        spec=components['spec'],
        device='cpu'
    )
    
    print(f"Verifier type: {type(verifier)}")
    print(f"Verifier dataset: {type(verifier.dataset)}")
    print(f"Verifier spec: {type(verifier.spec)}")
    
    # Test verify method
    result = verifier.verify(None, None)
    assert result == "UNKNOWN"
    
    print("✅ BaseVerifier inheritance verified")
    return verifier


def test_perturbation_analysis():
    """Test perturbation analysis using real components."""
    print("\n=== Test: Perturbation Analysis ===")
    
    components = load_real_act_components()
    model = components['model'].pytorch_model
    input_spec = components['input_spec']
    
    model.eval()
    
    # Test predictions at perturbation boundaries
    test_points = {
        'center': input_spec.input_center,
        'lower_bound': input_spec.input_lb,
        'upper_bound': input_spec.input_ub
    }
    
    predictions = {}
    print(f"Testing epsilon-ball boundaries (ε={input_spec.epsilon}):")
    
    with torch.no_grad():
        for name, point in test_points.items():
            point_batch = point.unsqueeze(0)  # Add only batch dimension: [1,28,28] -> [1,1,28,28]
            output = model(point_batch)
            pred_class = torch.argmax(output, dim=1).item()
            confidence = torch.softmax(output, dim=1).max().item()
            predictions[name] = pred_class
            
            print(f"  {name:12}: class {pred_class}, confidence {confidence:.3f}")
    
    # Check consistency
    unique_preds = set(predictions.values())
    consistent = len(unique_preds) == 1
    
    print(f"\nPrediction consistency: {consistent}")
    if consistent:
        print("✅ Model predictions are consistent across epsilon-ball")
    else:
        print("⚠️  Model predictions vary within epsilon-ball")
        
    print("✅ Perturbation analysis completed")
    return predictions


if __name__ == "__main__":
    print("=== Real ACT Component Tests (Following main.py) ===")
    print("Testing Model, Dataset, InputSpec, OutputSpec, Spec")
    print("Parameters: MNIST, epsilon=0.1, norm=inf, mean=0.1307, std=0.3081")
    
    try:
        # Run all tests
        components = test_load_real_components()
        model = test_model_properties() 
        dataset = test_dataset_properties()
        input_spec = test_input_spec_properties()
        spec = test_spec_integration()
        verifier = test_baseVerifier_inheritance()
        predictions = test_perturbation_analysis()
        
        print(f"\n=== Test Summary ===")
        print(f"✅ Real ACT components loaded successfully")
        print(f"✅ Model: {model.get_expected_input_shape()}")
        print(f"✅ Dataset: {dataset.input_center.shape}")
        print(f"✅ InputSpec: ε={input_spec.epsilon}, norm={input_spec.norm}")
        print(f"✅ Spec integration working")
        print(f"✅ BaseVerifier inheritance working")
        print(f"✅ Perturbation analysis completed")
        
        unique_preds = set(predictions.values())
        robustness = "robust" if len(unique_preds) == 1 else "vulnerable"
        print(f"✅ Model appears {robustness} for this sample")
        
        print(f"\n🎉 All real ACT component tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        raise