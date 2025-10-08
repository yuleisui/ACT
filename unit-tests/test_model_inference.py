#!/usr/bin/env python3
"""
Simple test script for demonstrating model inference with perturbed inputs.
This is a standalone test that shows the concepts without requiring the full ACT framework.
"""

import torch
import torch.nn as nn
import numpy as np


def create_simple_model():
    """Create a simple neural network for testing."""
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(784, 128)  # 28*28 = 784 for MNIST-like input
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(128, 10)   # 10 classes
            
        def forward(self, x):
            x = self.flatten(x)
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x
    
    return SimpleModel()


def perform_model_inference(model, normalized_input):
    """
    Simulate the BaseVerifier._perform_model_inference method.
    
    Args:
        model: PyTorch model
        normalized_input: Preprocessed tensor ready for model
        
    Returns:
        Predicted class index (int)
    """
    try:
        with torch.no_grad():
            model.eval()
            outputs = model(normalized_input)
            return torch.argmax(outputs, dim=1).item()
    except Exception as e:
        raise ValueError(f"Model inference failed: {e}") from e


def normalize_input_tensor_shape(input_tensor):
    """
    Normalize input tensor to single-batch format compatible with model.
    Simulates BaseVerifier._normalize_input_tensor_shape method.
    """
    dims = input_tensor.ndim
    
    if dims == 1:
        return input_tensor.unsqueeze(0)  # [784] -> [1, 784]
    elif dims == 2:
        if input_tensor.shape[0] == 1:
            return input_tensor  # Already [1, 784]
        return input_tensor[0:1]  # Take first batch
    elif dims == 3:
        return input_tensor.unsqueeze(0)  # [1, 28, 28] -> [1, 1, 28, 28]
    elif dims == 4:
        return input_tensor[0:1] if input_tensor.shape[0] != 1 else input_tensor
    
    raise RuntimeError(f"Unsupported input tensor dimensions: {dims}")

### Original image (simplified 2x2 example for clarity)
### original_image = torch.tensor([[[[0.5, 0.8],
###                                [0.3, 0.6]]]])  # Shape: [1, 1, 2, 2]
### epsilon = 0.1  # Perturbation radius
###

### input_lb = original_image - epsilon (clamped to [0,1])
### input_lb = torch.tensor([[[[0.4, 0.7],    # 0.5-0.1=0.4, 0.8-0.1=0.7
###                          [0.2, 0.5]]]])   # 0.3-0.1=0.2, 0.6-0.1=0.5

### input_ub = original_image + epsilon (clamped to [0,1])
### input_ub = torch.tensor([[[[0.6, 0.9],    # 0.5+0.1=0.6, 0.8+0.1=0.9
###                           [0.4, 0.7]]]])   # 0.3+0.1=0.4, 0.6+0.1=0.7

### input_center = (input_lb + input_ub) / 2.0
### input_center = torch.tensor([[[[0.5, 0.8],    # (0.4+0.6)/2=0.5, (0.7+0.9)/2=0.8
###                               [0.3, 0.6]]]])   # (0.2+0.4)/2=0.3, (0.5+0.7)/2=0.6

def run_perturbation_tests(model, original_image, original_prediction):
    """Run model inference tests with perturbed inputs using pre-computed original prediction."""
    
    print("\n🔬 Testing Model Inference with Perturbations")
    print("=" * 50)
    
    # Configuration
    epsilon = 0.1   # Perturbation radius
    true_label = 7  # Pretend this is the true label
    
    print(f"   Using pre-computed original prediction: Class {original_prediction}")
    print(f"   True label: {true_label}")
    print(f"   Epsilon (perturbation radius): {epsilon}")
    
    # Create input bounds (like ACT verification)
    print("\n📊 Creating Input Bounds for Verification:")
    input_lb = torch.clamp(original_image - epsilon, 0.0, 1.0)
    input_ub = torch.clamp(original_image + epsilon, 0.0, 1.0)
    input_center = (input_lb + input_ub) / 2.0
    
    print(f"   Lower bounds range: [{input_lb.min():.3f}, {input_lb.max():.3f}]")
    print(f"   Upper bounds range: [{input_ub.min():.3f}, {input_ub.max():.3f}]")
    print(f"   Center range: [{input_center.min():.3f}, {input_center.max():.3f}]")
    print(f"   Perturbation width: [{(input_ub - input_lb).min():.3f}, {(input_ub - input_lb).max():.3f}]")
    
    # Test perturbed image inference
    print("\n🔄 Testing Perturbed Image Inference:")
    
    perturbations = [
        ("Small positive", 0.05),
        ("Small negative", -0.05), 
        ("Max positive", epsilon),
        ("Max negative", -epsilon),
        ("Random uniform", None)
    ]
    
    consistent_predictions = 0
    total_tests = len(perturbations)
    
    for desc, perturbation_value in perturbations:
        print(f"\n   🎯 Testing {desc} perturbation:")
        
        if perturbation_value is None:
            # Random uniform perturbation within [-epsilon, +epsilon]
            perturbation = (torch.rand(size=original_image.shape) * 2 - 1) * epsilon
        else:
            # Constant perturbation
            perturbation = torch.full_like(original_image, perturbation_value)
        
        # Apply perturbation and clamp to valid range [0, 1]
        perturbed_image = torch.clamp(original_image + perturbation, 0.0, 1.0)
        
        print(f"      Perturbation range: [{perturbation.min():.3f}, {perturbation.max():.3f}]")
        print(f"      Perturbed image range: [{perturbed_image.min():.3f}, {perturbed_image.max():.3f}]")
        
        # Normalize and test
        normalized_perturbed = normalize_input_tensor_shape(perturbed_image)
        
        try:
            perturbed_prediction = perform_model_inference(model, normalized_perturbed)
            is_consistent = perturbed_prediction == original_prediction
            if is_consistent:
                consistent_predictions += 1
            status = "✅ CONSISTENT" if is_consistent else "⚠️  CHANGED"
            print(f"      Perturbed prediction: Class {perturbed_prediction} {status}")
        except Exception as e:
            print(f"      ❌ Perturbed inference failed: {e}")
    
    # Test verification corner cases (only sampling three cases here, darkest, lightest, center)
    print("\n📋 Testing Verification Corner Cases:")
    corner_tests = [
        ("Lower bound corner", input_lb),
        ("Upper bound corner", input_ub), 
        ("Center point", input_center)
    ]
    
    corner_consistent = 0
    for corner_desc, corner_input in corner_tests:
        normalized_corner = normalize_input_tensor_shape(corner_input)
        try:
            corner_prediction = perform_model_inference(model, normalized_corner)
            is_consistent = corner_prediction == original_prediction
            if is_consistent:
                corner_consistent += 1
            status = "✅ CONSISTENT" if is_consistent else "⚠️  INCONSISTENT"
            print(f"   {corner_desc}: Class {corner_prediction} {status}")
        except Exception as e:
            print(f"   {corner_desc}: ❌ Failed - {e}")
    
    # Summary
    print(f"\n📈 Test Summary:")
    print(f"   Perturbation consistency: {consistent_predictions}/{total_tests}")
    print(f"   Corner case consistency: {corner_consistent}/{len(corner_tests)}")
    
    if consistent_predictions == total_tests and corner_consistent == len(corner_tests):
        print("   🎉 All tests consistent - verification would likely succeed!")
    elif consistent_predictions >= total_tests * 0.8:
        print("   ⚠️  Mostly consistent - verification might succeed with refinement")
    else:
        print("   ❌ Many inconsistencies - verification would likely find counterexamples")
    
    print("\n💡 Verification Insight:")
    print(f"   This demonstrates how ACT verifiers analyze the region [{input_lb.min():.3f}, {input_ub.max():.3f}]")
    print(f"   instead of just testing individual points.")
    print(f"   Real verifiers use abstract domains to reason about ALL inputs in this region simultaneously.")
    
    print("\n✅ Model inference testing completed!")


def test_pytorch_basics():
    """Test basic PyTorch operations and original image inference."""
    model, original_image, original_prediction = run_pytorch_basics()
    # Pytest version doesn't return values, just validates that the function works
    assert model is not None
    assert original_image is not None
    assert original_prediction is not None


def run_pytorch_basics():
    """Test basic PyTorch operations and original image inference."""
    print("🔧 Testing PyTorch Basics and Model Inference:")
    
    # Test tensor creation
    x = torch.rand(2, 3)
    print(f"   Random tensor: {x.shape}")
    
    # Test model creation
    model = create_simple_model()
    print(f"   Model created: {type(model).__name__}")
    
    # Test forward pass with simple input
    test_input = torch.rand(1, 1, 28, 28)
    with torch.no_grad():
        output = model(test_input)
    print(f"   Forward pass: {test_input.shape} -> {output.shape}")
    
    prediction = torch.argmax(output, dim=1).item()
    print(f"   Basic prediction: Class {prediction}")
    
    # Test original image inference (moved from main test)
    print("\n🖼️  Testing Original Image Inference:")
    original_image = torch.rand(1, 1, 28, 28)  # Create test image
    print(f"   Original image shape: {original_image.shape}")
    print(f"   Original image range: [{original_image.min():.3f}, {original_image.max():.3f}]")
    
    normalized_original = normalize_input_tensor_shape(original_image)
    print(f"   Normalized shape: {normalized_original.shape}")
    
    try:
        original_prediction = perform_model_inference(model, normalized_original)
        print(f"   ✅ Original prediction: Class {original_prediction}")
        print("   ✅ PyTorch basics and inference working correctly!")
        return model, original_image, original_prediction
    except Exception as e:
        print(f"   ❌ Original inference failed: {e}")
        raise


if __name__ == "__main__":
    # Test PyTorch basics and get model components
    model, original_image, original_prediction = run_pytorch_basics()
    
    # Test perturbations using the pre-computed components
    run_perturbation_tests(model, original_image, original_prediction)


def test_model_inference_with_perturbations():
    """Pytest test function for model inference with perturbations."""
    # Test PyTorch basics and get model components
    model, original_image, original_prediction = run_pytorch_basics()
    
    # Test perturbations using the pre-computed components
    run_perturbation_tests(model, original_image, original_prediction)