
from __future__ import annotations
import numpy as np, torch
import torch.nn as nn
from typing import Optional, Tuple, List, Dict, Any
from act.front_end.model_inference import model_inference
from act.front_end.specs import InputSpec, OutputSpec, InKind, OutKind
from act.front_end.model_synthesis import InputLayer, InputAdapterLayer, InputSpecLayer, OutputSpecLayer
from act.front_end.device_manager import get_default_device, get_default_dtype
# -----------------------------------------------------------------------------
# Mock Wrapper Models for torch2act - Embedded with specs and input
# -----------------------------------------------------------------------------

def mock_wrapped_mlp_mnist(
    hidden_sizes: List[int] = [64, 32], 
    num_classes: int = 10,
    epsilon: float = 0.01,
    seed: int = 42
) -> nn.Sequential:
    """
    Create a mock wrapped MLP model for MNIST with embedded specs and input.
    Ready for torch2act conversion.
    
    Args:
        hidden_sizes: List of hidden layer sizes
        num_classes: Number of output classes
        epsilon: L∞ perturbation bound
        seed: Random seed for reproducibility
        
    Returns:
        nn.Sequential wrapped model with all necessary layers
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Get default device and dtype settings
    device = get_default_device()
    dtype = get_default_dtype()
    
    # Create mock MNIST sample (28x28 = 784 features)
    rng = np.random.default_rng(seed)
    center_data = torch.tensor(rng.random(784), dtype=dtype, device=device)  # Flattened MNIST image
    true_label = int(rng.integers(0, num_classes))
    
    # Input layer with shape declaration
    input_shape = (1, 784)  # Batch=1, features=784
    input_layer = InputLayer(shape=input_shape, center=center_data)
    
    # Simple identity adapter (no transformation needed for flattened MNIST)
    adapter = InputAdapterLayer()
    
    # Input specification (L∞ ball around center)
    input_spec = InputSpec(kind=InKind.LINF_BALL, center=center_data, eps=epsilon)
    input_spec_layer = InputSpecLayer(spec=input_spec)
    
    # Create MLP model
    layers = []
    in_features = 784
    for hidden_size in hidden_sizes:
        layers.extend([
            nn.Linear(in_features, hidden_size),
            nn.ReLU()
        ])
        in_features = hidden_size
    layers.append(nn.Linear(in_features, num_classes))  # Final output layer
    
    mlp_model = nn.Sequential(*layers)
    
    # Output specification (margin robustness)
    output_spec = OutputSpec(kind=OutKind.MARGIN_ROBUST, y_true=true_label, margin=0.0)
    output_spec_layer = OutputSpecLayer(spec=output_spec)
    
    # Assemble wrapped model
    wrapped_model = nn.Sequential(
        input_layer,           # INPUT
        adapter,              # INPUT_ADAPTER (identity)
        input_spec_layer,     # INPUT_SPEC
        nn.Flatten(),         # FLATTEN (for consistency)
        mlp_model,            # DENSE layers + RELU
        output_spec_layer     # ASSERT (output constraint)
    )
    
    return wrapped_model


def mock_wrapped_cnn_mnist(
    conv_channels: List[int] = [16, 32],
    kernel_size: int = 3,
    hidden_size: int = 64,
    num_classes: int = 10,
    epsilon: float = 0.01,
    seed: int = 42
) -> nn.Sequential:
    """
    Create a mock wrapped CNN model for MNIST with embedded specs and input.
    Ready for torch2act conversion.
    
    Args:
        conv_channels: List of convolutional channel sizes
        kernel_size: Convolution kernel size
        hidden_size: FC layer hidden size
        num_classes: Number of output classes
        epsilon: L∞ perturbation bound
        seed: Random seed for reproducibility
        
    Returns:
        nn.Sequential wrapped model with all necessary layers
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Get default device and dtype settings
    device = get_default_device()
    dtype = get_default_dtype()
    
    # Create mock MNIST sample (28x28 = 784 features, flattened for compatibility)
    rng = np.random.default_rng(seed)
    center_data = torch.tensor(rng.random(784), dtype=dtype, device=device)  # Flattened for compatibility
    true_label = int(rng.integers(0, num_classes))
    
    # Input layer with flattened shape for compatibility
    input_shape = (1, 784)  # Batch=1, features=784 (compatible with MLP)
    input_layer = InputLayer(shape=input_shape, center=center_data)
    
    # Simple adapter (identity)
    adapter = InputAdapterLayer()
    
    # Input specification (L∞ ball around center)
    input_spec = InputSpec(kind=InKind.LINF_BALL, center=center_data, eps=epsilon)
    input_spec_layer = InputSpecLayer(spec=input_spec)
    
    # Create simplified CNN that accepts flattened input
    layers = []
    
    # First layer reshapes flattened input to image format and applies conv
    layers.append(nn.Unflatten(1, (1, 28, 28)))  # (batch, 784) -> (batch, 1, 28, 28)
    
    in_channels = 1
    for out_channels in conv_channels:
        layers.extend([
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1),
            nn.ReLU()
        ])
        in_channels = out_channels
    
    # Global average pooling to reduce spatial dimensions
    layers.append(nn.AdaptiveAvgPool2d(1))  # Output: (batch, channels, 1, 1)
    layers.append(nn.Flatten())  # Output: (batch, channels)
    
    # Final classification layers
    layers.extend([
        nn.Linear(conv_channels[-1], hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, num_classes)
    ])
    
    cnn_model = nn.Sequential(*layers)
    
    # Output specification (margin robustness)
    output_spec = OutputSpec(kind=OutKind.MARGIN_ROBUST, y_true=true_label, margin=0.0)
    output_spec_layer = OutputSpecLayer(spec=output_spec)
    
    # Assemble wrapped model
    wrapped_model = nn.Sequential(
        input_layer,           # INPUT
        adapter,              # INPUT_ADAPTER (reshape)
        input_spec_layer,     # INPUT_SPEC
        cnn_model,            # CONV2D + RELU + MAXPOOL + FLATTEN + DENSE
        output_spec_layer     # ASSERT (output constraint)
    )
    
    return wrapped_model


def mock_wrapped_mlp_cifar(
    hidden_sizes: List[int] = [128, 64],
    num_classes: int = 10,
    epsilon: float = 0.01,
    seed: int = 42
) -> nn.Sequential:
    """
    Create a mock wrapped MLP model for CIFAR-10 with embedded specs and input.
    Ready for torch2act conversion.
    
    Args:
        hidden_sizes: List of hidden layer sizes
        num_classes: Number of output classes
        epsilon: L∞ perturbation bound
        seed: Random seed for reproducibility
        
    Returns:
        nn.Sequential wrapped model with all necessary layers
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Get default device and dtype settings
    device = get_default_device()
    dtype = get_default_dtype()
    
    # Create mock CIFAR-10 sample (3x32x32 = 3072 features)
    rng = np.random.default_rng(seed)
    center_data = torch.tensor(rng.random(3072), dtype=dtype, device=device)  # Flattened CIFAR-10 image
    true_label = int(rng.integers(0, num_classes))
    
    # Input layer with shape declaration
    input_shape = (1, 3072)  # Batch=1, features=3072
    input_layer = InputLayer(shape=input_shape, center=center_data)
    
    # Simple identity adapter (no transformation needed for flattened CIFAR-10)
    adapter = InputAdapterLayer()
    
    # Input specification (L∞ ball around center)
    input_spec = InputSpec(kind=InKind.LINF_BALL, center=center_data, eps=epsilon)
    input_spec_layer = InputSpecLayer(spec=input_spec)
    
    # Create MLP model
    layers = []
    in_features = 3072
    for hidden_size in hidden_sizes:
        layers.extend([
            nn.Linear(in_features, hidden_size),
            nn.ReLU()
        ])
        in_features = hidden_size
    layers.append(nn.Linear(in_features, num_classes))  # Final output layer
    
    mlp_model = nn.Sequential(*layers)
    
    # Output specification (margin robustness)
    output_spec = OutputSpec(kind=OutKind.MARGIN_ROBUST, y_true=true_label, margin=0.0)
    output_spec_layer = OutputSpecLayer(spec=output_spec)
    
    # Assemble wrapped model
    wrapped_model = nn.Sequential(
        input_layer,           # INPUT
        adapter,              # INPUT_ADAPTER (identity)
        input_spec_layer,     # INPUT_SPEC
        nn.Flatten(),         # FLATTEN (for consistency)
        mlp_model,            # DENSE layers + RELU
        output_spec_layer     # ASSERT (output constraint)
    )
    
    return wrapped_model


def mock_wrapped_models_collection(seed: int = 42) -> Dict[str, nn.Sequential]:
    """
    Create a collection of mock wrapped models for testing torch2act.
    
    Args:
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary of model_name -> wrapped_model (with model_inference compatible names)
    """
    models = {}
    
    # MNIST models (format: "x:model_type|x:dataset_name")
    models["x:mlp_small|x:mnist"] = mock_wrapped_mlp_mnist(
        hidden_sizes=[32, 16], epsilon=0.01, seed=seed
    )
    models["x:mlp_medium|x:mnist"] = mock_wrapped_mlp_mnist(
        hidden_sizes=[64, 32], epsilon=0.02, seed=seed+1
    )
    models["x:cnn_small|x:mnist"] = mock_wrapped_cnn_mnist(
        conv_channels=[8, 16], hidden_size=32, epsilon=0.01, seed=seed+2
    )
    models["x:cnn_medium|x:mnist"] = mock_wrapped_cnn_mnist(
        conv_channels=[16, 32], hidden_size=64, epsilon=0.02, seed=seed+3
    )
    
    # CIFAR-10 models (format: "x:model_type|x:dataset_name")
    models["x:mlp_small|x:cifar"] = mock_wrapped_mlp_cifar(
        hidden_sizes=[64, 32], epsilon=0.005, seed=seed+4
    )
    models["x:mlp_medium|x:cifar"] = mock_wrapped_mlp_cifar(
        hidden_sizes=[128, 64], epsilon=0.01, seed=seed+5
    )
    
    return models

if __name__ == "__main__":
    print("🧪 Testing Mock Wrapper Models - Using model_inference")
    print("=" * 50)
    
    # Create test models using the collection function
    test_models = mock_wrapped_models_collection(seed=42)
    
    print(f"📦 Created {len(test_models)} mock wrapped models from collection")
    print(f"� Models: {list(test_models.keys())}")
    
    # Prepare input data for model_inference (expected format: {dataset: {"x": tensor}})
    input_data = {}
    
    # Get the default device and dtype settings
    device = get_default_device()
    dtype = get_default_dtype()
    
    # Create appropriate input data for each dataset (using default dtype to match model weights)
    # MNIST: 784 features (28x28 flattened) - works for both MLP and CNN (CNN will reshape)
    mnist_rng = np.random.default_rng(42)
    mnist_sample = torch.tensor(mnist_rng.random(784), dtype=dtype, device=device).unsqueeze(0)  # (1, 784)
    input_data["mnist"] = {"x": mnist_sample}
    
    # CIFAR: 3072 features (3x32x32 flattened) - works for MLP models
    cifar_rng = np.random.default_rng(43)
    cifar_sample = torch.tensor(cifar_rng.random(3072), dtype=dtype, device=device).unsqueeze(0)  # (1, 3072)
    input_data["cifar"] = {"x": cifar_sample}
    
    print(f"📋 Prepared input data for datasets: {list(input_data.keys())}")
    print(f"   • mnist input shape: {input_data['mnist']['x'].shape}")
    print(f"   • cifar input shape: {input_data['cifar']['x'].shape}")
    
    # Use model_inference to test all models
    print(f"\n🔬 Running model_inference on {len(test_models)} models...")
    
    # First, let's debug by testing one model manually
    print(f"\n🔍 Debug: Testing first model manually...")
    first_model_name = list(test_models.keys())[0]
    first_model = test_models[first_model_name]
    dataset_name = first_model_name.split('|')[1].split(':')[1]
    test_input = input_data[dataset_name]["x"]
    
    print(f"   Model: {first_model_name}")
    print(f"   Input shape: {test_input.shape}")
    print(f"   Model structure: {len(first_model)} layers")
    
    try:
        with torch.no_grad():
            output = first_model(test_input)
        print(f"   ✅ Manual test successful! Output shape: {output.shape}")
    except Exception as e:
        print(f"   ❌ Manual test failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
    
    successful_models = model_inference(test_models, input_data)
    
    print(f"\n📊 INFERENCE RESULTS:")
    print(f"  ✅ Successful: {len(successful_models)}/{len(test_models)}")
    print(f"  📋 Successful models: {list(successful_models.keys())}")
    
    print(f"\n🎯 MOCK WRAPPER MODELS: COMPLETE ✅")
