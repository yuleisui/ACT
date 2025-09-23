# Models Directory

This directory contains pre-trained neural network models for verification benchmarking and testing. Models are organized by dataset.

## Directory Structure

### Sample_models (`Sample_models/`)
**Simple sample models for testing and demonstrations**

#### MNIST Sample Models (`Sample_models/MNIST/`)
- **`small_relu_mnist_cnn_model_1.onnx`**: Small CNN with ReLU activation
- **`small_sigmoid_mnist_cnn_model_1.onnx`**: Small CNN with Sigmoid activation
- **`small_tanh_mnist_cnn_model_1.onnx`**: Small CNN with Tanh activation

#### CIFAR-10 Sample Models (`Sample_models/CIFAR10/`)
- **`small_relu_cifar10_cnn_model_1.onnx`**: Small CNN with ReLU activation
- **`small_sigmoid_cifar10_cnn_model_1.onnx`**: Small CNN with Sigmoid activation
- **`small_tanh_cifar10_cnn_model_1.onnx`**: Small CNN with Tanh activation


## Model Format and Compatibility

All models are provided in **ONNX format** (.onnx files) for universal compatibility across verification backends.

### ACT Hybrid Zonotope Operation Support

The ACT Hybrid Zonotope verifier supports the following operations:
- **Convolutional layers**: `Conv2D`
- **Pooling**: `MaxPool`
- **Fully connected layers**: `Gemm`
- **Element-wise operations**: `Add`, `Sub`, `Mul`, `Div`
- **Activation functions**: `ReLU`, `Sigmoid`, `Tanh`

**Note**: For more complex operations (residual connections, batch normalisation, advanced activations), use ERAN or αβ-CROWN verifiers instead.

## Usage

For comprehensive usage examples, please refer to the main `README.md` file in the repository root. The main documentation contains detailed examples for all verification methods including:

- ACT Hybrid Zonotope verification examples
- CSV-based batch verification patterns
- VNNLIB specification examples
- External verifier integration (ERAN, αβ-CROWN)
- All supported datasets and model formats
