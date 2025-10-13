"""
🔧 Model Loading for Front-End Integration

Self-contained ONNX model loading and conversion for front-end preprocessing.
Independent of existing ACT model loading to maintain clean separation.
"""

from __future__ import annotations
import os
import torch
import torch.nn as nn
import onnx
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass

from act.front_end.preprocessor_base import ModelSignature, Preprocessor
from act.front_end.preprocessor_image import ImgPre
from act.front_end.preprocessor_text import TextPre


@dataclass
class ModelMetadata:
    """Metadata extracted from ONNX model"""
    input_names: List[str]
    output_names: List[str] 
    input_shapes: List[Tuple[int, ...]]
    output_shapes: List[Tuple[int, ...]]
    input_dtypes: List[str]
    output_dtypes: List[str]
    model_size_mb: float
    total_params: int


class ModelLoader:
    """Self-contained ONNX model loading for front-end preprocessing"""
    def __init__(self):
        """Initialize ModelLoader"""
        pass
                
    def load_onnx_model(self, onnx_path: str) -> torch.nn.Module:
        """
        Load ONNX model and convert to PyTorch for front-end use
        
        Args:
            onnx_path: Path to ONNX model file
            
        Returns:
            PyTorch model ready for inference
            
        Raises:
            FileNotFoundError: If ONNX file doesn't exist
            RuntimeError: If model conversion fails
        """
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
            
        try:
            # Load ONNX model
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            
            # Try to convert using torch.jit.trace as fallback
            try:
                # Attempt direct conversion (works for many simple models)
                import torch.onnx
                import io
                
                # Create a temporary PyTorch model structure
                pytorch_model = self._convert_onnx_to_pytorch(onnx_model)
                pytorch_model.eval()
                
                return pytorch_model
                
            except Exception as conv_error:
                print(f"⚠️  Direct conversion failed: {conv_error}")
                print("📋 For complex models, consider using onnx2torch library")
                raise RuntimeError(f"Model conversion failed: {conv_error}")
                
        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model {onnx_path}: {e}")
        
    def _convert_onnx_to_pytorch(self, onnx_model) -> torch.nn.Module:
        """
        Convert ONNX model to PyTorch Sequential model for feedforward networks.
        
        This implementation handles Conv2d, Linear, ReLU, and Flatten layers.
        """
        import onnx.numpy_helper as numpy_helper
        
        # Parse ONNX graph nodes
        graph = onnx_model.graph
        initializers = {init.name: numpy_helper.to_array(init) for init in graph.initializer}
        
        layers = []
        
        # Track the flow through the network
        for i, node in enumerate(graph.node):
            if node.op_type == "Conv":
                # Convolutional layer
                weight_name = node.input[1]  # Second input is weight
                bias_name = node.input[2] if len(node.input) > 2 else None
                
                if weight_name in initializers:
                    weight = torch.from_numpy(initializers[weight_name]).float()
                    # Conv weight shape: [out_channels, in_channels, kernel_h, kernel_w]
                    out_channels, in_channels, kernel_h, kernel_w = weight.shape
                    
                    # Extract conv attributes
                    kernel_size = (kernel_h, kernel_w)
                    stride = 1
                    padding = 0
                    
                    for attr in node.attribute:
                        if attr.name == "strides" and attr.ints:
                            stride = tuple(attr.ints) if len(attr.ints) > 1 else attr.ints[0]
                        elif attr.name == "pads" and attr.ints:
                            # ONNX pads format: [pad_top, pad_left, pad_bottom, pad_right]
                            # PyTorch expects: (pad_left, pad_right) or single value
                            pads = attr.ints
                            if len(pads) == 4:
                                padding = pads[0]  # Assume symmetric padding
                            elif len(pads) == 2:
                                padding = tuple(pads)
                            else:
                                padding = pads[0]
                        elif attr.name == "kernel_shape" and attr.ints:
                            kernel_size = tuple(attr.ints) if len(attr.ints) > 1 else attr.ints[0]
                    
                    conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, 
                                         stride=stride, padding=padding)
                    conv_layer.weight.data = weight
                    
                    if bias_name and bias_name in initializers:
                        bias = torch.from_numpy(initializers[bias_name]).float()
                        conv_layer.bias.data = bias
                    else:
                        conv_layer.bias.data = torch.zeros(out_channels)
                    
                    layers.append(conv_layer)
            
            elif node.op_type == "Gemm":
                # General matrix multiplication (fully connected layer)
                weight_name = node.input[1]  # Second input is weight
                bias_name = node.input[2] if len(node.input) > 2 else None
                
                if weight_name in initializers:
                    weight = torch.from_numpy(initializers[weight_name]).float()
                    # Gemm weight shape: [out_features, in_features]
                    out_features, in_features = weight.shape
                    
                    linear_layer = nn.Linear(in_features, out_features)
                    linear_layer.weight.data = weight
                    
                    if bias_name and bias_name in initializers:
                        bias = torch.from_numpy(initializers[bias_name]).float()
                        linear_layer.bias.data = bias
                    else:
                        linear_layer.bias.data = torch.zeros(out_features)
                    
                    layers.append(linear_layer)
            
            elif node.op_type == "MatMul":
                # Matrix multiplication (alternative to Gemm)
                weight_name = None
                for input_name in node.input:
                    if input_name in initializers:
                        weight_name = input_name
                        break
                
                if weight_name:
                    weight = torch.from_numpy(initializers[weight_name]).float()
                    if weight.dim() == 2:
                        out_features, in_features = weight.shape
                        linear_layer = nn.Linear(in_features, out_features)
                        linear_layer.weight.data = weight
                        linear_layer.bias.data = torch.zeros(out_features)
                        layers.append(linear_layer)
            
            elif node.op_type == "Add":
                # Bias addition (usually after MatMul)
                bias_name = None
                for input_name in node.input:
                    if input_name in initializers:
                        bias_name = input_name
                        break
                
                if bias_name and layers and isinstance(layers[-1], nn.Linear):
                    bias = torch.from_numpy(initializers[bias_name]).float()
                    layers[-1].bias.data = bias
            
            elif node.op_type == "LSTM":
                # LSTM layer
                input_weights = []
                hidden_weights = []
                biases = []
                
                # Extract LSTM parameters from initializers
                for input_name in node.input[1:]:  # Skip first input (data)
                    if input_name in initializers:
                        param = torch.from_numpy(initializers[input_name]).float()
                        if "W" in input_name:
                            if "input" in input_name.lower() or "i" in input_name.lower():
                                input_weights.append(param)
                            else:
                                hidden_weights.append(param)
                        elif "B" in input_name or "bias" in input_name.lower():
                            biases.append(param)
                
                if input_weights and hidden_weights:
                    # Reconstruct LSTM parameters
                    weight_ih = input_weights[0] if input_weights else None
                    weight_hh = hidden_weights[0] if hidden_weights else None
                    bias_ih = biases[0] if len(biases) > 0 else None
                    bias_hh = biases[1] if len(biases) > 1 else None
                    
                    # Get dimensions
                    if weight_ih is not None:
                        hidden_size = weight_ih.shape[0] // 4  # LSTM has 4 gates
                        input_size = weight_ih.shape[1]
                        
                        # Create LSTM layer
                        lstm_layer = nn.LSTM(input_size, hidden_size, batch_first=True)
                        
                        # Set weights
                        lstm_layer.weight_ih_l0.data = weight_ih
                        lstm_layer.weight_hh_l0.data = weight_hh
                        if bias_ih is not None:
                            lstm_layer.bias_ih_l0.data = bias_ih
                        if bias_hh is not None:
                            lstm_layer.bias_hh_l0.data = bias_hh
                        
                        layers.append(lstm_layer)
            
            elif node.op_type == "GRU":
                # GRU layer
                input_weights = []
                hidden_weights = []
                biases = []
                
                # Extract GRU parameters
                for input_name in node.input[1:]:
                    if input_name in initializers:
                        param = torch.from_numpy(initializers[input_name]).float()
                        if "W" in input_name:
                            if "input" in input_name.lower() or "i" in input_name.lower():
                                input_weights.append(param)
                            else:
                                hidden_weights.append(param)
                        elif "B" in input_name or "bias" in input_name.lower():
                            biases.append(param)
                
                if input_weights and hidden_weights:
                    weight_ih = input_weights[0]
                    weight_hh = hidden_weights[0]
                    bias_ih = biases[0] if len(biases) > 0 else None
                    bias_hh = biases[1] if len(biases) > 1 else None
                    
                    # Get dimensions
                    hidden_size = weight_ih.shape[0] // 3  # GRU has 3 gates
                    input_size = weight_ih.shape[1]
                    
                    # Create GRU layer
                    gru_layer = nn.GRU(input_size, hidden_size, batch_first=True)
                    
                    # Set weights
                    gru_layer.weight_ih_l0.data = weight_ih
                    gru_layer.weight_hh_l0.data = weight_hh
                    if bias_ih is not None:
                        gru_layer.bias_ih_l0.data = bias_ih
                    if bias_hh is not None:
                        gru_layer.bias_hh_l0.data = bias_hh
                    
                    layers.append(gru_layer)
            
            elif node.op_type == "RNN":
                # Vanilla RNN layer
                input_weights = []
                hidden_weights = []
                biases = []
                
                # Extract RNN parameters
                for input_name in node.input[1:]:
                    if input_name in initializers:
                        param = torch.from_numpy(initializers[input_name]).float()
                        if "W" in input_name:
                            if "input" in input_name.lower() or "i" in input_name.lower():
                                input_weights.append(param)
                            else:
                                hidden_weights.append(param)
                        elif "B" in input_name or "bias" in input_name.lower():
                            biases.append(param)
                
                if input_weights and hidden_weights:
                    weight_ih = input_weights[0]
                    weight_hh = hidden_weights[0]
                    bias_ih = biases[0] if len(biases) > 0 else None
                    bias_hh = biases[1] if len(biases) > 1 else None
                    
                    # Get dimensions
                    hidden_size = weight_ih.shape[0]
                    input_size = weight_ih.shape[1]
                    
                    # Create RNN layer (default to tanh)
                    rnn_layer = nn.RNN(input_size, hidden_size, batch_first=True, nonlinearity='tanh')
                    
                    # Set weights
                    rnn_layer.weight_ih_l0.data = weight_ih
                    rnn_layer.weight_hh_l0.data = weight_hh
                    if bias_ih is not None:
                        rnn_layer.bias_ih_l0.data = bias_ih
                    if bias_hh is not None:
                        rnn_layer.bias_hh_l0.data = bias_hh
                    
                    layers.append(rnn_layer)
            
            elif node.op_type == "Gather":
                # Embedding layer (uses Gather operation in ONNX)
                weight_name = None
                for input_name in node.input:
                    if input_name in initializers:
                        weight_name = input_name
                        break
                
                if weight_name:
                    weight = torch.from_numpy(initializers[weight_name]).float()
                    if weight.dim() == 2:
                        num_embeddings, embedding_dim = weight.shape
                        embedding_layer = nn.Embedding(num_embeddings, embedding_dim)
                        embedding_layer.weight.data = weight
                        layers.append(embedding_layer)
            
            elif node.op_type == "Relu":
                layers.append(nn.ReLU())
            
            elif node.op_type == "Flatten":
                layers.append(nn.Flatten())
                
            elif node.op_type == "Reshape":
                # Handle reshape operations
                layers.append(nn.Flatten())
        
        if not layers:
            raise ValueError(f"No supported layers found in ONNX model. Found operations: {[node.op_type for node in graph.node]}")
        
        # Create Sequential model
        model = nn.Sequential(*layers)
        
        print(f"🔧 Converted ONNX to PyTorch Sequential with {len(layers)} layers:")
        for i, layer in enumerate(layers):
            if isinstance(layer, nn.Conv2d):
                print(f"   {i}: {type(layer).__name__}({layer.in_channels}, {layer.out_channels}, "
                      f"kernel_size={layer.kernel_size}, stride={layer.stride}, padding={layer.padding})")
            elif isinstance(layer, nn.Linear):
                print(f"   {i}: {type(layer).__name__}({layer.in_features}, {layer.out_features})")
            elif isinstance(layer, nn.LSTM):
                print(f"   {i}: {type(layer).__name__}({layer.input_size}, {layer.hidden_size})")
            elif isinstance(layer, nn.GRU):
                print(f"   {i}: {type(layer).__name__}({layer.input_size}, {layer.hidden_size})")
            elif isinstance(layer, nn.RNN):
                print(f"   {i}: {type(layer).__name__}({layer.input_size}, {layer.hidden_size}, nonlinearity='{layer.nonlinearity}')")
            elif isinstance(layer, nn.Embedding):
                print(f"   {i}: {type(layer).__name__}({layer.num_embeddings}, {layer.embedding_dim})")
            else:
                print(f"   {i}: {type(layer).__name__}")
        
        return model
        
    def extract_model_signature(self, onnx_path: str) -> ModelSignature:
        """
        Extract input/output metadata from ONNX model
        
        Args:
            onnx_path: Path to ONNX model file
            
        Returns:
            ModelSignature with input/output shape information
        """
        onnx_model = onnx.load(onnx_path)
        metadata = self._extract_metadata(onnx_model)
        
        # Create ModelSignature from first input/output
        input_shape = tuple(metadata.input_shapes[0]) if metadata.input_shapes else (1,)
        output_shape = tuple(metadata.output_shapes[0]) if metadata.output_shapes else (1,)
        
        return ModelSignature(
            input_shape=input_shape,
            output_shape=output_shape,
            device=self.device,
            dtype=self.dtype
        )
        
    def _extract_metadata(self, onnx_model) -> ModelMetadata:
        """Extract comprehensive metadata from ONNX model"""
        
        # Extract input information
        input_names = []
        input_shapes = []
        input_dtypes = []
        
        for input_info in onnx_model.graph.input:
            input_names.append(input_info.name)
            
            # Extract shape
            shape = []
            for dim in input_info.type.tensor_type.shape.dim:
                if dim.dim_value > 0:
                    shape.append(dim.dim_value)
                else:
                    shape.append(-1)  # Dynamic dimension
            input_shapes.append(tuple(shape))
            
            # Extract dtype
            dtype_map = {1: "float32", 2: "uint8", 3: "int8", 6: "int32", 7: "int64", 11: "float64"}
            dtype_id = input_info.type.tensor_type.elem_type
            input_dtypes.append(dtype_map.get(dtype_id, "unknown"))
            
        # Extract output information
        output_names = []
        output_shapes = []
        output_dtypes = []
        
        for output_info in onnx_model.graph.output:
            output_names.append(output_info.name)
            
            # Extract shape
            shape = []
            for dim in output_info.type.tensor_type.shape.dim:
                if dim.dim_value > 0:
                    shape.append(dim.dim_value)
                else:
                    shape.append(-1)  # Dynamic dimension
            output_shapes.append(tuple(shape))
            
            # Extract dtype
            dtype_id = output_info.type.tensor_type.elem_type
            output_dtypes.append(dtype_map.get(dtype_id, "unknown"))
            
        # Calculate model size and parameters
        model_size_mb = os.path.getsize(onnx_model.SerializeToString()) / (1024 * 1024) if hasattr(onnx_model, 'SerializeToString') else 0.0
        total_params = sum(np.prod(init.dims) for init in onnx_model.graph.initializer)
        
        return ModelMetadata(
            input_names=input_names,
            output_names=output_names,
            input_shapes=input_shapes,
            output_shapes=output_shapes,
            input_dtypes=input_dtypes,
            output_dtypes=output_dtypes,
            model_size_mb=model_size_mb,
            total_params=total_params
        )
  
        
    def get_model_info(self, onnx_path: str) -> Dict[str, Any]:
        """
        Get comprehensive model information for debugging/analysis
        
        Returns:
            Dictionary with model metadata and analysis
        """
        try:
            onnx_model = onnx.load(onnx_path)
            metadata = self._extract_metadata(onnx_model)
            signature = self.extract_model_signature(onnx_path)
            
            return {
                "path": onnx_path,
                "metadata": metadata,
                "signature": signature,
                "device": self.device,
                "dtype": str(self.dtype),
                "status": "✅ Model loaded successfully"
            }
        except Exception as e:
            return {
                "path": onnx_path,
                "error": str(e),
                "status": "❌ Failed to load model"
            }