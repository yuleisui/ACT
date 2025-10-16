#!/usr/bin/env python3
"""
ONNX Model Comprehension Tool

A comprehensive script to analyze, interpret, and understand any ONNX model.
Provides detailed architectural analysis, parameter breakdown, tensor flow visualization,
and verification insights.

Usage:
    python models/analyze_onnx.py <model_path>
    python models/analyze_onnx.py models/Sample_models/MNIST/small_relu_mnist_cnn_model_1.onnx
    python models/analyze_onnx.py models/Sample_models/CIFAR10/small_relu_cifar10_cnn_model_1.onnx
    python models/analyze_onnx.py --help

Features:
- Complete model architecture analysis
- Parameter counting and distribution
- Tensor shape transformations
- Layer-by-layer breakdown
- Verification complexity assessment
- Memory and computational requirements
- Comparative analysis suggestions
"""

import argparse
import os
import sys
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

try:
    import onnx
    import onnx.helper
    from onnx import numpy_helper
    ONNX_AVAILABLE = True
except ImportError:
    print("❌ ONNX not available. Install with: pip install onnx")
    ONNX_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    print("❌ NumPy not available. Install with: pip install numpy")
    NUMPY_AVAILABLE = False

try:
    import onnxruntime as ort
    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    ONNXRUNTIME_AVAILABLE = False


class ONNXAnalyzer:
    """Comprehensive ONNX model analyzer and interpreter."""
    
    def __init__(self, model_path: str):
        """Initialize with ONNX model path."""
        self.model_path = Path(model_path)
        self.model = None
        self.graph = None
        self.analysis_results = {}
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX library is required")
        
        try:
            self.model = onnx.load(str(self.model_path))
            self.graph = self.model.graph
            print(f"✅ Loaded ONNX model: {self.model_path.name}")
        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model: {e}")
    
    def get_model_metadata(self) -> Dict[str, Any]:
        """Extract basic model metadata."""
        metadata = {
            "file_name": self.model_path.name,
            "file_size_mb": self.model_path.stat().st_size / (1024 * 1024),
            "ir_version": self.model.ir_version,
            "producer_name": self.model.producer_name,
            "producer_version": self.model.producer_version,
            "domain": self.model.domain,
            "doc_string": self.model.doc_string,
            "graph_name": self.graph.name,
            "num_nodes": len(self.graph.node),
            "num_inputs": len(self.graph.input),
            "num_outputs": len(self.graph.output),
            "num_initializers": len(self.graph.initializer)
        }
        return metadata
    
    def get_tensor_info(self, tensor) -> Dict[str, Any]:
        """Extract tensor shape and type information."""
        info = {
            "name": tensor.name,
            "shape": "Unknown",
            "dtype": "Unknown",
            "total_elements": 0
        }
        
        try:
            if tensor.type.tensor_type.shape.dim:
                shape = []
                total_elements = 1
                for dim in tensor.type.tensor_type.shape.dim:
                    if dim.dim_value:
                        shape.append(dim.dim_value)
                        total_elements *= dim.dim_value
                    elif dim.dim_param:
                        shape.append(dim.dim_param)
                        total_elements = -1  # Dynamic shape
                    else:
                        shape.append("?")
                        total_elements = -1
                
                info["shape"] = shape
                info["total_elements"] = total_elements
            
            # Data type mapping
            dtype_map = {
                1: "float32", 2: "uint8", 3: "int8", 4: "uint16", 5: "int16",
                6: "int32", 7: "int64", 8: "string", 9: "bool", 10: "float16",
                11: "float64", 12: "uint32", 13: "uint64"
            }
            dtype_int = tensor.type.tensor_type.elem_type
            info["dtype"] = dtype_map.get(dtype_int, f"Unknown({dtype_int})")
            
        except Exception as e:
            info["error"] = str(e)
        
        return info
    
    def get_inputs_outputs(self) -> Dict[str, List[Dict]]:
        """Analyze model inputs and outputs."""
        inputs = []
        outputs = []
        
        for inp in self.graph.input:
            # Skip initializers (they're not actual inputs)
            if not any(init.name == inp.name for init in self.graph.initializer):
                inputs.append(self.get_tensor_info(inp))
        
        for out in self.graph.output:
            outputs.append(self.get_tensor_info(out))
        
        return {"inputs": inputs, "outputs": outputs}
    
    def get_parameter_analysis(self) -> Dict[str, Any]:
        """Analyze model parameters (weights and biases)."""
        parameters = []
        total_params = 0
        param_breakdown = {}
        
        for init in self.graph.initializer:
            param_info = {
                "name": init.name,
                "shape": list(init.dims),
                "total_params": int(np.prod(init.dims)) if init.dims else 0,
                "data_type": init.data_type
            }
            
            # Categorize parameters
            param_type = "other"
            if "weight" in init.name:
                param_type = "weights"
            elif "bias" in init.name:
                param_type = "biases"
            elif "running_mean" in init.name or "running_var" in init.name:
                param_type = "batch_norm"
            
            param_info["type"] = param_type
            
            # Analyze layer type from parameter shape
            if "conv" in init.name.lower() and "weight" in init.name:
                if len(param_info["shape"]) == 4:
                    out_ch, in_ch, kh, kw = param_info["shape"]
                    param_info["layer_type"] = "conv2d"
                    param_info["description"] = f"Conv2D: {in_ch}→{out_ch} channels, {kh}×{kw} kernel"
                elif len(param_info["shape"]) == 5:
                    param_info["layer_type"] = "conv3d"
            elif ("fc" in init.name.lower() or "linear" in init.name.lower()) and "weight" in init.name:
                if len(param_info["shape"]) == 2:
                    out_feat, in_feat = param_info["shape"]
                    param_info["layer_type"] = "linear"
                    param_info["description"] = f"Linear: {in_feat}→{out_feat} neurons"
            
            parameters.append(param_info)
            total_params += param_info["total_params"]
            
            # Count by type
            param_breakdown[param_type] = param_breakdown.get(param_type, 0) + param_info["total_params"]
        
        return {
            "parameters": parameters,
            "total_parameters": total_params,
            "parameter_breakdown": param_breakdown,
            "memory_estimate_mb": total_params * 4 / (1024 * 1024)  # Assuming float32
        }
    
    def get_layer_analysis(self) -> List[Dict[str, Any]]:
        """Analyze each layer in the model."""
        layers = []
        layer_counts = {}
        
        # Get parameter information for matching with layers
        param_info = {}
        for init in self.graph.initializer:
            param_info[init.name] = {
                "shape": list(init.dims),
                "total_params": int(np.prod(init.dims)) if init.dims else 0,
                "data_type": init.data_type
            }
        
        for i, node in enumerate(self.graph.node):
            layer_info = {
                "index": i + 1,
                "name": node.name,
                "op_type": node.op_type,
                "inputs": list(node.input),
                "outputs": list(node.output),
                "attributes": {},
                "parameters": {},
                "computational_complexity": "Unknown"
            }
            
            # Extract attributes
            for attr in node.attribute:
                attr_value = self._get_attribute_value(attr)
                layer_info["attributes"][attr.name] = attr_value
            
            # Match parameters to this layer
            layer_params = []
            total_layer_params = 0
            for input_name in node.input:
                if input_name in param_info:
                    param = param_info[input_name]
                    param["name"] = input_name
                    layer_params.append(param)
                    total_layer_params += param["total_params"]
            
            layer_info["parameters"] = {
                "params": layer_params,
                "total_params": total_layer_params,
                "param_names": [p["name"] for p in layer_params]
            }
            
            # Add layer-specific analysis
            if node.op_type == "Conv":
                layer_info["analysis"] = self._analyze_conv_layer(layer_info["attributes"])
                # Estimate computational complexity for convolution
                if layer_params:
                    for param in layer_params:
                        if "weight" in param["name"] and len(param["shape"]) == 4:
                            out_ch, in_ch, kh, kw = param["shape"]
                            layer_info["computational_complexity"] = f"O(out_ch × in_ch × kh × kw × H × W)"
                            layer_info["filter_info"] = {
                                "input_channels": in_ch,
                                "output_channels": out_ch,
                                "kernel_height": kh,
                                "kernel_width": kw
                            }
                            
            elif node.op_type == "Gemm":
                layer_info["analysis"] = self._analyze_gemm_layer(layer_info["attributes"])
                # Estimate computational complexity for dense layer
                if layer_params:
                    for param in layer_params:
                        if "weight" in param["name"] and len(param["shape"]) == 2:
                            out_feat, in_feat = param["shape"]
                            layer_info["computational_complexity"] = f"O({in_feat} × {out_feat})"
                            layer_info["dense_info"] = {
                                "input_features": in_feat,
                                "output_features": out_feat
                            }
                            
            elif node.op_type in ["Relu", "Sigmoid", "Tanh", "LeakyRelu", "Elu", "Softmax"]:
                layer_info["analysis"] = self._analyze_activation_layer(node.op_type, layer_info["attributes"])
                layer_info["computational_complexity"] = "O(n) - element-wise"
                
            elif node.op_type == "Flatten":
                layer_info["analysis"] = self._analyze_flatten_layer(layer_info["attributes"])
                layer_info["computational_complexity"] = "O(1) - shape manipulation only"
                
            elif node.op_type in ["MaxPool", "AveragePool", "GlobalAveragePool"]:
                layer_info["analysis"] = self._analyze_pooling_layer(node.op_type, layer_info["attributes"])
                if "kernel_shape" in layer_info["attributes"]:
                    kernel = layer_info["attributes"]["kernel_shape"]
                    kernel_area = kernel[0] * kernel[1] if len(kernel) == 2 else "N/A"
                    layer_info["computational_complexity"] = f"O(kernel_area × H × W) = O({kernel_area} × H × W)"
                    
            elif node.op_type in ["BatchNormalization", "LayerNormalization", "InstanceNormalization"]:
                layer_info["analysis"] = self._analyze_normalization_layer(node.op_type, layer_info["attributes"])
                layer_info["computational_complexity"] = "O(n) - per element normalization"
                
            elif node.op_type in ["Reshape", "Transpose", "Squeeze", "Unsqueeze"]:
                layer_info["analysis"] = self._analyze_reshape_layer(node.op_type, layer_info["attributes"])
                layer_info["computational_complexity"] = "O(1) - no computation"
                
            else:
                # Generic analysis for unknown layer types
                layer_info["analysis"] = {
                    "layer_type": "unknown",
                    "description": f"Operation: {node.op_type}",
                    "verification": "Unknown complexity"
                }
            
            # Add verification difficulty assessment
            verification_difficulty = "Unknown"
            if "analysis" in layer_info and "verification" in layer_info["analysis"]:
                verification_text = layer_info["analysis"]["verification"]
                if "Easy" in verification_text or "Trivial" in verification_text:
                    verification_difficulty = "Easy"
                elif "Medium" in verification_text:
                    verification_difficulty = "Medium"
                elif "Hard" in verification_text:
                    verification_difficulty = "Hard"
            
            layer_info["verification_difficulty"] = verification_difficulty
            
            layers.append(layer_info)
            
            # Count layer types
            layer_counts[node.op_type] = layer_counts.get(node.op_type, 0) + 1
        
        return layers, layer_counts
    
    def _get_attribute_value(self, attr):
        """Extract attribute value based on type."""
        if attr.type == onnx.AttributeProto.INT:
            return attr.i
        elif attr.type == onnx.AttributeProto.FLOAT:
            return attr.f
        elif attr.type == onnx.AttributeProto.STRING:
            return attr.s.decode('utf-8')
        elif attr.type == onnx.AttributeProto.INTS:
            return list(attr.ints)
        elif attr.type == onnx.AttributeProto.FLOATS:
            return list(attr.floats)
        elif attr.type == onnx.AttributeProto.STRINGS:
            return [s.decode('utf-8') for s in attr.strings]
        else:
            return f"Type_{attr.type}"
    
    def _analyze_conv_layer(self, attributes: Dict) -> Dict[str, Any]:
        """Analyze convolution layer specifics."""
        analysis = {"layer_type": "convolution"}
        
        if "kernel_shape" in attributes:
            kernel = attributes["kernel_shape"]
            analysis["kernel_size"] = f"{kernel[0]}×{kernel[1]}" if len(kernel) == 2 else str(kernel)
            analysis["kernel_area"] = kernel[0] * kernel[1] if len(kernel) == 2 else "N/A"
        
        if "strides" in attributes:
            strides = attributes["strides"]
            analysis["stride"] = f"{strides[0]}×{strides[1]}" if len(strides) == 2 else str(strides)
            analysis["stride_product"] = strides[0] * strides[1] if len(strides) == 2 else "N/A"
        
        if "pads" in attributes:
            pads = attributes["pads"]
            if len(pads) == 4:
                analysis["padding"] = f"[{pads[0]},{pads[1]},{pads[2]},{pads[3]}]"
                analysis["padding_total"] = f"H: {pads[0]+pads[2]}, W: {pads[1]+pads[3]}"
            else:
                analysis["padding"] = str(pads)
        
        if "dilations" in attributes:
            analysis["dilation"] = str(attributes["dilations"])
        
        if "group" in attributes:
            analysis["groups"] = attributes["group"]
            if attributes["group"] > 1:
                analysis["grouped_conv"] = True
        
        # Calculate receptive field
        if "kernel_shape" in attributes and "dilations" in attributes:
            kernel = attributes["kernel_shape"]
            dilations = attributes["dilations"]
            if len(kernel) == 2 and len(dilations) == 2:
                eff_kernel_h = kernel[0] + (kernel[0] - 1) * (dilations[0] - 1)
                eff_kernel_w = kernel[1] + (kernel[1] - 1) * (dilations[1] - 1)
                analysis["effective_kernel"] = f"{eff_kernel_h}×{eff_kernel_w}"
        
        return analysis
    
    def _analyze_gemm_layer(self, attributes: Dict) -> Dict[str, Any]:
        """Analyze GEMM (matrix multiplication) layer."""
        analysis = {"layer_type": "linear/dense"}
        
        if "transA" in attributes:
            analysis["transpose_A"] = bool(attributes["transA"])
        if "transB" in attributes:
            analysis["transpose_B"] = bool(attributes["transB"])
        if "alpha" in attributes:
            analysis["alpha"] = attributes["alpha"]
        if "beta" in attributes:
            analysis["beta"] = attributes["beta"]
        
        # Explain the operation
        operation_parts = []
        alpha = attributes.get("alpha", 1.0)
        beta = attributes.get("beta", 1.0)
        
        if alpha != 1.0:
            operation_parts.append(f"{alpha} * ")
        operation_parts.append("A")
        if attributes.get("transA", 0):
            operation_parts.append("^T")
        operation_parts.append(" @ B")
        if attributes.get("transB", 0):
            operation_parts.append("^T")
        
        if beta != 0.0:
            if beta == 1.0:
                operation_parts.append(" + C")
            else:
                operation_parts.append(f" + {beta} * C")
        
        analysis["operation"] = "".join(operation_parts)
        analysis["description"] = "Matrix multiplication with optional bias"
        
        return analysis
    
    def _analyze_flatten_layer(self, attributes: Dict) -> Dict[str, Any]:
        """Analyze flatten layer."""
        analysis = {"layer_type": "reshape"}
        
        if "axis" in attributes:
            axis = attributes["axis"]
            analysis["axis"] = axis
            analysis["description"] = f"Flatten from axis {axis}"
            
            if axis == 1:
                analysis["flatten_mode"] = "Keep batch dimension, flatten rest"
            elif axis == 0:
                analysis["flatten_mode"] = "Flatten everything into 1D"
            else:
                analysis["flatten_mode"] = f"Flatten from dimension {axis} onwards"
        else:
            analysis["description"] = "Flatten with default axis=1"
            analysis["flatten_mode"] = "Keep batch dimension, flatten rest"
        
        return analysis
    
    def _analyze_activation_layer(self, op_type: str, attributes: Dict) -> Dict[str, Any]:
        """Analyze activation layers."""
        analysis = {"layer_type": "activation"}
        
        activation_info = {
            "Relu": {
                "function": "f(x) = max(0, x)",
                "properties": ["Piecewise linear", "Non-saturating", "Sparse activation"],
                "verification": "Easy - creates linear regions",
                "gradient": "1 if x > 0, else 0"
            },
            "Sigmoid": {
                "function": "f(x) = 1 / (1 + e^(-x))",
                "properties": ["Smooth", "Saturating", "Output range (0,1)"],
                "verification": "Harder - smooth non-linearity",
                "gradient": "σ(x) * (1 - σ(x))"
            },
            "Tanh": {
                "function": "f(x) = (e^x - e^(-x)) / (e^x + e^(-x))",
                "properties": ["Smooth", "Zero-centered", "Output range (-1,1)"],
                "verification": "Harder - smooth non-linearity",
                "gradient": "1 - tanh²(x)"
            },
            "LeakyRelu": {
                "function": "f(x) = x if x > 0 else α*x",
                "properties": ["Piecewise linear", "Non-saturating", "Fixes dying ReLU"],
                "verification": "Easy - piecewise linear",
                "gradient": "1 if x > 0 else α"
            },
            "Elu": {
                "function": "f(x) = x if x > 0 else α*(e^x - 1)",
                "properties": ["Smooth", "Zero-centered", "Non-saturating positive"],
                "verification": "Medium - smooth but simpler than sigmoid",
                "gradient": "1 if x > 0 else α*e^x"
            },
            "Softmax": {
                "function": "f(x_i) = e^(x_i) / Σ(e^(x_j))",
                "properties": ["Probability distribution", "Sum to 1", "Smooth"],
                "verification": "Hard - complex multi-input dependency",
                "gradient": "σ_i * (δ_ij - σ_j)"
            }
        }
        
        if op_type in activation_info:
            info = activation_info[op_type]
            analysis.update(info)
            analysis["name"] = op_type
        
        # Handle specific attributes
        if op_type == "LeakyRelu" and "alpha" in attributes:
            analysis["alpha"] = attributes["alpha"]
            analysis["function"] = f"f(x) = x if x > 0 else {attributes['alpha']}*x"
        elif op_type == "Elu" and "alpha" in attributes:
            analysis["alpha"] = attributes["alpha"]
            analysis["function"] = f"f(x) = x if x > 0 else {attributes['alpha']}*(e^x - 1)"
        
        return analysis
    
    def _analyze_pooling_layer(self, op_type: str, attributes: Dict) -> Dict[str, Any]:
        """Analyze pooling layers."""
        analysis = {"layer_type": "pooling"}
        
        if "kernel_shape" in attributes:
            kernel = attributes["kernel_shape"]
            analysis["kernel_size"] = f"{kernel[0]}×{kernel[1]}" if len(kernel) == 2 else str(kernel)
        
        if "strides" in attributes:
            strides = attributes["strides"]
            analysis["stride"] = f"{strides[0]}×{strides[1]}" if len(strides) == 2 else str(strides)
        
        if "pads" in attributes:
            pads = attributes["pads"]
            if len(pads) == 4:
                analysis["padding"] = f"[{pads[0]},{pads[1]},{pads[2]},{pads[3]}]"
        
        if op_type == "MaxPool":
            analysis["operation"] = "Select maximum value in each window"
            analysis["properties"] = ["Translation invariant", "Preserves important features"]
            analysis["verification"] = "Medium - piecewise linear but complex"
        elif op_type == "AveragePool":
            analysis["operation"] = "Compute average of values in each window"
            analysis["properties"] = ["Smooth operation", "Reduces noise"]
            analysis["verification"] = "Easy - linear operation"
        elif op_type == "GlobalAveragePool":
            analysis["operation"] = "Average entire feature map to single value"
            analysis["properties"] = ["Spatial dimension reduction", "Acts as regularizer"]
            analysis["verification"] = "Easy - linear operation"
        
        return analysis
    
    def _analyze_normalization_layer(self, op_type: str, attributes: Dict) -> Dict[str, Any]:
        """Analyze normalization layers."""
        analysis = {"layer_type": "normalization"}
        
        if op_type == "BatchNormalization":
            analysis["operation"] = "(x - μ) / √(σ² + ε) * γ + β"
            analysis["properties"] = ["Stabilizes training", "Reduces internal covariate shift"]
            analysis["verification"] = "Easy - linear transformation (during inference)"
            
            if "epsilon" in attributes:
                analysis["epsilon"] = attributes["epsilon"]
            if "momentum" in attributes:
                analysis["momentum"] = attributes["momentum"]
                
        elif op_type == "LayerNormalization":
            analysis["operation"] = "Normalize across feature dimension"
            analysis["properties"] = ["Layer-wise normalization", "Independent of batch size"]
            analysis["verification"] = "Medium - depends on input statistics"
            
        elif op_type == "InstanceNormalization":
            analysis["operation"] = "Normalize each channel independently"
            analysis["properties"] = ["Instance-wise normalization", "Good for style transfer"]
            analysis["verification"] = "Medium - channel-wise statistics"
        
        return analysis
    
    def _analyze_reshape_layer(self, op_type: str, attributes: Dict) -> Dict[str, Any]:
        """Analyze reshape/transformation layers."""
        analysis = {"layer_type": "transformation"}
        
        if op_type == "Reshape":
            analysis["operation"] = "Change tensor shape without changing data"
            if "shape" in attributes:
                analysis["target_shape"] = attributes["shape"]
            analysis["verification"] = "Trivial - no computation, just indexing"
            
        elif op_type == "Transpose":
            analysis["operation"] = "Permute tensor dimensions"
            if "perm" in attributes:
                analysis["permutation"] = attributes["perm"]
            analysis["verification"] = "Trivial - no computation, just reordering"
            
        elif op_type == "Squeeze":
            analysis["operation"] = "Remove dimensions of size 1"
            if "axes" in attributes:
                analysis["axes"] = attributes["axes"]
            analysis["verification"] = "Trivial - shape manipulation only"
            
        elif op_type == "Unsqueeze":
            analysis["operation"] = "Add dimensions of size 1"
            if "axes" in attributes:
                analysis["axes"] = attributes["axes"]
            analysis["verification"] = "Trivial - shape manipulation only"
        
        return analysis
    
    def calculate_tensor_flow(self) -> List[Dict[str, Any]]:
        """Calculate tensor shapes through the network."""
        if not ONNXRUNTIME_AVAILABLE:
            return [{"error": "ONNXRuntime not available for tensor flow calculation"}]
        
        try:
            session = ort.InferenceSession(str(self.model_path))
            
            # Get input/output details
            input_details = session.get_inputs()
            output_details = session.get_outputs()
            
            flow_info = []
            
            # Input shapes
            for inp in input_details:
                shape = inp.shape
                # Handle dynamic dimensions
                concrete_shape = [1 if (dim is None or isinstance(dim, str)) else dim for dim in shape]
                
                flow_info.append({
                    "stage": "input",
                    "name": inp.name,
                    "shape": concrete_shape,
                    "dtype": inp.type,
                    "total_elements": int(np.prod(concrete_shape))
                })
            
            # Create dummy input for forward pass
            dummy_inputs = {}
            for inp in input_details:
                shape = inp.shape
                concrete_shape = [1 if (dim is None or isinstance(dim, str)) else dim for dim in shape]
                dummy_data = np.random.randn(*concrete_shape).astype(np.float32)
                dummy_inputs[inp.name] = dummy_data
            
            # Run inference to get output shapes
            outputs = session.run(None, dummy_inputs)
            
            # Output shapes
            for i, (out_detail, output) in enumerate(zip(output_details, outputs)):
                flow_info.append({
                    "stage": "output",
                    "name": out_detail.name,
                    "shape": list(output.shape),
                    "dtype": out_detail.type,
                    "total_elements": int(output.size),
                    "value_range": [float(output.min()), float(output.max())]
                })
            
            return flow_info
            
        except Exception as e:
            return [{"error": f"Tensor flow calculation failed: {e}"}]
    
    def get_verification_insights(self) -> Dict[str, Any]:
        """Provide insights for neural network verification."""
        layers, layer_counts = self.get_layer_analysis()
        
        insights = {
            "verification_complexity": "unknown",
            "recommended_verifiers": [],
            "challenges": [],
            "advantages": [],
            "properties_to_test": []
        }
        
        # Analyze activation functions
        activation_layers = [layer for layer in layers if layer["op_type"] in 
                           ["Relu", "Sigmoid", "Tanh", "LeakyRelu", "Elu", "Softmax"]]
        
        if activation_layers:
            main_activation = activation_layers[0]["op_type"]
            
            if main_activation == "Relu":
                insights["verification_complexity"] = "low"
                insights["recommended_verifiers"] = ["ERAN", "αβ-CROWN", "Base"]
                insights["advantages"].append("ReLU is piecewise linear (verification-friendly)")
            elif main_activation in ["Sigmoid", "Tanh"]:
                insights["verification_complexity"] = "medium"
                insights["recommended_verifiers"] = ["Base", "αβ-CROWN"]
                insights["challenges"].append(f"{main_activation} is non-linear (harder to verify)")
        
        # Analyze network depth
        if len(layers) <= 5:
            insights["advantages"].append("Shallow network (fast verification)")
        elif len(layers) > 20:
            insights["challenges"].append("Deep network (complex verification)")
        
        # Analyze convolution layers
        conv_layers = [layer for layer in layers if layer["op_type"] == "Conv"]
        if conv_layers:
            for conv in conv_layers:
                if "analysis" in conv and "kernel_size" in conv["analysis"]:
                    kernel = conv["analysis"]["kernel_size"]
                    if "8×8" in kernel or "7×7" in kernel:
                        insights["challenges"].append("Large convolution kernels increase complexity")
        
        # Suggest verification properties
        io_info = self.get_inputs_outputs()
        if io_info["outputs"]:
            output_shape = io_info["outputs"][0]["shape"]
            if isinstance(output_shape, list) and len(output_shape) == 2 and output_shape[1] == 10:
                insights["properties_to_test"].extend([
                    "Local robustness (ε-ball around input)",
                    "Classification consistency",
                    "Adversarial robustness"
                ])
        
        return insights
    
    def generate_usage_examples(self) -> Dict[str, str]:
        """Generate usage examples for this model."""
        model_name = self.model_path.stem
        examples = {}
        
        # Determine dataset from model name
        if "mnist" in model_name.lower():
            dataset = "mnist"
            epsilon = "0.1"
            examples["description"] = "MNIST digit classification model"
        elif "cifar" in model_name.lower():
            dataset = "cifar10"
            epsilon = "8/255"
            examples["description"] = "CIFAR-10 object classification model"
        else:
            dataset = "custom"
            epsilon = "0.1"
            examples["description"] = "Custom classification model"
        
        # ACT framework verification
        examples["act_verification"] = f"""python verifier/main.py \\
  --model_path {self.model_path} \\
  --dataset {dataset} \\
  --spec_type local_lp \\
  --epsilon {epsilon} \\
  --norm inf \\
  --start 0 --end 10 \\
  --verifier base"""
        
        # Netron visualization
        examples["visualization"] = f"netron {self.model_path}"
        
        # Python inference
        examples["inference_test"] = f"""import onnxruntime as ort
import numpy as np

session = ort.InferenceSession('{self.model_path}')
input_shape = {self.get_inputs_outputs()['inputs'][0]['shape'] if self.get_inputs_outputs()['inputs'] else '[1, 3, 32, 32]'}
dummy_input = np.random.randn(*input_shape).astype(np.float32)
output = session.run(None, {{'input': dummy_input}})
print(f'Output shape: {{output[0].shape}}')"""
        
        return examples
    
    def analyze_complete(self) -> Dict[str, Any]:
        """Perform complete analysis of the ONNX model."""
        print("🔍 Performing comprehensive ONNX model analysis...")
        
        results = {}
        
        # Basic metadata
        results["metadata"] = self.get_model_metadata()
        
        # Input/Output analysis
        results["io_analysis"] = self.get_inputs_outputs()
        
        # Parameter analysis
        results["parameter_analysis"] = self.get_parameter_analysis()
        
        # Layer analysis
        layers, layer_counts = self.get_layer_analysis()
        results["layer_analysis"] = {"layers": layers, "layer_counts": layer_counts}
        
        # Tensor flow
        results["tensor_flow"] = self.calculate_tensor_flow()
        
        # Verification insights
        results["verification_insights"] = self.get_verification_insights()
        
        # Usage examples
        results["usage_examples"] = self.generate_usage_examples()
        
        self.analysis_results = results
        return results
    
    def print_analysis(self):
        """Print formatted analysis results."""
        if not self.analysis_results:
            self.analyze_complete()
        
        results = self.analysis_results
        
        print("\n" + "="*80)
        print(f"🎯 ONNX MODEL ANALYSIS: {results['metadata']['file_name']}")
        print("="*80)
        
        # Metadata
        meta = results["metadata"]
        print(f"\n📊 Model Metadata:")
        print(f"   File Size: {meta['file_size_mb']:.2f} MB")
        print(f"   Producer: {meta['producer_name']} {meta['producer_version']}")
        print(f"   IR Version: {meta['ir_version']}")
        print(f"   Graph: {meta['graph_name']} ({meta['num_nodes']} layers)")
        
        # Input/Output
        io = results["io_analysis"]
        print(f"\n📥📤 Input/Output Analysis:")
        for i, inp in enumerate(io["inputs"]):
            shape_str = str(inp["shape"]).replace("'", "")
            print(f"   Input {i}: {inp['name']} - {shape_str} ({inp['dtype']})")
        for i, out in enumerate(io["outputs"]):
            shape_str = str(out["shape"]).replace("'", "")
            print(f"   Output {i}: {out['name']} - {shape_str} ({out['dtype']})")
        
        # Parameters
        params = results["parameter_analysis"]
        print(f"\n🔢 Parameter Analysis:")
        print(f"   Total Parameters: {params['total_parameters']:,}")
        print(f"   Memory Estimate: {params['memory_estimate_mb']:.2f} MB")
        if params["parameter_breakdown"]:
            print("   Parameter Breakdown:")
            for param_type, count in params["parameter_breakdown"].items():
                percentage = (count / params["total_parameters"]) * 100
                print(f"     {param_type}: {count:,} ({percentage:.1f}%)")
        
        # Layer Architecture
        layer_analysis = results["layer_analysis"]
        print(f"\n🏗️ Layer Architecture:")
        for op_type, count in sorted(layer_analysis["layer_counts"].items()):
            print(f"   {op_type}: {count} layer(s)")
        
        # Detailed layer breakdown
        print(f"\n📋 Layer-by-Layer Breakdown:")
        for layer in layer_analysis["layers"][:10]:  # Show first 10 layers
            print(f"   {layer['index']}. {layer['op_type']} ({layer['name']})")
            if "analysis" in layer and "description" in layer["analysis"]:
                print(f"      {layer['analysis']['description']}")
            elif layer["op_type"] == "Conv" and layer["attributes"]:
                attrs = layer["attributes"]
                kernel = attrs.get("kernel_shape", "?")
                stride = attrs.get("strides", "?")
                print(f"      Kernel: {kernel}, Stride: {stride}")
        
        if len(layer_analysis["layers"]) > 10:
            print(f"   ... and {len(layer_analysis['layers']) - 10} more layers")
        
        # Tensor Flow
        if results["tensor_flow"] and "error" not in results["tensor_flow"][0]:
            print(f"\n📐 Tensor Flow:")
            for flow in results["tensor_flow"]:
                shape_str = str(flow["shape"]).replace("'", "")
                print(f"   {flow['stage'].title()}: {flow['name']} - {shape_str} ({flow['total_elements']:,} elements)")
        
        # Verification Insights
        verification = results["verification_insights"]
        print(f"\n🔒 Verification Insights:")
        print(f"   Complexity: {verification['verification_complexity']}")
        if verification["recommended_verifiers"]:
            print(f"   Recommended Verifiers: {', '.join(verification['recommended_verifiers'])}")
        if verification["advantages"]:
            print("   Verification Advantages:")
            for adv in verification["advantages"]:
                print(f"     ✅ {adv}")
        if verification["challenges"]:
            print("   Verification Challenges:")
            for challenge in verification["challenges"]:
                print(f"     ⚠️ {challenge}")
        
        # Usage Examples
        examples = results["usage_examples"]
        print(f"\n🚀 Usage Examples:")
        print(f"   {examples['description']}")
        print(f"\n   Verification Command:")
        for line in examples["act_verification"].split('\n'):
            print(f"     {line.strip()}")
        print(f"\n   Visualization: {examples['visualization']}")
        
        print("\n" + "="*80)
    
    def _print_layer_details(self, layer: Dict[str, Any], detail_level: str = "detailed") -> None:
        """Print detailed information about a single layer."""
        print(f"\n{'='*50}")
        print(f"Layer {layer['index']}: {layer['name']} ({layer['op_type']})")
        print(f"{'='*50}")
        
        # Basic information
        print(f"Operation Type: {layer['op_type']}")
        print(f"Verification Difficulty: {layer.get('verification_difficulty', 'Unknown')}")
        print(f"Computational Complexity: {layer.get('computational_complexity', 'Unknown')}")
        
        # Inputs and outputs
        print(f"\nInputs ({len(layer['inputs'])}): {', '.join(layer['inputs'])}")
        print(f"Outputs ({len(layer['outputs'])}): {', '.join(layer['outputs'])}")
        
        # Parameters information
        if layer.get('parameters', {}).get('total_params', 0) > 0:
            print(f"\nParameters:")
            print(f"  Total parameters: {layer['parameters']['total_params']:,}")
            if detail_level == "detailed":
                for param in layer['parameters']['params']:
                    print(f"  - {param['name']}: {param['shape']} ({param['total_params']:,} params)")
        else:
            print(f"\nParameters: None (no trainable parameters)")
        
        # Layer-specific information
        if layer['op_type'] == "Conv" and 'filter_info' in layer:
            info = layer['filter_info']
            print(f"\nConvolution Details:")
            print(f"  Input channels: {info['input_channels']}")
            print(f"  Output channels: {info['output_channels']}")
            print(f"  Kernel size: {info['kernel_height']}×{info['kernel_width']}")
            
        elif layer['op_type'] == "Gemm" and 'dense_info' in layer:
            info = layer['dense_info']
            print(f"\nDense Layer Details:")
            print(f"  Input features: {info['input_features']}")
            print(f"  Output features: {info['output_features']}")
        
        # Attributes
        if layer['attributes']:
            print(f"\nAttributes:")
            for name, value in layer['attributes'].items():
                if isinstance(value, list) and len(value) > 5:
                    print(f"  {name}: [{', '.join(map(str, value[:3]))}, ...] (length: {len(value)})")
                else:
                    print(f"  {name}: {value}")
        
        # Detailed analysis
        if 'analysis' in layer and detail_level == "detailed":
            analysis = layer['analysis']
            print(f"\nDetailed Analysis:")
            if 'description' in analysis:
                print(f"  Description: {analysis['description']}")
            if 'mathematical_operation' in analysis:
                print(f"  Mathematical Operation: {analysis['mathematical_operation']}")
            if 'verification' in analysis:
                print(f"  Verification Complexity: {analysis['verification']}")
            if 'properties' in analysis:
                print(f"  Properties: {analysis['properties']}")
            if 'considerations' in analysis:
                print(f"  Considerations: {analysis['considerations']}")
            
            # Additional analysis fields
            for key, value in analysis.items():
                if key not in ['description', 'mathematical_operation', 'verification', 'properties', 'considerations']:
                    print(f"  {key.replace('_', ' ').title()}: {value}")
                    
        print(f"{'='*50}")
        
    def _print_layer_summary(self, layer: Dict[str, Any]) -> None:
        """Print a concise summary of a layer."""
        params_info = ""
        if layer.get('parameters', {}).get('total_params', 0) > 0:
            params_count = layer['parameters']['total_params']
            params_info = f" | {params_count:,} params"
        
        complexity = layer.get('computational_complexity', 'Unknown')
        if len(complexity) > 30:
            complexity = complexity[:27] + "..."
            
        print(f"  {layer['index']:2d}. {layer['name']:<25} ({layer['op_type']:<15}) {params_info} | {complexity}")
        
        # Show key analysis info in summary mode
        if 'analysis' in layer and 'description' in layer['analysis']:
            desc = layer['analysis']['description']
            if len(desc) > 80:
                desc = desc[:77] + "..."
            print(f"      └─ {desc}")
        
    def print_layer_analysis(self, detail_level: str = "summary") -> None:
        """Print layer analysis with specified detail level."""
        layers, layer_counts = self.get_layer_analysis()
        
        print(f"\n{'='*70}")
        print(f"LAYER-BY-LAYER ANALYSIS ({len(layers)} layers)")
        print(f"{'='*70}")
        
        # Layer type summary
        print(f"\nLayer Type Distribution:")
        for op_type, count in sorted(layer_counts.items()):
            print(f"  {op_type}: {count} layer{'s' if count > 1 else ''}")
        
        print(f"\n{'-'*70}")
        
        if detail_level == "summary":
            print(f"Layer Summary (use --layer-detail detailed for more info):")
            print(f"{'':4}{'Name':<25} {'Type':<15} {'Parameters':<15} {'Complexity'}")
            print(f"{'-'*70}")
            for layer in layers:
                self._print_layer_summary(layer)
                
        elif detail_level == "detailed":
            print(f"Detailed Layer Analysis:")
            for layer in layers:
                self._print_layer_details(layer, detail_level)
        
        print(f"\n{'='*70}")
        print(f"END LAYER ANALYSIS")
        print(f"{'='*70}")
    
    def save_analysis(self, output_file: Optional[str] = None):
        """Save analysis results to JSON file."""
        if not self.analysis_results:
            self.analyze_complete()
        
        if output_file is None:
            output_file = f"{self.model_path.stem}_analysis.json"
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Clean results for JSON serialization
        clean_results = json.loads(json.dumps(self.analysis_results, default=convert_numpy_types))
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(clean_results, f, indent=2, default=str)
        
        print(f"📁 Analysis saved to: {output_file}")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Comprehensive ONNX Model Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis with detailed layers (default)
  python models/analyze_onnx.py models/Sample_models/MNIST/small_relu_mnist_cnn_model_1.onnx
  
  # Summary layer analysis only
  python models/analyze_onnx.py models/Sample_models/CIFAR10/small_relu_cifar10_cnn_model_1.onnx --layer-detail summary
  
  # Show only detailed layer analysis (skip other sections)
  python models/analyze_onnx.py my_model.onnx --layers-only
  
  # Save analysis to JSON
  python models/analyze_onnx.py my_model.onnx --save-json
  python models/analyze_onnx.py my_model.onnx --output analysis_results.json
        """
    )
    
    parser.add_argument("model_path", help="Path to ONNX model file")
    parser.add_argument("--save-json", action="store_true", 
                       help="Save analysis results to JSON file")
    parser.add_argument("--output", "-o", type=str, 
                       help="Output file for JSON analysis (default: <model_name>_analysis.json)")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Suppress detailed output, only save JSON")
    parser.add_argument("--layer-detail", choices=["summary", "detailed"], default="detailed",
                       help="Level of detail for layer analysis (default: detailed)")
    parser.add_argument("--layers-only", action="store_true",
                       help="Show only detailed layer analysis (skip other sections)")
    
    args = parser.parse_args()
    
    # Check dependencies
    if not ONNX_AVAILABLE:
        print("❌ ONNX library is required. Install with: pip install onnx")
        sys.exit(1)
    
    if not NUMPY_AVAILABLE:
        print("❌ NumPy library is required. Install with: pip install numpy")
        sys.exit(1)
    
    try:
        # Create analyzer
        analyzer = ONNXAnalyzer(args.model_path)
        
        # Perform analysis
        analyzer.analyze_complete()
        
        # Print results (unless quiet)
        if not args.quiet:
            if args.layers_only:
                # Show only detailed layer analysis
                analyzer.print_layer_analysis(args.layer_detail)
            else:
                # Show standard analysis
                analyzer.print_analysis()
                
                # Add detailed layer analysis if requested
                if args.layer_detail == "detailed":
                    analyzer.print_layer_analysis(args.layer_detail)
        
        # Save to JSON if requested
        if args.save_json or args.output:
            analyzer.save_analysis(args.output)
        
        if not ONNXRUNTIME_AVAILABLE and not args.quiet:
            print("\n💡 Install ONNXRuntime for tensor flow analysis: pip install onnxruntime")
        
    except Exception as e:
        print(f"❌ Error analyzing model: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()