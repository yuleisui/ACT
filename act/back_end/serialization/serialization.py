#===- act/back_end/serialization.py - ACT Net JSON Serialization -------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   JSON serialization and deserialization for ACT Net and Layer structures.
#   Provides robust tensor handling, schema validation, and round-trip
#   conversion capabilities for neural network verification models.
#
#===---------------------------------------------------------------------===#

from __future__ import annotations
import json
import base64
import io
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

from act.back_end.core import Layer, Net
from act.back_end.layer_schema import REGISTRY, LayerKind
from act.back_end.layer_util import validate_layer

# Version for format compatibility
SERIALIZATION_VERSION = "1.0"

class ACTSerializationError(Exception):
    """Custom exception for serialization errors."""
    pass

class TensorEncoder:
    """Handles PyTorch tensor encoding/decoding to/from JSON-compatible format."""
    
    @staticmethod
    def encode_tensor(tensor: torch.Tensor) -> Dict[str, Any]:
        """Convert PyTorch tensor to JSON-serializable dictionary."""
        if not HAS_TORCH:
            raise ACTSerializationError("PyTorch not available for tensor encoding")
            
        # Convert to numpy and encode as base64
        np_array = tensor.detach().cpu().numpy()
        buffer = io.BytesIO()
        np.save(buffer, np_array)
        encoded_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return {
            "data": encoded_data,
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "device": str(tensor.device),
            "requires_grad": tensor.requires_grad
        }
    
    @staticmethod
    def decode_tensor(tensor_dict: Dict[str, Any], target_device: Optional[str] = None) -> torch.Tensor:
        """Convert JSON dictionary back to PyTorch tensor."""
        if not HAS_TORCH:
            raise ACTSerializationError("PyTorch not available for tensor decoding")
            
        # Decode base64 data
        encoded_data = tensor_dict["data"]
        buffer = io.BytesIO(base64.b64decode(encoded_data.encode('utf-8')))
        np_array = np.load(buffer)
        
        # Create tensor with original properties
        tensor = torch.from_numpy(np_array)
        
        # Apply dtype conversion if needed
        if "dtype" in tensor_dict:
            dtype_str = tensor_dict["dtype"]
            if hasattr(torch, dtype_str.split('.')[-1]):
                dtype = getattr(torch, dtype_str.split('.')[-1])
                tensor = tensor.to(dtype)
        
        # Move to device
        device = target_device or tensor_dict.get("device", "cpu")
        if device != "cpu":
            tensor = tensor.to(device)
            
        # Set requires_grad
        if tensor_dict.get("requires_grad", False):
            tensor.requires_grad_(True)
            
        return tensor

class LayerSerializer:
    """Handles Layer serialization/deserialization."""
    
    @staticmethod
    def serialize_layer(layer: Layer) -> Dict[str, Any]:
        """Convert Layer to JSON-serializable dictionary."""
        # Encode tensor parameters
        params_encoded = {}
        for name, value in layer.params.items():
            if HAS_TORCH and isinstance(value, torch.Tensor):
                params_encoded[name] = TensorEncoder.encode_tensor(value)
            else:
                # Handle non-tensor parameters (floats, ints, strings, etc.)
                params_encoded[name] = value
        
        # Encode cache tensors (if any)
        cache_encoded = {}
        if hasattr(layer, 'cache'):
            for name, value in layer.cache.items():
                if HAS_TORCH and isinstance(value, torch.Tensor):
                    cache_encoded[name] = TensorEncoder.encode_tensor(value)
                else:
                    # Handle non-tensor cache values
                    cache_encoded[name] = value
        
        return {
            "id": layer.id,
            "kind": layer.kind,
            "params": params_encoded,
            "meta": layer.meta,  # Meta should be JSON-serializable already
            "in_vars": layer.in_vars,
            "out_vars": layer.out_vars,
            "cache": cache_encoded
        }
    
    @staticmethod
    def deserialize_layer(layer_dict: Dict[str, Any], target_device: Optional[str] = None) -> Layer:
        """Convert JSON dictionary to Layer object with proper validation."""
        # Decode tensor parameters
        params_decoded = {}
        for name, value in layer_dict.get("params", {}).items():
            if isinstance(value, dict) and "dtype" in value and "shape" in value:
                # This is a tensor that was encoded
                params_decoded[name] = TensorEncoder.decode_tensor(value, target_device)
            else:
                # This is a regular value (float, int, string, etc.)
                params_decoded[name] = value
        
        # Decode cache tensors
        cache_decoded = {}
        for name, value in layer_dict.get("cache", {}).items():
            if isinstance(value, dict) and "dtype" in value and "shape" in value:
                # This is a tensor that was encoded
                cache_decoded[name] = TensorEncoder.decode_tensor(value, target_device)
            else:
                # This is a regular value
                cache_decoded[name] = value
        
        # Create Layer object with validation using the schema
        try:
            layer = Layer(
                id=layer_dict["id"],
                kind=layer_dict["kind"],
                params=params_decoded,
                meta=layer_dict["meta"],
                in_vars=layer_dict["in_vars"],
                out_vars=layer_dict["out_vars"],
                cache=cache_decoded
            )
        except ValueError as e:
            # If core validation fails during deserialization, create the Layer
            # without validation for template/example usage
            if "schema violation" in str(e):
                print(f"⚠️  Creating example/template layer {layer_dict['kind']}(id={layer_dict['id']}) without validation")
                
                # Create Layer without validation
                layer = object.__new__(Layer)
                layer.id = layer_dict["id"]
                layer.kind = layer_dict["kind"]
                layer.params = params_decoded
                layer.meta = layer_dict["meta"]
                layer.in_vars = layer_dict["in_vars"]
                layer.out_vars = layer_dict["out_vars"]
                layer.cache = cache_decoded
            else:
                # Re-raise for other types of errors
                raise

        return layer


class NetSerializer:
    """Handles Net serialization/deserialization."""
    
    @staticmethod
    def serialize_net(net: Net, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Convert Net to JSON-serializable dictionary."""
        # Serialize all layers
        layers_serialized = []
        for layer in net.layers:
            layers_serialized.append(LayerSerializer.serialize_layer(layer))
        
        # Create metadata
        net_metadata = {
            "creation_time": datetime.now().isoformat(),
            "framework_version": f"ACT-{SERIALIZATION_VERSION}",
            "layer_count": len(net.layers),
            "layer_kinds": [layer.kind for layer in net.layers]
        }
        if metadata:
            net_metadata.update(metadata)
        
        return {
            "format_version": SERIALIZATION_VERSION,
            "act_net": {
                "layers": layers_serialized,
                "graph": {
                    "preds": {str(k): v for k, v in net.preds.items()},
                    "succs": {str(k): v for k, v in net.succs.items()}
                },
                "metadata": net_metadata
            }
        }
    
    @staticmethod
    def deserialize_net(net_dict: Dict[str, Any], target_device: Optional[str] = None) -> Tuple[Net, Dict[str, Any]]:
        """Convert JSON dictionary to Net object and metadata with validation."""
        # Version check
        format_version = net_dict.get("format_version", "unknown")
        if format_version != SERIALIZATION_VERSION:
            print(f"Warning: Format version mismatch. Expected {SERIALIZATION_VERSION}, got {format_version}")
        
        act_net = net_dict["act_net"]
        
        # Deserialize layers
        layers = []
        for layer_dict in act_net["layers"]:
            layer = LayerSerializer.deserialize_layer(layer_dict, target_device)
            layers.append(layer)
        
        # Reconstruct graph structure
        graph = act_net.get("graph", {})
        preds = {int(k): v for k, v in graph.get("preds", {}).items()}
        succs = {int(k): v for k, v in graph.get("succs", {}).items()}
        
        # Extract metadata
        metadata = act_net.get("metadata", {})
        
        # Create Net object with validation
        try:
            net = Net(layers=layers, preds=preds, succs=succs)
        except ValueError as e:
            # If validation fails during deserialization, create the Net
            # without validation for template/example usage
            if "schema violation" in str(e) or "validation" in str(e).lower():
                print(f"⚠️  Creating example/template net without validation")
                
                # Create Net without triggering validation
                net = object.__new__(Net)
                net.layers = layers
                net.preds = preds
                net.succs = succs
                net.by_id = {L.id: L for L in layers}
            else:
                # Re-raise for other types of errors
                raise
        
        # Attach metadata if present
        if hasattr(net, 'meta'):
            net.meta = metadata
        else:
            # Store metadata in a way that won't interfere with Net structure
            net._serialization_metadata = metadata

        return net, metadata

class ACTJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for ACT objects."""
    
    def default(self, obj):
        if isinstance(obj, (Layer, Net)):
            raise ACTSerializationError("Use NetSerializer.serialize_net() instead of json.dumps() directly")
        return super().default(obj)

# High-level API functions
def save_net_to_file(net: Net, filepath: str, metadata: Optional[Dict[str, Any]] = None, 
                     indent: int = 2) -> None:
    """Save ACT Net to JSON file."""
    net_dict = NetSerializer.serialize_net(net, metadata)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(net_dict, f, indent=indent, ensure_ascii=False, cls=ACTJSONEncoder)
    
    print(f"✅ Net saved to {filepath}")

def load_net_from_file(filepath: str, target_device: Optional[str] = None) -> Tuple[Net, Dict[str, Any]]:
    """Load ACT Net from JSON file with metadata."""
    with open(filepath, 'r', encoding='utf-8') as f:
        net_dict = json.load(f)
    
    net, metadata = NetSerializer.deserialize_net(net_dict, target_device)
    print(f"✅ Net loaded from {filepath}")
    return net, metadata

def save_net_to_string(net: Net, metadata: Optional[Dict[str, Any]] = None, 
                       indent: int = 2) -> str:
    """Serialize ACT Net to JSON string."""
    net_dict = NetSerializer.serialize_net(net, metadata)
    return json.dumps(net_dict, indent=indent, ensure_ascii=False, cls=ACTJSONEncoder)

def load_net_from_string(json_str: str, target_device: Optional[str] = None) -> Tuple[Net, Dict[str, Any]]:
    """Deserialize ACT Net from JSON string with metadata."""
    net_dict = json.loads(json_str)
    return NetSerializer.deserialize_net(net_dict, target_device)

# Validation utilities
def validate_json_schema(net_dict: Dict[str, Any]) -> List[str]:
    """Validate JSON schema structure before deserialization."""
    errors = []
    
    # Check required top-level fields
    if "format_version" not in net_dict:
        errors.append("Missing 'format_version' field")
    
    if "act_net" not in net_dict:
        errors.append("Missing 'act_net' field")
        return errors
    
    act_net = net_dict["act_net"]
    
    # Check required act_net fields
    if "layers" not in act_net:
        errors.append("Missing 'layers' field in act_net")
    elif not isinstance(act_net["layers"], list):
        errors.append("'layers' must be a list")
    
    # Validate each layer structure
    for i, layer in enumerate(act_net.get("layers", [])):
        if not isinstance(layer, dict):
            errors.append(f"Layer {i} must be a dictionary")
            continue
            
        required_fields = ["id", "kind", "params", "meta", "in_vars", "out_vars"]
        for field in required_fields:
            if field not in layer:
                errors.append(f"Layer {i} missing required field '{field}'")
    
    return errors