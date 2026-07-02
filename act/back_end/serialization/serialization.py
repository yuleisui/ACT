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
import torch

from act.back_end.core import Layer, Net
from act.back_end.layer_schema import REGISTRY, LayerKind
from act.back_end.layer_util import validate_layer

SERIALIZATION_VERSION = "2.0"

class ACTSerializationError(Exception):
    """Custom exception for serialization errors."""
    pass

class TensorEncoder:
    """Handles PyTorch tensor encoding/decoding to/from JSON-compatible format."""
    
    @staticmethod
    def encode_tensor(tensor: torch.Tensor) -> Dict[str, Any]:
        """Convert PyTorch tensor to JSON-serializable dictionary."""
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
        """Convert JSON dictionary back to PyTorch tensor.

        Floating-point tensors are normalized to device_manager's default
        dtype so JSONs serialized at float64 still load cleanly into a
        float32 runtime (and vice versa). Integer / bool dtypes are
        preserved as-is — coercing them to float breaks invariants in
        downstream consumers (e.g. ``y_true`` must stay ``torch.long`` for
        ``y[arange(B), y_true]`` advanced indexing in
        ``OutputSpecLayer.forward``).
        """
        encoded_data = tensor_dict["data"]
        buffer = io.BytesIO(base64.b64decode(encoded_data.encode('utf-8')))
        np_array = np.load(buffer)

        tensor = torch.from_numpy(np_array)
        if tensor.is_floating_point():
            from act.util.device_manager import get_default_dtype
            tensor = tensor.to(dtype=get_default_dtype())

        if target_device is None:
            from act.util.device_manager import get_default_device
            device = get_default_device()
        else:
            device = torch.device(target_device)
        tensor = tensor.to(device=device)

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
            if isinstance(value, torch.Tensor):
                params_encoded[name] = TensorEncoder.encode_tensor(value)
            else:
                # Handle non-tensor parameters (floats, ints, strings, etc.)
                params_encoded[name] = value
        
        # Encode cache tensors (if any)
        cache_encoded = {}
        if hasattr(layer, 'cache'):
            for name, value in layer.cache.items():
                if isinstance(value, torch.Tensor):
                    cache_encoded[name] = TensorEncoder.encode_tensor(value)
                else:
                    # Handle non-tensor cache values
                    cache_encoded[name] = value
        
        return {
            "id": layer.id,
            "kind": layer.kind,
            "params": params_encoded,
            "in_vars": layer.in_vars,
            "out_vars": layer.out_vars,
            "cache": cache_encoded
        }
    
    @staticmethod
    def deserialize_layer(layer_dict: Dict[str, Any], target_device: Optional[str] = None) -> Layer:
        """Convert JSON dictionary to Layer object."""
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
        
        try:
            layer = Layer(
                id=layer_dict["id"],
                kind=layer_dict["kind"],
                params=params_decoded,
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
    def serialize_net(net: Net) -> Dict[str, Any]:
        """Convert Net to JSON-serializable dictionary."""
        layers_serialized = []
        for layer in net.layers:
            layers_serialized.append(LayerSerializer.serialize_layer(layer))
        
        return {
            "format_version": SERIALIZATION_VERSION,
            "act_net": {
                "layers": layers_serialized,
                "graph": {
                    "preds": {str(k): v for k, v in net.preds.items()},
                    "succs": {str(k): v for k, v in net.succs.items()}
                }
            }
        }
    
    @staticmethod
    def deserialize_net(net_dict: Dict[str, Any], target_device: Optional[str] = None) -> Net:
        """Convert JSON dictionary to Net object."""
        format_version = net_dict.get("format_version", "unknown")
        if format_version != SERIALIZATION_VERSION:
            raise ValueError(f"Unsupported format version: {format_version}. Expected {SERIALIZATION_VERSION}")
        
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
        
        # Create Net object with validation
        try:
            return Net(layers=layers, preds=preds, succs=succs)
        except ValueError as e:
            # If validation fails during deserialization, create the Net
            # without validation for template/example usage
            if "schema violation" in str(e) or "validation" in str(e).lower():
                print(f"Warning: Creating net without validation")
                net = object.__new__(Net)
                net.layers = layers
                net.preds = preds
                net.succs = succs
                net.by_id = {L.id: L for L in layers}
                return net
            else:
                raise

class ACTJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for ACT objects."""
    
    def default(self, o):
        if isinstance(o, (Layer, Net)):
            raise ACTSerializationError("Use NetSerializer.serialize_net() instead of json.dumps() directly")
        return super().default(o)

# High-level API functions
def save_net_to_file(net: Net, filepath: str, indent: int = 2) -> None:
    """Save ACT Net to JSON file."""
    net_dict = NetSerializer.serialize_net(net)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(net_dict, f, indent=indent, ensure_ascii=False, cls=ACTJSONEncoder)

def load_net_from_file(filepath: str, target_device: Optional[str] = None) -> Net:
    """Load ACT Net from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        net_dict = json.load(f)
    return NetSerializer.deserialize_net(net_dict, target_device)

def save_net_to_string(net: Net, indent: int = 2) -> str:
    """Serialize ACT Net to JSON string."""
    net_dict = NetSerializer.serialize_net(net)
    return json.dumps(net_dict, indent=indent, ensure_ascii=False, cls=ACTJSONEncoder)

def load_net_from_string(json_str: str, target_device: Optional[str] = None) -> Net:
    """Deserialize ACT Net from JSON string."""
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
            
        required_fields = ["id", "kind", "params", "in_vars", "out_vars"]
        for field in required_fields:
            if field not in layer:
                errors.append(f"Layer {i} missing required field '{field}'")
    
    return errors
