#!/usr/bin/env python3
"""Concise YAML-driven network factory for ACT examples."""

import json
import yaml
import torch
from pathlib import Path
from typing import Dict, Any, List

from act.back_end.core import Layer, Net
from act.back_end.serialization.serialization import NetSerializer


class NetFactory:
    """Concise factory that reads config and generates models in nets folder."""
    
    def __init__(self, config_path: str = "act/back_end/examples/examples_config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.output_dir = Path("act/back_end/examples/nets")
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_weight_tensor(self, kind: str, meta: Dict[str, Any]) -> torch.Tensor:
        """Generate minimal weight tensors that satisfy schema requirements."""
        if kind == "DENSE":
            in_features = meta.get("in_features", 10)
            out_features = meta.get("out_features", 10)
            # Create minimal weight tensor W
            return torch.randn(out_features, in_features) * 0.1
        elif kind in ["CONV2D", "CONV1D", "CONV3D"]:
            in_channels = meta.get("in_channels", 1)
            out_channels = meta.get("out_channels", 1)
            kernel_size = meta.get("kernel_size", 3)
            if isinstance(kernel_size, int):
                if kind == "CONV1D":
                    weight_shape = (out_channels, in_channels, kernel_size)
                elif kind == "CONV2D":
                    weight_shape = (out_channels, in_channels, kernel_size, kernel_size)
                else:  # CONV3D
                    weight_shape = (out_channels, in_channels, kernel_size, kernel_size, kernel_size)
            else:
                # kernel_size is a tuple/list
                if kind == "CONV1D":
                    weight_shape = (out_channels, in_channels, kernel_size[0])
                elif kind == "CONV2D":
                    weight_shape = (out_channels, in_channels, kernel_size[0], kernel_size[1])
                else:  # CONV3D
                    weight_shape = (out_channels, in_channels, kernel_size[0], kernel_size[1], kernel_size[2])
            return torch.randn(*weight_shape) * 0.1
        return None
    
    def create_network(self, name: str, spec: Dict[str, Any]) -> Net:
        """Create network from YAML spec."""
        layers = []
        for i, layer_spec in enumerate(spec['layers']):
            # Simple sequential variable assignment
            in_vars = [i] if i > 0 else []
            out_vars = [i + 1]
            
            # Copy params and add required weight tensors if needed
            params = layer_spec.get('params', {}).copy()
            meta = layer_spec.get('meta', {})
            kind = layer_spec['kind']
            
            # Generate required weight tensors for layers that need them
            if kind == "DENSE" and "W" not in params:
                weight = self.generate_weight_tensor(kind, meta)
                if weight is not None:
                    params["W"] = weight
            elif kind.startswith("CONV") and "weight" not in params:
                weight = self.generate_weight_tensor(kind, meta)
                if weight is not None:
                    params["weight"] = weight
            
            layer = Layer(
                id=i,
                kind=kind,
                params=params,
                meta=meta,
                in_vars=in_vars,
                out_vars=out_vars
            )
            layers.append(layer)
        
        # Create graph structure for Net
        preds = {i: [i-1] if i > 0 else [] for i in range(len(layers))}
        succs = {i: [i+1] if i < len(layers)-1 else [] for i in range(len(layers))}
        
        net = Net(layers=layers, preds=preds, succs=succs)
        net.meta = {
            'name': name,
            'description': spec.get('description', ''),
            'architecture_type': spec.get('architecture_type', ''),
            'input_shape': spec.get('input_shape', [])
        }
        return net
    
    def save_network(self, net: Net, name: str) -> None:
        """Save network using proper ACT serialization with tensor encoding."""
        output_path = self.output_dir / f"{name}.json"
        
        # Use NetSerializer to properly handle tensors
        net_dict = NetSerializer.serialize_net(net, metadata={'generated_by': 'NetFactory'})
        
        with open(output_path, 'w') as f:
            json.dump(net_dict, f, indent=2)
        print(f"Saved: {output_path}")
    
    def generate_all(self) -> None:
        """Generate all networks from config."""
        networks = self.config['networks']
        print(f"Generating {len(networks)} networks...")
        
        for name, spec in networks.items():
            net = self.create_network(name, spec)
            self.save_network(net, name)
        
        print(f"All networks generated in {self.output_dir}")


if __name__ == "__main__":
    factory = NetFactory()
    factory.generate_all()