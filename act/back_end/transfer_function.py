"""
Transfer Function Interface

This module defines the abstract interface for transfer function implementations
in the ACT verification framework. Transfer functions compute bounds and constraints
for different layer types during the analysis phase.

The interface supports multiple implementations:
- IntervalTF: Interval-based bounds propagation  
- HybridzTF: HybridZ zonotope-based analysis with enhanced precision
"""

import torch
from abc import ABC, abstractmethod
from typing import Dict, List
from act.back_end.core import Bounds, Fact, Layer, Net


class AnalysisContext:
    """Context object providing access to network state during transfer function dispatch."""
    
    def __init__(self, net: Net, before: Dict[int, Fact], after: Dict[int, Fact]):
        self.net = net
        self.before = before 
        self.after = after
        
    def get_predecessor_bounds(self, layer_id: int, pred_index: int = 0) -> Bounds:
        """Get bounds from specific predecessor by index."""
        if layer_id not in self.net.preds or pred_index >= len(self.net.preds[layer_id]):
            raise IndexError(f"Layer {layer_id} has no predecessor at index {pred_index}")
        
        pred_id = self.net.preds[layer_id][pred_index]
        return self.after[pred_id].bounds if pred_id in self.after else self.before[pred_id].bounds
        
    def get_all_predecessor_bounds(self, layer_id: int) -> List[Bounds]:
        """Get bounds from all predecessors of the given layer."""
        if layer_id not in self.net.preds:
            return []
        return [self.get_predecessor_bounds(layer_id, i) 
                for i in range(len(self.net.preds[layer_id]))]
                
    def get_layer_bounds(self, layer_id: int, use_after: bool = True) -> Bounds:
        """Get current bounds for a specific layer."""
        facts = self.after if use_after else self.before
        return facts[layer_id].bounds if layer_id in facts else self.before[layer_id].bounds


class TransferFunction(ABC):
    """Abstract base class for transfer function implementations.
    
    Transfer functions compute output bounds and constraints for network layers
    during the analysis phase. Different implementations provide different
    precision/performance tradeoffs.
    """
    
    @abstractmethod
    def supports_layer(self, layer_kind: str) -> bool:
        """Check if this transfer function implementation supports the given layer kind.
        
        Args:
            layer_kind: Layer type (e.g., "DENSE", "RELU", "CONV2D")
            
        Returns:
            True if this implementation can handle the layer kind
        """
        pass
    
    @abstractmethod
    def apply(self, L: Layer, input_bounds: Bounds, net: Net, 
              before: Dict[int, Fact], after: Dict[int, Fact]) -> Fact:
        """Apply transfer function to compute output bounds and constraints.
        
        Args:
            L: Layer to process
            input_bounds: Input bounds for this layer  
            net: Complete network structure
            before: Pre-processing facts for all layers
            after: Post-processing facts for all layers
            
        Returns:
            Fact containing output bounds and generated constraints
        """
        pass
    
    @property
    @abstractmethod 
    def name(self) -> str:
        """Implementation name for debugging and logging."""
        pass


# Global transfer function management
_current_tf: TransferFunction = None


def set_transfer_function(tf_impl: TransferFunction) -> None:
    """Set the global transfer function implementation."""
    global _current_tf
    _current_tf = tf_impl


def get_transfer_function() -> TransferFunction:
    """Get the current transfer function implementation."""
    if _current_tf is None:
        raise RuntimeError("No transfer function implementation set. Call set_transfer_function() first.")
    return _current_tf


def set_transfer_function_mode(mode: str = "interval") -> None:
    """Set transfer function implementation by mode name.
    
    Args:
        mode: "interval" for IntervalTF, "hybridz" for HybridzTF
    """
    if mode == "interval":
        from act.back_end.interval_tf import IntervalTF
        set_transfer_function(IntervalTF())
    elif mode == "hybridz":
        from act.back_end.hybridz_tf import HybridzTF
        set_transfer_function(HybridzTF())
    else:
        raise ValueError(f"Unknown transfer function mode: {mode}. Use 'interval' or 'hybridz'.")


@torch.no_grad()
def dispatch_tf(L: Layer, before: Dict[int, Fact], after: Dict[int, Fact], net: Net) -> Fact:
    """Dispatch to current transfer function implementation.
    
    This is the main entry point called by analyze() for each layer.
    """
    tf_impl = get_transfer_function()
    input_bounds = before[L.id].bounds
    return tf_impl.apply(L, input_bounds, net, before, after)