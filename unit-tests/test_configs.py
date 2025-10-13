#!/usr/bin/env python3
"""
Single shared configuration file for bounds propagation testing.
Provides mock models, data, and specs to test real BoundsPropagate APIs.

This module provides:
- MockModelConfig, MockDataConfig, MockSpecConfig for consistent test data
- MockFactory for creating mock models and input bounds 
- get_unit_test_configs() for main unit testing configurations
- get_regression_test_configs() for performance regression testing
- Centralized configuration management for all bounds propagation tests

Used by:
- test_bounds_propagation.py (main unit tests)
- test_bounds_prop_regression.py (regression tests)
- regression_test.sh (test runner)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

# Import input parser types to understand real spec formats
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from act.base.input_parser.type import SpecType, LPNormType


@dataclass
class MockModelConfig:
    """Configuration for creating mock models."""
    name: str
    model_type: str  # 'linear', 'conv', 'mixed'
    input_shape: Tuple[int, ...]
    layers: List[int] = None
    conv_layers: List[Dict] = None
    activations: List[str] = None
    has_batchnorm: bool = False


@dataclass
class MockDataConfig:
    """Configuration for creating mock input bounds."""
    name: str
    shape: Tuple[int, ...]
    bound_type: str  # 'uniform', 'gaussian', 'adversarial'
    bound_width: float = 0.1
    center_value: float = 0.0
    norm_type: Optional[str] = None  # 'linf', 'l2'
    epsilon: Optional[float] = None


@dataclass
class MockSpecConfig:
    """Configuration for creating verification specifications."""
    name: str
    spec_type: str  # 'local_lp', 'local_vnnlib', 'set_vnnlib'
    input_config: str  # Reference to MockDataConfig
    output_constraints: Optional[Dict] = None
    classification_target: Optional[int] = None


class TestConfigurations:
    """Central configuration class for all bounds propagation tests."""
    
    # =============================================================================
    # MOCK MODEL CONFIGURATIONS
    # =============================================================================
    
    MOCK_MODELS = [
        MockModelConfig(
            name="simple_linear",
            model_type="linear", 
            input_shape=(10,),
            layers=[10, 20, 15, 5],
            activations=["relu", "relu", "linear"]
        ),
        MockModelConfig(
            name="small_cnn",
            model_type="conv",
            input_shape=(3, 32, 32),
            conv_layers=[
                {"in_channels": 3, "out_channels": 16, "kernel_size": 3, "padding": 1},
                {"in_channels": 16, "out_channels": 32, "kernel_size": 3, "stride": 2, "padding": 1}
            ],
            layers=[32*16*16, 128, 10],  # FC layers after conv
            activations=["relu", "relu", "linear"]
        ),
        MockModelConfig(
            name="mnist_mlp", 
            model_type="linear",
            input_shape=(784,),
            layers=[784, 256, 128, 64, 10],
            activations=["relu", "relu", "relu", "linear"]
        ),
        MockModelConfig(
            name="deep_conv",
            model_type="conv",
            input_shape=(1, 28, 28),
            conv_layers=[
                {"in_channels": 1, "out_channels": 8, "kernel_size": 5, "stride": 1, "padding": 2},
                {"in_channels": 8, "out_channels": 16, "kernel_size": 3, "stride": 2, "padding": 1}
            ],
            layers=[16*14*14, 64, 10],
            activations=["relu", "linear"],
            has_batchnorm=True
        ),
        MockModelConfig(
            name="mixed_network",
            model_type="mixed",
            input_shape=(3, 16, 16), 
            conv_layers=[
                {"in_channels": 3, "out_channels": 8, "kernel_size": 3, "padding": 1}
            ],
            layers=[8*16*16, 32, 8, 3],
            activations=["relu", "relu", "linear"]
        )
    ]
    
    # =============================================================================
    # MOCK DATA CONFIGURATIONS  
    # =============================================================================
    
    MOCK_DATA = [
        MockDataConfig(
            name="small_uniform",
            shape=(10,),
            bound_type="uniform",
            bound_width=0.1
        ),
        MockDataConfig(
            name="mnist_linf",
            shape=(784,),
            bound_type="uniform", 
            bound_width=0.3,
            norm_type="linf",
            epsilon=0.03
        ),
        MockDataConfig(
            name="cifar_linf",
            shape=(3, 32, 32),
            bound_type="uniform",
            bound_width=0.2,
            norm_type="linf", 
            epsilon=0.031  # 8/255
        ),
        MockDataConfig(
            name="mnist_l2",
            shape=(1, 28, 28),
            bound_type="gaussian",
            bound_width=0.5,
            norm_type="l2",
            epsilon=1.0
        ),
        MockDataConfig(
            name="adversarial_bounds",
            shape=(3, 16, 16),  # Match mixed_network model input_shape
            bound_type="adversarial",
            bound_width=0.1,
            center_value=0.5
        ),
        MockDataConfig(
            name="wide_bounds", 
            shape=(10,),  # Match simple_linear model input_shape
            bound_type="uniform",
            bound_width=2.0  # Very wide for stress testing
        )
    ]
    
    # =============================================================================
    # MOCK SPECIFICATION CONFIGURATIONS
    # =============================================================================
    
    MOCK_SPECS = [
        MockSpecConfig(
            name="mnist_robustness",
            spec_type="local_lp",
            input_config="mnist_linf",
            classification_target=7  # Target class for robustness
        ),
        MockSpecConfig(
            name="cifar_adversarial",
            spec_type="local_lp", 
            input_config="cifar_linf",
            classification_target=3
        ),
        MockSpecConfig(
            name="simple_bounds",
            spec_type="local_lp",
            input_config="small_uniform",
            output_constraints={"type": "classification", "num_classes": 5}
        ),
        MockSpecConfig(
            name="l2_robustness",
            spec_type="local_lp",
            input_config="mnist_l2",
            classification_target=1
        )
    ]
    
    # =============================================================================
    # TEST SUITE CONFIGURATIONS
    # =============================================================================
    
    UNIT_TEST_CONFIGS = {
        "layer_tests": [
            {"model": "simple_linear", "data": "small_uniform", "focus": "linear_layers"},
            {"model": "small_cnn", "data": "cifar_linf", "focus": "conv_layers"},
            {"model": "deep_conv", "data": "mnist_l2", "focus": "batchnorm"},
            {"model": "mixed_network", "data": "adversarial_bounds", "focus": "structural"}
        ],
        "end_to_end_tests": [
            {"model": "mnist_mlp", "data": "mnist_linf", "spec": "mnist_robustness"},
            {"model": "small_cnn", "data": "cifar_linf", "spec": "cifar_adversarial"},
            {"model": "simple_linear", "data": "small_uniform", "spec": "simple_bounds"}
        ],
        "property_tests": [
            {"model": "simple_linear", "data": "wide_bounds", "property": "monotonicity"},
            {"model": "mnist_mlp", "data": "mnist_linf", "property": "soundness"},
            {"model": "small_cnn", "data": "cifar_linf", "property": "bounds_ordering"}
        ]
    }
    
    REGRESSION_TEST_CONFIGS = {
        "performance_tests": [
            {"model": "simple_linear", "data": "small_uniform", "iterations": 5},
            {"model": "mnist_mlp", "data": "mnist_linf", "iterations": 3},
            {"model": "small_cnn", "data": "cifar_linf", "iterations": 2},
            {"model": "deep_conv", "data": "mnist_l2", "iterations": 2}
        ],
        "correctness_tests": [
            {"model": "simple_linear", "data": "small_uniform"},
            {"model": "mnist_mlp", "data": "mnist_linf"}, 
            {"model": "small_cnn", "data": "cifar_linf"},
            {"model": "mixed_network", "data": "adversarial_bounds"}
        ]
    }


class MockFactory:
    """Factory for creating mock inputs from configurations."""
    
    @staticmethod
    def create_model(config_name: str) -> nn.Module:
        """Create mock model from configuration."""
        config = next((c for c in TestConfigurations.MOCK_MODELS if c.name == config_name), None)
        if not config:
            raise ValueError(f"Model config '{config_name}' not found")
        
        if config.model_type == "linear":
            return MockFactory._create_linear_model(config)
        elif config.model_type == "conv":
            return MockFactory._create_conv_model(config)
        elif config.model_type == "mixed":
            return MockFactory._create_mixed_model(config)
        else:
            raise ValueError(f"Unknown model type: {config.model_type}")
    
    @staticmethod
    def create_data(config_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create mock input bounds from configuration."""
        config = next((c for c in TestConfigurations.MOCK_DATA if c.name == config_name), None)
        if not config:
            raise ValueError(f"Data config '{config_name}' not found")
        
        if config.bound_type == "uniform":
            return MockFactory._create_uniform_bounds(config)
        elif config.bound_type == "gaussian":
            return MockFactory._create_gaussian_bounds(config)
        elif config.bound_type == "adversarial":
            return MockFactory._create_adversarial_bounds(config)
        else:
            raise ValueError(f"Unknown bound type: {config.bound_type}")
    
    @staticmethod
    def create_spec(config_name: str) -> Dict[str, Any]:
        """Create mock specification from configuration."""
        config = next((c for c in TestConfigurations.MOCK_SPECS if c.name == config_name), None)
        if not config:
            raise ValueError(f"Spec config '{config_name}' not found")
        
        # Get associated data config
        data_config = next((c for c in TestConfigurations.MOCK_DATA if c.name == config.input_config), None)
        
        spec = {
            "spec_type": config.spec_type,
            "input_shape": data_config.shape if data_config else None,
            "norm_type": data_config.norm_type if data_config else None,
            "epsilon": data_config.epsilon if data_config else None,
            "classification_target": config.classification_target,
            "output_constraints": config.output_constraints
        }
        
        return spec
    
    # =============================================================================
    # PRIVATE FACTORY METHODS
    # =============================================================================
    
    @staticmethod
    def _create_linear_model(config: MockModelConfig) -> nn.Module:
        """Create linear model from config."""
        layers = []
        for i in range(len(config.layers) - 1):
            layers.append(nn.Linear(config.layers[i], config.layers[i + 1]))
            if i < len(config.activations) and config.activations[i] == "relu":
                layers.append(nn.ReLU())
        return nn.Sequential(*layers)
    
    @staticmethod
    def _create_conv_model(config: MockModelConfig) -> nn.Module:
        """Create convolutional model from config."""
        layers = []
        
        # Add conv layers
        for i, conv_config in enumerate(config.conv_layers):
            layers.append(nn.Conv2d(**conv_config))
            if config.has_batchnorm:
                layers.append(nn.BatchNorm2d(conv_config["out_channels"]))
            if i < len(config.activations) and config.activations[i] == "relu":
                layers.append(nn.ReLU())
        
        # Add flatten
        layers.append(nn.Flatten())
        
        # Add FC layers
        if config.layers:
            for i in range(len(config.layers) - 1):
                layers.append(nn.Linear(config.layers[i], config.layers[i + 1]))
                if i + len(config.conv_layers) < len(config.activations):
                    if config.activations[i + len(config.conv_layers)] == "relu":
                        layers.append(nn.ReLU())
        
        return nn.Sequential(*layers)
    
    @staticmethod
    def _create_mixed_model(config: MockModelConfig) -> nn.Module:
        """Create mixed conv+linear model from config."""
        return MockFactory._create_conv_model(config)  # Same as conv for now
    
    @staticmethod
    def _create_uniform_bounds(config: MockDataConfig) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create uniform random bounds."""
        center = torch.full(config.shape, config.center_value)
        if config.bound_width > 0:
            center += torch.randn(config.shape) * 0.1  # Small randomization
        
        half_width = config.bound_width / 2
        lb = center - half_width
        ub = center + half_width
        return lb, ub
    
    @staticmethod
    def _create_gaussian_bounds(config: MockDataConfig) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create Gaussian-distributed bounds."""
        center = torch.randn(config.shape) * 0.5 + config.center_value
        half_width = config.bound_width / 2
        lb = center - half_width
        ub = center + half_width
        return lb, ub
    
    @staticmethod
    def _create_adversarial_bounds(config: MockDataConfig) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create adversarial-style bounds (tighter in some dimensions)."""
        center = torch.full(config.shape, config.center_value)
        
        # Make some dimensions tighter
        widths = torch.full(config.shape, config.bound_width)
        if len(config.shape) > 1:
            # Tighten some channels/features
            widths[::2] *= 0.1  # Every other dimension much tighter
        
        lb = center - widths / 2
        ub = center + widths / 2
        return lb, ub


# =============================================================================
# CONFIGURATION ACCESS HELPERS
# =============================================================================

def get_model_config(name: str) -> MockModelConfig:
    """Get model configuration by name."""
    return next((c for c in TestConfigurations.MOCK_MODELS if c.name == name), None)

def get_data_config(name: str) -> MockDataConfig:
    """Get data configuration by name."""
    return next((c for c in TestConfigurations.MOCK_DATA if c.name == name), None)

def get_spec_config(name: str) -> MockSpecConfig:
    """Get spec configuration by name."""
    return next((c for c in TestConfigurations.MOCK_SPECS if c.name == name), None)

def get_unit_test_configs(category: str = None) -> List[Dict]:
    """Get unit test configurations."""
    if category:
        return TestConfigurations.UNIT_TEST_CONFIGS.get(category, [])
    return TestConfigurations.UNIT_TEST_CONFIGS

def get_regression_test_configs(category: str = None) -> List[Dict]:
    """Get regression test configurations."""
    if category:
        return TestConfigurations.REGRESSION_TEST_CONFIGS.get(category, [])
    return TestConfigurations.REGRESSION_TEST_CONFIGS