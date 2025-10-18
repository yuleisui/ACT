#===- act/pipeline/mock_factory.py - Mock Input Factory ----------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Mock input factory for generating test inputs from YAML configuration.
#   Provides configurable generation of mock data, input specifications,
#   output specifications, and neural network models for testing.
#
#===---------------------------------------------------------------------===#


import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Tuple, List, Callable, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging

from act.pipeline.config import load_config

logger = logging.getLogger(__name__)


@dataclass
class MockGenerationError(Exception):
    """Error in mock input generation."""
    generator_type: str
    config_name: str
    message: str
    
    def __str__(self):
        return f"Mock generation error for {self.generator_type} '{self.config_name}': {self.message}"


class BaseGenerator(ABC):
    """Base class for mock input generators."""
    
    @abstractmethod
    def generate(self, config: Dict[str, Any]) -> Any:
        """Generate mock input from configuration."""
        pass


class SampleDataGenerator(BaseGenerator):
    """Generate sample data (images, tensors) from configuration."""
    
    def generate(self, config: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate sample data and labels.
        
        Args:
            config: Configuration dictionary with 'type', 'shape', 'distribution', etc.
            
        Returns:
            Tuple of (sample_data, labels)
        """
        data_type = config.get("type", "image")
        shape = config.get("shape", [1, 28, 28])
        batch_size = config.get("batch_size", 10)
        distribution = config.get("distribution", "uniform")
        
        # Full shape including batch dimension
        full_shape = [batch_size] + shape
        
        # Generate data based on distribution
        if distribution == "uniform":
            range_min, range_max = config.get("range", [0, 1])
            data = torch.rand(full_shape) * (range_max - range_min) + range_min
            
        elif distribution == "normal":
            mean = config.get("mean", 0.0)
            std = config.get("std", 1.0)
            data = torch.randn(full_shape) * std + mean
            
        elif distribution == "gaussian_noise":
            base_value = config.get("base_value", 0.5)
            noise_level = config.get("noise_level", 0.1)
            data = torch.full(full_shape, base_value) + torch.randn(full_shape) * noise_level
            
        else:
            raise MockGenerationError("sample_data", str(config), f"Unknown distribution: {distribution}")
        
        # Clamp to valid range if specified
        if "range" in config:
            range_min, range_max = config["range"]
            data = torch.clamp(data, range_min, range_max)
        
        # Generate corresponding labels
        num_classes = config.get("num_classes", 10)
        labels = torch.randint(0, num_classes, (batch_size,))
        
        logger.debug(f"Generated {data_type} data: shape={data.shape}, range=[{data.min():.3f}, {data.max():.3f}]")
        
        return data, labels


class InputSpecGenerator(BaseGenerator):
    """Generate input specifications from configuration."""
    
    def generate(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate input specification.
        
        Args:
            config: Configuration dictionary with spec details
            
        Returns:
            Input specification dictionary
        """
        spec_type = config.get("spec_type", "LOCAL_LP")
        
        if spec_type == "LOCAL_LP":
            return self._generate_lp_spec(config)
        elif spec_type == "LOCAL_VNNLIB":
            return self._generate_vnnlib_spec(config)
        elif spec_type == "SET_BOX":
            return self._generate_box_spec(config)
        else:
            raise MockGenerationError("input_spec", str(config), f"Unknown spec type: {spec_type}")
    
    def _generate_lp_spec(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate LP (Lp norm) specification."""
        norm_type = config.get("norm_type", "LINF")
        epsilon = config.get("epsilon", 0.1)
        center_point = config.get("center_point", "auto")
        
        return {
            "type": "LOCAL_LP",
            "norm_type": norm_type,
            "epsilon": epsilon,
            "center_point": center_point
        }
    
    def _generate_vnnlib_spec(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate VNNLIB specification."""
        constraints = config.get("constraints", [])
        bounds = config.get("bounds", {"lower": [], "upper": []})
        
        return {
            "type": "LOCAL_VNNLIB",
            "constraints": constraints,
            "bounds": bounds
        }
    
    def _generate_box_spec(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate box constraint specification."""
        lower_bounds = config.get("lower_bounds", [])
        upper_bounds = config.get("upper_bounds", [])
        
        return {
            "type": "SET_BOX",
            "lower_bounds": lower_bounds,
            "upper_bounds": upper_bounds
        }


class OutputSpecGenerator(BaseGenerator):
    """Generate output specifications from configuration."""
    
    def generate(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate output specification.
        
        Args:
            config: Configuration dictionary with output spec details
            
        Returns:
            Output specification dictionary
        """
        spec_type = config.get("type", "classification")
        
        if spec_type == "classification":
            return self._generate_classification_spec(config)
        elif spec_type == "robustness":
            return self._generate_robustness_spec(config)
        else:
            raise MockGenerationError("output_spec", str(config), f"Unknown output spec type: {spec_type}")
    
    def _generate_classification_spec(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate classification output specification."""
        num_classes = config.get("num_classes", 10)
        target_label = config.get("target_label", None)
        
        return {
            "type": "classification",
            "num_classes": num_classes,
            "target_label": target_label
        }
    
    def _generate_robustness_spec(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate robustness output specification."""
        property_desc = config.get("property", "margin >= 0.1")
        margin_threshold = config.get("margin_threshold", 0.1)
        
        return {
            "type": "robustness",
            "property": property_desc,
            "margin_threshold": margin_threshold
        }


class ModelGenerator(BaseGenerator):
    """Generate neural network models from configuration."""
    
    def generate(self, config: Dict[str, Any]) -> nn.Module:
        """
        Generate neural network model.
        
        Args:
            config: Configuration dictionary with model architecture
            
        Returns:
            PyTorch neural network model
        """
        architecture = config.get("architecture", "feedforward")
        
        if architecture == "feedforward":
            return self._generate_feedforward_model(config)
        elif architecture == "cnn":
            return self._generate_cnn_model(config)
        else:
            raise MockGenerationError("model", str(config), f"Unknown architecture: {architecture}")
    
    def _generate_feedforward_model(self, config: Dict[str, Any]) -> nn.Module:
        """Generate feedforward neural network."""
        layers = config.get("layers", [784, 50, 10])
        activations = config.get("activations", ["relu", "linear"])
        
        if len(activations) != len(layers) - 1:
            raise MockGenerationError("model", str(config), 
                                    f"Number of activations ({len(activations)}) must be layers-1 ({len(layers)-1})")
        
        modules = []
        for i in range(len(layers) - 1):
            modules.append(nn.Linear(layers[i], layers[i + 1]))
            
            activation = activations[i].lower()
            if activation == "relu":
                modules.append(nn.ReLU())
            elif activation == "sigmoid":
                modules.append(nn.Sigmoid())
            elif activation == "tanh":
                modules.append(nn.Tanh())
            elif activation != "linear":
                raise MockGenerationError("model", str(config), f"Unknown activation: {activation}")
        
        model = nn.Sequential(*modules)
        logger.debug(f"Generated feedforward model: {layers} with activations {activations}")
        
        return model
    
    def _generate_cnn_model(self, config: Dict[str, Any]) -> nn.Module:
        """Generate convolutional neural network."""
        conv_layers = config.get("conv_layers", [[1, 16, 3], [16, 32, 3]])
        fc_layers = config.get("fc_layers", [32*6*6, 10])
        
        modules = []
        
        # Convolutional layers
        for i, (in_channels, out_channels, kernel_size) in enumerate(conv_layers):
            modules.append(nn.Conv2d(in_channels, out_channels, kernel_size))
            modules.append(nn.ReLU())
            if i < len(conv_layers) - 1:  # No pooling on last conv layer
                modules.append(nn.MaxPool2d(2))
        
        # Flatten for fully connected layers
        modules.append(nn.Flatten())
        
        # Fully connected layers
        for i in range(len(fc_layers) - 1):
            modules.append(nn.Linear(fc_layers[i], fc_layers[i + 1]))
            if i < len(fc_layers) - 2:  # No activation on output layer
                modules.append(nn.ReLU())
        
        model = nn.Sequential(*modules)
        logger.debug(f"Generated CNN model: conv_layers={conv_layers}, fc_layers={fc_layers}")
        
        return model


class MockInputFactory:
    """Factory for generating all types of mock inputs from YAML configuration."""
    
    def __init__(self, config_path: str = "mock_inputs"):
        """
        Initialize mock input factory.
        
        Args:
            config_path: Path to mock inputs configuration file
        """
        self.config = load_config(config_path)
        self.generators = {
            "sample_data": SampleDataGenerator(),
            "input_specs": InputSpecGenerator(),
            "output_specs": OutputSpecGenerator(),
            "models": ModelGenerator()
        }
        self.custom_generators: Dict[str, Callable] = {}
        
    def create_sample_data(self, config_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate sample data and labels from configuration.
        
        Args:
            config_name: Name of configuration in sample_data section
            
        Returns:
            Tuple of (sample_data, labels)
        """
        return self._generate("sample_data", config_name)
    
    def create_input_spec(self, config_name: str) -> Dict[str, Any]:
        """
        Generate input specification from configuration.
        
        Args:
            config_name: Name of configuration in input_specs section
            
        Returns:
            Input specification dictionary
        """
        return self._generate("input_specs", config_name)
    
    def create_output_spec(self, config_name: str) -> Dict[str, Any]:
        """
        Generate output specification from configuration.
        
        Args:
            config_name: Name of configuration in output_specs section
            
        Returns:
            Output specification dictionary
        """
        return self._generate("output_specs", config_name)
    
    def create_model(self, config_name: str) -> nn.Module:
        """
        Generate neural network model from configuration.
        
        Args:
            config_name: Name of configuration in models section
            
        Returns:
            PyTorch neural network model
        """
        return self._generate("models", config_name)
    
    def _generate(self, section: str, config_name: str) -> Any:
        """Generate mock input from specified section and config name."""
        if section not in self.config:
            raise MockGenerationError(section, config_name, f"Missing section '{section}' in configuration")
        
        if config_name not in self.config[section]:
            available = list(self.config[section].keys())
            raise MockGenerationError(section, config_name, 
                                    f"Config '{config_name}' not found. Available: {available}")
        
        config = self.config[section][config_name]
        
        # Check for custom generator
        generator_name = config.get("generator")
        if generator_name and generator_name in self.custom_generators:
            return self.custom_generators[generator_name](config)
        
        # Use built-in generator
        generator = self.generators[section]
        return generator.generate(config)
    
    def register_custom_generator(self, name: str, generator_func: Callable) -> None:
        """
        Register a custom generator function.
        
        Args:
            name: Name to identify the custom generator
            generator_func: Function that takes config dict and returns generated input
        """
        self.custom_generators[name] = generator_func
        logger.info(f"Registered custom generator: {name}")
    
    def list_available_configs(self) -> Dict[str, List[str]]:
        """
        List all available configurations.
        
        Returns:
            Dictionary mapping section names to lists of config names
        """
        available = {}
        for section in self.generators.keys():
            if section in self.config:
                available[section] = list(self.config[section].keys())
            else:
                available[section] = []
        
        return available
    
    def create_test_batch(self, scenario_config: Dict[str, str]) -> Dict[str, Any]:
        """
        Create a complete test batch from scenario configuration.
        
        Args:
            scenario_config: Dictionary with keys for sample_data, input_spec, output_spec, model
            
        Returns:
            Dictionary containing all generated test components
        """
        batch = {}
        
        if "sample_data" in scenario_config:
            batch["sample_data"], batch["labels"] = self.create_sample_data(scenario_config["sample_data"])
        
        if "input_spec" in scenario_config:
            batch["input_spec"] = self.create_input_spec(scenario_config["input_spec"])
        
        if "output_spec" in scenario_config:
            batch["output_spec"] = self.create_output_spec(scenario_config["output_spec"])
        
        if "model" in scenario_config:
            batch["model"] = self.create_model(scenario_config["model"])
        
        return batch
