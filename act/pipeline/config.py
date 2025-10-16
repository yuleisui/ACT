"""
Configuration loading and management for ACT pipeline testing framework.

This module provides centralized configuration management for the pipeline testing system,
including YAML loading, validation, and default configurations.
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ValidationError(Exception):
    """Configuration validation error."""
    field: str
    message: str
    
    def __str__(self):
        return f"Configuration error in '{self.field}': {self.message}"


@dataclass
class ConfigPaths:
    """Standard configuration file paths."""
    mock_inputs: str = "configs/mock_inputs.yaml"
    test_scenarios: str = "configs/test_scenarios.yaml"
    solver_settings: str = "configs/solver_settings.yaml"
    baselines: str = "configs/baselines.json"


class ConfigManager:
    """Centralized configuration management for pipeline testing."""
    
    def __init__(self, base_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            base_path: Base directory for configuration files. Defaults to pipeline directory.
        """
        if base_path is None:
            base_path = Path(__file__).parent
        self.base_path = Path(base_path)
        self.paths = ConfigPaths()
        self._config_cache: Dict[str, Any] = {}
        
    def load_config(self, config_name: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Load configuration from YAML/JSON file.
        
        Args:
            config_name: Name of config file or path relative to base_path
            use_cache: Whether to use cached configuration
            
        Returns:
            Loaded configuration dictionary
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValidationError: If config format is invalid
        """
        if use_cache and config_name in self._config_cache:
            return self._config_cache[config_name]
            
        config_path = self._resolve_config_path(config_name)
        
        try:
            if config_path.suffix.lower() == '.json':
                config = self._load_json(config_path)
            else:
                config = self._load_yaml(config_path)
                
            self._validate_config(config, config_name)
            
            if use_cache:
                self._config_cache[config_name] = config
                
            logger.info(f"Loaded configuration: {config_path}")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load config {config_path}: {e}")
            raise
    
    def _resolve_config_path(self, config_name: str) -> Path:
        """Resolve configuration file path."""
        # Check if it's a predefined config
        if hasattr(self.paths, config_name):
            relative_path = getattr(self.paths, config_name)
            config_path = self.base_path / relative_path
        else:
            # Treat as relative path
            config_path = self.base_path / config_name
            
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        return config_path
    
    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        """Load YAML configuration file."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ValidationError("yaml_format", f"Invalid YAML format: {e}")
    
    def _load_json(self, path: Path) -> Dict[str, Any]:
        """Load JSON configuration file."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ValidationError("json_format", f"Invalid JSON format: {e}")
    
    def _validate_config(self, config: Dict[str, Any], config_name: str) -> None:
        """Validate configuration structure."""
        if not isinstance(config, dict):
            raise ValidationError("root", "Configuration must be a dictionary")
            
        # Specific validations based on config type
        if "mock_inputs" in config_name:
            self._validate_mock_inputs_config(config)
        elif "test_scenarios" in config_name:
            self._validate_test_scenarios_config(config)
        elif "solver_settings" in config_name:
            self._validate_solver_settings_config(config)
    
    def _validate_mock_inputs_config(self, config: Dict[str, Any]) -> None:
        """Validate mock inputs configuration."""
        required_sections = ["sample_data", "input_specs", "output_specs", "models"]
        for section in required_sections:
            if section not in config:
                raise ValidationError(section, f"Missing required section: {section}")
            if not isinstance(config[section], dict):
                raise ValidationError(section, f"Section {section} must be a dictionary")
    
    def _validate_test_scenarios_config(self, config: Dict[str, Any]) -> None:
        """Validate test scenarios configuration."""
        if "scenarios" not in config:
            raise ValidationError("scenarios", "Missing required 'scenarios' section")
    
    def _validate_solver_settings_config(self, config: Dict[str, Any]) -> None:
        """Validate solver settings configuration."""
        if "solvers" not in config:
            raise ValidationError("solvers", "Missing required 'solvers' section")
    
    def get_default_config(self, config_type: str) -> Dict[str, Any]:
        """
        Get default configuration for a specific type.
        
        Args:
            config_type: Type of configuration ('mock_inputs', 'test_scenarios', etc.)
            
        Returns:
            Default configuration dictionary
        """
        defaults = {
            "mock_inputs": self._get_default_mock_inputs(),
            "test_scenarios": self._get_default_test_scenarios(),
            "solver_settings": self._get_default_solver_settings(),
        }
        
        return defaults.get(config_type, {})
    
    def _get_default_mock_inputs(self) -> Dict[str, Any]:
        """Default mock inputs configuration."""
        return {
            "sample_data": {
                "mnist_small": {
                    "type": "image",
                    "shape": [1, 28, 28],
                    "distribution": "uniform",
                    "range": [0, 1],
                    "batch_size": 10
                }
            },
            "input_specs": {
                "robust_l_inf": {
                    "spec_type": "LOCAL_LP",
                    "norm_type": "LINF",
                    "epsilon": 0.1
                }
            },
            "output_specs": {
                "classification": {
                    "type": "classification",
                    "num_classes": 10
                }
            },
            "models": {
                "simple_relu": {
                    "architecture": "feedforward",
                    "layers": [784, 50, 10],
                    "activations": ["relu", "linear"]
                }
            }
        }
    
    def _get_default_test_scenarios(self) -> Dict[str, Any]:
        """Default test scenarios configuration."""
        return {
            "scenarios": {
                "quick_smoke_test": {
                    "sample_data": "mnist_small",
                    "input_spec": "robust_l_inf",
                    "output_spec": "classification",
                    "model": "simple_relu",
                    "expected_result": "UNSAT",
                    "timeout": 30
                }
            }
        }
    
    def _get_default_solver_settings(self) -> Dict[str, Any]:
        """Default solver settings configuration."""
        return {
            "solvers": {
                "eran_deeppoly": {
                    "method": "deeppoly",
                    "timeout": 300
                }
            },
            "testing": {
                "parallel_workers": 4,
                "memory_limit_gb": 8,
                "retry_on_timeout": True
            }
        }
    
    def save_config(self, config: Dict[str, Any], config_name: str) -> None:
        """
        Save configuration to file.
        
        Args:
            config: Configuration dictionary to save
            config_name: Name of config file or path
        """
        config_path = self._resolve_config_path(config_name)
        
        try:
            # Ensure directory exists
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            if config_path.suffix.lower() == '.json':
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2)
            else:
                with open(config_path, 'w', encoding='utf-8') as f:
                    yaml.dump(config, f, default_flow_style=False, indent=2)
                    
            logger.info(f"Saved configuration: {config_path}")
            
            # Update cache
            self._config_cache[config_name] = config
            
        except Exception as e:
            logger.error(f"Failed to save config {config_path}: {e}")
            raise
    
    def clear_cache(self) -> None:
        """Clear configuration cache."""
        self._config_cache.clear()
    
    def list_available_configs(self) -> List[str]:
        """List all available configuration files."""
        configs = []
        config_dir = self.base_path / "configs"
        
        if config_dir.exists():
            for file_path in config_dir.glob("*.yaml"):
                configs.append(file_path.name)
            for file_path in config_dir.glob("*.json"):
                configs.append(file_path.name)
                
        return sorted(configs)


# Global configuration manager instance
config_manager = ConfigManager()

# Convenience functions
def load_config(config_name: str, use_cache: bool = True) -> Dict[str, Any]:
    """Load configuration using global config manager."""
    return config_manager.load_config(config_name, use_cache)

def get_default_config(config_type: str) -> Dict[str, Any]:
    """Get default configuration using global config manager."""
    return config_manager.get_default_config(config_type)
