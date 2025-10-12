"""
ACT Pipeline Testing Framework

A comprehensive testing framework for validating the Abstract Constraint Transformer (ACT)
abstraction verifier with configurable mock testing, property-based validation, and 
regression testing capabilities.

Main Components:
- config: Configuration loading and management
- mock_factory: Configurable mock input generation  
- correctness: Verifier correctness and property validation
- regression: Baseline capture and regression testing
- integration: Front-end integration bridge
- reporting: Results analysis and report generation
- utils: Shared utilities and performance profiling
- run_tests: Command-line interface

Usage:
    # Simple 3-line validation
    from act.pipeline import validate_abstraction_verifier
    result = validate_abstraction_verifier("configs/my_tests.yaml")
    print(f"Status: {'✅ PASSED' if result.success else '❌ FAILED'}")
    
    # Or even simpler with defaults
    from act.pipeline import quick_validate
    success = quick_validate()
"""

from .config import ConfigManager, load_config, get_default_config
from .mock_factory import MockInputFactory
from .correctness import (
    AbstractionVerifierValidator, 
    TestCase,
    ValidationResult,
    PropertyTestResult,
    PerformanceResult,
    VerifyResult
)
from .regression import (
    BaselineManager,
    RegressionTester,
    RegressionResult,
    TrendData
)
from .integration import (
    ACTFrontendBridge,
    IntegrationTestCase,
    ACTIntegrationConfig
)
from .reporting import (
    ReportGenerator,
    ReportConfig,
    TestSummary,
    PerformanceSummary,
    RegressionSummary
)
from .utils import (
    PerformanceProfiler,
    ParallelExecutor,
    print_memory_usage,
    clear_torch_cache,
    setup_logging,
    ProgressTracker
)

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Package version
__version__ = "1.0.0"

# Default configuration paths
DEFAULT_CONFIG_PATHS = {
    "mock_inputs": "configs/mock_inputs.yaml",
    "test_scenarios": "configs/test_scenarios.yaml", 
    "solver_settings": "configs/solver_settings.yaml"
}


def validate_abstraction_verifier(config_path: str = "configs/test_scenarios.yaml",
                                verifier_module: Optional[Any] = None,
                                log_level: str = "INFO",
                                device: str = "cpu") -> Dict[str, Any]:
    """
    Main validation function for abstraction verifier.
    
    Args:
        config_path: Path to test scenarios configuration
        verifier_module: Optional verifier module to test (uses mock if None)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        
    Returns:
        Dictionary containing comprehensive validation results
    """
    setup_logging(level=log_level)
    logger.info(f"Starting abstraction verifier validation with config: {config_path}")
    
    try:
        # Load test configuration
        config = load_config(config_path)
        
        # Initialize validator with real abstraction verifier
        validator = AbstractionVerifierValidator(device=device)
        
        # Run comprehensive validation
        results = validator.run_comprehensive_validation(config)
        
        logger.info("Abstraction verifier validation completed successfully")
        return results
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": 0,
            "validations": {}
        }


def quick_validate(verifier_module: Optional[Any] = None) -> bool:
    """
    Quick validation with default comprehensive configuration.
    
    Args:
        verifier_module: Optional verifier module to test (uses mock if None)
        
    Returns:
        True if all validations passed, False otherwise
    """
    try:
        # Use default comprehensive test scenarios
        config_manager = ConfigManager()
        
        # Create default config if files don't exist
        default_config = {
            "scenarios": {
                "quick_smoke_test": {
                    "sample_data": "mnist_small",
                    "input_spec": "robust_l_inf_small",
                    "output_spec": "classification", 
                    "model": "simple_relu",
                    "expected_result": "UNSAT",
                    "timeout": 30
                }
            },
            "run_correctness": True,
            "run_properties": False,  # Skip properties for quick test
            "run_performance": True,
            "run_bab": False
        }
        
        # Initialize validator with real abstraction verifier
        validator = AbstractionVerifierValidator()
        
        # Run validation
        results = validator.run_comprehensive_validation(default_config)
        
        # Check if all validations passed
        success = True
        for validation_name, validation_result in results.get("validations", {}).items():
            if hasattr(validation_result, 'success'):
                success = success and validation_result.success
            elif isinstance(validation_result, list):
                # Handle property test results
                success = success and all(r.success for r in validation_result)
        
        return success
        
    except Exception as e:
        logger.error(f"Quick validation failed: {e}")
        return False


def create_test_scenario(sample_data: str,
                        input_spec: str,
                        output_spec: str,
                        model: str,
                        expected_result: Optional[str] = None,
                        timeout: float = 300.0) -> Dict[str, Any]:
    """
    Create a test scenario configuration.
    
    Args:
        sample_data: Name of sample data configuration
        input_spec: Name of input specification configuration
        output_spec: Name of output specification configuration
        model: Name of model configuration
        expected_result: Expected verification result (SAT, UNSAT, etc.)
        timeout: Timeout in seconds
        
    Returns:
        Test scenario configuration dictionary
    """
    scenario = {
        "sample_data": sample_data,
        "input_spec": input_spec,
        "output_spec": output_spec,
        "model": model,
        "timeout": timeout
    }
    
    if expected_result:
        scenario["expected_result"] = expected_result
    
    return scenario


def run_mock_test_example():
    """Run a simple example using mock inputs."""
    logger.info("Running mock test example")
    
    # Initialize mock factory
    factory = MockInputFactory()
    
    # Generate test inputs
    sample_data, labels = factory.create_sample_data("mnist_small")
    input_spec = factory.create_input_spec("robust_l_inf")
    output_spec = factory.create_output_spec("classification")
    model = factory.create_model("simple_relu")
    
    logger.info(f"Generated test inputs:")
    logger.info(f"  Sample data shape: {sample_data.shape}")
    logger.info(f"  Labels shape: {labels.shape}")
    logger.info(f"  Input spec: {input_spec['type']} with epsilon={input_spec.get('epsilon', 'N/A')}")
    logger.info(f"  Output spec: {output_spec['type']} with {output_spec.get('num_classes', 'N/A')} classes")
    logger.info(f"  Model: {type(model).__name__} with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create test case
    test_case = TestCase(
        sample_data=sample_data,
        labels=labels,
        input_spec=input_spec,
        output_spec=output_spec,
        model=model,
        test_id="mock_example"
    )
    
    # Run validation
    validator = AbstractionVerifierValidator()
    result = validator.validate_correctness([test_case])
    
    logger.info(f"Mock test result: {'PASSED' if result.success else 'FAILED'}")
    logger.info(f"Execution time: {result.execution_time:.2f}s")
    logger.info(f"Memory usage: {result.memory_usage_mb:.1f} MB")
    
    return result


# Make key classes and functions available at package level
__all__ = [
    # Main validation functions
    "validate_abstraction_verifier",
    "quick_validate",
    "create_test_scenario",
    "run_mock_test_example",
    
    # Core classes
    "ConfigManager",
    "MockInputFactory", 
    "AbstractionVerifierValidator",
    "PipelineValidator",
    "PerformanceProfiler",
    "ParallelExecutor",
    
    # Data classes
    "TestCase",
    "ValidationResult",
    "PropertyTestResult", 
    "PerformanceResult",
    "VerifyResult",
    
    # Utility functions
    "load_config",
    "get_default_config",
    "print_memory_usage",
    "clear_torch_cache",
    "setup_logging",
    
    # Constants
    "__version__",
    "DEFAULT_CONFIG_PATHS"
]


# Initialize logging on import
setup_logging(level="INFO")