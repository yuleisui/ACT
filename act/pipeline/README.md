# ACT Pipeline Testing Framework

A comprehensive testing framework for validating the Abstract Constraint Transformer (ACT) abstraction verifier with configurable mock testing, property-based validation, and regression testing capabilities.

## Overview

The ACT Pipeline Testing Framework provides a robust testing infrastructure to validate the correctness, performance, and reliability of the ACT abstraction verifier. It supports multiple testing methodologies including mock input generation, real dataset integration, regression testing, and performance profiling.

## Architecture

### Core Components

```
act/pipeline/
├── __init__.py           # Main entry points and convenience functions
├── config.py             # Configuration management and validation
├── mock_factory.py       # Mock input generation from YAML configs
├── correctness.py        # Verifier correctness and property validation
├── regression.py         # Baseline capture and regression testing
├── integration.py        # Front-end integration bridge
├── reporting.py          # Results analysis and report generation  
├── utils.py              # Shared utilities and performance profiling
├── run_tests.py          # Command-line interface
├── configs/              # Configuration files
│   ├── mock_inputs.yaml  # Mock data generation templates
│   ├── test_scenarios.yaml # Complete test scenario definitions
│   ├── solver_settings.yaml # Solver configuration options
│   └── baselines.json    # Performance baseline storage
└── examples/             # Usage examples and quick tests
    ├── quick_test.py     # Simple validation examples
    ├── custom_inputs.py  # Custom input generation examples
    └── ci_setup.py       # CI/CD integration examples
```

## Design Principles

### 1. **Modular Testing Architecture**
- **Separation of Concerns**: Each module handles a specific aspect (mocking, validation, regression, etc.)
- **Pluggable Components**: Easy to extend with new test types and validation methods
- **Independent Testing**: Each test type can run independently or as part of a suite

### 2. **Configuration-Driven Testing**
- **YAML Configurations**: Human-readable test specifications without code changes
- **Flexible Mock Generation**: Generate diverse test inputs from configuration templates
- **Scenario Composition**: Combine different components into complete test scenarios

### 3. **Comprehensive Validation Coverage**
- **Correctness Testing**: Validate verifier results against expected outcomes
- **Property-Based Testing**: Test fundamental properties like soundness and completeness
- **Performance Testing**: Monitor execution time, memory usage, and resource consumption
- **Regression Testing**: Track changes and detect performance/correctness regressions

### 4. **Real-World Integration**
- **Front-End Bridge**: Integration with ACT's actual front-end loaders and specifications
- **Dataset Support**: Testing with real MNIST, CIFAR, and custom datasets
- **Model Support**: Testing with various neural network architectures

## Key Features

### Mock Input Generation (`mock_factory.py`)
```python
# Generate diverse test inputs from YAML configuration
factory = MockInputFactory()
data, labels = factory.generate_sample_data("mnist_small")
input_spec = factory.generate_input_spec("robust_l_inf")
model = factory.generate_model("simple_relu")
```

**Capabilities:**
- **Sample Data**: Images, tensors with configurable distributions (uniform, normal, gaussian noise)
- **Input Specifications**: L∞/L2 perturbations, box constraints, custom bounds
- **Output Specifications**: Classification robustness, margin constraints, custom properties
- **Neural Networks**: Various architectures (linear, ReLU, CNN, custom)

### Correctness Validation (`correctness.py`)
```python
# Validate verifier correctness with comprehensive testing
validator = AbstractionVerifierValidator()
result = validator.validate_correctness(test_cases)
```

**Validation Types:**
- **Basic Correctness**: Expected SAT/UNSAT results match actual outcomes
- **Property Testing**: Soundness (no false negatives) and completeness validation
- **Performance Testing**: Execution time, memory usage, resource consumption
- **BaB Refinement**: Branch-and-bound refinement effectiveness testing

### Regression Testing (`regression.py`)
```python
# Capture baselines and detect regressions
baseline_mgr = BaselineManager()
baseline_mgr.capture_baseline("mnist_cnn_v1", validation_results, performance_results)
regression_result = baseline_mgr.compare_to_baseline("mnist_cnn_v1", current_results)
```

**Features:**
- **Baseline Capture**: Store performance and correctness metrics as baselines
- **Trend Analysis**: Track metrics over time and detect degradation patterns
- **Regression Detection**: Automated detection of performance/correctness regressions
- **Threshold Configuration**: Configurable thresholds for regression sensitivity

### Integration Testing (`integration.py`)
```python
# Test with real ACT front-end components
bridge = ACTFrontendBridge()
test_case = IntegrationTestCase(
    dataset_name="mnist",
    model_path="models/mnist_cnn.onnx", 
    spec_type="local_lp",
    epsilon=0.1
)
result = bridge.run_test(test_case)
```

**Integration Points:**
- **Dataset Loaders**: MNIST, CIFAR, custom CSV datasets
- **Model Loaders**: ONNX models, PyTorch models
- **Specification Loaders**: VNNLIB, custom specifications
- **Device Management**: CPU/GPU testing with proper device handling

### Performance Profiling (`utils.py`)
```python
# Comprehensive performance monitoring
profiler = PerformanceProfiler()
profiler.start()
# ... run verification ...
metrics = profiler.stop()  # execution_time, peak_memory_mb, cpu_usage_percent
```

**Monitoring:**
- **Execution Time**: Precise timing of verification operations
- **Memory Usage**: Peak memory consumption tracking
- **CPU/GPU Usage**: Resource utilization monitoring
- **Parallel Execution**: Multi-threaded test execution with resource tracking

### Report Generation (`reporting.py`)
```python
# Generate comprehensive test reports
generator = ReportGenerator()
generator.generate_full_report(validation_results, performance_results, regression_results)
```

**Report Types:**
- **HTML Reports**: Interactive dashboards with plots and metrics
- **JSON Reports**: Machine-readable results for CI/CD integration
- **Performance Analysis**: Bottleneck identification and optimization suggestions
- **Trend Visualization**: Performance trends and regression analysis

## Usage Examples

### 1. Quick Validation (3 lines)
```python
from act.pipeline import validate_abstraction_verifier
result = validate_abstraction_verifier("configs/my_tests.yaml")
print(f"Status: {'✅ PASSED' if result.success else '❌ FAILED'}")
```

### 2. Ultra-Simple Validation (1 line)
```python
from act.pipeline import quick_validate
success = quick_validate()  # Uses sensible defaults
```

### 3. Custom Mock Testing
```python
from act.pipeline import MockInputFactory, AbstractionVerifierValidator

# Generate custom test inputs
factory = MockInputFactory()
test_data = factory.generate_from_config("configs/custom_mocks.yaml")

# Validate with custom inputs
validator = AbstractionVerifierValidator()
results = validator.run_validation_suite(test_data)
```

### 4. Regression Testing
```python
from act.pipeline import BaselineManager, RegressionTester

# Capture new baseline
baseline_mgr = BaselineManager()
baseline_mgr.capture_baseline("v2.1", validation_results, performance_results)

# Compare against previous baseline
regression_tester = RegressionTester()
regression_result = regression_tester.compare_baselines("v2.0", "v2.1")
```

### 5. Command-Line Usage
```bash
# Quick validation
python run_tests.py --quick

# Full test suite with reporting
python run_tests.py --comprehensive --report results.html

# CI mode (fast, essential tests)
python run_tests.py --ci --output ci_results.json

# Custom configuration
python run_tests.py --config my_tests.yaml --mock-config my_mocks.yaml
```

## Configuration System

### Test Scenarios (`configs/test_scenarios.yaml`)
```yaml
scenarios:
  quick_smoke_test:
    sample_data: "mnist_small"
    input_spec: "robust_l_inf_small" 
    output_spec: "classification"
    model: "simple_relu"
    expected_result: "UNSAT"
    timeout: 30
```

### Mock Inputs (`configs/mock_inputs.yaml`)
```yaml
sample_data:
  mnist_small:
    type: "image"
    shape: [1, 28, 28]
    distribution: "uniform"
    range: [0, 1]
    batch_size: 10
    num_classes: 10

input_specs:
  robust_l_inf_small:
    spec_type: "LOCAL_LP"
    norm_type: "inf"
    epsilon: 0.1
```

### Solver Settings (`configs/solver_settings.yaml`)
```yaml
solvers:
  torch_lp:
    enabled: true
    timeout: 300
    memory_limit: "8GB"
  
  gurobi:
    enabled: true
    timeout: 600
    threads: 4
```

## Testing Workflow

### 1. **Development Testing**
```python
# During development - quick feedback
from act.pipeline import quick_validate
assert quick_validate(), "Basic functionality broken"
```

### 2. **Feature Testing**
```python
# When adding new features - comprehensive validation
result = validate_abstraction_verifier("configs/feature_tests.yaml")
assert result.validations.correctness.success, "Correctness regression detected"
```

### 3. **Release Testing**
```bash
# Before releases - full test suite with baseline comparison
python run_tests.py --comprehensive --regression --report release_report.html
```

### 4. **CI/CD Integration**
```bash
# In CI pipelines - fast, reliable tests
python run_tests.py --ci --timeout 120 --output ci_results.json
```

## Extension Points

### Adding New Test Types
```python
class CustomValidator(BaseValidator):
    def validate(self, test_case: TestCase) -> ValidationResult:
        # Custom validation logic
        return ValidationResult(...)

# Register with framework
validator.register_custom_validator("my_test", CustomValidator())
```

### Custom Mock Generators
```python
class CustomGenerator(BaseGenerator):
    def generate(self, config: Dict[str, Any]) -> Any:
        # Custom generation logic
        return generated_data

# Register with factory
factory.register_generator("custom_type", CustomGenerator())
```

### Custom Report Formats
```python
class CustomReportGenerator:
    def generate(self, results: List[ValidationResult]) -> str:
        # Custom reporting logic
        return report_content

# Use with reporting system
generator.add_format("custom", CustomReportGenerator())
```

## Error Handling and Debugging

### Comprehensive Error Reporting
- **Detailed Error Messages**: Clear error descriptions with context
- **Stack Trace Capture**: Full debugging information for failures
- **Resource Monitoring**: Track resource usage during failures
- **Graceful Degradation**: Continue testing even when individual tests fail

### Debugging Support
```python
# Enable debug logging
import logging
logging.getLogger('act.pipeline').setLevel(logging.DEBUG)

# Memory debugging
from act.pipeline.utils import print_memory_usage
print_memory_usage("Before verification")
```

## Performance Considerations

### Parallel Execution
- **Multi-threaded Testing**: Parallel execution of independent tests
- **Resource Management**: Intelligent resource allocation and cleanup
- **Memory Optimization**: Efficient memory usage with automatic cleanup

### Scalability Features
- **Batch Processing**: Efficient handling of large test suites
- **Incremental Testing**: Only run tests affected by changes
- **Resource Limits**: Configurable memory and time limits

## Integration with ACT Framework

The pipeline seamlessly integrates with ACT's core components:

- **Back-End Integration**: Direct use of `act.back_end` verification components
- **Front-End Bridge**: Integration with `act.front_end` loaders and specifications  
- **Device Management**: Proper CUDA/CPU device handling
- **Configuration Compatibility**: Works with existing ACT configuration systems

This design provides a robust, extensible testing framework that ensures the reliability and performance of the ACT abstraction verifier while being easy to use and extend.