# Copilot Instructions for Abstract Constraint Transformer (ACT)

## Project Overview
ACT is a unified neural network verification framework integrating multiple state-of-the-art verifiers (ERAN, αβ-CROWN) with novel Hybrid Zonotope methods. The core architecture follows a **plugin-based verifier pattern** with standardized input/output specifications.

## Architecture Essentials

### Verifier Plugin Pattern
All verifiers inherit from `BaseVerifier` in `verifier/abstract_constraint_solver/interval/base_verifier.py`:
- **Core method**: `verify(proof, public_inputs)` - main entry point
- **Shared workflow**: Input validation → Abstract solving → BaB refinement (optional) → Result aggregation
- **Standard result types**: `VerifyResult.{SAT, UNSAT, UNKNOWN, TIMEOUT}`

### Input Pipeline Flow
1. **Dataset** (`input_parser/dataset.py`) - Loads MNIST/CIFAR/CSV/VNNLIB with normalization
2. **InputSpec** (`input_parser/spec.py`) - Handles L∞/L2 perturbations and VNNLIB constraints  
3. **OutputSpec** - Manages classification labels or custom output constraints
4. **Model** (`input_parser/model.py`) - ONNX model loading with PyTorch conversion

### Key Data Structures
- `SpecType`: `LOCAL_LP` (ε-ball), `LOCAL_VNNLIB` (anchored), `SET_VNNLIB` (global), `SET_BOX`
- `LPNormType`: `LINF`, `L2`, `L1` for perturbation bounds
- **VNNLIB parsing**: Complex constraint extraction in `vnnlib_parser.py`

## Novel Hybrid Zonotope Features

### Generator Merging Optimization
Located in `hybridz_operations.py::MergeParallelGenerators()`:
```python
# Enable with --enable_generator_merging --cosine_threshold 0.95
# Applied automatically at final Linear layer for performance
center, G_c, G_b, A_c, A_b, b = MergeParallelGenerators(...)
```

### Relaxation Mechanisms
- `--relaxation_ratio 0.0`: Full-precision MILP (exact)
- `--relaxation_ratio 1.0`: Fully-relaxed LP (fast) 
- `--relaxation_ratio 0.0-1.0`: Hybrid MILP+LP approach

### Memory Management
Automatic memory estimation with fallback to optimized algorithms:
```python
estimated_memory_gb, use_memory_optimized, debug_info = 
    MemoryUsageEstimationIntersection(abstract_transformer_hz, input_hz)
```

## Development Workflows

### Environment Setup
```bash
cd setup/
source setup.sh main  # Creates conda env 'act-main'
conda activate act-main
```

### Running Verification
```bash
cd verifier/
python main.py \
  --verifier hybridz --method hybridz_relaxed \
  --model_path ../models/Sample_models/MNIST/small_relu_mnist_cnn_model_1.onnx \
  --dataset mnist --spec_type local_lp \
  --start 0 --end 1 --epsilon 0.03 --norm inf \
  --mean 0.1307 --std 0.3081
```

### Testing
- **Unit tests**: `pytest unit-tests/` (uses `conftest.py` for path setup)
- **CI patterns**: Environment variable `ACT_CI_MODE=true` for lightweight CI installs
- **Mock patterns**: See `test_get_sample.py` for BaseVerifier mocking

## Configuration System

### Dataset Defaults
Config files in `configs/` provide dataset-specific defaults:
```ini
[MNIST]
mean = [0.1307]
std = [0.3081]
spec_type = "local_lp"
```

### Verifier Selection
- **Command loading**: `util/options.py` defines all CLI parameters
- **Auto-loading**: `load_verifier_default_configs()` in `main.py`
- **Backend routing**: Factory pattern in `main.py` based on `--verifier` flag

## Critical Conventions

### Path Handling
- **Project root**: Always use `verifier/` as working directory for execution
- **Import paths**: `sys.path` setup in `conftest.py` enables `from verifier.input_parser import ...`
- **Model paths**: Relative to project root (`../models/Sample_models/...`)

### Memory Patterns
- **Auto-cleanup**: Torch cache clearing in memory-intensive operations
- **Progress tracking**: `print_memory_usage()` throughout HybridZ pipeline
- **Batch processing**: Single-sample verification with result aggregation

### Error Handling
- **Graceful degradation**: Unknown results instead of crashes
- **Verification stats**: Comprehensive tracking in `clean_prediction_stats`
- **Timeout support**: Built into BaB refinement with configurable limits

## Integration Points

### External Tools
- **ERAN**: Subprocess execution with parameter translation
- **αβ-CROWN**: YAML config generation and temp file management  
- **Gurobi**: License required in `gurobi/gurobi.lic` for MILP optimization

### VNNLIB Compatibility
Full SMT-LIB format support with:
- Variable extraction (`X_i`, `Y_j`, `X_hat_k`)
- Constraint parsing (linear combinations, bounds)
- Local vs. global property detection

## Python Coding Standards

When writing code for this project, follow these Python best practices:

### Code Quality
- Use clear, descriptive variable and function names
- Always add type hints and docstrings for functions/classes
- Use dataclasses for simple data containers
- Apply abstract base classes (ABC) for extensible interfaces
- Use factory patterns for object creation when appropriate
- Handle errors with try/except and custom exceptions
- Manage resources with context managers (`with` statements)
- Use logging (not print) for diagnostics
- Write modular, testable code (single responsibility principle)
- Avoid global state and side effects
- Prefer list/dict comprehensions over loops when clear
- Use f-strings for formatting
- Follow PEP8 style and PEP257 docstrings
- Add comments for non-obvious logic

### Testing Requirements
- Write unit tests using `pytest` (preferred) or `unittest`
- Use fixtures for setup/teardown and mock external dependencies
- Test both typical and edge cases, including error handling
- Use parameterized tests for multiple scenarios
- Keep tests isolated and independent
- Add integration tests for critical verification workflows
- Mock external verifier calls (ERAN, αβ-CROWN) in unit tests
- Use the existing `conftest.py` pattern for test configuration

### Debugging Practices
- Use structured logging with appropriate levels
- Add assertions to check invariants
- Write clear, actionable error messages
- Use `pytest` fixtures for reproducible test scenarios
- Profile memory usage in HybridZ operations
- Add memory cleanup in long-running verification tasks

When implementing new features, follow the BaseVerifier plugin pattern and ensure compatibility with the unified CLI interface. The HybridZ verifier demonstrates the most advanced patterns for memory optimization and novel constraint handling.