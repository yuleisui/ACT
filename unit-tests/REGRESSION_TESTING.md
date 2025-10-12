# Bounds Propagation Regression Testing

## Overview
Intelligent automated regression testing system for `act/interval/bounds_propagation.py` with smart auto-update functionality, performance tracking, and enhanced stability validation.

**Note**: The bounds data structures have been moved from `util/bounds.py` to `act/util/bounds.py` to better organize the codebase under the ACT framework.

## Quick Start
```bash
# Standard regression test with auto-improvement detection
cd unit-tests
./regression_test.sh --test-regression

# Capture new baseline (for new features or major changes)
./regression_test.sh --save-baseline

# View help for all options
./regression_test.sh --help
```

## 3 Essential Triggering Ways

### 1. Unit Test (Manual)
```bash
# Run regression tests with smart auto-update
python unit-tests/test_bounds_prop_regression.py --test-regression

# Capture new baseline (recommended naming)
python unit-tests/test_bounds_prop_regression.py --save-baseline

# Force update baseline (override safety checks)
python unit-tests/test_bounds_prop_regression.py --update-baseline

# Daily workflow script
cd unit-tests && ./regression_test.sh
```

### 2. Git Hook (Automatic)
- Pre-commit hook automatically runs regression tests when `bounds_propagation.py` changes
- Located at `.git/hooks/pre-commit`
- Uses `--save-baseline` for consistent interface
- Prevents commits that break regression tests
- Can be disabled with: `git config hooks.regression-tests false`

### 3. VS Code Integration (IDE)
- **Ctrl+Shift+T**: Run regression tests with auto-update
- **Ctrl+Shift+B**: Capture new baseline using `--save-baseline`
- VS Code tasks available in Command Palette with unified naming

## Smart Auto-Update Logic
The `--test-regression` command automatically updates baselines when:
1. ✅ **Correctness validation passes**: Same results as baseline (within numerical tolerance)
2. ✅ **Performance improves**: Measurably faster execution time (>5% improvement) 
3. ✅ **Statistical stability**: Multiple warmup runs confirm consistent performance
4. ✅ **All regression checks pass**: No functionality regressions detected

### Performance Intelligence Features
- **Outlier Removal**: Automatically filters out performance outliers from timing measurements
- **Warmup Runs**: Performs initial runs to stabilize JIT compilation and caching
- **Statistical Validation**: Uses multiple runs to ensure performance improvements are real
- **Automated Baseline Updates**: Seamlessly updates performance baselines when improvements are detected

## Test Scenarios & Coverage
- **MNIST CNN**: Small ReLU model with ε=0.03 perturbation
- **Input validation**: Proper bounds propagation through layers
- **Performance tracking**: Execution time monitoring with statistical analysis
- **Correctness verification**: Output bounds comparison with numerical tolerance
- **Memory efficiency**: Tracking memory usage patterns
- **Stability testing**: Multiple runs to ensure consistent behavior

## Performance Optimization Features
The testing framework includes sophisticated performance optimizations:

### Performance Mode Integration
- **Smart Performance Mode**: Automatically enabled through `metadata_tracker.performance_mode`
- **Conditional Logging**: Reduces overhead by >1000x during performance mode
- **Selective Validation**: Skips expensive checks when performance is prioritized
- **Memory Optimization**: Efficient memory usage patterns during testing

### Measurement Accuracy
- **Multiple Runs**: Each test performs multiple iterations for statistical accuracy
- **Warmup Cycles**: Initial runs excluded to account for JIT compilation
- **Outlier Filtering**: Automatic removal of statistical outliers from timing data
- **Environment Stability**: Consistent testing environment for reliable measurements

## Architecture & Design

### Centralized Performance Control
- **Metadata Tracker**: Central performance mode control in `bounds_prop_helper.py`
- **Clean Separation**: Performance concerns separated from core logic
- **Unified Interface**: Single point of control for all performance optimizations

### Error Handling & Robustness
- **Graceful Degradation**: Tests continue even if some components fail
- **Detailed Reporting**: Comprehensive failure analysis and debugging information
- **Recovery Mechanisms**: Automatic fallback strategies for common issues

## Files & Components
- **`unit-tests/test_bounds_prop_regression.py`**: Main testing framework with intelligent auto-update logic
- **`unit-tests/regression_test.sh`**: Daily workflow script with unified command interface
- **`.git/hooks/pre-commit`**: Git integration with `--save-baseline` support
- **`.vscode/tasks.json`**: VS Code tasks with consistent naming convention
- **`.vscode/keybindings.json`**: Keyboard shortcuts for quick testing
- **`unit-tests/regression_baselines/`**: Stored baselines (auto-managed with version tracking)

## Core Dependencies & Architecture
- **`act/interval/bounds_propagation.py`**: Main bounds propagation implementation with performance optimizations
- **`act/util/bounds.py`**: Bounds data structures (moved from `util/bounds.py` for better organization)
- **`act/interval/bounds_prop_helper.py`**: Enhanced metadata tracker and stability validation
- **`act/interval/metadata_tracker.py`**: Central performance mode control and optimization settings

## Troubleshooting & Best Practices

### Common Issues
1. **Performance Regression**: Use `--save-baseline` to establish new baseline after intended changes
2. **Numerical Differences**: Small floating-point differences are expected and handled automatically
3. **Environment Changes**: Baselines may need updates after system or dependency changes

### Best Practices
1. **Regular Testing**: Run `./regression_test.sh --test-regression` daily
2. **Baseline Management**: Only use `--save-baseline` for intentional changes
3. **CI Integration**: Ensure tests pass before committing changes
4. **Performance Monitoring**: Monitor the auto-update logs for performance trends

### Advanced Usage
```bash
# Test with custom tolerance
python test_bounds_prop_regression.py --test-regression --tolerance 1e-6

# Verbose output for debugging
python test_bounds_prop_regression.py --test-regression --verbose

# Performance analysis mode
python test_bounds_prop_regression.py --test-regression --profile
```