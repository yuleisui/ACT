# Bounds Propagation Regression Testing

## Overview
Automated regression testing system for `act/interval/bounds_propagation.py` with smart auto-update functionality.

**Note**: The bounds data structures have been moved from `util/bounds.py` to `act/util/bounds.py` to better organize the codebase under the ACT framework.

## 3 Essential Triggering Ways

### 1. Unit Test (Manual)
```bash
# Run regression tests with smart auto-update
python unit-tests/test_bounds_prop_regression.py --test-regression

# Capture new baseline
python unit-tests/test_bounds_prop_regression.py --capture-baseline

# Force update baseline
python unit-tests/test_bounds_prop_regression.py --update-baseline

# Quick daily test script
bash unit-tests/regression_test.sh
```

### 2. Git Hook (Automatic)
- Pre-commit hook automatically runs regression tests when `bounds_propagation.py` changes
- Located at `.git/hooks/pre-commit`
- Prevents commits that break regression tests
- Can be disabled with: `git config hooks.regression-tests false`

### 3. VS Code Integration (IDE)
- **Ctrl+Shift+T**: Run regression tests with auto-update
- **Ctrl+Shift+B**: Capture new baseline
- VS Code tasks available in Command Palette

## Smart Auto-Update Logic
The `--test-regression` command automatically updates baselines when:
1. ✅ Correctness validation passes (same results as baseline)
2. ✅ Performance improves (faster execution time)
3. ✅ All regression checks pass

## Test Scenarios
- **MNIST CNN**: Small ReLU model with ε=0.03 perturbation
- **Input validation**: Proper bounds propagation through layers
- **Performance tracking**: Execution time monitoring
- **Correctness verification**: Output bounds comparison

## Files
- `unit-tests/test_bounds_prop_regression.py`: Main testing framework
- `unit-tests/regression_test.sh`: Daily workflow script
- `.git/hooks/pre-commit`: Git integration
- `.vscode/tasks.json`: VS Code tasks
- `.vscode/keybindings.json`: Keyboard shortcuts
- `unit-tests/regression_baselines/`: Stored baselines (auto-managed)

## Core Dependencies
- `act/interval/bounds_propagation.py`: Main bounds propagation implementation
- `act/util/bounds.py`: Bounds data structures (moved from `util/bounds.py`)
- `act/interval/bounds_prop_helper.py`: Enhanced metadata and stability validation