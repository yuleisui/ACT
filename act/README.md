# ACT Directory

This directory contains the core verification framework and interfaces for the Abstract Constraint Transformer (ACT) system. It provides unified access to multiple verification backends including ACT native methods, ERAN, and αβ-CROWN through a hierarchical modular architecture.

## Directory Structure

```
act/
├── main.py                         # Main unified verification interface
├── __init__.py                     # Package initialization
│
├── hybridz/                        # ACT Hybrid Zonotope module
│   ├── hybridz_verifier.py         # Hybrid Zonotope verification engine
│   ├── hybridz_transformers.py     # Hybrid Zonotope transformer classes
│   └── hybridz_operations.py       # MILP/LP operations and optimization
│
├── interval/                       # ACT Interval verification module
│   ├── base_verifier.py            # Base verifier class and common functionality
│   ├── bounds_propagation.py       # Interval bound propagation
│   └── outputs_evaluation.py       # Output bounds evaluation
│
├── input_parser/                   # Specification parsing and data handling
│   ├── dataset.py                  # Dataset loading and preprocessing
│   ├── model.py                    # Neural network model parsing
│   ├── spec.py                     # Specification handling and processing
│   ├── type.py                     # Type definitions and data structures
│   ├── vnnlib_parser.py            # VNNLIB format parser
│   └── adaptor.py                  # Input/output adaptors
│
├── refinement/                     # Advanced refinement algorithms
│   └── bab_spec_refinement.py      # Branch-and-bound specification refinement
│
├── util/                           # Utility modules
│   ├── options.py                  # Command-line argument parsing
│   ├── path_config.py              # Centralized path configuration
│   ├── stats.py                    # Statistics and performance tracking
│   └── inference.py                # Model inference utilities
│
└── wrapper_exts/                   # External verifier integrations
    ├── abcrown/                    # αβ-CROWN integration module
    │   ├── abcrown_verifier.py     # αβ-CROWN wrapper and interface
    │   └── abcrown_runner.py       # αβ-CROWN backend execution script
    └── eran/                       # ERAN integration module
        └── eran_verifier.py        # ERAN wrapper and interface
```

## Module Documentation

### **Main Interface**
- **`main.py`**: Primary entry point for all verification tasks
  - Unified command-line interface supporting all verifiers
  - Parameter parsing, validation, and backend routing
  - Comprehensive argument compatibility across different verification tools
  - Integration with configuration defaults from `../configs/`

### **`hybridz/` - ACT Hybrid Zonotope Verification**
- **`hybridz_verifier.py`**: Hybrid Zonotope verification engine
  - Novel tensorised hybrid verification method
  - Integration with Gurobi MILP solver for exact optimization
  - Support for different precision modes (full MILP, relaxed LP, mixed)

- **`hybridz_transformers.py`**: Hybrid Zonotope transformer classes
  - Complexity-simplified Hybrid Zonotope activation transformer representations
  - Element and grid-based zonotope implementations
  - Abstract transformation methods for neural network layers

- **`hybridz_operations.py`**: MILP/LP operations and optimization
  - MILP/LP relaxation strategies with configurable precision ratios
  - Parallel generator merging optimization (configurable)
  - Three core configurations: Full-precision MILP, Fully-relaxed LP, Partially-relaxed MILP+LP
  - Gurobi license management and solver integration

### **`interval/` - ACT Interval Verification**
- **`base_verifier.py`**: Base verifier class and common functionality
  - Abstract interface for all verification backends
  - Common parameter validation and preprocessing
  - Unified result formatting and error handling
  - Shared utility methods for model and specification processing
  - Standard interval arithmetic for neural network verification

- **`bounds_propagation.py`**: Interval bound propagation implementation
  - Fast but potentially loose bound computation
  - Layer-by-layer propagation of interval bounds through neural networks

- **`outputs_evaluation.py`**: Output bounds evaluation utilities
  - Final output bound analysis and verification result determination

### **`input_parser/` - Specification and Data Handling**
- **`dataset.py`**: Dataset loading and preprocessing utilities
  - MNIST and CIFAR-10 dataset handlers with automatic download
  - CSV file processing for batch verification scenarios
  - Data normalization and preprocessing pipelines
  - Anchor dataset management for VNNLIB specification anchoring

- **`model.py`**: Neural network model loading and parsing
  - ONNX model loading with comprehensive validation
  - TensorFlow/PyTorch model support and conversion
  - Model architecture analysis and layer extraction
  - Compatibility checking across verification backends

- **`spec.py`**: Specification handling and processing
  - Local robustness (Lp-norm) specification processing
  - VNNLIB specification integration and validation
  - Set-based constraint handling for complex properties
  - Box constraint specifications for input space restrictions

- **`type.py`**: Type definitions and data structures
  - Verification result enumerations (SAT/UNSAT/UNKNOWN/TIMEOUT)
  - Backend and method type definitions
  - Parameter validation and constraint types
  - Common data structures for cross-module communication

- **`vnnlib_parser.py`**: VNNLIB format parser
  - Complete VNNLIB specification parsing and validation
  - Property extraction with support for complex constraints
  - Integration bridge for external verification tool compatibility
  - Comprehensive error handling for malformed specifications

- **`adaptor.py`**: Input/output adaptors
  - Data format conversion utilities
  - Interface adaptors between different verification backends

### **`refinement/` - Advanced Refinement**
- **`bab_spec_refinement.py`**: Branch-and-bound specification refinement
  - Prototype BaB-based specification refinement system
  - Automatic refinement when initial verification returns UNKNOWN/UNSAT
  - Configurable depth limits, subproblem constraints, and time bounds
  - Seamless integration with hybrid zonotope verification methods

### **`util/` - Utility Modules**
- **`options.py`**: Command-line argument parsing
  - Comprehensive argument parser for all verification backends
  - Parameter validation and type checking
  - Integration with configuration file defaults
  - Support for verifier-specific parameter sets

- **`path_config.py`**: Centralized path configuration
  - Eliminates redundant sys.path manipulations across files
  - Provides consistent import resolution for hierarchical structure
  - Enables clean absolute imports throughout the codebase

- **`stats.py`**: Statistics and performance tracking
  - Memory usage monitoring and optimization
  - Performance profiling and benchmarking utilities
  - Verification statistics collection and reporting

- **`inference.py`**: Model inference utilities
  - Common model inference operations
  - Batch processing support
  - Output format standardization

### **`wrapper_exts/` - External Verifier Integrations**

#### **`abcrown/` - αβ-CROWN Integration**
- **`abcrown_verifier.py`**: αβ-CROWN wrapper and interface
  - Translates ACT parameters to αβ-CROWN format
  - Manages conda environment isolation for αβ-CROWN execution
  - Handles subprocess communication and result parsing
  - Provides error handling and comprehensive logging

- **`abcrown_runner.py`**: αβ-CROWN backend execution script
  - Contains code adapted from the open-source αβ-CROWN project with enhancements for ACT framework integration
  - Executes within isolated `act-abcrown` conda environment
  - Direct interface to αβ-CROWN complete verification engine
  - Independent execution without ACT path dependencies

#### **`eran/` - ERAN Integration**
- **`eran_verifier.py`**: ERAN wrapper and interface
  - Integration with ERAN abstract interpretation methods
  - Support for DeepPoly, DeepZono, and other ERAN domains
  - Parameter translation for ERAN backend compatibility


## Architecture Benefits

The flattened modular architecture provides several key advantages:

### **Modular Design**
- **Clear Separation**: Core ACT modules (hybridz, interval) are separated from external verifier wrappers
- **Independent Development**: Modules can be developed, tested, and maintained separately
- **Easy Extension**: Add new verifiers by creating new modules in `wrapper_exts/`
- **Clean Dependencies**: Centralized utilities in `util/` eliminate code duplication

### **Flat Structure Benefits**
- **Simplified Imports**: Direct module access without deep nesting (e.g., `from hybridz.hybridz_verifier import ...`)
- **Faster Development**: Quick navigation to core modules without traversing multiple subdirectories
- **Clear Responsibilities**: Each top-level directory has a specific, well-defined purpose
- **Better IDE Support**: Enhanced autocomplete and navigation in development environments

### **Configuration Management**
- **Centralized Defaults**: Configuration files in `../configs/` provide optimal parameters
- **Path Resolution**: `util/path_config.py` ensures consistent module import resolution
- **Environment Isolation**: Different verifiers can use separate conda environments
- **Parameter Management**: `util/options.py` provides unified command-line interface

### **Integration Flexibility**
- **Unified Interface**: Single entry point (`main.py`) for all verification tasks
- **Backend Abstraction**: Consistent API regardless of underlying verification method
- **Parameter Translation**: Automatic conversion between ACT and backend-specific formats
- **Result Standardization**: Uniform output format across all verification backends

### **Performance Optimization**
- **Memory Management**: `util/stats.py` provides comprehensive memory tracking and optimization
- **Utility Reuse**: Common operations centralized in `util/` modules
- **Import Efficiency**: Flat structure reduces import overhead and circular dependency risks