# Verifier Directory

This directory contains the core verification framework and interfaces for the Abstract Constraint Transformer (ACT) system. It provides unified access to multiple verification backends including ACT native methods, ERAN, and αβ-CROWN through a hierarchical modular architecture.

## Directory Structure

```
verifier/
├── main.py                         # Main unified verification interface
├── path_config.py                  # Centralized path configuration
│
├── abstract_constraint_solver/     # Verification algorithm implementations
│   ├── base_verifier.py            # Base verifier class and common functionality
│   │
│   ├── abcrown/                    # αβ-CROWN integration module
│   │   ├── abcrown_verifier.py     # αβ-CROWN wrapper and interface
│   │   ├── abcrown_runner.py       # αβ-CROWN backend execution script
│   │   └── empty_config.yaml       # Required αβ-CROWN configuration
│   │
│   ├── eran/                       # ERAN integration module
│   │   └── eran_verifier.py        # ERAN wrapper and interface
│   │
│   ├── hybridz/                    # ACT Hybrid Zonotope module
│   │   ├── hybridz_verifier.py     # Hybrid Zonotope verification engine
│   │   ├── hybridz_transformers.py # Hybrid Zonotope transformer classes
│   │   └── hybridz_operations.py   # MILP/LP operations and optimization
│   │
│   └── interval/                   # ACT Interval verification module
│       └── interval_verifier.py    # Interval arithmetic verification
│
├── input_parser/                    # Specification parsing and data handling
│   ├── dataset.py                  # Dataset loading and preprocessing
│   ├── model.py                    # Neural network model parsing
│   ├── spec.py                     # Specification handling and processing
│   ├── type.py                     # Type definitions and data structures
│   └── vnnlib_parser.py            # VNNLIB format parser
│
└── bab_refinement/                 # Advanced refinement algorithms
    └── bab_spec_refinement.py      # Branch-and-bound specification refinement
```

## Module Documentation

### **Main Interface**
- **`main.py`**: Primary entry point for all verification tasks
  - Unified command-line interface supporting all verifiers
  - Parameter parsing, validation, and backend routing
  - Comprehensive argument compatibility across different verification tools
  - Integration with configuration defaults from `../configs/`

- **`path_config.py`**: Centralized module path configuration
  - Eliminates redundant sys.path manipulations across files
  - Provides consistent import resolution for hierarchical structure
  - Enables clean absolute imports throughout the codebase

### **`abstract_constraint_solver/` - Verification Algorithms**

#### **Base Framework**
- **`base_verifier.py`**: Base verifier class and common functionality
  - Abstract interface for all verification backends
  - Common parameter validation and preprocessing
  - Unified result formatting and error handling
  - Shared utility methods for model and specification processing

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

- **`empty_config.yaml`**: Required αβ-CROWN configuration template
  - Enables αβ-CROWN CLI execution within ACT framework
  - Provides default parameter structure for αβ-CROWN backend

#### **`eran/` - ERAN Integration**
- **`eran_verifier.py`**: ERAN wrapper and interface
  - Integration with ERAN abstract interpretation methods
  - Support for DeepPoly, DeepZono, and other ERAN domains
  - Parameter translation for ERAN backend compatibility

#### **`hybridz/` - ACT Hybrid Zonotope Verification**
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

#### **`interval/` - ACT Interval Verification**
- **`interval_verifier.py`**: Interval arithmetic verification
  - Standard interval arithmetic for neural network verification
  - Fast but potentially loose bound computation
  - Baseline verification method for comparison

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

### **`bab_refinement/` - Advanced Refinement**
- **`bab_spec_refinement.py`**: Branch-and-bound specification refinement
  - Prototype BaB-based specification refinement system
  - Automatic refinement when initial verification returns UNKNOWN/UNSAT
  - Configurable depth limits, subproblem constraints, and time bounds
  - Seamless integration with hybrid zonotope verification methods


## Architecture Benefits

The hierarchical modular architecture provides several key advantages:

### **Modular Design**
- **Clear Separation**: Each verification algorithm isolated in dedicated modules
- **Independent Development**: Modules can be developed, tested, and maintained separately
- **Easy Extension**: Add new verifiers by creating new modules in `abstract_constraint_solver/`
- **Clean Dependencies**: Centralized path management eliminates import complexity

### **Configuration Management**
- **Centralized Defaults**: Configuration files in `../configs/` provide optimal parameters
- **Path Resolution**: `path_config.py` ensures consistent module import resolution
- **Environment Isolation**: Different verifiers can use separate conda environments

### **Integration Flexibility**
- **Unified Interface**: Single entry point (`main.py`) for all verification tasks
- **Backend Abstraction**: Consistent API regardless of underlying verification method
- **Parameter Translation**: Automatic conversion between ACT and backend-specific formats
- **Result Standardization**: Uniform output format across all verification backends