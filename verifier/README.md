# Verifier Directory

This directory contains the core verification framework and interfaces for the Abstract Constraint Transformer (ACT) system. It provides unified access to multiple verification backends including ACT native methods, ERAN, and αβ-CROWN.

## Files Overview

### Main Interface
- **`main.py`**: Primary entry point for all verification tasks
  - Unified command-line interface for all verifiers
  - Contains code adapted from the open-source αβ-CROWN project with enhancements for ACT framework integration
  - Parameter parsing and validation
  - Backend selection and routing
  - Comprehensive argument compatibility across tools

### Backend Integrations
- **`abcrown_runner.py`**: αβ-CROWN integration wrapper
  - Contains code adapted from the open-source αβ-CROWN project with enhancements for ACT framework integration
  - Translates ACT parameters to αβ-CROWN format
  - Manages αβ-CROWN process execution
  - Handles result parsing and formatting
  - Provides error handling and logging

### ACT Native Implementation
- **`hybridz_verifier.py`**: ACT Hybrid Zonotope verification engine
- **`hybridz_transformers.py`**: Hybrid Zonotope transformer classes  
- **`hybridz_operations.py`**: Hybrid Zonotope operation methods
  - Novel tensorised hybrid verification method
  - Complexity-simplified Hybrid Zonotope activation transformer representation forms
  - MILP/LP relaxation strategies with configurable ratios
  - Parallel generator merging optimization - disabled by default
  - Three core configurations: Full-precision MILP, Fully-relaxed LP, Partially-relaxed MILP+LP

- **`bab_spec_refinement.py`**: Prototype Branch-and-bound specification refinement
  - BaB-based specification refinement if enabled by users
  - Automatic refinement when initial verification returns UNKNOWN/UNSAT
  - Configurable depth, subproblem limits, and time constraints
  - Integration with hybrid zonotope methods

### Utilities and Data Handling
- **`dataset.py`**: Dataset loading and preprocessing utilities
  - MNIST and CIFAR-10 dataset handlers
  - CSV file processing for batch verification
  - Data normalisation and preprocessing
  - Anchor dataset management for specification anchoring

- **`model.py`**: Neural network model loading and parsing
  - ONNX model loading and validation
  - TensorFlow/PyTorch model support
  - Model architecture analysis and layer extraction

- **`spec.py`**: Specification handling and parsing
  - Local robustness (Lp-norm) specifications
  - VNNLIB specification processing
  - Set-based constraint handling
  - Box constraint specifications

- **`vnnlib_parser.py`**: VNNLIB format parser
  - Complete VNNLIB specification parsing
  - Property extraction and validation
  - Integration with verification backends
  - Error handling for malformed specifications

### Configuration and Types
- **`type.py`**: Type definitions and data structures
  - Verification result types
  - Backend enumeration
  - Parameter validation types
  - Common data structures

- **`empty_config.yaml`**: αβ-CROWN configuration template
  - Required configuration file for αβ-CROWN backend operation
  - Enables αβ-CROWN CLI command execution within ACT framework
  - Patched during setup process for seamless integration


## Integration Architecture

The verifier directory implements a modular architecture:

1. **Unified Interface Layer**: `main.py` provides consistent API
2. **Backend Abstraction**: Runner modules translate between ACT and external tools
3. **Native Implementation**: ACT-specific verification algorithms
4. **Utility Layer**: Common data handling, parsing, and type definitions