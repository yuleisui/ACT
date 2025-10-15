# ACT Directory

This directory contains the core verification framework for the Abstract Constraint Transformer (ACT) system. It implements a modern three-tier architecture: Front-End (data/model/spec processing), Back-End (verification core), and Pipeline (testing/integration) with PyTorch-native verification capabilities.

## Directory Structure

```
act/
├── main.py                         # Main unified verification interface
├── __init__.py                     # Package initialization
│
├── front_end/                      # Front-End: User-facing data processing
│   ├── loaders/                    # Data, model, and specification loaders
│   │   ├── data_loader.py          # Dataset loading (MNIST/CIFAR/CSV/VNNLIB)
│   │   ├── model_loader.py         # Neural network model loading (ONNX/PyTorch)
│   │   └── spec_loader.py          # Specification loading and processing
│   ├── specs.py                    # InputSpec/OutputSpec with InKind/OutKind enums
│   ├── wrapper_layers.py           # PyTorch verification wrapper modules
│   ├── model_synthesis.py          # Advanced model generation and optimization
│   ├── device_manager.py           # GPU-first CUDA device handling
│   ├── preprocessor_image.py       # Image preprocessing and normalization
│   ├── preprocessor_text.py        # Text preprocessing utilities
│   ├── preprocessor_base.py        # Base preprocessor interface
│   ├── utils_image.py              # Image utility functions
│   ├── model_inference.py          # Model inference utilities
│   ├── mocks.py                    # Mock data generation for testing
│   └── README.md                   # Front-end documentation
│
├── back_end/                       # Back-End: Core verification engine
│   ├── core.py                     # Net, Layer, Bounds, Con, ConSet data structures
│   ├── verifier.py                 # Spec-free verification: verify_once(), verify_bab()
│   ├── layer_schema.py             # Layer type definitions and validation rules
│   ├── layer_validation.py         # Layer validation and creation utilities
│   ├── bab.py                      # Branch-and-bound refinement with CE validation
│   ├── utils.py                    # Backend utility functions
│   ├── analyze.py                  # Network analysis and bounds propagation
│   ├── cons_exportor.py            # Constraint export to solvers
│   ├── device_manager.py           # Backend device management
│   ├── solver/                     # MILP/LP optimization solvers
│   │   ├── solver_base.py          # Base solver interface
│   │   ├── solver_gurobi.py        # Gurobi MILP solver integration
│   │   └── solver_torch.py         # PyTorch-based LP solver
│   ├── transfer_funs/              # Transfer functions for different layer types
│   │   ├── tf_mlp.py               # MLP layer transfer functions
│   │   ├── tf_cnn.py               # CNN layer transfer functions
│   │   ├── tf_rnn.py               # RNN layer transfer functions
│   │   └── tf_transformer.py       # Transformer layer transfer functions
│   └── README.md                   # Back-end documentation
│
├── pipeline/                       # Pipeline: Testing framework and integration
│   ├── torch2act.py                # Automatic PyTorch→ACT Net conversion
│   ├── correctness.py              # Verifier correctness validation
│   ├── regression.py               # Baseline capture and regression testing
│   ├── integration.py              # Front-end integration bridge
│   ├── mock_factory.py             # Configurable mock input generation
│   ├── config.py                   # YAML-based test scenario management
│   ├── reporting.py                # Results analysis and report generation
│   ├── utils.py                    # Shared utilities and performance profiling
│   ├── run_tests.py                # Command-line testing interface
│   ├── configs/                    # Configuration files
│   │   ├── mock_inputs.yaml        # Mock data generation templates
│   │   ├── test_scenarios.yaml     # Complete test scenario definitions
│   │   ├── solver_settings.yaml    # Solver configuration options
│   │   └── baselines.json          # Performance baseline storage
│   ├── examples/                   # Example usage and quick tests
│   ├── log/                        # Test execution logs
│   ├── reports/                    # Generated test reports
│   └── README.md                   # Pipeline documentation
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

### **`front_end/` - User-Facing Data Processing**
- **`loaders/`**: Comprehensive data loading and preprocessing
  - **`data_loader.py`**: MNIST/CIFAR-10 dataset handlers with automatic download, CSV batch processing
  - **`model_loader.py`**: ONNX/PyTorch model loading with validation and conversion
  - **`spec_loader.py`**: VNNLIB specification loading and local robustness property handling

- **`specs.py`**: Specification data structures and enums
  - `InputSpec`/`OutputSpec` classes with `InKind`/`OutKind` type safety
  - Support for BOX, L_INF, LIN_POLY input constraints and SAFETY, ASSERT output properties

- **`wrapper_layers.py`**: PyTorch verification wrapper modules
  - `InputLayer`: Declares symbolic input blocks for verification
  - `InputAdapterLayer`: Config-driven input preprocessing (permute/reorder/slice/pad/affine/linear-proj)
  - `InputSpecLayer`: Wraps ACT InputSpec as nn.Module for seamless integration
  - `OutputSpecLayer`: Wraps ACT OutputSpec as nn.Module for property specification

- **`model_synthesis.py`**: Advanced model generation and optimization
  - Neural architecture synthesis and domain-specific model generation
  - Model optimization utilities and synthesis pipeline

- **`device_manager.py`**: GPU-first CUDA device handling
  - Automatic device detection and management
  - GPU memory optimization and fallback strategies

- **Preprocessors**: Modular preprocessing pipeline
  - **`preprocessor_image.py`**: Image normalization, augmentation, and format conversion
  - **`preprocessor_text.py`**: Text preprocessing utilities
  - **`preprocessor_base.py`**: Base preprocessor interface and common functionality

### **`back_end/` - Core Verification Engine**
- **`core.py`**: Fundamental ACT data structures
  - `Net`: Network representation with layers and graph connectivity
  - `Layer`: Individual layer with params, metadata, and variable mappings
  - `Bounds`: Box constraints with lb/ub tensors for variable ranges
  - `Con`/`ConSet`: Constraint representation and management

- **`verifier.py`**: Spec-free verification engine
  - `verify_once()`: Single-shot verification using embedded ACT constraints
  - `verify_bab()`: Branch-and-bound refinement with counterexample validation
  - No external input specs required - all constraints extracted from ACT Net

- **`layer_schema.py`**: Layer type definitions and validation rules
  - Comprehensive schema definitions for all supported layer types
  - Parameter validation and metadata requirements

- **`bab.py`**: Branch-and-bound refinement implementation
  - BaB tree management with priority queues
  - Counterexample validation and refinement strategies
  - Configurable depth limits and timeout handling

- **`solver/`**: MILP/LP optimization backend
  - **`solver_gurobi.py`**: Gurobi MILP solver integration with license management
  - **`solver_torch.py`**: PyTorch-based LP solver for lightweight optimization
  - **`solver_base.py`**: Unified solver interface and status handling

- **`transfer_funs/`**: Layer-specific analysis functions
  - **`tf_mlp.py`**: Multi-layer perceptron transfer functions
  - **`tf_cnn.py`**: Convolutional neural network layer analysis
  - **`tf_rnn.py`**: Recurrent neural network transfer functions
  - **`tf_transformer.py`**: Transformer block analysis and attention handling

### **`pipeline/` - Testing Framework and Integration**
- **`torch2act.py`**: Automatic PyTorch→ACT Net conversion
  - Seamless conversion from PyTorch nn.Module to ACT Net representation
  - Preserves all verification constraints and model semantics
  - Support for complex wrapper layer patterns

- **Testing Framework**: Comprehensive validation and regression testing
  - **`correctness.py`**: Verifier correctness validation with property-based testing
  - **`regression.py`**: Baseline capture and performance regression detection
  - **`mock_factory.py`**: Configurable mock input generation from YAML templates

- **`integration.py`**: Front-end integration bridge
  - Real ACT component integration for testing
  - Bridge between pipeline framework and ACT front-end loaders
  - Complete test case generation and validation

- **`config.py`**: YAML-based test scenario management
  - Configuration loading and validation
  - Test scenario composition and parameter management

- **Performance and Reporting**:
  - **`utils.py`**: Performance profiling, memory tracking, and optimization utilities
  - **`reporting.py`**: Results analysis and comprehensive report generation
  - **`run_tests.py`**: Command-line testing interface with parallel execution

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

The three-tier modular architecture provides several key advantages:

### **Three-Tier Design**
- **Front-End Separation**: User-facing data processing isolated from core verification logic
- **Back-End Focus**: Pure verification engine with PyTorch-native analysis and optimization
- **Pipeline Integration**: Comprehensive testing framework and Torch→ACT conversion bridge
- **Clean Boundaries**: Clear interfaces between data processing, verification, and testing

### **Modern Verification Features**
- **Spec-Free Verification**: All constraints embedded in PyTorch models via wrapper layers
- **PyTorch-Native**: Verification engine operates directly on PyTorch tensors for performance
- **Automatic Conversion**: Seamless PyTorch→ACT Net conversion preserving all semantics
- **GPU-First**: Optimized CUDA device management with automatic fallback strategies

### **Modular Design**
- **Clear Separation**: Front-end, back-end, and pipeline modules have distinct responsibilities
- **Independent Development**: Modules can be developed, tested, and maintained separately
- **Easy Extension**: Add new verifiers by creating new modules in `wrapper_exts/`
- **Reusable Components**: Shared utilities and interfaces enable code reuse

### **Testing and Validation**
- **Comprehensive Testing**: Pipeline framework provides correctness, regression, and performance testing
- **Mock Generation**: Configurable mock input generation from YAML templates
- **Integration Testing**: Real ACT component testing with front-end bridge
- **Continuous Validation**: Baseline capture and regression detection for quality assurance

### **Configuration Management**
- **Centralized Defaults**: Configuration files in `../configs/` provide optimal parameters
- **Device Management**: Intelligent GPU/CPU device selection and memory optimization
- **Environment Isolation**: Different verifiers can use separate conda environments
- **Parameter Management**: Unified command-line interface with type validation

### **Integration Flexibility**
- **Unified Interface**: Single entry point (`main.py`) for all verification tasks
- **Backend Abstraction**: Consistent API regardless of underlying verification method
- **Parameter Translation**: Automatic conversion between ACT and backend-specific formats
- **Result Standardization**: Uniform output format across all verification backends

### **Performance Optimization**
- **Memory Management**: Comprehensive memory tracking and optimization throughout pipeline
- **Utility Reuse**: Common operations centralized to eliminate code duplication
- **Efficient Imports**: Modular structure reduces import overhead and circular dependencies
- **GPU Acceleration**: PyTorch-native verification leverages GPU computation where beneficial