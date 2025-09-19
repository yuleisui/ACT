# Abstract Constraint Transformer (ACT)

A unified neural network verification framework that integrates multiple state-of-the-art verifiers including ERAN, αβ-CROWN, and the novel Hybrid Zonotope methods. ACT provides a tensorised implementation for efficient verification of deep neural networks with support for various specification formats and robustness properties.

## Overview

ACT combines three powerful verification approaches:
- **[ERAN](https://github.com/eth-sri/eran)**: ETH Robustness Analyser with ab### Set-based VNNLIB (`set_vnnlib`)
- **[αβ-CROWN](https://github.com/Verified-Intelligence/alpha-beta-CROWN)**: Complete neural network verifier with Branch-and-Bound (BaB)
- **Hybrid Zonotope**: Novel tensorised hybrid verification method with MILP/LP relaxation strategies, incorporating three core configurations (Full-precision MILP, Fully-relaxed LP, Partially-relaxed MILP+LP)

The framework supports multiple input formats (ONNX, TF, Pytorch models, VNNLIB specifications) and provides comprehensive verification capabilities for classification tasks.

## Project Structure

```
Abstract-Constraint-Transformer/
├── README.md                           # This file - comprehensive project documentation
├── .gitmodules                         # Git submodule configuration for ERAN and αβ-CROWN
├── .gitignore                          # Git ignore patterns for Python, conda environments, etc
│
├── setup/                              # Environment setup scripts and requirements
│   ├── README.md                       # Setup documentation and troubleshooting guide
│   ├── setup.sh                        # Main automated setup script for all environments
│   ├── eran_env_setup.sh               # ERAN-specific environment setup script
│   ├── main_requirements.txt           # ACT main environment Python dependencies
│   ├── abcrown_requirements.txt        # αβ-CROWN environment Python dependencies
│   └── eran_requirements.txt           # ERAN environment Python dependencies
│
├── verifier/                           # Main verification framework and interfaces
│   ├── README.md                       # Verifier module documentation and usage guide
│   ├── verifier_tensorised.py          # Main unified verification interface (entry point)
│   ├── abcrown_runner.py               # αβ-CROWN integration and execution wrapper
│   ├── hybridz_tensorised.py           # ACT Hybrid Zonotope verification implementation
│   ├── bab_spec_refinement.py          # Branch-and-bound specification refinement
│   ├── dataset.py                      # Dataset loading and preprocessing utilities
│   ├── model.py                        # Neural network model loading and parsing
│   ├── spec.py                         # Specification handling (VNNLIB, local robustness)
│   ├── type.py                         # Type definitions and data structures
│   ├── vnnlib_parser.py                # VNNLIB specification format parser
│   └── empty_config.yaml               # Required configuration file for αβ-CROWN backend operation
│
├── modules/                            # External verifier submodules
│   ├── README.md                       # Submodules overview and integration notes
│   ├── abcrown/                        # αβ-CROWN complete verifier submodule
│   └── eran/                           # ERAN abstract interpretation verifier submodule
│
├── models/                             # Pre-trained neural network models
│   ├── README.md                       # Model collection overview and usage guide
│   ├── Sample_models/                  # Custom sample models for testing and development
│   │   ├── MNIST/                      # Sample MNIST CNN models
│   │   │   ├── small_relu_mnist_cnn_model_1.onnx       # Small CNN with ReLU activation
│   │   │   ├── small_sigmoid_mnist_cnn_model_1.onnx    # Small CNN with Sigmoid activation
│   │   │   └── small_tanh_mnist_cnn_model_1.onnx       # Small CNN with Tanh activation
│   │   │
│   │   └── CIFAR10/                                    # Sample CIFAR-10 CNN models
│   │       ├── small_relu_cifar10_cnn_model_1.onnx     # Small CNN with ReLU activation
│   │       ├── small_sigmoid_cifar10_cnn_model_1.onnx  # Small CNN with Sigmoid activation
│   │       └── small_tanh_cifar10_cnn_model_1.onnx     # Small CNN with Tanh activation
│
└── data/                                               # Sample datasets and verification specifications
    ├── README.md                                       # Dataset documentation and format specifications
    ├── MNIST_csv/                                      # MNIST dataset in CSV format
    │   └── mnist_first_100_samples.csv                 # First 100 MNIST test samples for verification
    │
    ├── CIFAR10_csv/                                    # CIFAR-10 dataset in CSV format
    │   └── cifar10_first_100_samples.csv               # First 100 CIFAR-10 test samples for verification
    │
    ├── anchor/                                         # Anchor datasets for specification anchoring
    │   └── mnist_csv.csv                               # MNIST anchor points for VNNLIB specification anchoring
    │
    ├── vnnlib/                                         # VNNLIB specification examples
    │   ├── set_vnnlib_example.vnnlib                   # Set-based property specification example (e.g., ACAS Xu style)
    │   └── local_vnnlib_example.vnnlib                 # Local robustness property specification example
    │
    └── json/                                           # JSON-format specification examples
        └── test_global_box.json                        # Box constraint specification in JSON format
```

## Installation

### Prerequisites

Before installation, ensure you have the following:
- **Git**: For cloning the repository with submodules
- **Linux system**: Recommended for ERAN dependencies (Ubuntu/Debian preferred)
- **Miniconda or Anaconda**: Required for environment management
  - Follow the steps: https://www.anaconda.com/docs/getting-started/miniconda/install#quickstart-install-instructions

### Gurobi License Setup (Required)

**Important**: ACT requires Gurobi optimizer for MILP/LP solving. Academic users can obtain **free licenses**.

1. **Academic License**: Visit https://www.gurobi.com/academia/ to obtain a free academic license
2. **License Installation**: Once obtained, place your `gurobi.lic` file at: ./gurobi/gurobi.lic

**Note**: Without a valid Gurobi license, verification methods that rely on MILP/LP solving (HybridZ methods) will not function.

### 1. Clone Repository

Clone the repository with all submodules:

```bash
git clone --recursive https://github.com/doctormeeee/Abstract-Constraint-Transformer.git
cd Abstract-Constraint-Transformer
```

### 2. Environment Setup

Run the automated setup script to create all required conda environments:

```bash
cd setup/
bash setup.sh
```

This script will create three separate conda environments:

#### Main Environment (`act-main`)
- **Purpose**: Primary ACT framework with Hybrid Zonotope verification and BaB-based specification refinement
- **Python Version**: 3.9
- **Key Dependencies**: PyTorch 2.x, ONNX tools, Gurobi, NumPy
- **Requirements**: `main_requirements.txt`

```bash
# Main environment packages include:
torch>=2.0.0,<2.4.0
torchvision>=0.12.0,<0.19.0
numpy>=1.20,<2.0
onnx, onnxruntime>=1.15
gurobipy>=10
tensorflow
```

#### αβ-CROWN Environment (`act-abcrown`)
- **Purpose**: αβ-CROWN complete verification
- **Python Version**: 3.9
- **Key Dependencies**: PyTorch 2.x, ONNX tools, Gurobi
- **Requirements**: `abcrown_requirements.txt` (derived from official αβ-CROWN repository setup scripts)

```bash
# αβ-CROWN environment packages include:
torch>=2.0.0,<2.4.0
onnxsim>=0.4.31
gurobipy>=10
sortedcontainers>=2.4
```

#### ERAN Environment (`act-eran`)
- **Purpose**: ERAN abstract interpretation methods
- **Python Version**: 3.8 (for ONNX 1.8.0 compatibility)
- **Key Dependencies**: TensorFlow 2.9.3, ONNX 1.8.0, ELINA, Gurobi
- **Requirements**: `eran_requirements.txt` (derived from official ERAN repository setup scripts)

```bash
# ERAN environment packages include:
numpy==1.23.5
tensorflow==2.9.3
onnx==1.8.0
torch, torchvision
pycddlib
```

The setup script automatically:
- Checks for conda installation
- Creates three separate environments with appropriate Python versions
- Installs all required dependencies
- Compiles ERAN system dependencies (GMP, MPFR, ELINA, DeepG)
- Configures Gurobi for optimisation
- Patches αβ-CROWN imports to prevent conflicts

## Usage

### Basic Verification Command

Activate the main environment and run verification:

```bash
conda activate act-main
cd verifier/
python verifier_tensorised.py [OPTIONS]
```

### Command Line Arguments

ACT provides a comprehensive command-line interface with parameters organized into ACT native features and external tool compatibility. All parameters are designed to work seamlessly across different verification backends.

#### Core Verification Parameters

```bash
--verifier {interval,eran,abcrown,hybridz}    # Verification backend selection
                                              # - interval: ACT native interval analysis
                                              # - eran: ERAN external verifier
                                              # - abcrown: αβ-CROWN external verifier  
                                              # - hybridz: ACT novel hybrid zonotope verifier

--method METHOD                               # Verification method (depends on verifier):
                                              # ERAN: deepzono, refinezono, deeppoly, refinepoly
                                              # αβ-CROWN: alpha, beta, alpha_beta
                                              # ACT-HybridZ: hybridz, hybridz_relaxed, hybridz_relaxed_with_bab
                                              # ACT-Interval: interval

--device {cpu,cuda}                           # Computation device (default: cpu)
                                              # Note: CUDA support currently limited to CPU only
```

#### Model and Data Parameters

```bash
--model_path PATH                             # Path to neural network model file (.onnx/.tf/.pt format supported)
--dataset {mnist,cifar10} or PATH             # Dataset name or path to CSV file
--anchor PATH                                 # Anchor dataset path for data point anchoring in specifications

--start INT                                   # Start from the i-th property in dataset (default: 0)
--end INT                                     # End with the (i-1)-th property in dataset (default: 10000)
--select_instance INT [INT ...]               # Select specific instances to verify (list of indices)
--num_outputs INT                             # Number of output classes (default: 10)

--mean FLOAT [FLOAT ...]                      # Mean values for data preprocessing normalisation
                                              # (auto-detected: MNIST=[0.1307], CIFAR10=[0.4914,0.4822,0.4465])
--std FLOAT [FLOAT ...]                       # Standard deviation for data preprocessing normalisation  
                                              # (auto-detected: MNIST=[0.3081], CIFAR10=[0.2023,0.1994,0.2010])
```

#### Specification Parameters

```bash
--spec_type {local_lp,local_vnnlib,set_vnnlib,set_box}
                                              # Verification specification type:
                                              # - local_lp: Lp norm around data points
                                              # - local_vnnlib: VNNLIB with anchor points  
                                              # - set_vnnlib: Set-based VNNLIB (e.g., AcasXu)
                                              # - set_box: Box constraints

--norm {1,2,inf}                              # Lp-norm for epsilon perturbation (default: inf)
--epsilon FLOAT                               # Perturbation bound (Lp norm)
--epsilon_min FLOAT                           # Minimum perturbation bound (default: 0.0)

--robustness_type {verified-acc,runnerup,clean-acc,specify-target,all-positive}
                                              # Robustness verification target:
                                              # - verified-acc: Verify against all labels
                                              # - runnerup: Verify against runner-up labels only

--vnnlib_path PATH                            # Path to VNNLIB specification file
--vnnlib_path_prefix PATH                     # Prefix for VNNLIB specification paths
--input_lb FLOAT [FLOAT ...]                  # Lower bounds for set-based input specification  
--input_ub FLOAT [FLOAT ...]                  # Upper bounds for set-based input specification
```

#### ACT Hybrid Zonotope Parameters (Novel Features)

```bash
--relaxation_ratio FLOAT                      # ACT Hybrid Zonotope relaxation ratio (default: 1.0)
                                              # - 0.0: Full-precision MILP
                                              # - 1.0: Fully-relaxed LP
                                              # - 0.0~1.0: Partially-relaxed MILP+LP
                                              # Note: hybridz_relaxed_with_bab forces 1.0

--enable_generator_merging                    # ACT parallel generator merging optimisation in final 
                                              # linear layer. Note: This feature introduces noticeable performance  
                                              # overhead due to nested-loop cosine similarity calculations, 
                                              # so it is disabled by default.

--cosine_threshold FLOAT                      # ACT Cosine similarity threshold for parallel generator 
                                              # detection (default: 0.95, range: 0.0-1.0)
```

#### ACT Specification Refinement Parameters (Branch-and-Bound)

```bash
--enable_spec_refinement                      # ACT implementation: Enable specification refinement BaB verification
                                              # Automatically triggers when initial verification returns UNKNOWN/UNSAT

--bab_max_depth INT                           # ACT BaB: Maximum search depth (default: 8)
--bab_max_subproblems INT                     # ACT BaB: Maximum number of subproblems (default: 500)
--bab_time_limit FLOAT                        # ACT BaB: Time limit in seconds (default: 300.0)
--bab_split_tolerance FLOAT                   # ACT BaB: Split tolerance (default: 1e-6)
--bab_verbose                                 # ACT BaB: Enable verbose output (default: True)
```

#### αβ-CROWN Parameters

```bash
--root_path PATH                              # Root path prefix for relative file paths
--pkl_path PATH                               # Load verification properties from pickle file
--filter_path PATH                            # Pickle file containing examples to skip
--data_idx_file PATH                          # Text file with list of example IDs to run
--load_model PATH                             # Load pretrained model from specified path (αβ-CROWN compatibility)
--rhs_offset FLOAT                            # Offset to add to right-hand side of constraints (advanced usage)
```

#### Parameter Compatibility Notes

- **ERAN Verifier**: Supports datasets: `mnist`, `cifar10`, `acasxu`. Does not support specification refinement BaB.
- **αβ-CROWN Verifier**: Supports datasets: `mnist`, `cifar`, `eran`. Supports BaB.
- **ACT Interval Verifier**: Supports all datasets and all ACT native features including BaB-based specification refinement.
- **ACT Hybrid Zonotope Verifier**: Supports all datasets and all ACT native features. Specification refinement BaB only available with `hybridz_relaxed_with_bab` method.

**CSV File Usage**:
- **For all verifiers**: Use `--dataset path/to/data.csv` for verification datasets containing data points and labels

### Example Usage

ACT provides three main verification backends with specific dataset and method support:

#### 1. ERAN Verifier (External Tool Integration)

**Supported Datasets**: Torchvision datasets (`--dataset mnist` or `--dataset cifar10`)  
**Methods**: `deepzono`, `refinezono`, `deeppoly`, `refinepoly`  

```bash
# MNIST with DeepPoly
python verifier_tensorised.py \
    --verifier eran \
    --method deeppoly \
    --model_path ../models/Sample_models/MNIST/small_relu_mnist_cnn_model_1.onnx \
    --dataset mnist \
    --spec_type local_lp \
    --start 0 \
    --end 10 \
    --epsilon 0.1 \
    --norm inf \
    --mean 0.1307 \
    --std 0.3081

# CIFAR-10 with DeepZono
python verifier_tensorised.py \
    --verifier eran \
    --method deepzono \
    --model_path ../models/Sample_models/CIFAR10/small_relu_cifar10_cnn_model_1.onnx \
    --dataset cifar10 \
    --spec_type local_lp \
    --start 0 \
    --end 5 \
    --epsilon 0.01 \
    --norm inf \
    --mean 0.4914 0.4822 0.4465 \
    --std 0.2023 0.1994 0.2010
```

#### 2. αβ-CROWN Verifier (External Tool Integration)

**Supported Datasets**: Torchvision datasets only (`--dataset mnist` or `--dataset cifar`)  
**Methods**: `alpha`, `beta`, `alpha_beta`  

```bash
# MNIST with Alpha-Beta-Crown
python verifier_tensorised.py \
    --verifier abcrown \
    --method alpha_beta \
    --model_path ../models/Sample_models/MNIST/small_relu_mnist_cnn_model_1.onnx \
    --dataset mnist \
    --spec_type local_lp \
    --start 0 \
    --end 10 \
    --epsilon 0.1 \
    --norm inf \
    --mean 0.1307 \
    --std 0.3081

# CIFAR-10 with Beta-Crown
python verifier_tensorised.py \
    --verifier abcrown \
    --method beta \
    --model_path ../models/Sample_models/CIFAR10/small_relu_cifar10_cnn_model_1.onnx \
    --dataset cifar \
    --spec_type local_lp \
    --start 0 \
    --end 5 \
    --epsilon 0.01 \
    --norm inf \
    --mean 0.4914 0.4822 0.4465 \
    --std 0.2023 0.1994 0.2010
```

#### 3. HybridZ Verifier (ACT Native Implementation)

**Supported Datasets**: Three types supported:
- **Torchvision**: `--dataset mnist` or `--dataset cifar10`
- **CSV files**: `--dataset path/to/file.csv --spec_type local_lp`
- **VNNLIB**: `--vnnlib_path path/to/file.vnnlib --spec_type local_vnnlib/set_vnnlib`

**Methods Available**:
- `hybridz` → **HybridZ-Full** (Full-precision MILP)
- `hybridz_relaxed --relaxation_ratio 1.0` → **HybridZ-Relaxed** (Fully-relaxed LP)
- `hybridz_relaxed --relaxation_ratio 0.5` → **HybridZ-Partial** (Partial relaxation MILP+LP)
- `hybridz_relaxed_with_bab --enable_spec_refinement` → **HybridZ+BaB** (Prototype with specification refinement)

##### 3.1 HybridZ with Torchvision Datasets

```bash
# HybridZ-Full: MNIST verification
python verifier_tensorised.py \
    --verifier hybridz \
    --method hybridz \
    --model_path ../models/Sample_models/MNIST/small_relu_mnist_cnn_model_1.onnx \
    --dataset mnist \
    --spec_type local_lp \
    --start 0 \
    --end 1 \
    --epsilon 0.1 \
    --norm inf \
    --mean 0.1307 \
    --std 0.3081

# HybridZ-Relaxed: CIFAR-10 verification  
python verifier_tensorised.py \
    --verifier hybridz \
    --method hybridz_relaxed \
    --model_path ../models/Sample_models/CIFAR10/small_relu_cifar10_cnn_model_1.onnx \
    --dataset cifar10 \
    --spec_type local_lp \
    --start 0 \
    --end 5 \
    --epsilon 0.01 \
    --norm inf \
    --relaxation_ratio 1.0 \
    --mean 0.4914 0.4822 0.4465 \
    --std 0.2023 0.1994 0.2010

# HybridZ-Partial: Mixed MILP+LP with 50% relaxation
python verifier_tensorised.py \
    --verifier hybridz \
    --method hybridz_relaxed \
    --model_path ../models/Sample_models/MNIST/small_relu_mnist_cnn_model_1.onnx \
    --dataset mnist \
    --spec_type local_lp \
    --start 0 \
    --end 5 \
    --epsilon 0.1 \
    --norm inf \
    --relaxation_ratio 0.5 \
    --mean 0.1307 \
    --std 0.3081
```

##### 3.2 HybridZ with CSV Files

```bash
# CSV-based batch verification
python verifier_tensorised.py \
    --verifier hybridz \
    --method hybridz \
    --model_path ../models/Sample_models/MNIST/small_relu_mnist_cnn_model_1.onnx \
    --dataset ../data/MNIST_csv/mnist_first_100_samples.csv \
    --spec_type local_lp \
    --start 0 \
    --end 10 \
    --epsilon 0.1 \
    --norm inf \
    --mean 0.1307 \
    --std 0.3081

# Relaxed verification with CSV
python verifier_tensorised.py \
    --verifier hybridz \
    --method hybridz_relaxed \
    --model_path ../models/Sample_models/MNIST/small_relu_mnist_cnn_model_1.onnx \
    --dataset ../data/MNIST_csv/mnist_first_100_samples.csv \
    --spec_type local_lp \
    --start 0 \
    --end 10 \
    --epsilon 0.1 \
    --norm inf \
    --relaxation_ratio 1.0 \
    --mean 0.1307 \
    --std 0.3081
```

##### 3.3 HybridZ with VNNLIB Specifications

```bash
# Local VNNLIB with anchor points
python verifier_tensorised.py \
    --verifier hybridz \
    --method hybridz \
    --model_path ../models/Sample_models/MNIST/small_relu_mnist_cnn_model_1.onnx \
    --vnnlib_path ../data/vnnlib/local_vnnlib_example.vnnlib \
    --spec_type local_vnnlib \
    --anchor ../data/anchor/mnist_csv.csv

# Set-based VNNLIB verification
python verifier_tensorised.py \
    --verifier hybridz \
    --method hybridz_relaxed \
    --model_path ../models/Sample_models/MNIST/small_relu_mnist_cnn_model_1.onnx \
    --vnnlib_path ../data/vnnlib/set_vnnlib_example.vnnlib \
    --spec_type set_vnnlib \
    --relaxation_ratio 1.0
```

##### 3.4 HybridZ+BaB Specification Refinement (Prototype)

**Important**: This method requires `--enable_spec_refinement` flag and is currently a prototype feature.

```bash
# HybridZ with BaB specification refinement
python verifier_tensorised.py \
    --verifier hybridz \
    --method hybridz_relaxed_with_bab \
    --model_path ../models/Sample_models/MNIST/small_relu_mnist_cnn_model_1.onnx \
    --dataset mnist \
    --spec_type local_lp \
    --start 0 \
    --end 5 \
    --epsilon 0.1 \
    --norm inf \
    --enable_spec_refinement \
    --bab_max_depth 8 \
    --bab_max_subproblems 500 \
    --bab_time_limit 300 \
    --mean 0.1307 \
    --std 0.3081

# HybridZ+BaB with VNNLIB
python verifier_tensorised.py \
    --verifier hybridz \
    --method hybridz_relaxed_with_bab \
    --model_path ../models/Sample_models/MNIST/small_relu_mnist_cnn_model_1.onnx \
    --vnnlib_path ../data/vnnlib/local_vnnlib_example.vnnlib \
    --spec_type local_vnnlib \
    --anchor ../data/anchor/mnist_csv.csv \
    --enable_spec_refinement \
    --bab_max_depth 5 \
    --bab_time_limit 120
```

## Verification Methods

### ACT Native Methods

#### Interval Verification
- **Method**: `interval`
- **Features**: Fast interval analysis with tight bounds, supports all ACT native features including specification refinement

#### HybridZ-Full
**Full-precision MILP verification**
```bash
python verifier_tensorised.py \
    --verifier hybridz \
    --method hybridz \
    --model_path ../models/Sample_models/MNIST/small_relu_mnist_cnn_model_1.onnx \
    --dataset ../data/MNIST_csv/mnist_first_100_samples.csv \
    --spec_type local_lp \
    --start 0 \
    --end 5 \
    --epsilon 0.1 \
    --norm inf \
    --mean 0.1307 \
    --std 0.3081
```

#### HybridZ-Relaxed  
**Fully-relaxed LP verification (verifier=hybridz, method=hybridz_relaxed, relaxation_ratio=1)**
```bash
python verifier_tensorised.py \
    --verifier hybridz \
    --method hybridz_relaxed \
    --model_path ../models/Sample_models/MNIST/small_relu_mnist_cnn_model_1.onnx \
    --dataset ../data/MNIST_csv/mnist_first_100_samples.csv \
    --spec_type local_lp \
    --start 0 \
    --end 5 \
    --epsilon 0.1 \
    --norm inf \
    --relaxation_ratio 1.0 \
    --mean 0.1307 \
    --std 0.3081
```

#### HybridZ-Partial
**Partial relaxation MILP+LP verification (verifier=hybridz, method=hybridz_relaxed, relaxation_ratio 0-1)**
```bash
python verifier_tensorised.py \
    --verifier hybridz \
    --method hybridz_relaxed \
    --model_path ../models/Sample_models/MNIST/small_relu_mnist_cnn_model_1.onnx \
    --dataset ../data/MNIST_csv/mnist_first_100_samples.csv \
    --spec_type local_lp \
    --start 0 \
    --end 5 \
    --epsilon 0.1 \
    --norm inf \
    --relaxation_ratio 0.5 \
    --mean 0.1307 \
    --std 0.3081
```

#### HybridZ+BaB Spec Refinement
**Hybrid zonotope with branch-and-bound specification refinement (prototype)**
```bash
python verifier_tensorised.py \
    --verifier hybridz \
    --method hybridz_relaxed_with_bab \
    --model_path ../models/Sample_models/MNIST/small_relu_mnist_cnn_model_1.onnx \
    --dataset ../data/MNIST_csv/mnist_first_100_samples.csv \
    --spec_type local_lp \
    --start 1 \
    --end 2 \
    --epsilon 0.39 \
    --enable_spec_refinement \
    --mean 0.1307 \
    --std 0.3081
```
- **Features**: 
  - Configurable BaB parameters (depth, subproblems, time limits)
  - Automatic specification refinement when initial verification is inconclusive

### External Tool Integration

#### ERAN Methods
- **deeppoly**: Precise abstract interpretation with polyhedra
- **deepzono**: Zonotope-based abstract interpretation  
- **refinepoly**: Refinement-based DeepPoly
- **refinezono**: Refinement-based DeepZono
- **Limitations**: Only supports torchvision datasets (`mnist`, `cifar10`, `acasxu`)

#### αβ-CROWN Methods
- **alpha**: Fast incomplete verification with linear bounds
- **beta**: Complete verification with branch-and-bound
- **alpha_beta**: Combined α-β approach for efficiency with reused intermediate bounds
## Specification Formats

ACT supports multiple specification formats for different types of verification problems:

### Local Robustness (`local_lp`)
**Lp-norm perturbation around specific data points**
```bash
--spec_type local_lp --norm inf --epsilon 0.1 --dataset mnist
```
- Creates Lp-norm balls around input data points
- Suitable for robustness verification of classification models
- Requires: `--norm`, `--epsilon`, and a dataset

### VNNLIB with Anchor Points (`local_vnnlib`)  
**VNNLIB specifications anchored to dataset points**
```bash
--spec_type local_vnnlib --vnnlib_path spec.vnnlib --anchor ../data/anchor/mnist_csv.csv
```
- Combines VNNLIB specification with specific anchor data points
- Useful for targeted verification around known inputs
- Requires: `--vnnlib_path` and `--anchor`

### Set-based VNNLIB (`set_vnnlib`)
**Pure VNNLIB specifications without anchor points**
```bash
--spec_type set_vnnlib --vnnlib_path ../data/vnnlib/set_vnnlib_example.vnnlib
```
- Direct VNNLIB specification verification
- Suitable for competition-style verification problems (e.g., VNN-COMP)
- Requires: `--vnnlib_path`

### Box Constraints (`set_box`)
**Direct input bound specification**
```bash
--spec_type set_box --input_lb 0.0 0.0 0.0 --input_ub 1.0 1.0 1.0
```
- Simple box constraints on input variables
- Useful for basic reachability analysis
- Requires: `--input_lb` and `--input_ub`

### CSV-based Batch Verification
**Batch verification from CSV files**
```bash
--dataset ../data/MNIST_csv/mnist_first_100_samples.csv --spec_type local_lp
```
- Processes multiple verification instances from CSV
- CSV format: `label,pixel_0,pixel_1,...,pixel_N`
- Can be combined with any specification type

## Performance Optimisation


### ACT Parallel Generator Merging - disabled by default due to performance overhead from nested-loop cosine similarity calculations

**Optimise final layer computation:**
```bash
--enable_generator_merging --cosine_threshold 0.95
```
- ACT implementation for hybrid zonotope verification
- Merges parallel generators in final linear layer
- `cosine_threshold`: Controls similarity detection (higher = stricter)

### Relaxation Strategy Control
**Simplified Activation Transformers with controlled MILP/LP trade-off for hybrid methods:**
```bash
--relaxation_ratio 0.5  # 50% MILP + 50% LP relaxation
--relaxation_ratio 0.0  # Full-precision MILP (slower, more precise) - equivalent to hybridz
--relaxation_ratio 1.0  # Fully-relaxed LP (faster, less precise)
```
- Only applicable to `hybridz_relaxed` method

### Specification Refinement Optimisation
**Configure BaB parameters for efficiency:**
```bash
--enable_spec_refinement \
--bab_max_depth 5 \          # Reduce for faster verification
--bab_max_subproblems 100 \  # Limit subproblem explosion
--bab_time_limit 60          # Set time bounds
```

## Output and Results

### Verification Status

ACT displays comprehensive verification results directly in the command line:

**Verification Status Values**:
- **SAT (Satisfied)**: Property holds for all inputs in specification
- **UNSAT (Unsatisfied)**: Property violated, counterexample may be available
- **UNKNOWN**: Verification inconclusive (timeout, resource limits, method limitations)
- **TIMEOUT**: Verification exceeded time limits
- **CLEAN_FAILURE**: Clean prediction on input center point failed

**Command Line Output Includes**:
- Individual sample verification results
- Overall verification statistics and percentages
- Total verification time and performance metrics
- Method-specific information (HybridZ relaxation ratios, BaB refinement statistics)
- Memory usage and layer-by-layer processing times

## Troubleshooting

### Environment Issues
If environments fail to activate:
```bash
conda init
source ~/.bashrc  # or ~/.zshrc
```

### ERAN Dependencies
For ERAN compilation issues, ensure system dependencies:
```bash
sudo apt-get update
sudo apt-get install build-essential cmake m4 autoconf libtool
```

## Contributing

This framework integrates multiple verification tools. For contributions:
- Framework improvements: Submit PRs to this repository
- ERAN issues: Report to ETH-SRI ERAN repository
- αβ-CROWN issues: Report to Verified-Intelligence αβ-CROWN repository

## Licence

This project integrates multiple tools with different licences:
- ACT Framework: BSD 3-Clause
- ERAN: Apache 2.0
- αβ-CROWN: BSD 3-Clause

<!-- ## Citation

If you use ACT in your research, please cite the relevant papers for the verification methods employed. -->

## Attribution and Acknowledgments

**Important**: ACT provides a unified interface for neural network verification whilst integrating established tools. All attribution details are provided below and within the source code comments.

### External Tool Integration

ACT integrates with and provides compatible interfaces for:

- **[αβ-CROWN](https://github.com/Verified-Intelligence/alpha-beta-CROWN)**: State-of-the-art complete neural network verifier
  - Copyright: (C) 2021-2025 The α,β-CROWN Team  
  - Licence: BSD 3-Clause Licence
  - Authors: Huan Zhang (UIUC), Zhouxing Shi (UCLA), Xiangru Zhong (UIUC), and the αβ-CROWN development team

- **[ERAN](https://github.com/eth-sri/eran)**: ETH Robustness Analyser for Neural Networks
  - Copyright: ETH Zurich
  - Licence: Apache 2.0 Licence
  - Authors / Institution: ETH Zurich, SRI Lab

### ACT Native Contributions

- **Hybrid Zonotope Verification**: Novel tensorised implementation with complexity-simplification strategies
- **Specification Refinement**: Prototype BaB-based refinement framework
- **Unified Interface**: Cross-verifier parameter compatibility and workflow integration

### Interface Design Philosophy

ACT adopts compatible parameter structures from established verification tools to:
1. Reduce learning curve for existing tool users
2. Enable seamless integration in verification workflows  
3. Maintain ecosystem consistency
4. Facilitate research reproducibility

All parameter adaptations are documented and attributed to original sources.
