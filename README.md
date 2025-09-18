# Abstract Constraint Transformer (ACT)

A unified neural network verification framework that integrates multiple state-of-the-art verifiers including ERAN, αβ-CROWN, and the novel Hybrid Zonotope methods. ACT provides a tensorised implementation for efficient verification of deep neural networks with support for various specification formats and robustness properties.

## Overview

ACT combines three powerful verification approaches:
- **ERAN**: ETH Robustness Analyser with abstract interpretation methods (DeepPoly, DeepZono, etc) https://github.com/eth-sri/eran
- **αβ-CROWN**: Complete neural network verifier with Branch-and-Bound (BaB) https://github.com/Verified-Intelligence/alpha-beta-CROWN
- **Hybrid Zonotope**: Novel tensorised hybrid verification method with MILP/LP relaxation strategies， incorporating three core configurations (Full-precision MILP, Fully-relaxed LP, Partially-relaxed MILP+LP)

The framework supports multiple input formats (ONNX, TF, Pytorch models, VNNLIB specifications) and provides comprehensive verification capabilities for classification tasks.

## Project Structure

```
Abstract-Constraint-Transformer/
├── README.md                 # This file
├── setup/                    # Environment setup scripts and requirements
├── verifier/                 # Main verification framework
├── modules/                  # Submodules (ERAN, αβ-CROWN)
├── models/                   # Pre-trained models
└── data/                     # Sample datasets and specifications
```

## Installation

### Prerequisites

Before installation, ensure you have the following:
- **Miniconda or Anaconda**: Required for environment management
  - Download from: https://www.anaconda.com/docs/getting-started/miniconda/install#quickstart-install-instructions
- **Git**: For cloning the repository with submodules
- **Linux system**: Recommended for ERAN dependencies (Ubuntu/Debian preferred)

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

#### Core Arguments

```bash
--verifier {interval,eran,abcrown,hybridz}                  # Verification backend
--method {interval,                                         # Verification method for interval and Hybrid Zonotope
          deepzono,refinezono,deeppoly,refinepoly,          # Verification method for ERAN
          alpha,beta,alpha_beta,                            # Verification method for αβ-CROWN
          hybridz,hybridz_relaxed,hybridz_relaxed_with_bab  # Verification method for Hybrid Zonotope
         }
--device {cpu,cuda}                    # Computation device
```

#### Model and Data Arguments

```bash
--model_path PATH                      # Path to .onnx/.tf/.pt model file
--dataset PATH                         # Dataset name or CSV file path
--csv_name PATH                        # CSV file with verification instances
--spec_type {local_lp,local_vnnlib,    # Specification format
            set_vnnlib,set_box}
```

#### Robustness Verification Arguments

```bash
--norm {1,2,inf}                       # Lp-norm for perturbation
--epsilon FLOAT                        # Perturbation bound
--robustness_type {verified-acc,       # Verification target
                   runnerup,clean-acc,
                   specify-target,all-positive}
```

#### VNNLIB Specification Arguments

```bash
--vnnlib_path PATH                     # Path to VNNLIB specification file
--input_lb FLOAT [FLOAT ...]           # Lower bounds for input
--input_ub FLOAT [FLOAT ...]           # Upper bounds for input
```

#### Hybrid Zonotope Arguments

```bash
--relaxation_ratio FLOAT               # Relaxation ratio (0.0=MILP, 1.0=LP)
--enable_generator_merging             # Enable parallel generator optimisation
--cosine_threshold FLOAT               # Cosine similarity threshold (0.0-1.0)
```

#### Branch-and-Bound Arguments

```bash
--enable_spec_refinement               # Enable BaB refinement verification
--bab_max_depth INT                    # Maximum BaB search depth
--bab_max_subproblems INT              # Maximum number of subproblems
--bab_time_limit FLOAT                 # Time limit in seconds
--bab_split_tolerance FLOAT            # Split tolerance
--bab_verbose                          # Verbose BaB output
```

#### Data Processing Arguments

```bash
--mean FLOAT [FLOAT ...]               # Mean values for normalisation
--std FLOAT [FLOAT ...]                # Standard deviation for normalisation
--start INT                            # Start index in dataset
--end INT                              # End index in dataset
--num_outputs INT                      # Number of output classes
```

### Example Usage

#### MNIST Verification with ERAN DeepPoly

```bash
python verifier_tensorised.py \
    --verifier eran \
    --method deeppoly \
    --model_path ../models/ERAN\ Models/MNIST/ffnnRELU__Point_6_500.onnx \
    --dataset mnist \
    --spec_type local_lp \
    --norm inf \
    --epsilon 0.1 \
    --start 0 \
    --end 100
```

#### CIFAR-10 Verification with αβ-CROWN

```bash
python verifier_tensorised.py \
    --verifier abcrown \
    --method alpha_beta \
    --model_path ../models/ERAN\ Models/CIFAR10/convSmallRELU__Point.onnx \
    --dataset cifar10 \
    --spec_type local_lp \
    --norm inf \
    --epsilon 0.01 \
    --device cuda \
    --start 0 \
    --end 50
```

#### Hybrid Zonotope with BaB Refinement

```bash
python verifier_tensorised.py \
    --verifier eran \
    --method hybridz_relaxed_with_bab \
    --model_path ../models/test/mnist_all_layers.onnx \
    --dataset mnist \
    --spec_type local_lp \
    --norm inf \
    --epsilon 0.1 \
    --enable_spec_refinement \
    --bab_max_depth 10 \
    --bab_time_limit 600 \
    --enable_generator_merging \
    --device cpu
```

#### VNNLIB Specification Verification

```bash
python verifier_tensorised.py \
    --verifier abcrown \
    --method alpha_beta \
    --model_path ../models/test/mnist_all_layers.onnx \
    --vnnlib_path ../data/vnnlib/global_vnnlib_example.vnnlib \
    --spec_type set_vnnlib \
    --device cuda
```

#### CSV-based Batch Verification

```bash
python verifier_tensorised.py \
    --verifier eran \
    --method deeppoly \
    --csv_name ../data/MNIST_csv/mnist_first_100_samples.csv \
    --spec_type local_lp \
    --norm inf \
    --results_file mnist_results.txt
```

## Verification Methods

### ERAN Methods
- **DeepPoly**: Precise abstract interpretation with polyhedra
- **DeepZono**: Zonotope-based abstract interpretation
- **RefineZono/RefinePoly**: Refinement-based methods

### αβ-CROWN Methods
- **Alpha**: Fast incomplete verification with linear bounds
- **Beta**: Complete verification with branch-and-bound
- **Alpha-Beta**: Combined approach for efficiency

### Hybrid Methods
- **hybridz_relaxed_with_bab**: Novel tensorised hybrid approach with automatic BaB refinement

## Specification Formats

### Local Robustness (`local_lp`)
Lp-norm ball around input data points:
```bash
--spec_type local_lp --norm inf --epsilon 0.1
```

### VNNLIB Specifications
Standard verification competition format:
```bash
--spec_type local_vnnlib --vnnlib_path spec.vnnlib    # With anchor points
--spec_type set_vnnlib --vnnlib_path spec.vnnlib      # Set-based properties
```

### Box Constraints
Direct input bound specification:
```bash
--spec_type set_box --input_lb 0.0 0.0 --input_ub 1.0 1.0
```

## Performance Optimisation

### GPU Acceleration
Enable CUDA for supported methods:
```bash
--device cuda
```

### Parallel Generator Merging
Optimise final layer computation:
```bash
--enable_generator_merging --cosine_threshold 0.95
```

### Relaxation Strategy
Control MILP/LP trade-off for hybrid methods:
```bash
--relaxation_ratio 0.5  # 50% relaxation
```

## Output and Results

Verification results are saved to specified files:
- `--results_file`: Text summary of verification results
- `--output_file`: Detailed results in pickle format

Example output includes:
- Verification status (SAT/UNSAT/UNKNOWN/TIMEOUT)
- Verification time
- Number of neurons/constraints
- BaB statistics (if applicable)

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

### Gurobi Licence
Academic users can obtain free licences at: https://www.gurobi.com/academia/

### Memory Issues
For large models, consider:
- Using CPU instead of GPU
- Reducing batch size
- Limiting BaB depth and subproblems

## Contributing

This framework integrates multiple verification tools. For contributions:
- Framework improvements: Submit PRs to this repository
- ERAN issues: Report to ETH-SRI ERAN repository
- αβ-CROWN issues: Report to Verified-Intelligence αβ-CROWN repository

## Licence

This project integrates multiple tools with different licences:
- ACT Framework: [Check repository licence]
- ERAN: Apache 2.0
- αβ-CROWN: BSD 3-Clause

## Citation

If you use ACT in your research, please cite the relevant papers for the verification methods employed.

## Attribution and Acknowledgments

**Important**: ACT provides a unified interface for neural network verification whilst integrating established tools. All attribution details are provided below and within the source code comments.

### External Tool Integration

ACT integrates with and provides compatible interfaces for:

- **[α,β-CROWN](https://github.com/Verified-Intelligence/alpha-beta-CROWN)**: State-of-the-art complete neural network verifier
  - Copyright: (C) 2021-2025 The α,β-CROWN Team  
  - License: BSD 3-Clause License
  - Authors: Huan Zhang (UIUC), Zhouxing Shi (UCLA), Xiangru Zhong (UIUC)

- **[ERAN](https://github.com/eth-sri/eran)**: ETH Robustness Analyzer for Neural Networks
  - Copyright: ETH Zurich
  - License: Apache 2.0 License
  - ETH SRI Lab

### ACT Native Contributions

- **Hybrid Zonotope Verification**: Novel tensorised implementation with parallel optimisations
- **Specification Refinement**: Branch-and-bound refinement framework
- **Unified Interface**: Cross-tool parameter compatibility and workflow integration

### Interface Design Philosophy

ACT adopts compatible parameter structures from established verification tools to:
1. Reduce learning curve for existing tool users
2. Enable seamless integration in verification workflows  
3. Maintain ecosystem consistency
4. Facilitate research reproducibility

All parameter adaptations are documented and attributed to original sources.
