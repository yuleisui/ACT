# Setup Directory

This directory contains all environment setup scripts and dependency requirements for the Abstract Constraint Transformer (ACT) framework.

## Files Overview

### Main Setup Script
- **`setup.sh`**: Automated setup script that creates all required conda environments
  - Creates `act-main`, `act-abcrown`, and `act-eran` environments
  - Installs all Python dependencies from requirement files
  - Invokes ERAN-specific setup (GMP, MPFR, ELINA, DeepG)
  - Configures Gurobi optimizer
  - Patches αβ-CROWN imports to prevent conflicts

### ERAN-Specific Setup
- **`eran_env_setup.sh`**: Specialised setup script for ERAN environment
  - Handles ERAN's linux system dependencies
  - Compiles mathematical libraries (GMP, MPFR)
  - Builds ELINA abstract interpretation library
  - Configures DeepG optimization components

### Python Requirements
- **`main_requirements.txt`**: Dependencies for ACT main environment (`act-main`)
  - PyTorch 2.x with CUDA support
  - ONNX tools and runtime
  - Gurobi Python interface
  - NumPy, TensorFlow, and other scientific computing libraries

- **`abcrown_requirements.txt`**: Dependencies for αβ-CROWN environment (`act-abcrown`)
  - PyTorch 2.x compatible with αβ-CROWN
  - ONNX simplification tools
  - Gurobi optimizer
  - Sorted containers for efficient data structures

- **`eran_requirements.txt`**: Dependencies for ERAN environment (`act-eran`)
  - Python 3.8 compatible packages
  - TensorFlow 2.9.3 for ERAN compatibility
  - ONNX 1.8.0 (specific version required by ERAN)
  - Mathematical libraries (pycddlib for convex hull operations)

## Usage

### Quick Setup
```bash
cd setup/
source setup.sh
```

### Manual ERAN Setup (if needed)
```bash
cd setup/
source eran_env_setup.sh
```

## Troubleshooting

### Environment Creation Failures
If conda environments fail to create:
```bash
conda clean --all
conda update conda
```

### Linux System Dependencies for ERAN (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install build-essential cmake m4 autoconf libtool
sudo apt-get install libgmp-dev libmpfr-dev
```

### Gurobi License Issues
- Academic users: https://www.gurobi.com/academia/
- Ensure license is activated in each conda environment
- Check license with: `python -c "import gurobipy; print('Gurobi OK')"`

### ERAN Compilation Issues
Common solutions:
1. Ensure all system dependencies are installed
2. Check GCC version compatibility (GCC 7-9 recommended)
3. Verify ELINA library compilation logs
4. Clear any cached builds: `rm -rf modules/eran/ELINA/build/`

## Environment Specifications

### act-main (Python 3.9)
Primary ACT framework with:
- Hybrid Zonotope verification
- Specification refinement BaB
- Full ONNX/PyTorch/TensorFlow model support

### act-abcrown (Python 3.9)
αβ-CROWN integration with:
- Complete neural network verification
- Advanced branch-and-bound algorithms
- Linear relaxation techniques

### act-eran (Python 3.8)
ERAN integration with:
- Abstract interpretation methods
- DeepPoly and DeepZono domains
