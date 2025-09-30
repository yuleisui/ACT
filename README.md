# Abstract Constraint Transformer (ACT)

[![Continuous Integration (CI) Status](https://github.com/doctormeeee/Abstract-Constraint-Transformer/actions/workflows/ci.yml/badge.svg)](https://github.com/doctormeeee/Abstract-Constraint-Transformer/actions/workflows/ci.yml)

A unified neural network verification framework that integrates multiple state-of-the-art verifiers including ERAN, αβ-CROWN, and the novel Hybrid Zonotope methods.

## Documentation

| 📝 About | 📁 Project Structure | ⚙️ Setup | 🚀 Usage | 🏗️ Design |
|----------|----------------------|----------|----------|-----------|
| [Project overview and core features](https://github.com/doctormeeee/Abstract-Constraint-Transformer/wiki/About-ACT) | [Complete project directory guide](https://github.com/doctormeeee/Abstract-Constraint-Transformer/wiki/Project-Structure-of-ACT) | [Installation and configuration guide](https://github.com/doctormeeee/Abstract-Constraint-Transformer/wiki/Setup-Guide) | [User guide and examples](https://github.com/doctormeeee/Abstract-Constraint-Transformer/wiki/Usage-Guide) | [Architecture and implementation](https://github.com/doctormeeee/Abstract-Constraint-Transformer/wiki/Design-of-ACT) |

## Quick Start

For the full detailed setup and configuration guide, please refer to the [Setup wiki page](https://github.com/doctormeeee/Abstract-Constraint-Transformer/wiki/Setup-Guide) and [Usage wiki page](https://github.com/doctormeeee/Abstract-Constraint-Transformer/wiki/Usage-Guide).
The following provides a minimal Quick Start with step-by-step commands to get the project running.

```bash
# 1. Clone repository
git clone --recursive https://github.com/doctormeeee/Abstract-Constraint-Transformer.git
cd Abstract-Constraint-Transformer

# 2. Place Gurobi license (Required for MILP optimization)
# Place your gurobi.lic file in ./gurobi/ directory
cp /path/to/your/gurobi.lic ./gurobi/gurobi.lic

# 3. Set up environments (inside setup/)
cd setup
source setup.sh

# 4. Activate environment
conda activate act-main

# 5. Run verification (inside verifier/)
cd ../verifier
python main.py \
  --verifier hybridz --method hybridz \
  --model_path ../models/Sample_models/MNIST/small_relu_mnist_cnn_model_1.onnx \
  --dataset mnist --spec_type local_lp \
  --start 0 --end 1 --epsilon 0.03 --norm inf \
  --mean 0.1307 --std 0.3081

# 6. Example output
The verifier will run with the full-precision MILP Hybrid Zonotope and report SAT/UNSAT or timeout results.
```

## License

This project integrates multiple tools with different licenses:
- ACT Framework: BSD 3-Clause
- ERAN: Apache 2.0
- αβ-CROWN: BSD 3-Clause