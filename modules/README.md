# Modules Directory

This directory contains external verifier submodules integrated into the Abstract Constraint Transformer (ACT) framework. These are established neural network verification tools that ACT provides unified interfaces for.

## Submodules Overview

### αβ-CROWN (`abcrown/`)
**Complete Neural Network Verifier with Branch-and-Bound**

- **Repository**: https://github.com/Verified-Intelligence/alpha-beta-CROWN
- **License**: BSD 3-Clause License
- **Integration**: Git submodule
- **Purpose**: State-of-the-art complete verification with advanced branch-and-bound algorithms

#### ACT Integration Notes:
- Parameters mapped to ACT unified interface
- Configuration managed through ACT's parameter system
- Compatible with ACT's specification refinement framework

### ERAN (`eran/`)
**ETH Robustness Analyzer for Neural Networks**

- **Repository**: https://github.com/eth-sri/eran
- **License**: Apache 2.0 License
- **Integration**: Git submodule
- **Purpose**: Abstract interpretation-based verification with multiple domains


#### Abstract Domains Supported:
- **DeepPoly**: Polyhedra-based abstract interpretation
- **DeepZono**: Zonotope-based abstract interpretation
- **RefinePoly**: Refinement-based DeepPoly with MILP
- **RefineZono**: Refinement-based DeepZono with optimization

#### ACT Integration Notes:
- Environment isolation due to Python 3.8/TensorFlow 2.9.3 requirements
- Parameter translation for ERAN's command-line interface

## Submodule Management

### Initialisation
Submodules are automatically initialised during setup:
```bash
git clone --recursive https://github.com/doctormeeee/Abstract-Constraint-Transformer.git
```

### Manual Submodule Updates
```bash
git submodule update --init --recursive
git submodule update --remote
```

### Submodule Status
Check submodule status:
```bash
git submodule status
```

## Integration Architecture

### Parameter Mapping
ACT translates its unified parameters to each tool's native format:
- Common parameters (model, dataset, epsilon) mapped directly
- Tool-specific parameters preserved when using respective backends
- Default values provided for missing parameters

### Environment Isolation
- **ERAN**: Isolated Python 3.8 environment (`act-eran`)
- **αβ-CROWN**: Shared Python 3.9 environment (`act-abcrown`)
- **ACT**: Main Python 3.9 environment (`act-main`)


## Compatibility Matrix

| Feature | ERAN | αβ-CROWN | ACT Native |
|---------|------|----------|------------|
| MNIST | ✓ | ✓ | ✓ |
| CIFAR-10 | ✓ | ✓ | ✓ |
| VNNLIB | x | ✓ | ✓ |
| BaB Refinement | x | ✓ | ✓ |

## Troubleshooting

### Submodule Issues
```bash
# Reset submodules to clean state
git submodule deinit --all -f
git submodule update --init --recursive
```

### ERAN Environment Issues
```bash
# Rebuild ERAN environment
conda env remove -n act-eran
cd setup/
source eran_env_setup.sh
```

### αβ-CROWN Import Conflicts
The setup script automatically patches αβ-CROWN imports to prevent conflicts. If issues persist:
```bash
cd modules/abcrown/
git checkout .  # Reset any local changes
cd ../../setup/
source setup.sh   # Re-run setup to reapply patches
```

## Contributing to Submodules

### ERAN Issues
Report issues directly to: https://github.com/eth-sri/eran

### αβ-CROWN Issues  
Report issues directly to: https://github.com/Verified-Intelligence/alpha-beta-CROWN

### Integration Issues
Report ACT-specific integration issues to the main ACT repository.
