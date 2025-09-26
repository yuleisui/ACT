# Data Directory

This directory contains sample datasets and verification specifications used for neural network verification testing and benchmarking. Data is organized by format and purpose to support various verification scenarios.

## Directory Structure

### CSV Datasets (`*_csv/`)
**Preprocessed datasets in CSV format for batch verification**

#### MNIST CSV (`MNIST_csv/`)
- **`mnist_first_100_samples.csv`**: First 100 MNIST test samples
  - Format: `label,pixel_0,pixel_1,...,pixel_783`
  - Dimensions: 100 rows × 785 columns (1 label + 784 pixels)
  - Pixel values: Normalised to [0,1] range
  - Use case: Batch verification testing and benchmarking
  - Labels: Ground truth classifications (0-9)

#### CIFAR-10 CSV (`CIFAR10_csv/`)
- **`cifar10_first_100_samples.csv`**: First 100 CIFAR-10 test samples
  - Format: `label,r_0,g_0,b_0,r_1,g_1,b_1,...,r_1023,g_1023,b_1023`
  - Dimensions: 100 rows × 3073 columns (1 label + 3072 RGB values)
  - Color values: Normalized to [0,1] range
  - Use case: Color image verification testing
  - Labels: Ground truth classifications (0-9)

### Anchor Datasets (`anchor/`)
**Reference datasets for VNNLIB specification anchoring**

- **`mnist_csv.csv`**: MNIST anchor points for specification anchoring
  - Format: Same as MNIST CSV but serves as anchor reference
  - Purpose: Provides specific data points for local robustness specifications
  - Use case: `--spec_type local_vnnlib --anchor anchor/mnist_csv.csv`

### VNNLIB Specifications (`vnnlib/`)
**Standard verification specification files in VNNLIB format**

- **`set_vnnlib_example.vnnlib`**: Set-based property specification example
  - Property type: Set-based reachability or safety property (e.g., ACAS Xu style)
  - Input constraints: Set-based input bounds
  - Output constraints: Set-based property requirements
  - Use case: `--spec_type set_vnnlib --vnnlib_path vnnlib/set_vnnlib_example.vnnlib`
  - Example content: Aircraft collision avoidance properties, competition-style verification

- **`local_vnnlib_example.vnnlib`**: Local robustness property specification
  - Property type: Local robustness around specific points
  - Input constraints: Lp-norm balls around anchor points
  - Output constraints: Label preservation requirements
  - Use case: `--spec_type local_vnnlib --vnnlib_path vnnlib/local_vnnlib_example.vnnlib`
  - Example content: ε-robustness verification

### JSON Specifications (`json/`)
**Alternative specification format for simple constraints**

- **`test_global_box.json`**: Box constraint specification in JSON format
  - Format: Simple JSON with input bounds
  - Structure: `{"input_lb": [...], "input_ub": [...], "output_constraints": [...]}`
  - Use case: Simple box constraint verification
  - Advantage: Human-readable, easy to modify

## Data Formats

### CSV Format Specification
```
label,feature_0,feature_1,...,feature_n
0,0.123,0.456,...,0.789
1,0.234,0.567,...,0.890
...
```
- **Header**: Column names (label + feature indices)
- **Values**: Comma-separated, normalised to [0,1] unless specified
- **Encoding**: UTF-8 text format

### VNNLIB Format Specification
VNNLIB follows the VNN-COMP standard:
```lisp
; Input variable declarations
(declare-fun X_0 () Real)
(declare-fun X_1 () Real)
...

; Input constraints
(assert (>= X_0 0.0))
(assert (<= X_0 1.0))
...

; Output variable declarations  
(declare-fun Y_0 () Real)
(declare-fun Y_1 () Real)
...

; Output constraints (property to verify)
(assert (or
    (>= Y_0 Y_1)
    (>= Y_0 Y_2)
    ...
))
```

### JSON Format Specification
```json
{
  "lb": [0.0, 0.1, 0.2],
  "ub": [1.0, 0.9, 0.8],
  "output_constraints": {
    "A": [
      [1.0, -1.0, 0.0],
      [0.0, 1.0, -1.0]
    ],
    "b": [0.5, -0.2]
  }
}
```

## Usage Examples

All comprehensive usage examples have been moved to the main README.md file for better organization and centralized documentation. Please refer to the main README for:

- CSV-based batch verification examples
- VNNLIB specification verification examples  
- Torchvision dataset examples
- HybridZ method variations
- All other verification patterns

### Quick Reference

For data format specifications and structure, see the sections above. For actual usage commands and examples, please refer to the **Example Usage** section in the main `README.md` file.

## Data Preprocessing

### Normalisation
All datasets use standard normalisation:
- **MNIST**: mean=[0.1307], std=[0.3081]
- **CIFAR-10**: mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
- **Custom datasets**: normalised to [0,1] range unless specified

