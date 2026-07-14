# VNNLIB Benchmark Data

This directory contains downloaded VNNLIB benchmark instances from VNN-COMP (Verification of Neural Networks Competition).

## Directory Structure

Each category is organized as follows:

```
vnnlib/
├── acasxu_2023/
│   ├── onnx/                        # ONNX model files
│   │   ├── ACASXU_run2a_1_1_batch_2000.onnx
│   │   ├── ACASXU_run2a_1_2_batch_2000.onnx
│   │   └── ...
│   ├── vnnlib/                      # VNNLIB property files
│   │   ├── prop_1.vnnlib
│   │   ├── prop_2.vnnlib
│   │   └── ...
│   └── instances.csv                # Instance mapping (onnx, vnnlib, timeout)
│
├── cifar100_2024/
│   ├── onnx/
│   ├── vnnlib/
│   └── instances.csv
│
└── ... (26 total categories)
```

## Instance File Format

Each category contains an `instances.csv` file mapping ONNX models to VNNLIB properties:

```csv
onnx,vnnlib,timeout
onnx/ACASXU_run2a_1_1_batch_2000.onnx,vnnlib/prop_1.vnnlib,60
onnx/ACASXU_run2a_1_1_batch_2000.onnx,vnnlib/prop_2.vnnlib,60
onnx/ACASXU_run2a_1_2_batch_2000.onnx,vnnlib/prop_1.vnnlib,60
```

- **onnx**: Relative path to ONNX model file
- **vnnlib**: Relative path to VNNLIB property file
- **timeout**: Verification timeout in seconds

## VNNLIB Categories

### Overview Statistics

- **Total Categories**: 26
- **Competition Years**: 2022-2025
- **Total Instances**: Varies by category (tens to thousands)
- **Model Formats**: ONNX (converted to PyTorch by ACT)
- **Property Formats**: VNNLIB (SMT-LIB-based)

### Categories by Domain

#### Image Classification (5 categories)
| Category | Year | Input Size | Classes | Description |
|----------|------|------------|---------|-------------|
| cifar100_2024 | 2024 | 3×32×32 | 100 | CIFAR-100 adversarial robustness |
| tinyimagenet_2024 | 2024 | 3×64×64 | 200 | TinyImageNet L-inf robustness |
| vggnet16_2022 | 2022 | 3×224×224 | 1000 | VGG-16 adversarial examples |
| cersyve | 2024 | Variable | Variable | Certified robustness benchmarks |
| traffic_signs_recognition_2023 | 2023 | 3×32×32 | Variable | Traffic sign safety-critical verification |

#### Safety-Critical Systems (3 categories)
| Category | Year | Domain | Description |
|----------|------|--------|-------------|
| acasxu_2023 | 2023 | Aviation | Aircraft collision avoidance (45 networks) |
| collins_aerospace_benchmark | 2024 | Aerospace | Industrial aerospace control systems |
| collins_rul_cnn_2022 | 2022 | Aerospace | Remaining Useful Life prediction |

#### Object Detection (2 categories)
| Category | Year | Architecture | Description |
|----------|------|--------------|-------------|
| yolo_2023 | 2023 | YOLOv3/v5 | Object detection robustness |
| cctsdb_yolo_2023 | 2023 | YOLO | Chinese traffic sign detection |

#### Control Systems (3 categories)
| Category | Year | Type | Description |
|----------|------|------|-------------|
| lsnc_relu | 2024 | Controller | Learning-enabled state-space control |
| nn4sys | 2024 | System ID | Neural networks for modeling |
| cora_2024 | 2024 | Reachability | CORA reachability analysis |

#### Advanced Architectures (2 categories)
| Category | Year | Architecture | Description |
|----------|------|--------------|-------------|
| vit_2023 | 2023 | Vision Transformer | ViT robustness verification |
| safenlp_2024 | 2024 | Transformer | NLP safety and fairness |

#### Specialized Domains (11 categories)
- **cgan_2023**: Conditional GAN verification
- **ml4acopf_2024**: Power grid optimization
- **malbeware**: Malware detection evasion
- **metaroom_2023**: 3D scene reconstruction
- **dist_shift_2023**: Distribution shift robustness
- **tllverifybench_2023**: Transfer learning verification
- **linearizenn_2024**: Network linearization
- **relusplitter**: ReLUSplitter algorithm benchmarks
- **sat_relu**: SAT-based verification
- **soundnessbench**: Verifier soundness testing
- **test**: Small test cases

## Download Instructions

### Using Unified CLI (Recommended)

```bash
# Auto-detect and download VNNLIB category
python -m act.front_end --download acasxu_2023

# List all categories (TorchVision + VNNLIB)
python -m act.front_end --list

# Show category details
python -m act.front_end --info cifar100_2024

# Search categories
python -m act.front_end --search yolo
```

### Using VNNLIB-Specific CLI

```bash
# Download specific category
python -m act.front_end.vnnlib_loader.cli --download acasxu_2023

# Download with instance limit
python -m act.front_end.vnnlib_loader.cli --download cifar100_2024 --max 10

# List all VNNLIB categories
python -m act.front_end.vnnlib_loader.cli --list

# List downloads
python -m act.front_end.vnnlib_loader.cli --list-downloads
```

### Programmatic Download

```python
from act.front_end.vnnlib_loader.data_model_loader import download_vnnlib_category

# Download category
result = download_vnnlib_category(
    category="acasxu_2023",
    max_instances=None,  # Download all
    force_redownload=False
)

print(f"Downloaded to: {result['category_path']}")
print(f"ONNX models: {result['num_onnx']}")
print(f"VNNLIB files: {result['num_vnnlib']}")
print(f"Instances: {result['num_instances']}")
```

## VNNLIB Format Details

### Input Constraints (BOX)

VNNLIB files define box constraints for input variables:

```lisp
; Declare input variables
(declare-const X_0 Real)
(declare-const X_1 Real)
(declare-const X_2 Real)

; Box constraints: X_i ∈ [lower_i, upper_i]
(assert (>= X_0 0.6))
(assert (<= X_0 0.6798577687))
(assert (>= X_1 -0.5))
(assert (<= X_1 -0.4528301887))
(assert (>= X_2 0.0))
(assert (<= X_2 0.5))
```

### Output Properties (LINEAR_LE)

Output properties are expressed as disjunctions of linear constraints:

```lisp
; Declare output variables
(declare-const Y_0 Real)
(declare-const Y_1 Real)
(declare-const Y_2 Real)

; Safety property: Y_0 should be maximal
(assert (or
    (and (>= Y_0 Y_1))     ; Y_0 >= Y_1
    (and (>= Y_0 Y_2))     ; Y_0 >= Y_2
))
```

ACT converts these to:
- **InputSpec**: BOX kind with lb/ub tensors
- **OutputSpec**: LINEAR_LE kind (A·y ≤ b)

## File Sizes

Approximate sizes for common categories (uncompressed):

| Category | ONNX Models | VNNLIB Files | Total Size | Instances |
|----------|-------------|--------------|------------|-----------|
| acasxu_2023 | 45 | 100s | ~50 MB | ~1,000 |
| cifar100_2024 | 10s | 100s | ~200 MB | ~500 |
| yolo_2023 | 5-10 | 100s | ~500 MB | ~200 |
| vggnet16_2022 | 10s | 100s | ~1 GB | ~1,000 |
| test | 5 | 10 | ~1 MB | 20 |

**Note**: Actual sizes vary. Use `--max` parameter to limit downloads.

## Data Source

All benchmarks are sourced from VNN-COMP:
- **Repository**: https://github.com/VNN-COMP/vnncomp2025_benchmarks
- **License**: Varies by category (typically MIT/Apache 2.0)
- **Citation**: Please cite VNN-COMP papers when using these benchmarks

## Integration with ACT

### Loading Downloaded Data

```python
from act.front_end.vnnlib_loader.data_model_loader import load_vnnlib_pair

# Load a specific instance
result = load_vnnlib_pair(
    category="acasxu_2023",
    instance_id="instance_0"  # From instances.csv
)

pytorch_model = result['model']      # ONNX → PyTorch
input_spec = result['input_spec']    # BOX constraints
output_spec = result['output_spec']  # LINEAR_LE constraints
```

### Creating Specifications

```python
from act.front_end.vnnlib_loader.create_specs import VNNLibSpecCreator

# Create specs from downloaded data
creator = VNNLibSpecCreator()
results = creator.create_specs_for_data_model_pairs(
    categories=["acasxu_2023"],
    max_instances=10
)

for category, instance_id, model, inputs, specs in results:
    print(f"{category}/{instance_id}: {len(specs)} spec pairs")
```

## Comparison with TorchVision Data

| Aspect | TorchVision (`data/torchvision/`) | VNNLIB (`data/vnnlib/`) |
|--------|-----------------------------------|-------------------------|
| **Source** | PyTorch official datasets | VNN-COMP benchmarks |
| **Models** | PyTorch native | ONNX (converted) |
| **Properties** | Generated (ε-balls, polytopes) | Parsed from VNNLIB files |
| **Structure** | `DATASET/MODEL/{train,test}/` | `CATEGORY/{onnx,vnnlib}/` |
| **Instances** | Dataset samples | (ONNX, VNNLIB) pairs |
| **Specs** | Created at runtime | Pre-defined in VNNLIB |
| **Size** | Dataset-dependent (GBs) | Category-dependent (MBs-GBs) |
| **Use Case** | Custom verification | Standard benchmarks |

## Maintenance

### Cleaning Up

```bash
# Remove specific category
rm -rf data/vnnlib/acasxu_2023

# Remove all VNNLIB data
rm -rf data/vnnlib/*
```

### Updating Categories

VNN-COMP releases new benchmarks yearly. To update:

1. Check VNN-COMP repository for new categories
2. Update `category_mapping.py` with new metadata
3. Download new categories using CLI or programmatic API

## See Also

- **VNNLIB Creator**: `act/front_end/vnnlib/README.md`
- **TorchVision Data**: `data/torchvision/README.md`
- **Unified CLI**: `act/front_end/README.md`
- **VNN-COMP Website**: https://www.vnncomp.com/
