# 🔗 Front-End to Abstraction Integration Plan
========================================================

This document outlines the complete integration plan for connecting the ACT front-end preprocessing module with the abstraction verification framework.

## 📋 Current Status Assessment

### ✅ Front-End Module (Self-Contained & Ready)
- **Specification System**: Unified `InputSpec`/`OutputSpec` classes with `InKind`/`OutKind` enums
- **Preprocessors**: `ImgPre`, `TextPre` with device-aware processing
- **Batch Processing**: `run_batch()` pipeline with verification interface
- **Data Handling**: Raw → Model → Flat → Verification format conversion
- **Mock System**: Complete testing infrastructure
- **Device Management**: CPU/CUDA support with proper tensor handling

### ✅ Abstraction Framework (Verified Working)
- **Verification Engine**: `verify_once()` with constraint generation
- **Specification Support**: Compatible with front-end specs (unified classes)
- **Solver Integration**: Gurobi MILP + PyTorch LP solvers
- **Model Support**: PyTorch models, ACT `Net` representations
- **Branch-and-Bound**: Advanced verification with refinement

### 🔄 Integration Gaps Identified
1. **Model Loading Bridge**: ONNX models → ACT framework
2. **Data Loading Pipeline**: CSV/VNNLIB → Front-end samples
3. **Batch Verification**: Front-end batches → Abstraction verification
4. **Result Mapping**: Counterexamples back to raw space
5. **Configuration Integration**: Unified command-line interface

## 🛠️ Integration Steps

### Step 1: Model Loading Integration
**Objective**: Bridge ONNX models with abstraction framework

**Components to Create:**
```python
# act/front_end/models.py
class ModelLoader:
    \"\"\"Self-contained ONNX model loading for front-end preprocessing\"\"\"
    
    def load_onnx_model(self, onnx_path: str) -> torch.nn.Module:
        \"\"\"Load ONNX as PyTorch model for front-end use\"\"\"
        
    def extract_model_signature(self, onnx_path: str) -> ModelSignature:
        \"\"\"Extract input/output shapes and metadata from ONNX\"\"\"
        
    def create_preprocessor_from_model(self, model_path: str) -> Preprocessor:
        \"\"\"Auto-create appropriate preprocessor based on model signature\"\"\"
        
    def prepare_for_abstraction(self, pytorch_model: torch.nn.Module) -> Any:
        \"\"\"Convert PyTorch model to abstraction framework format\"\"\"
```

**Files to Create:**
- Create `act/front_end/models.py` (new self-contained model handling)
- Update front-end `__init__.py` to export model utilities
- Keep existing `act/input_parser/model.py` unchanged

### Step 2: Data Loading Pipeline  
**Objective**: Separate data loading from specification generation

**Components to Create:**
```python
# act/front_end/data_loader.py
class DatasetLoader:
    \"\"\"Pure data loading without specifications\"\"\"
    
    def load_csv_data(self, csv_path: str) -> List[Tuple[Any, int]]:
        \"\"\"Load CSV data as (sample, label) pairs\"\"\"
        
    def load_image_directory(self, image_dir: str) -> List[Tuple[Any, int]]:
        \"\"\"Load images from directory as (image, label) pairs\"\"\"
        
    def load_vnnlib_anchor_data(self, anchor_path: str) -> List[Tuple[Any, int]]:
        \"\"\"Load anchor points for VNNLIB specifications\"\"\"

# act/front_end/spec_loader.py  
class SpecLoader:
    \"\"\"Specification generation and loading\"\"\"
    
    def create_input_specs(self, samples: List[Any], spec_config: dict) -> List[InputSpec]:
        \"\"\"Generate input specifications for samples\"\"\"
        
    def create_output_specs(self, labels: List[int], spec_config: dict) -> List[OutputSpec]:
        \"\"\"Generate output specifications for labels\"\"\"
        
    def load_vnnlib_specs(self, vnnlib_path: str) -> List[Tuple[InputSpec, OutputSpec]]:
        \"\"\"Load specifications from VNNLIB file\"\"\"
        
    def combine_data_and_specs(self, data_pairs: List[Tuple], 
                              input_specs: List[InputSpec], 
                              output_specs: List[OutputSpec]) -> List[SampleRecord]:
        \"\"\"Combine data with specifications into SampleRecord objects\"\"\"
```

**Files to Create:**
- Create `act/front_end/data_loader.py` (pure data loading)
- Create `act/front_end/spec_loader.py` (specification handling)
- Integrate with existing `data/` analysis tools
- Add VNNLIB parsing support

### Step 3: Batch Verification Bridge
**Objective**: Connect front-end batches with abstraction verification

**Components to Create:**
```python
# act/front_end/verification_bridge.py
class VerificationBridge:
    \"\"\"Bridge front-end batch processing with abstraction verification\"\"\"
    
    def create_abstraction_verifier(self, net, solver_config: dict):
        \"\"\"Create abstraction-compatible verifier function\"\"\"
        
    def run_abstraction_batch(self, items: List[SampleRecord], 
                            model_path: str, config: dict) -> List[ItemResult]:
        \"\"\"Complete pipeline: preprocessing → verification → results\"\"\"
        
    def map_counterexample_to_raw(self, ce_x: np.ndarray, 
                                preprocessor: Preprocessor) -> Any:
        \"\"\"Map counterexample back to raw input space\"\"\"
```

**Files to Modify:**
- Create `act/front_end/verification_bridge.py`
- Extend `batch.py` with abstraction support
- Update `ItemResult` with raw-space counterexamples

### Step 4: Configuration Integration
**Objective**: Unified command-line interface and configuration

**Components to Create:**
```python
# act/front_end/config.py
class FrontEndConfig:
    \"\"\"Configuration for integrated front-end + abstraction verification\"\"\"
    
    @classmethod
    def from_args(cls, args) -> 'FrontEndConfig':
        \"\"\"Create config from command-line arguments\"\"\"
        
    def create_preprocessor(self) -> Preprocessor:
        \"\"\"Create appropriate preprocessor from config\"\"\"
        
    def create_model_loader(self) -> ModelLoader:
        \"\"\"Create configured model loader\"\"\"
        
    def create_data_loader(self) -> DatasetLoader:
        \"\"\"Create configured data loader\"\"\"
        
    def create_spec_loader(self) -> SpecLoader:
        \"\"\"Create configured specification loader\"\"\"
        
    def create_verification_bridge(self, net) -> VerificationBridge:
        \"\"\"Create configured verification bridge\"\"\"
```

**Files to Modify:**
- Create `act/front_end/config.py`
- Extend `act/util/options.py` with front-end arguments
- Update `act/main.py` to support front-end mode

### Step 5: Comprehensive Integration Driver
**Objective**: Complete end-to-end verification pipeline

**Components to Create:**
```python
# act/front_end/integrated_driver.py
def main_integrated_verification(args):
    \"\"\"Main entry point for integrated front-end + abstraction verification\"\"\"
    
    # 1. Load and configure
    config = FrontEndConfig.from_args(args)
    model_loader = ModelLoader()
    data_loader = DatasetLoader()
    spec_loader = SpecLoader()
    verif_bridge = VerificationBridge()
    
    # 2. Setup model and preprocessor
    pytorch_model = model_loader.load_onnx_model(args.model_path)
    abstraction_net = model_loader.prepare_for_abstraction(pytorch_model)
    preprocessor = config.create_preprocessor()
    
    # 3. Load dataset and specifications separately
    if args.dataset_csv:
        data_pairs = data_loader.load_csv_data(args.dataset_csv)
        input_specs = spec_loader.create_input_specs([pair[0] for pair in data_pairs], config.spec_config)
        output_specs = spec_loader.create_output_specs([pair[1] for pair in data_pairs], config.spec_config)
        items = spec_loader.combine_data_and_specs(data_pairs, input_specs, output_specs)
    elif args.vnnlib_path:
        if args.anchor_path:
            data_pairs = data_loader.load_vnnlib_anchor_data(args.anchor_path)
        vnnlib_specs = spec_loader.load_vnnlib_specs(args.vnnlib_path)
        items = spec_loader.combine_data_and_specs(data_pairs, *zip(*vnnlib_specs))
    else:
        data_pairs = data_loader.load_image_directory(args.image_dir)
        input_specs = spec_loader.create_input_specs([pair[0] for pair in data_pairs], config.spec_config)
        output_specs = spec_loader.create_output_specs([pair[1] for pair in data_pairs], config.spec_config)
        items = spec_loader.combine_data_and_specs(data_pairs, input_specs, output_specs)
    
    # 4. Run verification
    results = verif_bridge.run_abstraction_batch(items, abstraction_net, config.dict())
    
    # 5. Report results
    report_verification_results(results, config)
```

**Files to Modify:**
- Create `act/front_end/integrated_driver.py`
- Add to `act/main.py` as verification mode
- Create result reporting utilities

## 📊 Integration Testing Plan

### Test 1: Model Loading Integration
```bash
# Test ONNX model loading with front-end
python -c "
from act.front_end.models import ModelLoader
loader = ModelLoader()
model = loader.load_onnx_model('models/Sample_models/MNIST/small_relu_mnist_cnn_model_1.onnx')
signature = loader.extract_model_signature('models/Sample_models/MNIST/small_relu_mnist_cnn_model_1.onnx')
print(f'Model loaded: {type(model)}, Input shape: {signature.input_shape}')
"
```

### Test 2: Data and Specification Loading  
```bash
# Test separated data and spec loading
python -c "
from act.front_end.data_loader import DatasetLoader
from act.front_end.spec_loader import SpecLoader
from act.front_end.preprocessor_image import ImgPre

# Load data separately
data_loader = DatasetLoader()
data_pairs = data_loader.load_csv_data('data/MNIST_csv/mnist_first_100_samples.csv')
print(f'Loaded {len(data_pairs)} data samples')

# Generate specifications separately
spec_loader = SpecLoader()
samples = [pair[0] for pair in data_pairs]
labels = [pair[1] for pair in data_pairs]
input_specs = spec_loader.create_input_specs(samples, {'type': 'linf_ball', 'epsilon': 0.03})
output_specs = spec_loader.create_output_specs(labels, {'type': 'margin_robust'})
print(f'Generated {len(input_specs)} input specs and {len(output_specs)} output specs')

# Combine into SampleRecord objects
items = spec_loader.combine_data_and_specs(data_pairs, input_specs, output_specs)
print(f'Created {len(items)} SampleRecord objects')
"
```

### Test 3: Integrated Verification
```bash
# Test complete pipeline
python -m act.front_end.integrated_driver \
    --model models/Sample_models/MNIST/small_relu_mnist_cnn_model_1.onnx \
    --dataset_csv data/MNIST_csv/mnist_first_100_samples.csv \
    --spec_type linf_ball --epsilon 0.03 \
    --solver auto --start 0 --end 5
```

### Test 4: VNNLIB Integration
```bash
# Test VNNLIB specification handling
python -m act.front_end.integrated_driver \
    --model models/Sample_models/MNIST/small_relu_mnist_cnn_model_1.onnx \
    --vnnlib_path data/vnnlib/local_vnnlib_example.vnnlib \
    --anchor_path data/anchor/mnist_csv.csv \
    --solver gurobi --time_limit 60
```

## 🎯 Expected Integration Benefits

### For Users:
- **Unified Interface**: Single command for complete verification workflows
- **Real Data Support**: Easy processing of CSV, images, VNNLIB specifications
- **Flexible Preprocessing**: Automatic handling of normalization, resizing, tokenization
- **Rich Results**: Counterexamples mapped back to interpretable raw space

### For Developers:
- **Modular Design**: Clean separation between preprocessing and verification
- **Extensible**: Easy to add new preprocessors, data formats, specifications
- **Testing**: Comprehensive mock system for development and debugging
- **Performance**: Device-aware processing with GPU acceleration

### For Research:
- **Dataset Compatibility**: Support for standard verification benchmarks
- **Specification Flexibility**: Complex constraints via VNNLIB + simple epsilon-balls
- **Batch Processing**: Efficient large-scale verification experiments
- **Result Analysis**: Detailed statistics and counterexample analysis

## 🔄 Implementation Priority

### Phase 1 (Essential): Core Integration
1. ✅ **Model Bridge**: ONNX loading for abstraction framework
2. ✅ **Basic Data Loader**: CSV dataset support
3. ✅ **Verification Bridge**: Front-end → abstraction pipeline  
4. ✅ **Simple Driver**: Command-line interface

### Phase 2 (Enhanced): Advanced Features
5. **VNNLIB Integration**: Complex specification support
6. **Image Directory**: Direct image file processing
7. **Result Enhancement**: Rich counterexample mapping
8. **Performance Optimization**: Memory and speed improvements

### Phase 3 (Production): Polish & Testing
9. **Comprehensive Tests**: Full test suite
10. **Documentation**: User guides and examples
11. **Error Handling**: Robust error recovery
12. **Configuration**: Advanced configuration options

## 💡 Next Steps

The front-end module is **fully functional and self-contained**. The integration plan is comprehensive and ready for implementation. 

**Recommended Starting Point**: Begin with Phase 1, Step 1 (Model Loading Integration) to establish the foundation for ONNX model support in the unified pipeline.

All components are designed to be backward-compatible and non-intrusive to existing functionality.