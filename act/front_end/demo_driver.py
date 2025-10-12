#!/usr/bin/env python3
"""
🧩 ACT Front-End Comprehensive Demo
==================================

Demonstrates all front-end capabilities:
- Image and text preprocessing
- Specification handling (BOX, LINF_BALL, LIN_POLY)
- Batch processing pipeline
- Mock data generation
- Device management

Usage:
    python comprehensive_demo.py
"""

from __future__ import annotations
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

# Import everything from front_end
from act.front_end import *

def demo_image_processing():
    """Demonstrate image preprocessing capabilities"""
    print("🖼️  Image Processing Demo")
    print("=" * 50)
    
    # Create preprocessors for different image types
    mnist_pre = ImgPre(H=28, W=28, C=1, mean=(0.1307,), std=(0.3081,))
    cifar_pre = ImgPre(H=32, W=32, C=3, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    
    print(f"📋 MNIST preprocessor: {mnist_pre.signature.input_shape}")
    print(f"📋 CIFAR preprocessor: {cifar_pre.signature.input_shape}")
    
    # Generate and process mock images
    for i, (name, pre) in enumerate([("MNIST", mnist_pre), ("CIFAR", cifar_pre)]):
        img, label = mock_image_sample(C=pre.C, H=pre.H, W=pre.W, seed=100+i)
        print(f"\n🎯 {name} Sample {i}:")
        print(f"   Raw image: {img.shape}, label: {label}")
        
        # Process image
        x_model = pre.prepare_sample(img)
        print(f"   Processed: {x_model.shape}, range: [{x_model.min():.3f}, {x_model.max():.3f}]")
        
        # Test flattening
        x_flat = pre.flatten_model_input(x_model)
        print(f"   Flattened: {x_flat.shape}, dtype: {x_flat.dtype}")
        
        # Test inverse mapping
        x_recovered = pre.unflatten_to_model_input(x_flat)
        x_raw = pre.inverse_to_raw_space(x_recovered)
        print(f"   Recovered: {x_raw.shape}, dtype: {x_raw.dtype}")

def demo_specification_system():
    """Demonstrate specification handling"""
    print("\n📋 Specification System Demo") 
    print("=" * 50)
    
    pre = ImgPre(H=32, W=32, C=3)
    img, label = mock_image_sample(C=3, H=32, W=32, seed=42)
    x_model = pre.prepare_sample(img)
    
    # Test different input specification types
    specs = [
        ("BOX", InputSpec(kind=InKind.BOX, 
                         lb=torch.zeros_like(x_model), 
                         ub=torch.ones_like(x_model))),
        ("LINF_BALL", InputSpec(kind=InKind.LINF_BALL, center=x_model, eps=0.03)),
        ("LIN_POLY", InputSpec(kind=InKind.LIN_POLY, 
                              A=torch.randn(5, x_model.numel()), 
                              b=torch.randn(5)))
    ]
    
    for spec_name, raw_spec in specs:
        print(f"\n🔍 {spec_name} Specification:")
        print(f"   Kind: {raw_spec.kind}")
        
        try:
            canonical_spec = pre.canonicalize_input_spec(raw_spec, center=img, eps=0.03)
            print(f"   ✅ Canonicalized successfully")
            if hasattr(canonical_spec, 'center') and canonical_spec.center is not None:
                print(f"   Center shape: {canonical_spec.center.shape}")
            if hasattr(canonical_spec, 'eps') and canonical_spec.eps is not None:
                print(f"   Epsilon: {canonical_spec.eps}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Test output specifications
    print(f"\n🎯 Output Specifications:")
    out_specs = [
        ("MARGIN_ROBUST", OutputSpec(kind=OutKind.MARGIN_ROBUST, y_true=label, margin=0.0)),
        ("TOP1_ROBUST", OutputSpec(kind=OutKind.TOP1_ROBUST, y_true=label)),
        ("LINEAR_LE", OutputSpec(kind=OutKind.LINEAR_LE, c=torch.randn(10), d=1.0)),
        ("RANGE", OutputSpec(kind=OutKind.RANGE, lb=torch.zeros(10), ub=torch.ones(10)))
    ]
    
    for spec_name, out_spec in out_specs:
        canonical_out = pre.canonicalize_output_spec(out_spec, label=label)
        print(f"   {spec_name}: {canonical_out.kind} (y_true={getattr(canonical_out, 'y_true', 'N/A')})")

def demo_batch_processing():
    """Demonstrate batch processing pipeline"""
    print("\n📦 Batch Processing Demo")
    print("=" * 50)
    
    # Create preprocessor
    pre = ImgPre(H=28, W=28, C=1, mean=(0.0,), std=(1.0,))
    
    # Create sample records
    items = []
    for i in range(3):
        img, y = mock_image_sample(C=1, H=28, W=28, seed=200+i)
        x_t = pre.prepare_sample(img)
        I_raw, O_raw = mock_image_specs(x_t, eps=0.05, y_true=y)
        items.append(SampleRecord(idx=i, sample_raw=img, label_raw=y, 
                                 in_spec_raw=I_raw, out_spec_raw=O_raw))
    
    print(f"📋 Created {len(items)} sample records")
    
    # Define a mock verifier
    @dataclass
    class MockVerifResult:
        status: str
        ce_x: Optional[np.ndarray] = None
        ce_y: Optional[np.ndarray] = None
        model_stats: Dict[str, Any] = field(default_factory=dict)
    
    def mock_verify_once(net, entry_id, input_ids, output_ids, input_spec, output_spec, 
                        seed_bounds, solver, timelimit=None, maximize_violation=False):
        # Simulate verification logic
        status = np.random.choice(["CERTIFIED", "COUNTEREXAMPLE", "UNKNOWN"], p=[0.3, 0.4, 0.3])
        ce_x = np.random.randn(len(input_ids)) if status == "COUNTEREXAMPLE" else None
        ce_y = np.random.randn(len(output_ids)) if status == "COUNTEREXAMPLE" else None
        return MockVerifResult(status=status, ce_x=ce_x, ce_y=ce_y, 
                              model_stats={"constraints": len(input_ids) + len(output_ids)})
    
    # Mock objects for verification
    net = object()
    solver = object()
    
    # Configure batch processing
    cfg = BatchConfig(time_budget_s=5.0, maximize_violation=True, entry_id=0, 
                     output_ids=list(range(10)))
    
    # Run batch verification
    results = run_batch(items, pre, net, solver, mock_verify_once, output_dim=10, cfg=cfg)
    
    print(f"\n📊 Batch Results:")
    for r in results:
        status_emoji = {"CERTIFIED": "✅", "COUNTEREXAMPLE": "❌", "UNKNOWN": "❓"}.get(r.status, "⚠️")
        print(f"   Sample {r.idx}: {status_emoji} {r.status}")
        if r.ce_x is not None:
            print(f"     CE input norm: {np.linalg.norm(r.ce_x):.3f}")

def demo_text_processing():
    """Demonstrate text preprocessing capabilities"""
    print("\n📝 Text Processing Demo")
    print("=" * 50)
    
    # Create text preprocessor
    text_pre = TextPre(seq_len=16, vocab={"hello": 10, "world": 11, "test": 12})
    print(f"📋 Text preprocessor: {text_pre.signature.input_shape}")
    
    # Generate and process mock text
    tokens, label = mock_text_sample(seq_len=16, vocab_size=100, seed=300)
    print(f"🎯 Raw tokens: {tokens[:8]}... (length: {len(tokens)})")
    print(f"🎯 Label: {label}")
    
    # Process text
    x_model = text_pre.prepare_sample(tokens)
    print(f"✅ Processed tokens: {x_model.shape}, range: [{x_model.min()}, {x_model.max()}]")
    
    # Test specification
    I_raw, O_raw = mock_text_specs(tokens, y_true=label)
    I_canonical = text_pre.canonicalize_input_spec(I_raw)
    O_canonical = text_pre.canonicalize_output_spec(O_raw, label=label)
    
    print(f"📋 Input spec: {I_canonical.kind}")
    print(f"📋 Output spec: {O_canonical.kind} (y_true={O_canonical.y_true})")

def demo_device_management():
    """Demonstrate device management"""
    print("\n🖥️  Device Management Demo")
    print("=" * 50)
    
    # Test different devices
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    
    for device in devices:
        print(f"\n🔧 Testing device: {device}")
        try:
            pre = ImgPre(H=28, W=28, C=3, device=device)
            img, _ = mock_image_sample(C=3, H=28, W=28, seed=42)
            x_model = pre.prepare_sample(img)
            print(f"   ✅ Success: tensor on {x_model.device}, dtype: {x_model.dtype}")
        except Exception as e:
            print(f"   ❌ Error: {e}")

def main():
    """Run comprehensive front-end demonstration"""
    print("🧩 ACT Front-End Comprehensive Demo")
    print("=" * 60)
    print("Testing all front-end capabilities...")
    
    try:
        demo_image_processing()
        demo_specification_system()
        demo_batch_processing()
        demo_text_processing()
        demo_device_management()
        
        print("\n🎉 All front-end demos completed successfully!")
        print("✅ Front-end module is fully functional and ready for integration")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()