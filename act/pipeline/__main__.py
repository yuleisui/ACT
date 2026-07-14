#!/usr/bin/env python3
"""
ACT Pipeline Entry Point.

Provides unified command-line interface for ACT pipeline operations including:
- ACTFuzzer: Inference-based whitebox fuzzing
- Model synthesis and inference
- Verification workflows

Usage:
    # List available VNNLib benchmarks
    python -m act.pipeline --list
    
    # Fuzz CIFAR-100 VNNLib benchmark
    python -m act.pipeline --fuzz --category cifar100_2024 --timeout 60
    
    # Fuzz with custom device/dtype
    python -m act.pipeline --fuzz --category acasxu_2023 --device cpu --dtype float32

Copyright (C) 2025 SVF-tools/ACT
License: AGPLv3+
"""

from act.config.pipeline_cli import main

if __name__ == "__main__":
    main()
