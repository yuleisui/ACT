"""
🔧 Model Inference and Testing Framework

Functions for testing synthesized models, analyzing failures, and providing user-friendly
explanations for architecture mismatches in the ACT verification pipeline.
"""

from __future__ import annotations
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple, Union

# ------------------- Model Inference Function --------------------------------
# Main model inference function
# -----------------------------------------------------------------------------
def model_inference(models: Dict[str, nn.Sequential], input_data: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, nn.Sequential]:
    """
    Test all wrapped models with their respective inputs and provide execution statistics.
    
    Args:
        models: Dict[combo_id, nn.Sequential] - Synthesized wrapped models to test
        input_data: Dict[dataset_name, data_pack] - Input data for testing models
        
    Returns:
        Dict[combo_id, nn.Sequential] - Successfully inferred models only
    """
    print(f"\n🔧 Testing {len(models)} models...")
    
    # Handle case where no models were generated
    if not models:
        print("⚠️  No models to test - check spec generation and synthesis configuration")
        return {}
    
    # Group by dataset for organized testing
    by_dataset = {}
    for combo_id, model in models.items():
        dataset = combo_id.split('|')[1].split(':')[1]  # Extract from x:dataset_name
        if dataset not in by_dataset:
            by_dataset[dataset] = []
        by_dataset[dataset].append((combo_id, model))
    
    success_count = 0
    failure_count = 0
    failure_summary = {}  # Track unique failure types
    successful_models = {}  # Track successfully inferred models
    
    for dataset_name, models_list in by_dataset.items():
        test_input = input_data[dataset_name]["x"]
        dataset_successes = 0
        
        for combo_id, model in models_list:
            model_name = combo_id.split('|')[0].split(':')[1]
            try:
                with torch.no_grad():
                    output = model(test_input)
                    success_count += 1
                    dataset_successes += 1
                    successful_models[combo_id] = model  # Store successful model
            except Exception as e:
                failure_count += 1
                # Track unique failure patterns
                pattern = f"{model_name.split('_')[0]} + {dataset_name}"  # e.g., "mnist + cifar10"
                if pattern not in failure_summary:
                    failure_summary[pattern] = {'count': 0, 'error': str(e)[:100]}
                failure_summary[pattern]['count'] += 1
        
        print(f"   {dataset_name}: {dataset_successes}/{len(models_list)} successful")
    
    success_rate = (success_count / len(models)) * 100 if len(models) > 0 else 0
    print(f"\n📊 Overall: {success_count}/{len(models)} successful ({success_rate:.1f}%)")
    
    # Show concise failure analysis
    if failure_summary:
        print(f"\n❌ Failure patterns:")
        for pattern, info in failure_summary.items():
            print(f"   • {pattern}: {info['count']} failures (architecture mismatch)")
        print(f"   💡 Tip: Use domain-matched combinations (mnist+mnist, cifar+cifar) for 100% success")
        
        # Optional: Add detailed explanation for first failure (can be enabled if needed)
        # if "--verbose" in sys.argv:
        #     first_pattern = list(failure_summary.keys())[0]
        #     model_name, dataset_name = first_pattern.split(" + ")
        #     print(f"\n🔍 DETAILED ANALYSIS (example):")
        #     print(explain_architecture_mismatch(model_name, dataset_name, list(failure_summary.values())[0]['error']))
    
    return successful_models
        
# -----------------------------------------------------------------------------
# Helper functions for user-friendly error explanations
# -----------------------------------------------------------------------------
def extract_shape_info(error_msg: str) -> dict:
    """Extract shape information from error messages for detailed explanations."""
    import re
    
    info = {"input_features": None, "expected_features": None, "input_shape": None}
    
    # Pattern for "mat1 and mat2 shapes cannot be multiplied (1x180 and 245x10)"
    mat_pattern = r"mat1 and mat2 shapes cannot be multiplied \(1x(\d+) and (\d+)x\d+\)"
    match = re.search(mat_pattern, error_msg)
    if match:
        info["input_features"] = int(match.group(1))
        info["expected_features"] = int(match.group(2))
    
    # Pattern for input shape errors
    shape_pattern = r"input\[([^\]]+)\]"
    match = re.search(shape_pattern, error_msg)
    if match:
        info["input_shape"] = match.group(1)
    
    return info


def get_model_architecture_info(model_domain: str) -> dict:
    """Get detailed architecture information for different model domains."""
    arch_info = {
        "mnist": {
            "input_size": "28×28 pixels",
            "channels": "1 (grayscale)",
            "total_pixels": "784 features",
            "architecture": "CNN optimized for handwritten digits",
            "typical_features_before_fc": "196 (after conv/pool layers)"
        },
        "cifar10": {
            "input_size": "32×32 pixels", 
            "channels": "3 (RGB)",
            "total_pixels": "3072 features",
            "architecture": "CNN optimized for natural images",
            "typical_features_before_fc": "245 (after conv/pool layers)"
        },
        "unknown": {
            "input_size": "unknown",
            "channels": "unknown",
            "total_pixels": "unknown",
            "architecture": "unknown architecture",
            "typical_features_before_fc": "unknown"
        }
    }
    return arch_info.get(model_domain, arch_info["unknown"])


def get_domain_info(domain: str) -> str:
    """Get descriptive information about a domain."""
    domain_info = {
        "mnist": "28×28 grayscale handwritten digits",
        "cifar10": "32×32 RGB natural images",
        "unknown": "unknown image format"
    }
    return domain_info.get(domain, "unknown format")


def explain_architecture_mismatch(model_name: str, dataset_name: str, error_msg: str) -> str:
    """Provide concise explanations for architecture mismatches."""
    
    # Extract domains
    model_domain = "mnist" if "mnist" in model_name.lower() else "cifar10" if "cifar10" in model_name.lower() else "unknown"
    data_domain = "mnist" if "mnist" in dataset_name.lower() else "cifar10" if "cifar10" in dataset_name.lower() else "unknown"
    
    # Extract key error info
    import re
    shape_match = re.search(r"mat1 and mat2 shapes cannot be multiplied \(1x(\d+) and (\d+)x\d+\)", error_msg)
    
    if shape_match:
        actual_features = shape_match.group(1)
        expected_features = shape_match.group(2)
        explanation = f"""
🔍 MISMATCH: {model_name} + {dataset_name}
   Model expects {expected_features} features, got {actual_features}
   Cause: {model_domain.upper()} model designed for {get_domain_info(model_domain)}
          {data_domain.upper()} data provides {get_domain_info(data_domain)}
   Fix: Use {model_domain}+{model_domain} combinations for compatibility"""
    else:
        explanation = f"""
🔍 MISMATCH: {model_name} + {dataset_name}
   Architecture incompatibility between {model_domain.upper()} model and {data_domain.upper()} data
   Fix: Use domain-matched combinations"""
    
    return explanation.strip()

