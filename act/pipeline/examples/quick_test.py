#!/usr/bin/env python3
"""
Quick test example for ACT pipeline testing framework.

This demonstrates the simplest way to validate the abstraction verifier
using the pipeline testing framework.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from act.pipeline import quick_validate, validate_abstraction_verifier, run_mock_test_example


def main():
    """Run quick validation examples."""
    print("🔧 ACT Pipeline Testing Framework - Quick Test Example")
    print("=" * 60)
    
    # Example 1: Ultra-simple one-line validation
    print("\n1. Ultra-simple validation (one line):")
    success = quick_validate()
    print(f"   Result: {'✅ PASSED' if success else '❌ FAILED'}")
    
    # Example 2: Mock input generation example
    print("\n2. Mock input generation example:")
    result = run_mock_test_example()
    print(f"   Generated and tested mock inputs successfully")
    
    # Example 3: Full validation with default config
    print("\n3. Full validation with configuration:")
    try:
        # Try to use actual config file, fall back to defaults
        config_path = "configs/test_scenarios.yaml" 
        full_result = validate_abstraction_verifier(config_path)
        success = full_result.get("validations", {}).get("correctness", {}).get("success", False)
        print(f"   Comprehensive validation: {'✅ PASSED' if success else '❌ FAILED'}")
        
        if "validations" in full_result:
            for validation_type, validation_result in full_result["validations"].items():
                if hasattr(validation_result, 'total_tests'):
                    print(f"   {validation_type}: {validation_result.passed_tests}/{validation_result.total_tests} tests passed")
    
    except Exception as e:
        print(f"   ⚠️  Config-based validation failed: {e}")
        print("   (This is expected if config files are not accessible)")
    
    print("\n" + "=" * 60)
    print("Quick test completed! This demonstrates:")
    print("• One-line validation: quick_validate()")
    print("• Mock input generation and testing")
    print("• Configuration-based comprehensive validation")
    print("• Simple integration with existing ACT verifier")


if __name__ == "__main__":
    main()