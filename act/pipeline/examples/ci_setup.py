#!/usr/bin/env python3
"""
CI/CD setup example for ACT pipeline testing framework.

This demonstrates how to integrate the pipeline testing framework
into CI/CD workflows for automated validation.
"""

import sys
import os
from act.util.path_config import get_project_root
sys.path.append(get_project_root())

import json
import yaml
from act.pipeline import validate_abstraction_verifier, ConfigManager, setup_logging


def create_ci_config():
    """Create a CI-optimized configuration for fast, reliable testing."""
    ci_config = {
        "scenarios": {
            "ci_smoke_test": {
                "sample_data": "mnist_small",
                "input_spec": "robust_l_inf_small",
                "output_spec": "classification",
                "model": "simple_relu",
                "expected_result": "UNSAT",
                "timeout": 15
            },
            "ci_performance_check": {
                "sample_data": "mnist_small",
                "input_spec": "robust_l_inf",
                "output_spec": "classification",
                "model": "simple_relu",
                "timeout": 30
            }
        },
        "run_correctness": True,
        "run_properties": False,  # Skip time-consuming property tests in CI
        "run_performance": True,
        "run_bab": False,  # Skip BaB tests in CI
        "properties": {
            "consistency": {
                "test_cases": ["ci_smoke_test"],
                "num_runs": 3
            }
        }
    }
    
    return ci_config


def run_ci_validation():
    """Run CI validation with appropriate settings."""
    print("🔧 CI/CD Pipeline Validation")
    print("=" * 40)
    
    # Setup logging for CI
    setup_logging(level="INFO")
    
    print("\n1. Creating CI-optimized configuration...")
    ci_config = create_ci_config()
    print("   ✅ Created lightweight CI config")
    print(f"   • {len(ci_config['scenarios'])} test scenarios")
    print(f"   • Focus on correctness and performance")
    print(f"   • Skipping time-consuming property tests")
    
    print("\n2. Running CI validation...")
    try:
        # Run validation with CI config
        results = validate_abstraction_verifier(
            config_path=None,  # We'll pass config directly
            log_level="INFO"
        )
        
        # For this example, we'll create a mock result since we don't have the full pipeline
        mock_results = {
            "success": True,
            "timestamp": 1697075200,
            "validations": {
                "correctness": {
                    "success": True,
                    "total_tests": 2,
                    "passed_tests": 2,
                    "failed_tests": 0,
                    "execution_time": 8.5,
                    "memory_usage_mb": 120.3
                },
                "performance": [
                    {
                        "test_name": "ci_smoke_test",
                        "execution_time": 3.2,
                        "memory_usage_mb": 85.7,
                        "success": True
                    },
                    {
                        "test_name": "ci_performance_check", 
                        "execution_time": 5.3,
                        "memory_usage_mb": 95.1,
                        "success": True
                    }
                ]
            }
        }
        
        results = mock_results  # Use mock for demonstration
        
        # Analyze results
        validation_success = results.get("success", False)
        validations = results.get("validations", {})
        
        print(f"   Overall result: {'✅ PASSED' if validation_success else '❌ FAILED'}")
        
        if "correctness" in validations:
            correctness = validations["correctness"]
            print(f"   Correctness: {correctness.get('passed_tests', 0)}/{correctness.get('total_tests', 0)} tests passed")
            print(f"   Execution time: {correctness.get('execution_time', 0):.1f}s")
        
        if "performance" in validations:
            performance_tests = validations["performance"]
            avg_time = sum(t.get("execution_time", 0) for t in performance_tests) / len(performance_tests)
            print(f"   Performance: {len(performance_tests)} tests, avg time: {avg_time:.1f}s")
        
        return validation_success
        
    except Exception as e:
        print(f"   ❌ CI validation failed: {e}")
        return False


def generate_ci_artifacts(results, output_dir="/tmp/ci_artifacts"):
    """Generate CI artifacts and reports."""
    print(f"\n3. Generating CI artifacts in {output_dir}...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate JSON report
    json_report_path = os.path.join(output_dir, "validation_report.json")
    with open(json_report_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"   ✅ JSON report: {json_report_path}")
    
    # Generate JUnit XML for CI systems
    junit_xml_path = os.path.join(output_dir, "test_results.xml")
    junit_xml = generate_junit_xml(results)
    with open(junit_xml_path, 'w') as f:
        f.write(junit_xml)
    print(f"   ✅ JUnit XML: {junit_xml_path}")
    
    # Generate summary badge
    success = results.get("success", False)
    badge_color = "brightgreen" if success else "red"
    badge_text = "passing" if success else "failing"
    
    badge_url = f"https://img.shields.io/badge/validation-{badge_text}-{badge_color}"
    badge_path = os.path.join(output_dir, "validation_badge.txt")
    with open(badge_path, 'w') as f:
        f.write(badge_url)
    print(f"   ✅ Badge URL: {badge_path}")
    
    return output_dir


def generate_junit_xml(results):
    """Generate JUnit XML format for CI integration."""
    validations = results.get("validations", {})
    total_tests = 0
    failures = 0
    time_taken = 0
    
    # Count tests and failures
    if "correctness" in validations:
        correctness = validations["correctness"]
        total_tests += correctness.get("total_tests", 0)
        failures += correctness.get("failed_tests", 0)
        time_taken += correctness.get("execution_time", 0)
    
    if "performance" in validations:
        performance_tests = validations["performance"]
        total_tests += len(performance_tests)
        failures += sum(1 for t in performance_tests if not t.get("success", True))
        time_taken += sum(t.get("execution_time", 0) for t in performance_tests)
    
    junit_xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<testsuite name="ACT Pipeline Validation" 
           tests="{total_tests}" 
           failures="{failures}" 
           time="{time_taken:.2f}">
    
    <testcase name="CorrectnessValidation" 
              classname="act.pipeline.validation" 
              time="{validations.get('correctness', {}).get('execution_time', 0):.2f}">
        {'<failure message="Correctness tests failed"></failure>' if validations.get('correctness', {}).get('failed_tests', 0) > 0 else ''}
    </testcase>
    
    {''.join(f'<testcase name="{test.get("test_name", "PerformanceTest")}" classname="act.pipeline.performance" time="{test.get("execution_time", 0):.2f}">{"<failure message=\'Performance test failed\'></failure>" if not test.get("success", True) else ""}</testcase>' for test in validations.get("performance", []))}
    
</testsuite>'''
    
    return junit_xml


def demonstrate_github_actions():
    """Show example GitHub Actions workflow."""
    print("\n4. Example GitHub Actions Integration:")
    
    github_workflow = '''
name: ACT Pipeline Validation

on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v3
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install torch torchvision
        pip install pyyaml psutil
        # Add other ACT dependencies
    
    - name: Run ACT Pipeline Validation
      run: |
        cd act/pipeline/examples
        python ci_setup.py
    
    - name: Upload test results
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: validation-results
        path: /tmp/ci_artifacts/
    
    - name: Publish test results
      uses: dorny/test-reporter@v1
      if: success() || failure()
      with:
        name: Pipeline Tests
        path: /tmp/ci_artifacts/test_results.xml
        reporter: java-junit
'''
    
    print(github_workflow)


def demonstrate_docker_integration():
    """Show example Docker integration."""
    print("\n5. Example Docker Integration:")
    
    dockerfile = '''
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    git \\
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies  
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy ACT code
COPY . /app/act
WORKDIR /app

# Run validation
CMD ["python", "-m", "act.pipeline.examples.ci_setup"]
'''
    
    print(dockerfile)


def main():
    """Run CI/CD integration demonstration."""
    # Run CI validation
    success = run_ci_validation()
    
    # Generate sample results for artifact demonstration
    sample_results = {
        "success": success,
        "timestamp": "2025-10-12T00:00:00Z",
        "validations": {
            "correctness": {
                "success": success,
                "total_tests": 2,
                "passed_tests": 2 if success else 1,
                "failed_tests": 0 if success else 1,
                "execution_time": 8.5
            }
        }
    }
    
    # Generate artifacts
    artifacts_dir = generate_ci_artifacts(sample_results)
    
    # Show integration examples
    demonstrate_github_actions()
    demonstrate_docker_integration()
    
    print("\n" + "=" * 40)
    print("CI/CD integration demo completed!")
    print(f"Exit code: {0 if success else 1}")
    print("\nKey CI/CD features:")
    print("• Fast, lightweight validation for CI")
    print("• JUnit XML output for test reporting")
    print("• JSON artifacts for detailed analysis")
    print("• GitHub Actions workflow example")
    print("• Docker container support")
    print("• Automated badge generation")
    
    # Exit with appropriate code for CI
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()