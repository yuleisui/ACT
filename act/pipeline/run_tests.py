#!/usr/bin/env python3
#===- act/pipeline/run_tests.py - ACT Pipeline Testing CLI --------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Command-line interface for ACT pipeline testing framework. Provides
#   a comprehensive CLI for running various types of validation.
#
#===---------------------------------------------------------------------===#


import argparse
import sys
import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from act.util.path_config import get_project_root

# Add project root to path for imports
sys.path.append(get_project_root())

from act.pipeline.config import ConfigManager, load_config
from act.pipeline.correctness import PipelineValidator
from act.pipeline.mock_factory import MockInputFactory
from act.pipeline.utils import setup_logging, print_memory_usage
from act.pipeline import validate_abstraction_verifier, quick_validate, __version__

logger = logging.getLogger(__name__)


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="ACT Pipeline Testing Framework - Validate abstraction verifier correctness and performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick validation with defaults
  python run_tests.py --quick
  
  # Run specific test scenarios
  python run_tests.py --config configs/test_scenarios.yaml
  
  # Run only correctness tests
  python run_tests.py --correctness-only
  
  # Run with custom mock inputs
  python run_tests.py --mock-config configs/mock_inputs.yaml
  
  # Generate performance report
  python run_tests.py --performance --output results.json
  
  # CI mode (fast, essential tests only)
  python run_tests.py --ci
        """
    )
    
    # Main command options
    parser.add_argument("--version", action="version", version=f"ACT Pipeline Testing Framework {__version__}")
    
    # Test execution modes
    test_modes = parser.add_mutually_exclusive_group()
    test_modes.add_argument("--quick", action="store_true", 
                           help="Run quick validation with default settings")
    test_modes.add_argument("--ci", action="store_true",
                           help="Run CI mode (fast, essential tests only)")
    test_modes.add_argument("--comprehensive", action="store_true",
                           help="Run comprehensive validation with all tests")
    
    # Configuration files
    parser.add_argument("--config", type=str, default="configs/test_scenarios.yaml",
                       help="Path to test scenarios configuration file")
    parser.add_argument("--mock-config", type=str, default="configs/mock_inputs.yaml",
                       help="Path to mock inputs configuration file")
    parser.add_argument("--solver-config", type=str, default="configs/solver_settings.yaml",
                       help="Path to solver settings configuration file")
    
    # Test type selection
    parser.add_argument("--correctness-only", action="store_true",
                       help="Run only correctness validation tests")
    parser.add_argument("--properties-only", action="store_true", 
                       help="Run only property-based tests")
    parser.add_argument("--performance-only", action="store_true",
                       help="Run only performance measurement tests")
    parser.add_argument("--regression-only", action="store_true",
                       help="Run only regression tests")
    
    # Test configuration
    parser.add_argument("--timeout", type=float, default=300.0,
                       help="Default timeout for individual tests (seconds)")
    parser.add_argument("--parallel", type=int, default=4,
                       help="Number of parallel workers for testing")
    parser.add_argument("--memory-limit", type=float, default=8.0,
                       help="Memory limit in GB")
    
    # Output options
    parser.add_argument("--output", "-o", type=str,
                       help="Output file path for results (JSON format)")
    parser.add_argument("--report", type=str,
                       help="Generate HTML report at specified path")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Minimize output (only errors and final results)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output and debug logging")
    
    # Development and debugging
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be tested without running")
    parser.add_argument("--list-configs", action="store_true",
                       help="List available configurations and exit")
    parser.add_argument("--validate-config", action="store_true",
                       help="Validate configuration files and exit")
    
    return parser


def configure_logging(args: argparse.Namespace) -> None:
    """Configure logging based on command line arguments."""
    if args.quiet:
        level = "ERROR"
    elif args.verbose:
        level = "DEBUG"
    else:
        level = "INFO"
    
    setup_logging(level=level)


def load_test_configuration(args: argparse.Namespace) -> Dict[str, Any]:
    """Load and merge test configuration from files and arguments."""
    config = {}
    
    # Load base configuration
    if args.config and Path(args.config).exists():
        config = load_config(args.config)
        logger.info(f"Loaded test configuration from {args.config}")
    else:
        logger.warning(f"Configuration file not found: {args.config}, using defaults")
        config = {
            "scenarios": {
                "default_test": {
                    "sample_data": "mnist_small",
                    "input_spec": "robust_l_inf",
                    "output_spec": "classification",
                    "model": "simple_relu",
                    "timeout": args.timeout
                }
            }
        }
    
    # Apply command line overrides
    if args.correctness_only:
        config["run_correctness"] = True
        config["run_properties"] = False
        config["run_performance"] = False
        config["run_bab"] = False
    elif args.properties_only:
        config["run_correctness"] = False
        config["run_properties"] = True
        config["run_performance"] = False
        config["run_bab"] = False
    elif args.performance_only:
        config["run_correctness"] = False
        config["run_properties"] = False
        config["run_performance"] = True
        config["run_bab"] = False
    elif args.regression_only:
        config["run_correctness"] = False
        config["run_properties"] = False
        config["run_performance"] = False
        config["run_regression"] = True
    elif args.ci:
        # CI mode: fast, essential tests only
        config["run_correctness"] = True
        config["run_properties"] = False
        config["run_performance"] = True
        config["run_bab"] = False
        config["run_regression"] = False
    elif args.comprehensive:
        # Comprehensive mode: all tests
        config["run_correctness"] = True
        config["run_properties"] = True
        config["run_performance"] = True
        config["run_bab"] = True
        config["run_regression"] = True
    
    # Apply timeout and parallel settings
    config["default_timeout"] = args.timeout
    config["parallel_workers"] = args.parallel
    config["memory_limit_gb"] = args.memory_limit
    
    return config


def list_available_configurations(args: argparse.Namespace) -> None:
    """List all available configurations."""
    print("📋 Available Configurations")
    print("=" * 50)
    
    # List mock input configurations
    try:
        factory = MockInputFactory(args.mock_config)
        available = factory.list_available_configs()
        
        print("\n🔧 Mock Input Configurations:")
        for section, configs in available.items():
            print(f"  {section}:")
            for config_name in configs:
                print(f"    - {config_name}")
    except Exception as e:
        print(f"  ❌ Could not load mock configurations: {e}")
    
    # List test scenarios
    try:
        if Path(args.config).exists():
            test_config = load_config(args.config)
            scenarios = test_config.get("scenarios", {})
            
            print(f"\n📊 Test Scenarios ({len(scenarios)} available):")
            for scenario_name, scenario_config in scenarios.items():
                print(f"  - {scenario_name}")
                if isinstance(scenario_config, dict):
                    print(f"    Sample: {scenario_config.get('sample_data', 'N/A')}")
                    print(f"    Model: {scenario_config.get('model', 'N/A')}")
                elif isinstance(scenario_config, list):
                    print(f"    Multiple scenarios: {len(scenario_config)} items")
        else:
            print(f"\n📊 Test Scenarios: Configuration file not found ({args.config})")
    except Exception as e:
        print(f"  ❌ Could not load test scenarios: {e}")
    
    # List solver configurations
    try:
        if Path(args.solver_config).exists():
            solver_config = load_config(args.solver_config)
            solvers = solver_config.get("solvers", {})
            
            print(f"\n⚙️  Solver Configurations ({len(solvers)} available):")
            for solver_name, solver_settings in solvers.items():
                method = solver_settings.get("method", "unknown")
                timeout = solver_settings.get("timeout", "N/A")
                print(f"  - {solver_name}: {method} (timeout: {timeout}s)")
        else:
            print(f"\n⚙️  Solver Configurations: Configuration file not found ({args.solver_config})")
    except Exception as e:
        print(f"  ❌ Could not load solver configurations: {e}")


def validate_configurations(args: argparse.Namespace) -> bool:
    """Validate all configuration files."""
    print("🔍 Validating Configuration Files")
    print("=" * 50)
    
    valid = True
    
    # Validate mock inputs config
    print(f"\n1. Mock Inputs Config: {args.mock_config}")
    try:
        factory = MockInputFactory(args.mock_config)
        available = factory.list_available_configs()
        total_configs = sum(len(configs) for configs in available.values())
        print(f"   ✅ Valid - {total_configs} configurations available")
    except Exception as e:
        print(f"   ❌ Invalid - {e}")
        valid = False
    
    # Validate test scenarios config
    print(f"\n2. Test Scenarios Config: {args.config}")
    try:
        if Path(args.config).exists():
            config = load_config(args.config)
            scenarios = config.get("scenarios", {})
            print(f"   ✅ Valid - {len(scenarios)} scenarios available")
        else:
            print(f"   ⚠️  File not found (will use defaults)")
    except Exception as e:
        print(f"   ❌ Invalid - {e}")
        valid = False
    
    # Validate solver settings config
    print(f"\n3. Solver Settings Config: {args.solver_config}")
    try:
        if Path(args.solver_config).exists():
            config = load_config(args.solver_config)
            solvers = config.get("solvers", {})
            print(f"   ✅ Valid - {len(solvers)} solver configurations")
        else:
            print(f"   ⚠️  File not found (will use defaults)")
    except Exception as e:
        print(f"   ❌ Invalid - {e}")
        valid = False
    
    return valid


def run_dry_run(args: argparse.Namespace, config: Dict[str, Any]) -> None:
    """Show what would be tested without actually running tests."""
    print("🔍 Dry Run - Test Plan")
    print("=" * 40)
    
    print(f"\nConfiguration:")
    print(f"  Test scenarios config: {args.config}")
    print(f"  Mock inputs config: {args.mock_config}")
    print(f"  Solver settings config: {args.solver_config}")
    print(f"  Timeout: {args.timeout}s")
    print(f"  Parallel workers: {args.parallel}")
    print(f"  Memory limit: {args.memory_limit}GB")
    
    print(f"\nTest types to run:")
    print(f"  ✓ Correctness validation: {config.get('run_correctness', True)}")
    print(f"  ✓ Property-based testing: {config.get('run_properties', True)}")
    print(f"  ✓ Performance measurement: {config.get('run_performance', True)}")
    print(f"  ✓ BaB refinement testing: {config.get('run_bab', False)}")
    print(f"  ✓ Regression testing: {config.get('run_regression', False)}")
    
    scenarios = config.get("scenarios", {})
    print(f"\nTest scenarios ({len(scenarios)}):")
    for scenario_name, scenario_config in scenarios.items():
        if isinstance(scenario_config, dict):
            print(f"  - {scenario_name}")
            print(f"    Data: {scenario_config.get('sample_data', 'N/A')}")
            print(f"    Model: {scenario_config.get('model', 'N/A')}")
            print(f"    Expected: {scenario_config.get('expected_result', 'N/A')}")
        elif isinstance(scenario_config, list):
            print(f"  - {scenario_name} ({len(scenario_config)} sub-scenarios)")
    
    print(f"\nOutput:")
    if args.output:
        print(f"  JSON results: {args.output}")
    if args.report:
        print(f"  HTML report: {args.report}")
    if not args.output and not args.report:
        print(f"  Console output only")


def save_results(results: Dict[str, Any], output_path: str) -> None:
    """Save results to JSON file."""
    try:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save results to {output_path}: {e}")


def main() -> int:
    """Main CLI entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Configure logging
    configure_logging(args)
    
    # Handle special commands
    if args.list_configs:
        list_available_configurations(args)
        return 0
    
    if args.validate_config:
        valid = validate_configurations(args)
        return 0 if valid else 1
    
    # Load configuration
    try:
        config = load_test_configuration(args)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return 1
    
    # Handle dry run
    if args.dry_run:
        run_dry_run(args, config)
        return 0
    
    # Print initial memory usage
    print_memory_usage("Initial: ")
    
    # Run validation
    try:
        if args.quick:
            logger.info("Running quick validation")
            success = quick_validate()
            results = {"success": success, "mode": "quick"}
        else:
            logger.info("Running comprehensive validation")
            # For full implementation, would use actual config file
            # results = validate_abstraction_verifier(args.config)
            
            # Mock implementation for demonstration
            validator = PipelineValidator()
            results = validator.run_comprehensive_validation(config)
        
        # Save results if requested
        if args.output:
            save_results(results, args.output)
        
        # Generate HTML report if requested
        if args.report:
            logger.info(f"HTML report generation would be implemented")
            logger.info(f"Report would be saved to: {args.report}")
        
        # Print summary
        success = results.get("success", False)
        if not args.quiet:
            print("\n" + "=" * 50)
            print(f"🔧 ACT Pipeline Validation {'✅ COMPLETED' if success else '❌ FAILED'}")
            print("=" * 50)
            
            if "validations" in results:
                for validation_type, validation_result in results["validations"].items():
                    if hasattr(validation_result, 'success'):
                        status = "✅ PASSED" if validation_result.success else "❌ FAILED"
                        print(f"{validation_type}: {status}")
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        logger.info("Validation interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())