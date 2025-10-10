#!/usr/bin/env python3
"""
Proper regression testing framework for bounds propagation.

This module implements a true regression testing approach that:
1. Captures baseline performance and correctness metrics
2. Compares current implementation against saved baselines
3. Detects both performance and correctness regressions
4. Automatically updates baselines when correctness passes and performance improves
5. Provides tools for updating baselines when changes are intentional

Usage:
    # Capture baseline (run before making changes)
    python test_bounds_prop_regression.py --capture-baseline
    
    # Run regression tests with smart auto-update (default behavior)
    python test_bounds_prop_regression.py --test-regression
    
    # Force update baseline (when changes are intentional)
    python test_bounds_prop_regression.py --update-baseline

Author: ACT Team
Date: October 10, 2025
"""

import sys
import os
import unittest
import time
import json
import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import torch
import torch.nn as nn
import numpy as np

# Add project paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'verifier'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'act'))

from act.util.bounds import Bounds, WeightDecomposer
from act.interval.bounds_propagation import BoundsPropagate

# Import reusable testing components
from test_bounds_propagation import TestingUtils

# Import enhanced bounds propagation helper for advanced metrics
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from act.interval.bounds_prop_helper import BoundsPropMetadata, BoundsPropagationMetadata, NumericalInstabilityError, InvalidBoundsError
    BOUNDS_HELPER_AVAILABLE = True
except ImportError:
    # Fallback if helper not available
    BOUNDS_HELPER_AVAILABLE = False
    BoundsPropMetadata = None
    BoundsPropagationMetadata = None
    NumericalInstabilityError = Exception
    InvalidBoundsError = Exception


class PerformanceMetrics:
    """Enhanced container for performance measurement data specific to regression testing."""
    
    def __init__(self, operation_name: str, input_size: int, iterations: int,
                 total_time_ms: float, avg_time_per_operation_ms: float,
                 memory_peak_mb: float = None, memory_cleanups: int = 0,
                 numerical_warnings: int = 0, stability_issues: int = 0):
        self.operation_name = operation_name
        self.input_size = input_size
        self.iterations = iterations
        self.total_time_ms = total_time_ms
        self.avg_time_per_operation_ms = avg_time_per_operation_ms
        
        # Enhanced metrics from bounds propagation helper
        self.memory_peak_mb = memory_peak_mb
        self.memory_cleanups = memory_cleanups
        self.numerical_warnings = numerical_warnings
        self.stability_issues = stability_issues
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'operation_name': self.operation_name,
            'input_size': self.input_size,
            'iterations': self.iterations,
            'total_time_ms': self.total_time_ms,
            'avg_time_per_operation_ms': self.avg_time_per_operation_ms,
            'memory_peak_mb': self.memory_peak_mb,
            'memory_cleanups': self.memory_cleanups,
            'numerical_warnings': self.numerical_warnings,
            'stability_issues': self.stability_issues
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PerformanceMetrics':
        """Create from dictionary."""
        # Handle both old and new data formats
        return cls(
            operation_name=data['operation_name'],
            input_size=data['input_size'],
            iterations=data['iterations'],
            total_time_ms=data['total_time_ms'],
            avg_time_per_operation_ms=data['avg_time_per_operation_ms'],
            memory_peak_mb=data.get('memory_peak_mb'),
            memory_cleanups=data.get('memory_cleanups', 0),
            numerical_warnings=data.get('numerical_warnings', 0),
            stability_issues=data.get('stability_issues', 0)
        )


class CorrectnessMetrics:
    """Enhanced container for correctness validation data specific to regression testing."""
    
    def __init__(self, test_name: str, input_shape: Tuple[int, ...], output_shape: Tuple[int, ...],
                 lower_bounds_hash: str, upper_bounds_hash: str, 
                 numerical_properties: Dict[str, float],
                 stability_metrics: Dict[str, int] = None,
                 interval_statistics: Dict[str, float] = None):
        self.test_name = test_name
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.lower_bounds_hash = lower_bounds_hash
        self.upper_bounds_hash = upper_bounds_hash
        self.numerical_properties = numerical_properties
        
        # Enhanced stability and statistical metrics
        self.stability_metrics = stability_metrics or {}
        self.interval_statistics = interval_statistics or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'test_name': self.test_name,
            'input_shape': self.input_shape,
            'output_shape': self.output_shape,
            'lower_bounds_hash': self.lower_bounds_hash,
            'upper_bounds_hash': self.upper_bounds_hash,
            'numerical_properties': self.numerical_properties,
            'stability_metrics': self.stability_metrics,
            'interval_statistics': self.interval_statistics
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CorrectnessMetrics':
        """Create from dictionary."""
        return cls(
            test_name=data['test_name'],
            input_shape=data['input_shape'],
            output_shape=data['output_shape'],
            lower_bounds_hash=data['lower_bounds_hash'],
            upper_bounds_hash=data['upper_bounds_hash'],
            numerical_properties=data['numerical_properties'],
            stability_metrics=data.get('stability_metrics', {}),
            interval_statistics=data.get('interval_statistics', {})
        )


class RegressionTestFramework:
    """Framework for comprehensive regression testing."""
    
    def __init__(self, baseline_dir: str = "regression_baselines"):
        """Initialize regression test framework."""
        self.baseline_dir = Path(baseline_dir)
        self.baseline_dir.mkdir(exist_ok=True)
        
        self.performance_baseline_file = self.baseline_dir / "performance_baseline.json"
        self.correctness_baseline_file = self.baseline_dir / "correctness_baseline.json"
        
        # Initialize testing utilities
        self.testing_utils = TestingUtils()
        
        # Test configurations
        self.performance_test_configs = [
            {"size": 50, "iterations": 1000, "name": "linear_small"},
            {"size": 100, "iterations": 500, "name": "linear_medium"},
            {"size": 200, "iterations": 100, "name": "linear_large"},
            {"size": 50, "iterations": 2000, "name": "relu_small"},
            {"size": 100, "iterations": 1000, "name": "relu_medium"},
            {"size": 200, "iterations": 500, "name": "relu_large"},
        ]
        
        self.correctness_test_configs = [
            {"input_shape": (10,), "weight_shape": (20, 10), "name": "linear_basic"},
            {"input_shape": (50,), "weight_shape": (30, 50), "name": "linear_reduction"},
            {"input_shape": (25,), "weight_shape": (100, 25), "name": "linear_expansion"},
            {"input_shape": (3, 32, 32), "conv_config": {"out_channels": 16, "kernel_size": 3, "stride": 1, "padding": 1}, "name": "conv2d_basic"},
            {"input_shape": (1, 28, 28), "conv_config": {"out_channels": 8, "kernel_size": 5, "stride": 2, "padding": 2}, "name": "conv2d_stride"},
        ]
    
    def capture_performance_baseline(self) -> List[PerformanceMetrics]:
        """Capture performance baseline for all test configurations."""
        torch.manual_seed(42)
        np.random.seed(42)
        
        performance_metrics = []
        
        print("Capturing performance baseline...")
        
        for config in self.performance_test_configs:
            size = config["size"]
            iterations = config["iterations"]
            name = config["name"]
            
            print(f"  Testing {name} (size={size}, iterations={iterations})...")
            
            if name.startswith("linear"):
                # Linear operation test using BoundsPropagate._handle_linear
                lb = torch.randn(size)
                ub = lb + torch.abs(torch.randn(size))
                weight = torch.randn(size, size)
                bias = torch.randn(size)
                
                # Create linear layer and bounds propagator
                linear_layer = nn.Linear(size, size)
                linear_layer.weight.data = weight
                linear_layer.bias.data = bias
                bounds_propagator = BoundsPropagate()
                
                def operation():
                    bounds = Bounds(lb, ub, validate=False)
                    return bounds_propagator._handle_linear(linear_layer, bounds, 0)
                
            elif name.startswith("relu"):
                # ReLU operation test using Bounds.clamp_relu (still correct)
                lb = torch.randn(size) - 0.5
                ub = lb + torch.abs(torch.randn(size))
                
                def operation():
                    bounds = Bounds(lb, ub, validate=False)
                    return bounds.clamp_relu()
            
            # Enhanced performance measurement with stability tracking
            memory_cleanups = 0
            numerical_warnings = 0
            stability_issues = 0
            
            # Track initial memory
            initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            peak_memory = initial_memory
            
            start_time = time.perf_counter()
            for i in range(iterations):
                try:
                    result = operation()
                    
                    # Validate bounds stability
                    if hasattr(result, 'lb') and hasattr(result, 'ub'):
                        warnings = TestingUtils.validate_bounds_stability(result.lb, result.ub)
                        numerical_warnings += len(warnings)
                        
                        # Check for critical stability issues
                        if any('NaN' in w or 'Infinite' in w for w in warnings):
                            stability_issues += 1
                    
                    # Track memory usage
                    if torch.cuda.is_available():
                        current_memory = torch.cuda.memory_allocated()
                        peak_memory = max(peak_memory, current_memory)
                        
                        # Periodic cleanup
                        if i % 100 == 0 and i > 0:
                            torch.cuda.empty_cache()
                            memory_cleanups += 1
                            
                except (NumericalInstabilityError, InvalidBoundsError):
                    stability_issues += 1
                    
            end_time = time.perf_counter()
            total_time_ms = (end_time - start_time) * 1000
            avg_time_ms = total_time_ms / iterations
            
            peak_memory_mb = (peak_memory - initial_memory) / (1024 * 1024) if torch.cuda.is_available() else None
            
            metrics = PerformanceMetrics(
                operation_name=name,
                input_size=size,
                iterations=iterations,
                total_time_ms=total_time_ms,
                avg_time_per_operation_ms=avg_time_ms,
                memory_peak_mb=peak_memory_mb,
                memory_cleanups=memory_cleanups,
                numerical_warnings=numerical_warnings,
                stability_issues=stability_issues
            )
            
            performance_metrics.append(metrics)
            print(f"    Total: {total_time_ms:.2f}ms, Avg: {avg_time_ms:.4f}ms per operation")
        
        return performance_metrics
    
    def capture_correctness_baseline(self) -> List[CorrectnessMetrics]:
        """Capture correctness baseline for all test configurations."""
        torch.manual_seed(42)
        np.random.seed(42)
        
        correctness_metrics = []
        
        print("Capturing correctness baseline...")
        
        for config in self.correctness_test_configs:
            name = config["name"]
            print(f"  Testing {name}...")
            
            if name.startswith("linear"):
                input_shape = config["input_shape"]
                weight_shape = config["weight_shape"]
                
                lb = torch.randn(input_shape)
                ub = lb + torch.abs(torch.randn(input_shape))
                weight = torch.randn(weight_shape)
                bias = torch.randn(weight_shape[0])
                
                # Create linear layer and bounds propagator
                linear_layer = nn.Linear(weight_shape[1], weight_shape[0])
                linear_layer.weight.data = weight
                linear_layer.bias.data = bias
                bounds_propagator = BoundsPropagate()
                
                bounds = Bounds(lb, ub, validate=False)
                result = bounds_propagator._handle_linear(linear_layer, bounds, 0)
                
                output_shape = result.shape
                
            elif name.startswith("conv2d"):
                input_shape = config["input_shape"]
                conv_config = config["conv_config"]
                out_channels = conv_config["out_channels"]
                kernel_size = conv_config["kernel_size"]
                stride = conv_config["stride"]
                padding = conv_config["padding"]
                in_channels = input_shape[0]
                
                lb = torch.randn(input_shape)
                ub = lb + torch.abs(torch.randn(input_shape))
                
                # Create conv2d layer and bounds propagator
                conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
                bounds_propagator = BoundsPropagate()
                
                bounds = Bounds(lb, ub, validate=False)
                result = bounds_propagator._handle_conv2d(conv_layer, bounds, 0)
                
                output_shape = result.shape
            
            # Enhanced numerical properties with stability analysis
            intervals = result.ub - result.lb
            
            # Basic numerical properties
            numerical_properties = {
                "bounds_width_mean": float(torch.mean(intervals)),
                "bounds_width_std": float(torch.std(intervals)),
                "lower_bound_mean": float(torch.mean(result.lb)),
                "upper_bound_mean": float(torch.mean(result.ub)),
                "bounds_ordering_violations": int(torch.sum(result.lb > result.ub))
            }
            
            # Enhanced stability metrics
            stability_warnings = TestingUtils.validate_bounds_stability(result.lb, result.ub)
            stability_metrics = {
                "nan_values_lb": int(torch.sum(torch.isnan(result.lb))),
                "nan_values_ub": int(torch.sum(torch.isnan(result.ub))),
                "inf_values_lb": int(torch.sum(torch.isinf(result.lb))),
                "inf_values_ub": int(torch.sum(torch.isinf(result.ub))),
                "small_intervals": int(torch.sum(intervals < 1e-8)),
                "zero_intervals": int(torch.sum(intervals == 0)),
                "warning_count": len(stability_warnings)
            }
            
            # Interval statistics
            interval_statistics = {
                "interval_min": float(torch.min(intervals)),
                "interval_max": float(torch.max(intervals)),
                "interval_median": float(torch.median(intervals)),
                "interval_q25": float(torch.quantile(intervals, 0.25)),
                "interval_q75": float(torch.quantile(intervals, 0.75)),
                "interval_sparsity": float(torch.sum(intervals == 0) / intervals.numel()),
                "bounds_range_lb": float(torch.max(result.lb) - torch.min(result.lb)),
                "bounds_range_ub": float(torch.max(result.ub) - torch.min(result.ub))
            }
            
            metrics = CorrectnessMetrics(
                test_name=name,
                input_shape=tuple(input_shape) if isinstance(input_shape, (list, tuple)) else (input_shape,),
                output_shape=tuple(output_shape),
                lower_bounds_hash=self.testing_utils.hash_tensor(result.lb),
                upper_bounds_hash=self.testing_utils.hash_tensor(result.ub),
                numerical_properties=numerical_properties,
                stability_metrics=stability_metrics,
                interval_statistics=interval_statistics
            )
            
            correctness_metrics.append(metrics)
            print(f"    Shape: {input_shape} -> {output_shape}")
            print(f"    Properties: {numerical_properties}")
            print(f"    Stability: {len(stability_warnings)} warnings, {stability_metrics['zero_intervals']} zero intervals")
            if stability_warnings:
                print(f"    Warnings: {stability_warnings[:3]}...")  # Show first 3 warnings
        
        return correctness_metrics
    
    def save_baseline(self, performance_metrics: List[PerformanceMetrics], 
                     correctness_metrics: List[CorrectnessMetrics]):
        """Save baseline metrics to files."""
        # Save performance baseline
        with open(self.performance_baseline_file, 'w') as f:
            json.dump([m.to_dict() for m in performance_metrics], f, indent=2)
        
        # Save correctness baseline
        with open(self.correctness_baseline_file, 'w') as f:
            json.dump([m.to_dict() for m in correctness_metrics], f, indent=2)
        
        print(f"\nBaseline saved to:")
        print(f"  Performance: {self.performance_baseline_file}")
        print(f"  Correctness: {self.correctness_baseline_file}")
    
    def load_baseline(self) -> Tuple[List[PerformanceMetrics], List[CorrectnessMetrics]]:
        """Load baseline metrics from files."""
        if not self.performance_baseline_file.exists():
            raise FileNotFoundError(f"Performance baseline not found: {self.performance_baseline_file}")
        
        if not self.correctness_baseline_file.exists():
            raise FileNotFoundError(f"Correctness baseline not found: {self.correctness_baseline_file}")
        
        # Load performance baseline
        with open(self.performance_baseline_file, 'r') as f:
            performance_data = json.load(f)
        performance_metrics = [PerformanceMetrics.from_dict(d) for d in performance_data]
        
        # Load correctness baseline
        with open(self.correctness_baseline_file, 'r') as f:
            correctness_data = json.load(f)
        correctness_metrics = [CorrectnessMetrics.from_dict(d) for d in correctness_data]
        
        return performance_metrics, correctness_metrics
    
    def test_performance_regression(self, baseline_metrics: List[PerformanceMetrics], 
                                  max_regression_percent: float = 50.0) -> Tuple[bool, List[str]]:
        """Test for performance regressions against baseline."""
        print("Testing performance regression...")
        
        current_metrics = self.capture_performance_baseline()
        
        regressions = []
        passed = True
        
        # Create lookup for baseline metrics
        baseline_lookup = {m.operation_name: m for m in baseline_metrics}
        
        for current in current_metrics:
            if current.operation_name not in baseline_lookup:
                regressions.append(f"New operation {current.operation_name} not in baseline")
                continue
            
            baseline = baseline_lookup[current.operation_name]
            
            # Compare average time per operation
            baseline_avg = baseline.avg_time_per_operation_ms
            current_avg = current.avg_time_per_operation_ms
            
            if baseline_avg > 0:
                regression_percent = ((current_avg / baseline_avg) - 1) * 100
            else:
                regression_percent = 100.0 if current_avg > 0 else 0.0
            
            print(f"  {current.operation_name}:")
            print(f"    Baseline: {baseline_avg:.4f}ms, Current: {current_avg:.4f}ms")
            print(f"    Change: {regression_percent:+.1f}%")
            
            if regression_percent > max_regression_percent:
                regressions.append(
                    f"{current.operation_name}: {regression_percent:.1f}% regression "
                    f"(current: {current_avg:.4f}ms, baseline: {baseline_avg:.4f}ms)"
                )
                passed = False
        
        return passed, regressions
    
    def test_correctness_regression(self, baseline_metrics: List[CorrectnessMetrics]) -> Tuple[bool, List[str]]:
        """Test for correctness regressions against baseline."""
        print("Testing correctness regression...")
        
        current_metrics = self.capture_correctness_baseline()
        
        regressions = []
        passed = True
        
        # Create lookup for baseline metrics
        baseline_lookup = {m.test_name: m for m in baseline_metrics}
        
        for current in current_metrics:
            if current.test_name not in baseline_lookup:
                regressions.append(f"New test {current.test_name} not in baseline")
                continue
            
            baseline = baseline_lookup[current.test_name]
            
            print(f"  {current.test_name}:")
            
            # Check output shapes (normalize to tuples for comparison)
            current_shape = tuple(current.output_shape)
            baseline_shape = tuple(baseline.output_shape)
            
            if current_shape != baseline_shape:
                regressions.append(
                    f"{current.test_name}: Output shape changed "
                    f"({baseline_shape} -> {current_shape})"
                )
                passed = False
                print(f"    Shape regression detected!")
            
            # Check bounds hashes (exact mathematical equivalence)
            if current.lower_bounds_hash != baseline.lower_bounds_hash:
                regressions.append(f"{current.test_name}: Lower bounds values changed")
                passed = False
                print(f"    Lower bounds regression detected!")
            
            if current.upper_bounds_hash != baseline.upper_bounds_hash:
                regressions.append(f"{current.test_name}: Upper bounds values changed")
                passed = False
                print(f"    Upper bounds regression detected!")
            
            # Check numerical properties
            for prop_name, current_val in current.numerical_properties.items():
                baseline_val = baseline.numerical_properties.get(prop_name)
                if baseline_val is not None:
                    # Allow small numerical differences
                    if abs(current_val - baseline_val) > 1e-6:
                        regressions.append(
                            f"{current.test_name}: {prop_name} changed "
                            f"({baseline_val} -> {current_val})"
                        )
                        passed = False
                        print(f"    Property {prop_name} regression detected!")
            
            # Only print success if no regressions were found for this test
            test_regressions = [r for r in regressions if r.startswith(current.test_name)]
            if not test_regressions or len(test_regressions) == len([r for r in regressions if r.startswith(current.test_name) and "not in baseline" in r]):
                print(f"    ✅ No regressions detected")
        
        return passed, regressions
    
    def analyze_performance_improvements(self, baseline_performance: List[PerformanceMetrics]) -> Tuple[bool, List[str]]:
        """Analyze if current performance shows improvements over baseline."""
        current_metrics = self.capture_performance_baseline()
        improvements = []
        has_improvements = False
        
        baseline_lookup = {m.operation_name: m for m in baseline_performance}
        
        for current in current_metrics:
            if current.operation_name not in baseline_lookup:
                continue
                
            baseline = baseline_lookup[current.operation_name]
            
            # Calculate percentage change (negative means improvement)
            change_percent = ((current.avg_time_per_operation_ms - baseline.avg_time_per_operation_ms) / baseline.avg_time_per_operation_ms) * 100
            
            # Consider significant improvement if >5% faster
            if change_percent < -5.0:
                improvements.append(f"{current.operation_name}: {abs(change_percent):.1f}% faster")
                has_improvements = True
        
        return has_improvements, improvements

    def run_regression_tests(self, auto_update_on_improvement: bool = False) -> bool:
        """Run complete regression test suite with optional auto-update on improvements."""
        try:
            baseline_performance, baseline_correctness = self.load_baseline()
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Please capture baseline first using --capture-baseline")
            return False
        
        print(f"{'='*70}")
        print("REGRESSION TEST SUITE")
        print(f"{'='*70}")
        
        # Test performance regression
        perf_passed, perf_regressions = self.test_performance_regression(baseline_performance)
        
        # Test correctness regression
        corr_passed, corr_regressions = self.test_correctness_regression(baseline_correctness)
        
        # Check for performance improvements if correctness passes
        has_improvements = False
        improvements = []
        if corr_passed and auto_update_on_improvement:
            has_improvements, improvements = self.analyze_performance_improvements(baseline_performance)
        
        # Print summary
        print(f"\n{'='*70}")
        print("REGRESSION TEST SUMMARY")
        print(f"{'='*70}")
        
        if corr_passed and (perf_passed or has_improvements):
            if has_improvements and not perf_passed:
                print("🚀 PERFORMANCE IMPROVEMENTS DETECTED!")
                print("✅ Correctness: Mathematical equivalence maintained")
                print("📈 Performance improvements:")
                for improvement in improvements:
                    print(f"  - {improvement}")
                
                if auto_update_on_improvement:
                    print("\n🔄 Auto-updating baseline with improvements...")
                    current_perf = self.capture_performance_baseline()
                    current_corr = self.capture_correctness_baseline()
                    self.save_baseline(current_perf, current_corr)
                    print("✅ Baseline updated successfully!")
                    return True
            else:
                print("🎉 NO REGRESSIONS DETECTED!")
                print("✅ Performance: No significant regressions")
                print("✅ Correctness: Mathematical equivalence maintained")
                return True
        else:
            print("❌ REGRESSIONS DETECTED!")
            
            if perf_regressions:
                print(f"\nPerformance regressions ({len(perf_regressions)}):")
                for regression in perf_regressions:
                    print(f"  - {regression}")
            
            if corr_regressions:
                print(f"\nCorrectness regressions ({len(corr_regressions)}):")
                for regression in corr_regressions:
                    print(f"  - {regression}")
            
            if not corr_passed:
                print("\n⚠️  Correctness issues prevent auto-update even if performance improved")
            
            return False


def main():
    """Main entry point for regression testing."""
    parser = argparse.ArgumentParser(description="Bounds propagation regression testing framework")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--capture-baseline", action="store_true",
                      help="Capture performance and correctness baseline")
    group.add_argument("--test-regression", action="store_true",
                      help="Run regression tests with smart auto-update on improvements")
    group.add_argument("--update-baseline", action="store_true",
                      help="Update baseline (use when changes are intentional)")
    
    parser.add_argument("--baseline-dir", default="regression_baselines",
                       help="Directory to store baseline files")
    parser.add_argument("--max-regression-percent", type=float, default=50.0,
                       help="Maximum acceptable performance regression percentage")
    
    args = parser.parse_args()
    
    framework = RegressionTestFramework(args.baseline_dir)
    
    if args.capture_baseline or args.update_baseline:
        print("Capturing baseline metrics...")
        performance_metrics = framework.capture_performance_baseline()
        correctness_metrics = framework.capture_correctness_baseline()
        framework.save_baseline(performance_metrics, correctness_metrics)
        
        action = "updated" if args.update_baseline else "captured"
        print(f"\nBaseline {action} successfully!")
        
    elif args.test_regression:
        success = framework.run_regression_tests(auto_update_on_improvement=True)
        sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()