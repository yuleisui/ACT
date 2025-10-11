#!/usr/bin/env python3
"""
Regression tests for bounds propagation APIs.
Tests REAL production BoundsPropagate class using shared configurations.

This module provides:
- Performance regression detection for bounds propagation APIs
- Correctness regression validation using production code
- Baseline capture and comparison using shared test configurations
- Integration with MockFactory for consistent test data generation

Uses shared test configurations from test_configs.py to provide
comprehensive regression testing across various models and input scenarios.
"""

import json
import time
import hashlib
import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime

from act.interval.bounds_propagation import BoundsPropagate
from test_configs import MockFactory, get_regression_test_configs


@dataclass
class RegressionBaseline:
    performance: dict
    correctness: dict
    timestamp: str
    test_configs: dict


class BoundsRegressionTester:
    """Regression tester using shared configurations."""
    
    def __init__(self):
        self.baseline_dir = Path(__file__).parent / "regression_baselines"
        self.baseline_dir.mkdir(exist_ok=True)
        self.baseline_file = self.baseline_dir / "bounds_baseline.json"
    
    def capture_baseline(self):
        """Capture baseline using shared configurations."""
        torch.manual_seed(42)
        propagator = BoundsPropagate(performance_mode=True)
        
        performance_data = {}
        correctness_data = {}
        
        # Performance tests
        perf_configs = get_regression_test_configs("performance_tests")
        for config in perf_configs:
            test_name = f"{config['model']}_{config['data']}"
            print(f"📊 Performance baseline: {test_name}")
            
            model = MockFactory.create_model(config["model"])
            lb, ub = MockFactory.create_data(config["data"])
            
            # Warmup runs to eliminate first-time overhead
            for _ in range(2):
                propagator.propagate_bounds(model, lb, ub)
            
            # Measure performance with isolated timing
            times = []
            for _ in range(config["iterations"]):
                # Ensure clean state
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                start_time = time.perf_counter()
                result, metadata = propagator.propagate_bounds(model, lb, ub)
                elapsed = time.perf_counter() - start_time
                times.append(elapsed)
            
            # Remove outliers (first run might have initialization overhead)
            if len(times) > 3:
                times = sorted(times)[1:-1]  # Remove fastest and slowest
            
            performance_data[test_name] = {
                'mean_time': float(np.mean(times)),
                'std_time': float(np.std(times)),
                'iterations': config["iterations"],
                'output_shape': list(result.shape)
            }
        
        # Correctness tests
        correct_configs = get_regression_test_configs("correctness_tests")
        for config in correct_configs:
            test_name = f"{config['model']}_{config['data']}"
            print(f"📊 Correctness baseline: {test_name}")
            
            model = MockFactory.create_model(config["model"])
            lb, ub = MockFactory.create_data(config["data"])
            
            result, metadata = propagator.propagate_bounds(model, lb, ub)
            
            correctness_data[test_name] = {
                'lb_hash': self._hash_tensor(result.lb),
                'ub_hash': self._hash_tensor(result.ub),
                'bounds_width': float(torch.mean(result.ub - result.lb)),
                'output_shape': list(result.shape)
            }
        
        # Save baseline
        baseline = RegressionBaseline(
            performance=performance_data,
            correctness=correctness_data,
            timestamp=datetime.now().isoformat(),
            test_configs={
                'performance': perf_configs,
                'correctness': correct_configs
            }
        )
        
        with open(self.baseline_file, 'w') as f:
            json.dump(asdict(baseline), f, indent=2)
        
        print(f"✅ Baseline saved to {self.baseline_file}")
    
    def test_regression(self, auto_update_on_improvement: bool = False) -> bool:
        """Test for regressions using shared configurations."""
        if not self.baseline_file.exists():
            print("❌ No baseline found. Run with --capture-baseline first.")
            return False
        
        with open(self.baseline_file) as f:
            baseline = RegressionBaseline(**json.load(f))
        
        torch.manual_seed(42)
        propagator = BoundsPropagate(performance_mode=True)
        
        success = True
        performance_failures = []
        correctness_failures = []
        performance_improvements = []
        
        # Test performance regressions
        perf_configs = get_regression_test_configs("performance_tests")
        current_performance = {}
        
        for config in perf_configs:
            test_name = f"{config['model']}_{config['data']}"
            if test_name not in baseline.performance:
                continue
            
            print(f"🔍 Performance test: {test_name}")
            
            model = MockFactory.create_model(config["model"])
            lb, ub = MockFactory.create_data(config["data"])
            
            # Run current test with improved timing
            # Warmup run to eliminate first-time overhead
            propagator.propagate_bounds(model, lb, ub)
            
            # Multiple timing runs for better accuracy
            times = []
            for _ in range(3):
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                start_time = time.perf_counter()
                result, metadata = propagator.propagate_bounds(model, lb, ub)
                elapsed = time.perf_counter() - start_time
                times.append(elapsed)
            
            # Use median time to avoid outliers
            current_time = sorted(times)[len(times)//2]
            
            # Store current performance for potential baseline update
            current_performance[test_name] = {
                'mean_time': current_time,
                'std_time': float(np.std(times)),
                'iterations': config["iterations"],
                'output_shape': list(result.shape)
            }
            
            # Check for performance changes
            baseline_time = baseline.performance[test_name]['mean_time']
            if current_time > baseline_time * 1.2:
                regression_percent = ((current_time - baseline_time) / baseline_time) * 100
                failure_msg = f"Performance regression in {test_name}: {current_time:.4f}s vs baseline {baseline_time:.4f}s (+{regression_percent:.1f}% above 20% threshold)"
                print(f"⚠️  {failure_msg}")
                performance_failures.append(failure_msg)
                success = False
            elif current_time < baseline_time * 0.8:  # 20% improvement threshold
                improvement_percent = ((baseline_time - current_time) / baseline_time) * 100
                improvement_msg = f"Performance improvement in {test_name}: {current_time:.4f}s vs baseline {baseline_time:.4f}s (-{improvement_percent:.1f}% improvement)"
                print(f"🚀 {improvement_msg}")
                performance_improvements.append(improvement_msg)
        
        # Test correctness regressions
        correct_configs = get_regression_test_configs("correctness_tests")
        current_correctness = {}
        correctness_maintained = True
        
        for config in correct_configs:
            test_name = f"{config['model']}_{config['data']}"
            if test_name not in baseline.correctness:
                continue
            
            print(f"🔍 Correctness test: {test_name}")
            
            model = MockFactory.create_model(config["model"])
            lb, ub = MockFactory.create_data(config["data"])
            
            result, metadata = propagator.propagate_bounds(model, lb, ub)
            
            # Store current correctness for potential baseline update
            current_correctness[test_name] = {
                'lb_hash': self._hash_tensor(result.lb),
                'ub_hash': self._hash_tensor(result.ub),
                'bounds_width': float(torch.mean(result.ub - result.lb)),
                'output_shape': list(result.shape)
            }
            
            # Check correctness regression
            current_lb_hash = self._hash_tensor(result.lb)
            current_ub_hash = self._hash_tensor(result.ub)
            
            baseline_correct = baseline.correctness[test_name]
            if (current_lb_hash != baseline_correct['lb_hash'] or
                current_ub_hash != baseline_correct['ub_hash']):
                failure_msg = f"Correctness regression in {test_name}: tensor bounds changed (lb_hash: {current_lb_hash[:8]}... vs baseline {baseline_correct['lb_hash'][:8]}..., ub_hash: {current_ub_hash[:8]}... vs baseline {baseline_correct['ub_hash'][:8]}...)"
                print(f"⚠️  {failure_msg}")
                correctness_failures.append(failure_msg)
                correctness_maintained = False
                success = False
        
        # Auto-update baseline if performance improved and correctness maintained
        if auto_update_on_improvement and performance_improvements and correctness_maintained and not correctness_failures:
            print(f"\n🎯 Auto-updating baseline due to performance improvements...")
            self._update_baseline_with_improvements(baseline, current_performance, current_correctness, performance_improvements)
            print(f"✅ Baseline auto-updated with {len(performance_improvements)} performance improvements!")
        
        # Print detailed failure summary
        if not success:
            print(f"\n❌ Regression Test Failures Summary:")
            print("=" * 50)
            
            if performance_failures:
                print(f"\n🚀 Performance Regressions ({len(performance_failures)}):")
                for i, failure in enumerate(performance_failures, 1):
                    print(f"  {i}. {failure}")
            
            if correctness_failures:
                print(f"\n🎯 Correctness Regressions ({len(correctness_failures)}):")
                for i, failure in enumerate(correctness_failures, 1):
                    print(f"  {i}. {failure}")
            
            print(f"\n💡 Recommendations:")
            if performance_failures:
                print("  • Performance: Check for algorithmic changes, memory leaks, or suboptimal implementations")
                print("  • Re-capture baseline if performance changes are intentional: --capture-baseline")
            if correctness_failures:
                print("  • Correctness: Verify numerical stability, floating-point precision, or algorithm changes")
                print("  • Check for unintended modifications to bounds computation logic")
        elif performance_improvements:
            print(f"\n🎉 Performance Improvements Detected ({len(performance_improvements)}):")
            for i, improvement in enumerate(performance_improvements, 1):
                print(f"  {i}. {improvement}")
            if not auto_update_on_improvement:
                print(f"\n💡 To automatically update baseline with improvements, use --test-regression")
        
        if success:
            print("✅ All regression tests passed!")
        
        return success

    def _update_baseline_with_improvements(self, baseline: RegressionBaseline, 
                                         current_performance: dict, 
                                         current_correctness: dict,
                                         improvements: list) -> None:
        """Update baseline with improved performance while maintaining correctness."""
        # Update performance data with improvements
        for test_name, perf_data in current_performance.items():
            if test_name in baseline.performance:
                baseline.performance[test_name] = perf_data
        
        # Update correctness data to match current (since correctness was maintained)
        for test_name, correct_data in current_correctness.items():
            if test_name in baseline.correctness:
                baseline.correctness[test_name] = correct_data
        
        # Update metadata
        baseline.metadata.update({
            'last_auto_update': datetime.now().isoformat(),
            'auto_update_improvements': len(improvements),
            'auto_update_reason': 'Performance improvement with correctness maintained'
        })
        
        # Write updated baseline
        with open(self.baseline_file, 'w') as f:
            json.dump(asdict(baseline), f, indent=2)
        
        print(f"📝 Updated baseline saved to {self.baseline_file}")
    
    def _hash_tensor(self, tensor: torch.Tensor) -> str:
        """Create deterministic hash of tensor."""
        np_array = tensor.detach().cpu().numpy()
        rounded = np.round(np_array, decimals=6)
        return hashlib.md5(rounded.tobytes()).hexdigest()


def main():
    import sys
    tester = BoundsRegressionTester()
    
    if "--capture-baseline" in sys.argv:
        tester.capture_baseline()
    elif "--test-regression" in sys.argv:
        # Enable auto-update for --test-regression 
        success = tester.test_regression(auto_update_on_improvement=True)
        sys.exit(0 if success else 1)
    else:
        print("Usage: python test_bounds_prop_regression.py [--capture-baseline|--test-regression]")
        print("  --capture-baseline    : Capture new performance and correctness baseline")
        print("  --test-regression     : Test for regressions (auto-updates baseline on improvements)")


if __name__ == "__main__":
    main()