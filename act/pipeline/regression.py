"""
Baseline management and regression testing for ACT pipeline.

This module provides comprehensive regression testing capabilities including baseline
capture, performance tracking, correctness regression detection, and trend analysis.
"""

import os
import json
import time
import hashlib
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
import logging

from act.pipeline.utils import PerformanceProfiler, clear_torch_cache
from act.pipeline.correctness import ValidationResult, PerformanceResult, VerifyResult

logger = logging.getLogger(__name__)


@dataclass
class BaselineMetrics:
    """Metrics captured for baseline comparison."""
    accuracy: float
    avg_execution_time: float
    avg_memory_usage_mb: float
    sat_rate: float
    unsat_rate: float
    unknown_rate: float
    timeout_rate: float
    test_count: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    model_hash: str = ""
    config_hash: str = ""


@dataclass
class RegressionResult:
    """Result from regression testing."""
    baseline_name: str
    current_metrics: BaselineMetrics
    baseline_metrics: BaselineMetrics
    performance_regression: bool
    correctness_regression: bool
    improvement: bool
    details: Dict[str, Any] = field(default_factory=dict)
    threshold_violations: List[str] = field(default_factory=list)


@dataclass
class TrendData:
    """Historical trend data for metrics."""
    metric_name: str
    values: List[float]
    timestamps: List[str]
    trend_direction: str  # "improving", "degrading", "stable"
    avg_change_rate: float


class BaselineManager:
    """Manage baselines for regression testing."""
    
    def __init__(self, baseline_dir: str = "baselines"):
        """
        Initialize baseline manager.
        
        Args:
            baseline_dir: Directory to store baseline files
        """
        self.baseline_dir = Path(baseline_dir)
        self.baseline_dir.mkdir(exist_ok=True)
        
        # Default thresholds for regression detection
        self.default_thresholds = {
            "execution_time_increase": 0.25,  # 25% slower allowed
            "memory_increase": 0.30,          # 30% more memory allowed
            "accuracy_decrease": 0.05,        # 5% accuracy drop threshold
            "timeout_increase": 0.10,         # 10% more timeouts allowed
        }
    
    def capture_baseline(
        self,
        name: str,
        validation_results: List[ValidationResult],
        performance_results: List[PerformanceResult],
        model_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> BaselineMetrics:
        """
        Capture current performance as baseline.
        
        Args:
            name: Baseline name
            validation_results: Validation test results
            performance_results: Performance test results
            model_path: Path to model file for hash calculation
            config: Configuration dictionary for hash calculation
            
        Returns:
            Captured baseline metrics
        """
        logger.info(f"Capturing baseline: {name}")
        
        # Calculate aggregate metrics
        total_tests = sum(r.total_tests for r in validation_results)
        passed_tests = sum(r.passed_tests for r in validation_results)
        accuracy = passed_tests / total_tests if total_tests > 0 else 0.0
        
        # Calculate result distribution
        result_counts = {result.name: 0 for result in VerifyResult}
        for val_result in validation_results:
            for test_result in val_result.results:
                result_type = test_result.get('result', 'UNKNOWN')
                if result_type in result_counts:
                    result_counts[result_type] += 1
        
        sat_rate = result_counts['SAT'] / total_tests if total_tests > 0 else 0.0
        unsat_rate = result_counts['UNSAT'] / total_tests if total_tests > 0 else 0.0
        unknown_rate = result_counts['UNKNOWN'] / total_tests if total_tests > 0 else 0.0
        timeout_rate = result_counts['TIMEOUT'] / total_tests if total_tests > 0 else 0.0
        
        # Calculate performance metrics
        avg_execution_time = np.mean([r.execution_time for r in performance_results]) if performance_results else 0.0
        avg_memory_usage = np.mean([r.memory_usage_mb for r in performance_results]) if performance_results else 0.0
        
        # Calculate hashes for model and config
        model_hash = self._calculate_model_hash(model_path) if model_path else ""
        config_hash = self._calculate_config_hash(config) if config else ""
        
        baseline = BaselineMetrics(
            accuracy=accuracy,
            avg_execution_time=avg_execution_time,
            avg_memory_usage_mb=avg_memory_usage,
            sat_rate=sat_rate,
            unsat_rate=unsat_rate,
            unknown_rate=unknown_rate,
            timeout_rate=timeout_rate,
            test_count=total_tests,
            model_hash=model_hash,
            config_hash=config_hash
        )
        
        # Save baseline
        self._save_baseline(name, baseline)
        logger.info(f"Baseline '{name}' captured with {total_tests} tests, {accuracy:.1%} accuracy")
        
        return baseline
    
    def load_baseline(self, name: str) -> Optional[BaselineMetrics]:
        """
        Load baseline from storage.
        
        Args:
            name: Baseline name
            
        Returns:
            Loaded baseline metrics or None if not found
        """
        baseline_file = self.baseline_dir / f"{name}.json"
        if not baseline_file.exists():
            logger.warning(f"Baseline '{name}' not found")
            return None
        
        try:
            with open(baseline_file, 'r') as f:
                data = json.load(f)
            return BaselineMetrics(**data)
        except Exception as e:
            logger.error(f"Failed to load baseline '{name}': {e}")
            return None
    
    def list_baselines(self) -> List[str]:
        """
        List available baselines.
        
        Returns:
            List of baseline names
        """
        baseline_files = list(self.baseline_dir.glob("*.json"))
        return [f.stem for f in baseline_files]
    
    def compare_to_baseline(
        self,
        baseline_name: str,
        current_results: Tuple[List[ValidationResult], List[PerformanceResult]],
        model_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        custom_thresholds: Optional[Dict[str, float]] = None
    ) -> RegressionResult:
        """
        Compare current results to baseline.
        
        Args:
            baseline_name: Name of baseline to compare against
            current_results: Tuple of (validation_results, performance_results)
            model_path: Path to current model
            config: Current configuration
            custom_thresholds: Custom regression thresholds
            
        Returns:
            Regression test result
        """
        logger.info(f"Comparing to baseline: {baseline_name}")
        
        baseline = self.load_baseline(baseline_name)
        if baseline is None:
            raise ValueError(f"Baseline '{baseline_name}' not found")
        
        validation_results, performance_results = current_results
        
        # Capture current metrics
        current = self.capture_baseline(
            name=f"temp_{int(time.time())}",
            validation_results=validation_results,
            performance_results=performance_results,
            model_path=model_path,
            config=config
        )
        
        # Use custom or default thresholds
        thresholds = custom_thresholds or self.default_thresholds
        
        # Check for regressions
        performance_regression = False
        correctness_regression = False
        improvement = False
        violations = []
        details = {}
        
        # Performance regression checks
        time_increase = (current.avg_execution_time - baseline.avg_execution_time) / baseline.avg_execution_time
        if time_increase > thresholds["execution_time_increase"]:
            performance_regression = True
            violations.append(f"Execution time increased by {time_increase:.1%}")
        
        memory_increase = (current.avg_memory_usage_mb - baseline.avg_memory_usage_mb) / baseline.avg_memory_usage_mb
        if memory_increase > thresholds["memory_increase"]:
            performance_regression = True
            violations.append(f"Memory usage increased by {memory_increase:.1%}")
        
        # Correctness regression checks
        accuracy_decrease = baseline.accuracy - current.accuracy
        if accuracy_decrease > thresholds["accuracy_decrease"]:
            correctness_regression = True
            violations.append(f"Accuracy decreased by {accuracy_decrease:.1%}")
        
        timeout_increase = current.timeout_rate - baseline.timeout_rate
        if timeout_increase > thresholds["timeout_increase"]:
            correctness_regression = True
            violations.append(f"Timeout rate increased by {timeout_increase:.1%}")
        
        # Check for improvements
        if (time_increase < -0.05 or  # 5% faster
            memory_increase < -0.05 or  # 5% less memory
            accuracy_decrease < -0.01):  # 1% more accurate
            improvement = True
        
        # Store detailed comparison
        details = {
            "execution_time_change": time_increase,
            "memory_change": memory_increase,
            "accuracy_change": -accuracy_decrease,
            "timeout_rate_change": timeout_increase,
            "baseline_timestamp": baseline.timestamp,
            "current_timestamp": current.timestamp,
            "model_hash_match": current.model_hash == baseline.model_hash,
            "config_hash_match": current.config_hash == baseline.config_hash,
        }
        
        result = RegressionResult(
            baseline_name=baseline_name,
            current_metrics=current,
            baseline_metrics=baseline,
            performance_regression=performance_regression,
            correctness_regression=correctness_regression,
            improvement=improvement,
            details=details,
            threshold_violations=violations
        )
        
        # Clean up temporary baseline
        temp_file = self.baseline_dir / f"temp_{int(time.time())}.json"
        if temp_file.exists():
            temp_file.unlink()
        
        logger.info(f"Regression check complete: {'REGRESSION' if performance_regression or correctness_regression else 'PASS'}")
        
        return result
    
    def analyze_trends(
        self,
        baseline_names: Optional[List[str]] = None,
        days_back: int = 30
    ) -> Dict[str, TrendData]:
        """
        Analyze performance trends over time.
        
        Args:
            baseline_names: Specific baselines to analyze (default: all recent)
            days_back: Number of days to look back
            
        Returns:
            Dictionary of metric trends
        """
        if baseline_names is None:
            baseline_names = self.list_baselines()
        
        # Filter baselines by date
        cutoff_date = datetime.now() - timedelta(days=days_back)
        baselines = []
        
        for name in baseline_names:
            baseline = self.load_baseline(name)
            if baseline and datetime.fromisoformat(baseline.timestamp) >= cutoff_date:
                baselines.append(baseline)
        
        if not baselines:
            logger.warning("No recent baselines found for trend analysis")
            return {}
        
        # Sort by timestamp
        baselines.sort(key=lambda b: b.timestamp)
        
        trends = {}
        metrics = ['accuracy', 'avg_execution_time', 'avg_memory_usage_mb', 'timeout_rate']
        
        for metric in metrics:
            values = [getattr(b, metric) for b in baselines]
            timestamps = [b.timestamp for b in baselines]
            
            if len(values) < 2:
                continue
            
            # Calculate trend direction
            recent_avg = np.mean(values[-min(3, len(values)):])
            older_avg = np.mean(values[:min(3, len(values))])
            
            if recent_avg > older_avg * 1.05:  # 5% threshold
                direction = "degrading" if metric in ['avg_execution_time', 'avg_memory_usage_mb', 'timeout_rate'] else "improving"
            elif recent_avg < older_avg * 0.95:
                direction = "improving" if metric in ['avg_execution_time', 'avg_memory_usage_mb', 'timeout_rate'] else "degrading"
            else:
                direction = "stable"
            
            # Calculate average change rate
            if len(values) > 1:
                changes = [values[i] - values[i-1] for i in range(1, len(values))]
                avg_change_rate = np.mean(changes)
            else:
                avg_change_rate = 0.0
            
            trends[metric] = TrendData(
                metric_name=metric,
                values=values,
                timestamps=timestamps,
                trend_direction=direction,
                avg_change_rate=avg_change_rate
            )
        
        return trends
    
    def _save_baseline(self, name: str, baseline: BaselineMetrics):
        """Save baseline to file."""
        baseline_file = self.baseline_dir / f"{name}.json"
        with open(baseline_file, 'w') as f:
            json.dump(asdict(baseline), f, indent=2)
    
    def _calculate_model_hash(self, model_path: str) -> str:
        """Calculate hash of model file."""
        try:
            with open(model_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()[:16]
        except Exception:
            return ""
    
    def _calculate_config_hash(self, config: Dict[str, Any]) -> str:
        """Calculate hash of configuration."""
        try:
            config_str = json.dumps(config, sort_keys=True)
            return hashlib.sha256(config_str.encode()).hexdigest()[:16]
        except Exception:
            return ""


class RegressionTester:
    """High-level interface for regression testing."""
    
    def __init__(self, baseline_manager: Optional[BaselineManager] = None):
        """Initialize regression tester."""
        self.baseline_manager = baseline_manager or BaselineManager()
    
    def run_regression_test(
        self,
        test_name: str,
        test_function: callable,
        baseline_name: Optional[str] = None,
        create_baseline: bool = False,
        **test_kwargs
    ) -> RegressionResult:
        """
        Run a regression test.
        
        Args:
            test_name: Name of the test
            test_function: Function that returns (validation_results, performance_results)
            baseline_name: Baseline to compare against (default: test_name)
            create_baseline: Whether to create a new baseline
            **test_kwargs: Arguments to pass to test function
            
        Returns:
            Regression test result
        """
        baseline_name = baseline_name or test_name
        
        logger.info(f"Running regression test: {test_name}")
        
        # Run the test
        start_time = time.time()
        try:
            validation_results, performance_results = test_function(**test_kwargs)
        except Exception as e:
            logger.error(f"Test function failed: {e}")
            raise
        
        execution_time = time.time() - start_time
        logger.info(f"Test completed in {execution_time:.2f}s")
        
        # Create baseline if requested
        if create_baseline:
            self.baseline_manager.capture_baseline(
                name=baseline_name,
                validation_results=validation_results,
                performance_results=performance_results
            )
            logger.info(f"Created baseline: {baseline_name}")
        
        # Compare to baseline
        if not create_baseline:
            return self.baseline_manager.compare_to_baseline(
                baseline_name=baseline_name,
                current_results=(validation_results, performance_results)
            )
        else:
            # Return dummy success result for baseline creation
            current_metrics = self.baseline_manager.capture_baseline(
                name=f"{baseline_name}_current",
                validation_results=validation_results,
                performance_results=performance_results
            )
            
            return RegressionResult(
                baseline_name=baseline_name,
                current_metrics=current_metrics,
                baseline_metrics=current_metrics,
                performance_regression=False,
                correctness_regression=False,
                improvement=False,
                details={"baseline_created": True}
            )
    
    def continuous_monitoring(
        self,
        test_functions: Dict[str, callable],
        baseline_prefix: str = "nightly",
        **test_kwargs
    ) -> Dict[str, RegressionResult]:
        """
        Run continuous monitoring tests.
        
        Args:
            test_functions: Dictionary of test name -> test function
            baseline_prefix: Prefix for baseline names
            **test_kwargs: Arguments to pass to test functions
            
        Returns:
            Dictionary of test results
        """
        results = {}
        
        for test_name, test_function in test_functions.items():
            baseline_name = f"{baseline_prefix}_{test_name}"
            
            try:
                result = self.run_regression_test(
                    test_name=test_name,
                    test_function=test_function,
                    baseline_name=baseline_name,
                    **test_kwargs
                )
                results[test_name] = result
                
                if result.performance_regression or result.correctness_regression:
                    logger.warning(f"Regression detected in {test_name}")
                elif result.improvement:
                    logger.info(f"Performance improvement in {test_name}")
                    
            except Exception as e:
                logger.error(f"Failed to run test {test_name}: {e}")
        
        return results