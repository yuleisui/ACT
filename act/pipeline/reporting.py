#===- act/pipeline/reporting.py - Result Analysis and Reporting ---------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Result analysis and report generation for ACT pipeline testing.
#   Provides comprehensive analysis and reporting capabilities for test results,
#   including performance analysis, correctness summaries, regression reports.
#
#===---------------------------------------------------------------------===#


import os
import json
import time
import numpy as np

# Optional plotting dependencies
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    plt = None
    sns = None
    # Note: Plotting functionality will be disabled. To enable plots, run:
    # pip install matplotlib seaborn

from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
import logging
from collections import defaultdict

from act.pipeline.correctness import ValidationResult, PerformanceResult, VerifyResult
from act.pipeline.regression import RegressionResult, TrendData, BaselineMetrics

logger = logging.getLogger(__name__)

# Set matplotlib backend for headless environments
if PLOTTING_AVAILABLE:
    plt.switch_backend('Agg')


@dataclass
class ReportConfig:
    """Configuration for report generation."""
    output_dir: str = "reports"  # Will be resolved relative to pipeline directory
    include_plots: bool = True
    include_details: bool = True
    format: str = "html"  # html, json, markdown
    theme: str = "default"
    timestamp: bool = True


@dataclass
class TestSummary:
    """Summary of test execution."""
    total_tests: int
    passed_tests: int
    failed_tests: int
    success_rate: float
    avg_execution_time: float
    total_execution_time: float
    memory_usage_stats: Dict[str, float]
    result_distribution: Dict[str, int]
    error_summary: Dict[str, int] = field(default_factory=dict)


@dataclass
class PerformanceSummary:
    """Summary of performance metrics."""
    avg_execution_time: float
    min_execution_time: float
    max_execution_time: float
    std_execution_time: float
    avg_memory_usage_mb: float
    max_memory_usage_mb: float
    avg_cpu_usage: float
    gpu_usage: Optional[Dict[str, float]] = None
    bottlenecks: List[str] = field(default_factory=list)


@dataclass
class RegressionSummary:
    """Summary of regression analysis."""
    total_comparisons: int
    regressions_detected: int
    improvements_detected: int
    performance_regressions: int
    correctness_regressions: int
    critical_violations: List[str] = field(default_factory=list)


class ReportGenerator:
    """Generate comprehensive test reports."""
    
    def __init__(self, config: Optional[ReportConfig] = None):
        """
        Initialize report generator.
        
        Args:
            config: Report configuration
        """
        self.config = config or ReportConfig()
        # Resolve output directory relative to pipeline folder
        pipeline_dir = Path(__file__).parent
        self.output_dir = pipeline_dir / self.config.output_dir
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup plotting style
        if self.config.include_plots and PLOTTING_AVAILABLE:
            self._setup_plotting()
        elif self.config.include_plots and not PLOTTING_AVAILABLE:
            logger.warning("Plotting requested but matplotlib/seaborn not available. Plots will be skipped.")
            logger.info("To enable plotting, install the required packages: pip install matplotlib seaborn")
    
    def _setup_plotting(self):
        """Setup matplotlib and seaborn styling."""
        if not PLOTTING_AVAILABLE:
            return
            
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Configure matplotlib for better reports
        plt.rcParams.update({
            'figure.figsize': (10, 6),
            'figure.dpi': 100,
            'savefig.dpi': 150,
            'savefig.bbox': 'tight',
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
        })
    
    def generate_test_summary(
        self,
        validation_results: List[ValidationResult],
        performance_results: List[PerformanceResult]
    ) -> TestSummary:
        """
        Generate test execution summary.
        
        Args:
            validation_results: List of validation results
            performance_results: List of performance results
            
        Returns:
            Test summary
        """
        # Aggregate validation results
        total_tests = sum(r.total_tests for r in validation_results)
        passed_tests = sum(r.passed_tests for r in validation_results)
        failed_tests = sum(r.failed_tests for r in validation_results)
        success_rate = passed_tests / total_tests if total_tests > 0 else 0.0
        total_exec_time = sum(r.execution_time for r in validation_results)
        avg_exec_time = total_exec_time / len(validation_results) if validation_results else 0.0
        
        # Memory usage statistics
        memory_values = [r.memory_usage_mb for r in performance_results if r.memory_usage_mb > 0]
        memory_stats = {
            "avg": np.mean(memory_values) if memory_values else 0.0,
            "max": np.max(memory_values) if memory_values else 0.0,
            "min": np.min(memory_values) if memory_values else 0.0,
            "std": np.std(memory_values) if memory_values else 0.0,
        }
        
        # Result distribution
        result_counts = defaultdict(int)
        error_counts = defaultdict(int)
        
        for val_result in validation_results:
            for test_result in val_result.results:
                result_type = test_result.get('result', 'UNKNOWN')
                result_counts[result_type] += 1
                
                if 'error' in test_result:
                    error_type = type(test_result['error']).__name__
                    error_counts[error_type] += 1
        
        return TestSummary(
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            success_rate=success_rate,
            avg_execution_time=avg_exec_time,
            total_execution_time=total_exec_time,
            memory_usage_stats=memory_stats,
            result_distribution=dict(result_counts),
            error_summary=dict(error_counts)
        )
    
    def generate_performance_summary(
        self,
        performance_results: List[PerformanceResult]
    ) -> PerformanceSummary:
        """
        Generate performance analysis summary.
        
        Args:
            performance_results: List of performance results
            
        Returns:
            Performance summary
        """
        if not performance_results:
            return PerformanceSummary(
                avg_execution_time=0.0,
                min_execution_time=0.0,
                max_execution_time=0.0,
                std_execution_time=0.0,
                avg_memory_usage_mb=0.0,
                max_memory_usage_mb=0.0,
                avg_cpu_usage=0.0
            )
        
        exec_times = [r.execution_time for r in performance_results]
        memory_usage = [r.memory_usage_mb for r in performance_results if r.memory_usage_mb > 0]
        cpu_usage = [r.cpu_usage_percent for r in performance_results if r.cpu_usage_percent > 0]
        
        # Identify bottlenecks
        bottlenecks = []
        if exec_times:
            avg_time = np.mean(exec_times)
            slow_tests = [r.test_name for r in performance_results if r.execution_time > avg_time * 2]
            if slow_tests:
                bottlenecks.append(f"Slow tests: {', '.join(slow_tests[:3])}")
        
        if memory_usage:
            avg_memory = np.mean(memory_usage)
            memory_heavy = [r.test_name for r in performance_results if r.memory_usage_mb > avg_memory * 2]
            if memory_heavy:
                bottlenecks.append(f"Memory-heavy tests: {', '.join(memory_heavy[:3])}")
        
        # GPU usage summary
        gpu_usage = None
        gpu_memory_values = [r.gpu_memory_mb for r in performance_results if r.gpu_memory_mb]
        if gpu_memory_values:
            gpu_usage = {
                "avg_memory_mb": np.mean(gpu_memory_values),
                "max_memory_mb": np.max(gpu_memory_values),
                "utilization_rate": len(gpu_memory_values) / len(performance_results)
            }
        
        return PerformanceSummary(
            avg_execution_time=np.mean(exec_times) if exec_times else 0.0,
            min_execution_time=np.min(exec_times) if exec_times else 0.0,
            max_execution_time=np.max(exec_times) if exec_times else 0.0,
            std_execution_time=np.std(exec_times) if exec_times else 0.0,
            avg_memory_usage_mb=np.mean(memory_usage) if memory_usage else 0.0,
            max_memory_usage_mb=np.max(memory_usage) if memory_usage else 0.0,
            avg_cpu_usage=np.mean(cpu_usage) if cpu_usage else 0.0,
            gpu_usage=gpu_usage,
            bottlenecks=bottlenecks
        )
    
    def generate_regression_summary(
        self,
        regression_results: List[RegressionResult]
    ) -> RegressionSummary:
        """
        Generate regression analysis summary.
        
        Args:
            regression_results: List of regression results
            
        Returns:
            Regression summary
        """
        if not regression_results:
            return RegressionSummary(
                total_comparisons=0,
                regressions_detected=0,
                improvements_detected=0,
                performance_regressions=0,
                correctness_regressions=0
            )
        
        total_comparisons = len(regression_results)
        performance_regressions = sum(1 for r in regression_results if r.performance_regression)
        correctness_regressions = sum(1 for r in regression_results if r.correctness_regression)
        improvements = sum(1 for r in regression_results if r.improvement)
        total_regressions = sum(1 for r in regression_results if r.performance_regression or r.correctness_regression)
        
        # Critical violations
        critical_violations = []
        for result in regression_results:
            for violation in result.threshold_violations:
                if "accuracy decreased" in violation.lower() or "timeout" in violation.lower():
                    critical_violations.append(f"{result.baseline_name}: {violation}")
        
        return RegressionSummary(
            total_comparisons=total_comparisons,
            regressions_detected=total_regressions,
            improvements_detected=improvements,
            performance_regressions=performance_regressions,
            correctness_regressions=correctness_regressions,
            critical_violations=critical_violations
        )
    
    def create_performance_plots(
        self,
        performance_results: List[PerformanceResult],
        output_dir: Optional[Path] = None
    ) -> List[str]:
        """
        Create performance visualization plots.
        
        Args:
            performance_results: List of performance results
            output_dir: Directory to save plots
            
        Returns:
            List of generated plot filenames
        """
        if not self.config.include_plots or not performance_results or not PLOTTING_AVAILABLE:
            if not PLOTTING_AVAILABLE and self.config.include_plots:
                logger.warning("Plotting requested but matplotlib not available")
                logger.info("Install plotting dependencies with: pip install matplotlib seaborn")
            return []
        
        output_dir = output_dir or self.output_dir
        plot_files = []
        
        try:
            # Execution time distribution
            exec_times = [r.execution_time for r in performance_results]
            test_names = [r.test_name for r in performance_results]
            
            if exec_times:
                plt.figure(figsize=(12, 6))
                plt.subplot(1, 2, 1)
                plt.hist(exec_times, bins=20, alpha=0.7, edgecolor='black')
                plt.xlabel('Execution Time (s)')
                plt.ylabel('Frequency')
                plt.title('Execution Time Distribution')
                
                plt.subplot(1, 2, 2)
                top_indices = np.argsort(exec_times)[-10:]  # Top 10 slowest
                plt.barh([test_names[i][:20] for i in top_indices], [exec_times[i] for i in top_indices])
                plt.xlabel('Execution Time (s)')
                plt.title('Slowest Tests')
                plt.tight_layout()
                
                plot_file = output_dir / "execution_time_analysis.png"
                plt.savefig(plot_file)
                plt.close()
                plot_files.append(str(plot_file))
            
            # Memory usage analysis
            memory_values = [r.memory_usage_mb for r in performance_results if r.memory_usage_mb > 0]
            if memory_values:
                plt.figure(figsize=(10, 6))
                plt.plot(range(len(memory_values)), memory_values, marker='o', alpha=0.7)
                plt.xlabel('Test Index')
                plt.ylabel('Memory Usage (MB)')
                plt.title('Memory Usage Over Tests')
                plt.axhline(y=np.mean(memory_values), color='r', linestyle='--', label=f'Average: {np.mean(memory_values):.1f} MB')
                plt.legend()
                
                plot_file = output_dir / "memory_usage_trend.png"
                plt.savefig(plot_file)
                plt.close()
                plot_files.append(str(plot_file))
            
            # Success rate by performance
            success_data = [(r.execution_time, r.memory_usage_mb, r.success) for r in performance_results]
            if success_data:
                exec_times, memories, successes = zip(*success_data)
                
                plt.figure(figsize=(10, 6))
                colors = ['green' if s else 'red' for s in successes]
                plt.scatter(exec_times, memories, c=colors, alpha=0.6)
                plt.xlabel('Execution Time (s)')
                plt.ylabel('Memory Usage (MB)')
                plt.title('Test Success by Performance Metrics')
                plt.legend(['Success', 'Failure'])
                
                plot_file = output_dir / "performance_success_correlation.png"
                plt.savefig(plot_file)
                plt.close()
                plot_files.append(str(plot_file))
        
        except Exception as e:
            logger.warning(f"Failed to create performance plots: {e}")
        
        return plot_files
    
    def create_regression_plots(
        self,
        trend_data: Dict[str, TrendData],
        output_dir: Optional[Path] = None
    ) -> List[str]:
        """
        Create regression trend plots.
        
        Args:
            trend_data: Dictionary of trend data
            output_dir: Directory to save plots
            
        Returns:
            List of generated plot filenames
        """
        if not self.config.include_plots or not trend_data:
            return []
        
        output_dir = output_dir or self.output_dir
        plot_files = []
        
        try:
            # Trend plots for each metric
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.flatten()
            
            for i, (metric_name, trend) in enumerate(trend_data.items()):
                if i >= len(axes):
                    break
                
                ax = axes[i]
                timestamps = [datetime.fromisoformat(ts) for ts in trend.timestamps]
                
                ax.plot(timestamps, trend.values, marker='o', linewidth=2)
                ax.set_title(f'{metric_name.replace("_", " ").title()} Trend')
                ax.set_xlabel('Time')
                ax.set_ylabel(metric_name.replace("_", " ").title())
                ax.grid(True, alpha=0.3)
                
                # Color code by trend direction
                if trend.trend_direction == "improving":
                    ax.plot(timestamps, trend.values, color='green', marker='o', linewidth=2)
                elif trend.trend_direction == "degrading":
                    ax.plot(timestamps, trend.values, color='red', marker='o', linewidth=2)
                else:
                    ax.plot(timestamps, trend.values, color='blue', marker='o', linewidth=2)
                
                # Rotate x-axis labels for readability
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            plt.tight_layout()
            plot_file = output_dir / "regression_trends.png"
            plt.savefig(plot_file)
            plt.close()
            plot_files.append(str(plot_file))
        
        except Exception as e:
            logger.warning(f"Failed to create regression plots: {e}")
        
        return plot_files
    
    def generate_html_report(
        self,
        test_summary: TestSummary,
        performance_summary: PerformanceSummary,
        regression_summary: Optional[RegressionSummary] = None,
        plot_files: Optional[List[str]] = None,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate HTML test report.
        
        Args:
            test_summary: Test execution summary
            performance_summary: Performance summary
            regression_summary: Regression summary
            plot_files: List of plot file paths
            additional_data: Additional data to include
            
        Returns:
            Path to generated HTML report
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        plot_files = plot_files or []
        additional_data = additional_data or {}
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>ACT Pipeline Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
        .header {{ background-color: #f4f4f4; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .success {{ color: green; }}
        .failure {{ color: red; }}
        .warning {{ color: orange; }}
        .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #f9f9f9; border-radius: 3px; }}
        .plot {{ text-align: center; margin: 20px 0; }}
        .plot img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .status-pass {{ background-color: #d4edda; }}
        .status-fail {{ background-color: #f8d7da; }}
        .status-warn {{ background-color: #fff3cd; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>ACT Pipeline Test Report</h1>
        <p><strong>Generated:</strong> {timestamp}</p>
        <p><strong>Test Suite:</strong> Abstraction Verifier Validation</p>
    </div>
    
    <div class="section">
        <h2>Test Execution Summary</h2>
        <div class="metric">
            <strong>Total Tests:</strong> {test_summary.total_tests}
        </div>
        <div class="metric">
            <strong>Passed:</strong> <span class="success">{test_summary.passed_tests}</span>
        </div>
        <div class="metric">
            <strong>Failed:</strong> <span class="failure">{test_summary.failed_tests}</span>
        </div>
        <div class="metric">
            <strong>Success Rate:</strong> {test_summary.success_rate:.1%}
        </div>
        <div class="metric">
            <strong>Total Execution Time:</strong> {test_summary.total_execution_time:.2f}s
        </div>
        <div class="metric">
            <strong>Average Time per Test:</strong> {test_summary.avg_execution_time:.2f}s
        </div>
        
        <h3>Result Distribution</h3>
        <table>
            <tr><th>Result Type</th><th>Count</th><th>Percentage</th></tr>
        """
        
        for result_type, count in test_summary.result_distribution.items():
            percentage = (count / test_summary.total_tests * 100) if test_summary.total_tests > 0 else 0
            html_content += f"<tr><td>{result_type}</td><td>{count}</td><td>{percentage:.1f}%</td></tr>"
        
        html_content += """
        </table>
    </div>
    
    <div class="section">
        <h2>Performance Summary</h2>
        <div class="metric">
            <strong>Average Execution Time:</strong> {:.2f}s
        </div>
        <div class="metric">
            <strong>Execution Time Range:</strong> {:.2f}s - {:.2f}s
        </div>
        <div class="metric">
            <strong>Average Memory Usage:</strong> {:.1f} MB
        </div>
        <div class="metric">
            <strong>Peak Memory Usage:</strong> {:.1f} MB
        </div>
        <div class="metric">
            <strong>Average CPU Usage:</strong> {:.1f}%
        </div>
        """.format(
            performance_summary.avg_execution_time,
            performance_summary.min_execution_time,
            performance_summary.max_execution_time,
            performance_summary.avg_memory_usage_mb,
            performance_summary.max_memory_usage_mb,
            performance_summary.avg_cpu_usage
        )
        
        if performance_summary.gpu_usage:
            html_content += f"""
        <div class="metric">
            <strong>GPU Memory Usage:</strong> {performance_summary.gpu_usage['avg_memory_mb']:.1f} MB (avg)
        </div>
        <div class="metric">
            <strong>GPU Utilization:</strong> {performance_summary.gpu_usage['utilization_rate']:.1%}
        </div>
            """
        
        if performance_summary.bottlenecks:
            html_content += "<h3>Performance Bottlenecks</h3><ul>"
            for bottleneck in performance_summary.bottlenecks:
                html_content += f"<li class='warning'>{bottleneck}</li>"
            html_content += "</ul>"
        
        html_content += "</div>"
        
        # Regression summary
        if regression_summary:
            status_class = "status-fail" if regression_summary.regressions_detected > 0 else "status-pass"
            html_content += f"""
    <div class="section {status_class}">
        <h2>Regression Analysis</h2>
        <div class="metric">
            <strong>Total Comparisons:</strong> {regression_summary.total_comparisons}
        </div>
        <div class="metric">
            <strong>Regressions Detected:</strong> <span class="failure">{regression_summary.regressions_detected}</span>
        </div>
        <div class="metric">
            <strong>Performance Regressions:</strong> <span class="failure">{regression_summary.performance_regressions}</span>
        </div>
        <div class="metric">
            <strong>Correctness Regressions:</strong> <span class="failure">{regression_summary.correctness_regressions}</span>
        </div>
        <div class="metric">
            <strong>Improvements Detected:</strong> <span class="success">{regression_summary.improvements_detected}</span>
        </div>
            """
            
            if regression_summary.critical_violations:
                html_content += "<h3>Critical Violations</h3><ul>"
                for violation in regression_summary.critical_violations:
                    html_content += f"<li class='failure'>{violation}</li>"
                html_content += "</ul>"
            
            html_content += "</div>"
        
        # Plots
        if plot_files:
            html_content += '<div class="section"><h2>Performance Visualizations</h2>'
            for plot_file in plot_files:
                plot_name = Path(plot_file).stem.replace('_', ' ').title()
                html_content += f'<div class="plot"><h3>{plot_name}</h3><img src="{plot_file}" alt="{plot_name}"></div>'
            html_content += "</div>"
        
        # Additional data
        if additional_data:
            html_content += '<div class="section"><h2>Additional Information</h2>'
            for key, value in additional_data.items():
                html_content += f"<div class='metric'><strong>{key}:</strong> {value}</div>"
            html_content += "</div>"
        
        html_content += """
</body>
</html>
        """
        
        # Save report
        report_file = self.output_dir / f"test_report_{int(time.time())}.html"
        with open(report_file, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Generated HTML report: {report_file}")
        return str(report_file)
    
    def generate_json_report(
        self,
        test_summary: TestSummary,
        performance_summary: PerformanceSummary,
        regression_summary: Optional[RegressionSummary] = None,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate JSON test report.
        
        Args:
            test_summary: Test execution summary
            performance_summary: Performance summary
            regression_summary: Regression summary
            additional_data: Additional data to include
            
        Returns:
            Path to generated JSON report
        """
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "test_summary": asdict(test_summary),
            "performance_summary": asdict(performance_summary),
            "additional_data": additional_data or {}
        }
        
        if regression_summary:
            report_data["regression_summary"] = asdict(regression_summary)
        
        report_file = self.output_dir / f"test_report_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"Generated JSON report: {report_file}")
        return str(report_file)
    
    def generate_complete_report(
        self,
        validation_results: List[ValidationResult],
        performance_results: List[PerformanceResult],
        regression_results: Optional[List[RegressionResult]] = None,
        trend_data: Optional[Dict[str, TrendData]] = None,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """
        Generate complete test report with all components.
        
        Args:
            validation_results: List of validation results
            performance_results: List of performance results
            regression_results: List of regression results
            trend_data: Historical trend data
            additional_data: Additional data to include
            
        Returns:
            Dictionary of generated report files
        """
        logger.info("Generating complete test report")
        
        # Generate summaries
        test_summary = self.generate_test_summary(validation_results, performance_results)
        performance_summary = self.generate_performance_summary(performance_results)
        
        regression_summary = None
        if regression_results:
            regression_summary = self.generate_regression_summary(regression_results)
        
        # Generate plots
        plot_files = []
        if self.config.include_plots:
            plot_files.extend(self.create_performance_plots(performance_results))
            if trend_data:
                plot_files.extend(self.create_regression_plots(trend_data))
        
        # Generate reports
        generated_files = {}
        
        if self.config.format in ["html", "all"]:
            html_file = self.generate_html_report(
                test_summary, performance_summary, regression_summary, plot_files, additional_data
            )
            generated_files["html"] = html_file
        
        if self.config.format in ["json", "all"]:
            json_file = self.generate_json_report(
                test_summary, performance_summary, regression_summary, additional_data
            )
            generated_files["json"] = json_file
        
        logger.info(f"Report generation complete. Files: {list(generated_files.values())}")
        return generated_files


class DashboardGenerator:
    """Generate interactive dashboards for test results."""
    
    def __init__(self, output_dir: str = "dashboard"):
        """Initialize dashboard generator."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def create_live_dashboard(
        self,
        data_source: str,
        refresh_interval: int = 30
    ) -> str:
        """
        Create live updating dashboard.
        
        Args:
            data_source: Path to data source
            refresh_interval: Refresh interval in seconds
            
        Returns:
            Path to dashboard HTML file
        """
        dashboard_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>ACT Pipeline Live Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
        .dashboard-header {{ background-color: #2c3e50; color: white; padding: 20px; margin: -20px -20px 20px -20px; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
        .metric-card {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; text-align: center; }}
        .plot-container {{ margin: 20px 0; height: 400px; }}
        .status-indicator {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 5px; }}
        .status-pass {{ background-color: #28a745; }}
        .status-fail {{ background-color: #dc3545; }}
        .status-warn {{ background-color: #ffc107; }}
    </style>
</head>
<body>
    <div class="dashboard-header">
        <h1>ACT Pipeline Live Dashboard</h1>
        <p>Last updated: <span id="last-update"></span></p>
        <p><span class="status-indicator status-pass"></span> Auto-refresh every {refresh_interval}s</p>
    </div>
    
    <div class="metrics-grid">
        <div class="metric-card">
            <h3>Test Success Rate</h3>
            <div id="success-rate" style="font-size: 2em; font-weight: bold;">--</div>
        </div>
        <div class="metric-card">
            <h3>Average Execution Time</h3>
            <div id="avg-time" style="font-size: 2em; font-weight: bold;">--</div>
        </div>
        <div class="metric-card">
            <h3>Memory Usage</h3>
            <div id="memory-usage" style="font-size: 2em; font-weight: bold;">--</div>
        </div>
        <div class="metric-card">
            <h3>Active Tests</h3>
            <div id="active-tests" style="font-size: 2em; font-weight: bold;">--</div>
        </div>
    </div>
    
    <div class="plot-container" id="performance-trend"></div>
    <div class="plot-container" id="success-rate-trend"></div>
    
    <script>
        function updateDashboard() {{
            // Simulated data update - replace with actual data fetching
            document.getElementById('last-update').textContent = new Date().toLocaleString();
            document.getElementById('success-rate').textContent = Math.floor(Math.random() * 20 + 80) + '%';
            document.getElementById('avg-time').textContent = (Math.random() * 10 + 5).toFixed(2) + 's';
            document.getElementById('memory-usage').textContent = Math.floor(Math.random() * 500 + 100) + 'MB';
            document.getElementById('active-tests').textContent = Math.floor(Math.random() * 10 + 1);
        }}
        
        // Initialize dashboard
        updateDashboard();
        setInterval(updateDashboard, {refresh_interval * 1000});
        
        // Create sample plots
        Plotly.newPlot('performance-trend', [{{
            x: ['Test 1', 'Test 2', 'Test 3', 'Test 4', 'Test 5'],
            y: [1.2, 2.1, 1.8, 2.5, 1.9],
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Execution Time'
        }}], {{
            title: 'Performance Trend',
            xaxis: {{ title: 'Test Cases' }},
            yaxis: {{ title: 'Time (s)' }}
        }});
    </script>
</body>
</html>
        """
        
        dashboard_file = self.output_dir / "live_dashboard.html"
        with open(dashboard_file, 'w') as f:
            f.write(dashboard_html)
        
        logger.info(f"Created live dashboard: {dashboard_file}")
        return str(dashboard_file)