#===- util.stats.py ----ACT Statistics -------------------------------#
#
#                 ACT: Abstract Constraints Transformer
#
# Copyright (C) <2025->  ACT Team
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Purpose:
#   Utilities for the Abstract Constraints Transformer (ACT), including memory usage tracking, verification result
#   summaries, and performance reporting across the verification framework.
#
#===----------------------------------------------------------------------===#

import os
import torch
from typing import Dict, Any, List

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class ACTStats:
    """
    Utility class for monitoring and reporting memory usage during verification.
    
    Provides comprehensive memory monitoring including system memory, process memory,
    and GPU memory statistics. Gracefully handles missing dependencies.
    """
    
    @staticmethod
    def print_memory_usage(stage_name: str = "") -> float:
        """
        Print comprehensive memory usage statistics for the current process and system.
        
        Reports process memory usage, system memory statistics, and GPU memory
        information when available. Handles missing psutil dependency gracefully.
        
        Args:
            stage_name: Optional label for the memory report stage
            
        Returns:
            float: Process memory usage in MB, or 0 if psutil is unavailable
        """
        if not PSUTIL_AVAILABLE:
            print(f"⚠️ [{stage_name}] psutil not available, cannot monitor memory")
            return 0.0
            
        try:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024

            system_memory = psutil.virtual_memory()
            total_mb = system_memory.total / 1024 / 1024
            available_mb = system_memory.available / 1024 / 1024
            used_percent = (memory_mb / total_mb) * 100

            print(f"[{stage_name}] Memory Usage:")
            print(f"Process: {memory_mb:.1f} MB ({used_percent:.1f}% of total)")
            print(f"System: {total_mb:.1f} MB total, {available_mb:.1f} MB available")

            if torch.cuda.is_available():
                gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
                gpu_cached_mb = torch.cuda.memory_reserved() / 1024 / 1024
                print(f"GPU: {gpu_memory_mb:.1f} MB allocated, {gpu_cached_mb:.1f} MB cached")

            return memory_mb
            
        except Exception as e:
            print(f"⚠️ [{stage_name}] Error monitoring memory: {e}")
            return 0.0
    
    @classmethod
    def get_memory_usage_mb(cls) -> float:
        """
        Get current process memory usage in MB without printing.
        
        Returns:
            float: Process memory usage in MB, or 0 if psutil is unavailable
        """
        if not PSUTIL_AVAILABLE:
            return 0.0
            
        try:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            return memory_info.rss / 1024 / 1024
        except Exception:
            return 0.0
    
    @classmethod
    def get_current_memory_usage(cls) -> float:
        """
        Get current memory usage in MB (GPU if available, otherwise CPU).
        
        This is an enhanced version that automatically detects the best memory
        source (GPU vs CPU) for tracking during bounds propagation.
        
        Returns:
            float: Current memory usage in MB
        """
        try:
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                # Get GPU memory usage
                gpu_memory_bytes = torch.cuda.memory_allocated()
                return gpu_memory_bytes / (1024 * 1024)  # Convert to MB
            elif PSUTIL_AVAILABLE:
                # Get CPU memory usage for current process
                process = psutil.Process(os.getpid())
                cpu_memory_bytes = process.memory_info().rss
                return cpu_memory_bytes / (1024 * 1024)  # Convert to MB
            else:
                # Fallback if psutil not available
                return 0.0
        except Exception:
            # Fallback if memory monitoring fails
            return 0.0
    
    @classmethod
    def get_gpu_memory_info(cls) -> tuple[float, float]:
        """
        Get GPU memory info (available, total) in MB.
        
        Returns:
            tuple: (available_memory_mb, total_memory_mb) - (0.0, 0.0) if no GPU
        """
        try:
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                total_memory = torch.cuda.get_device_properties(0).total_memory
                allocated_memory = torch.cuda.memory_allocated()
                available_memory = total_memory - allocated_memory
                return available_memory / (1024 * 1024), total_memory / (1024 * 1024)
            else:
                return 0.0, 0.0
        except Exception:
            return 0.0, 0.0
    
    @classmethod
    def get_cpu_memory_usage(cls) -> float:
        """
        Get CPU memory usage for current process in MB.
        
        Returns:
            float: CPU memory usage in MB, or 0.0 if psutil unavailable
        """
        try:
            if PSUTIL_AVAILABLE:
                process = psutil.Process(os.getpid())
                return process.memory_info().rss / (1024 * 1024)
            else:
                return 0.0
        except Exception:
            return 0.0
    
    @classmethod
    def is_psutil_available(cls) -> bool:
        """
        Check if psutil is available for memory monitoring.
        
        Returns:
            bool: True if psutil is available, False otherwise
        """
        return PSUTIL_AVAILABLE
    
    @staticmethod
    def print_verification_stats(prediction_stats: Dict[str, Any]) -> None:
        """
        Print verification statistics from the prediction stats.
        
        Args:
            prediction_stats: Dictionary containing various verification statistics
        """
        # Hardcode verification_result_stat_dict as empty for now
        verification_result_stat_dict = {}  
        
        print("\n📊 Verification Results Summary:")
        
        # Display known statistics
        known_keys = ['total', 'safe', 'unsafe', 'timeout', 'clean_failure']
        for key in known_keys:
            if key in prediction_stats:
                print(f"  {key}: {prediction_stats[key]}")
        
        # Display any additional statistics
        additional_stats = {k: v for k, v in prediction_stats.items() if k not in known_keys}
        if additional_stats:
            print("  Additional stats:")
            for key, value in additional_stats.items():
                print(f"    {key}: {value}")
        
        # Display verification result statistics if available
        if verification_result_stat_dict:
            print("\n🔍 Verification VerifyResult Distribution:")
            for status, count in verification_result_stat_dict.items():
                print(f"  {status}: {count}")
        
        print("-" * 50)

    @staticmethod
    def print_final_verification_summary(results: List) -> Any:
        """
        Print final verification results summary and return the overall verdict.
        
        Args:
            results: List of VerifyResult for all verified samples
            
        Returns:
            Overall VerifyResult (SAT if all safe, UNSAT if any unsafe, UNKNOWN otherwise)
        """
        # Import VerifyResult locally to avoid circular imports
        from act.input_parser.type import VerifyResult
        
        print("\n" + "🏆" + "="*70 + "🏆")
        print("📊 Final verification results summary")
        print("🏆" + "="*70 + "🏆")

        for idx, result in enumerate(results):
            print(f"Sample {idx+1}: {result.name}")

        print("-" * 60)

        sat_count = sum(1 for r in results if r == VerifyResult.SAT)
        unsat_count = sum(1 for r in results if r == VerifyResult.UNSAT)
        clean_failure_count = sum(1 for r in results if r == VerifyResult.CLEAN_FAILURE)
        unknown_count = sum(1 for r in results if r == VerifyResult.UNKNOWN)
        total_count = len(results)

        valid_count = total_count - clean_failure_count

        print("Verification statistics:")
        print(f"Total samples: {total_count}")
        print(f"✅ SAT (safe): {sat_count} ")
        print(f"❌ UNSAT (unsafe): {unsat_count} ")
        print(f"⚠️  CLEAN_FAILURE (clean prediction failed): {clean_failure_count} ")
        print(f"❓ UNKNOWN: {unknown_count} ")
        print(f"🔍 Valid verification samples: {valid_count} ")

        if valid_count > 0:
            sat_percentage = (sat_count / valid_count) * 100
            unsat_percentage = (unsat_count / valid_count) * 100
            print(f"📊 SAT over valid samples: {sat_percentage:.2f}% ({sat_count}/{valid_count})")
            print(f"📊 UNSAT over valid samples: {unsat_percentage:.2f}% ({unsat_count}/{valid_count})")
        else:
            print("  ⚠️  No valid verification samples")

        if total_count > 0:
            sat_total_percentage = (sat_count / total_count) * 100
            unsat_total_percentage = (unsat_count / total_count) * 100
            clean_failure_percentage = (clean_failure_count / total_count) * 100
            print(f"📊 SAT over total: {sat_total_percentage:.2f}% ({sat_count}/{total_count})")
            print(f"📊 UNSAT over total: {unsat_total_percentage:.2f}% ({unsat_count}/{total_count})")
            print(f"📊 CLEAN_FAILURE over total: {clean_failure_percentage:.2f}% ({clean_failure_count}/{total_count})")

        print("-" * 60)

        if all(r == VerifyResult.SAT for r in results):
            final_result = VerifyResult.SAT
            print("✅ Final Result: SAT - all samples verified safe")
        elif any(r == VerifyResult.UNSAT for r in results):
            final_result = VerifyResult.UNSAT
            print("❌ Final Result: UNSAT - at least one sample violates the property")
        else:
            final_result = VerifyResult.UNKNOWN
            print("❓ Final Result: UNKNOWN - inconclusive")

        print("🏆" + "="*70 + "🏆")
        return final_result


class ACTLog:
    """
    Centralized logging utilities for the Abstract Constraints Transformer (ACT).
    
    Provides structured logging methods for prediction validation, constraint application,
    and verification processes with consistent formatting and emoji indicators.
    """
    
    @staticmethod
    def log_correct_prediction(
        predicted_label: int, 
        ground_truth_label: int, 
        sample_index: int,
        verbose: bool = True
    ) -> None:
        """Log successful prediction with verbose output."""
        if verbose:
            print(f"✅ Sample {sample_index + 1} Clean Prediction: Correct "
                  f"(pred: {predicted_label}, true: {ground_truth_label})")
    
    @staticmethod
    def log_incorrect_prediction(
        predicted_label: int, 
        ground_truth_label: int, 
        sample_index: int,
        verbose: bool = True
    ) -> None:
        """Log failed prediction with verbose output."""
        if verbose:
            print(f"❌ Sample {sample_index + 1} Clean Prediction: Incorrect "
                  f"(pred: {predicted_label}, true: {ground_truth_label})")
            print("⚠️  Skipping verification - clean prediction already incorrect")
    
    @staticmethod
    def log_prediction_failure(error: Exception) -> None:
        """Log prediction validation failure."""
        print(f"❌ Clean prediction validation failed: {error}")
    
    @staticmethod
    def log_constraint_application_start(cache_type_name: str, constraint_count: int, verbose: bool = True) -> None:
        """Log the start of constraint application process."""
        if verbose:
            print(f"[Bounds Fix] Applying {constraint_count} ReLU constraints "
                  f"to layer bounds cache")
            print(f"[Bounds Fix] Using {cache_type_name} bounds cache")
    
    @staticmethod
    def log_constraint_processing(
        layer_name: str, 
        neuron_index: int, 
        activation_type: str,
        verbose: bool = True
    ) -> None:
        """Log individual constraint processing."""
        if verbose:
            print(f"[Bounds Fix] Processing constraint: "
                  f"{layer_name}[{neuron_index}] = {activation_type}")
    
    @staticmethod
    def log_constraint_application_complete(verbose: bool = True) -> None:
        """Log successful completion of constraint application."""
        if verbose:
            print("✅ [Bounds Fix] ReLU constraint application done")
    
    @staticmethod
    def log_no_bounds_cache_warning(verbose: bool = True) -> None:
        """Log warning when no bounds cache is available."""
        if verbose:
            print("⚠️  No layer bounds cache found, skipping constraint application")
    
    @staticmethod
    def log_label_warning(sample_index: int) -> None:
        """Log warning when ground truth label cannot be found."""
        print(f"⚠️  Could not get true label for sample {sample_index+1}, using default label 0")
    
    @staticmethod
    def log_relu_constraints_set(constraints: List, verbose: bool = True) -> None:
        """Log ReLU constraint configuration."""
        if verbose:
            if constraints:
                print(f"Set ReLU constraints: {len(constraints)} constraints")
                for constraint in constraints:
                    print(f"{constraint['layer']}[{constraint['neuron_idx']}] = {constraint['constraint_type']}")
            else:
                print(f"Cleared ReLU constraints")
    
    @staticmethod
    def log_constraint_error(message: str, verbose: bool = True) -> None:
        """Log constraint processing errors."""
        if verbose:
            print(f"⚠️  {message}")
    
    @staticmethod
    def log_constraint_bound_update(
        constraint_type: str, 
        neuron_index: int, 
        original_bound: float, 
        new_bound: float, 
        verbose: bool = True
    ) -> None:
        """Log individual bound constraint updates."""
        if verbose:
            print(f"{constraint_type} constraint: neuron {neuron_index} "
                  f"{'ub' if constraint_type == 'inactive' else 'lb'}: {original_bound:.6f} → {new_bound:.6f}")
    
    @staticmethod
    def log_verification_info(message: str) -> None:
        """Log general verification information."""
        print(f"{message}")
    
    @staticmethod
    def log_verification_warning(message: str) -> None:
        """Log verification warnings."""
        print(f"Warning: {message}")
    
    @staticmethod
    def log_bab_start(sample_idx: int) -> None:
        """Log BaB specification refinement start."""
        print(f"Starting generic BaB specification refinement verification (sample {sample_idx})")
        print(f"Framework: theoretically-aligned specification refinement")
    
    @staticmethod
    def log_bab_results(
        status_name: str, 
        total_subproblems: int, 
        spurious_count: int, 
        has_real_counterexample: bool, 
        max_depth: int, 
        total_time: float
    ) -> None:
        """Log BaB verification results."""
        print(f"Specification refinement verification finished: {status_name}")
        print(f"Total subproblems: {total_subproblems}")
        print(f"Spurious counterexamples: {spurious_count}")
        print(f"Real counterexample: {'Yes' if has_real_counterexample else 'No'}")
        print(f"Max depth: {max_depth}")
        print(f"Total time: {total_time:.2f}s")
    
    @staticmethod
    def log_bab_error(error: Exception) -> None:
        """Log BaB verification errors."""
        print(f"⚠️ Specification refinement verification error: {error}")