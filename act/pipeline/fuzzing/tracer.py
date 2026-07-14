"""
Execution tracer for ACTFuzzer with configurable detail levels.

Unified tracing architecture supporting 4 detail levels:
- Level 0: Lightweight metrics (iteration, mutation, coverage)
- Level 1: + Input tensors and seed genealogy
- Level 2: + Full layer activations
- Level 3: + Gradients and loss values

Copyright (C) 2025 SVF-tools/ACT
License: AGPLv3+
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, Optional, Any
import time
import torch

from act.pipeline.fuzzing.trace_storage import create_storage, TraceStorage
from act.pipeline.fuzzing.checker import Counterexample
from act.util.format_utils import rule


class ExecutionTracer:
    """
    Unified execution tracer for all detail levels.

    Single class handles all tracing levels - the level determines what data
    gets stored. This avoids code duplication and ensures consistent API.

    Tracing Levels:
        0: Metrics only (iteration, mutation, coverage, violations)
           ~100 bytes/iter, ~0.5% overhead

        1: + Input tensors (before/after mutation) + seed genealogy
           ~50 KB/iter, ~2% overhead with async I/O

        2: + Full activations from all layers
           ~500 KB/iter, ~10-20% overhead with async I/O

        3: + Gradients + loss values
           ~1-5 MB/iter, ~30-50% overhead with async I/O

    Performance Optimizations:
        - Async I/O by default (background writes)
        - HDF5 with gzip compression (5x smaller than JSON)
        - Sampling support (trace every Nth iteration)
        - Lazy gradient capture (only when needed)

    Example:
        >>> tracer = ExecutionTracer(
        ...     level=1,
        ...     sample_rate=10,
        ...     storage_backend="hdf5",
        ...     output_path=Path("fuzzing_results/traces.h5")
        ... )
        >>>
        >>> for iteration in range(10000):
        ...     if tracer.should_trace(iteration):
        ...         tracer.record_iteration(
        ...             iteration=iteration,
        ...             timestamp=time.time(),
        ...             mutation_strategy="gradient",
        ...             violation=None,
        ...             coverage=0.67,
        ...             coverage_delta=0.02,
        ...             energy=5.2,
        ...             seed_id="seed_123",
        ...             input_before=seed_tensor,
        ...             input_after=mutated_tensor,
        ...             ...
        ...         )
        >>>
        >>> tracer.close()
    """

    def __init__(
        self, level: int, sample_rate: int, storage_backend: str, output_path: Path
    ):
        """
        Initialize execution tracer.

        Args:
            level: Tracing detail level (0-3)
            sample_rate: Capture every Nth iteration (1 = all iterations)
            storage_backend: Storage type ("hdf5" or "json")
            output_path: Path for trace file

        Raises:
            ValueError: If level not in [0, 1, 2, 3]
        """
        if level not in [0, 1, 2, 3]:
            raise ValueError(f"Invalid trace level: {level}. Must be 0, 1, 2, or 3.")

        self.level = level
        self.sample_rate = sample_rate
        self.output_path = output_path

        # Create storage with async wrapper
        self.storage: TraceStorage = create_storage(
            backend=storage_backend,
            path=output_path,
            async_write=True,  # Always use async for performance
        )

        # Statistics
        self.traces_captured = 0
        self.traces_skipped = 0
        self.start_time = time.time()

    def should_trace(self, iteration: int) -> bool:
        """
        Check if this iteration should be traced based on sampling rate.

        Args:
            iteration: Current fuzzing iteration

        Returns:
            True if this iteration should be traced
        """
        return iteration % self.sample_rate == 0

    def record_iteration(
        self,
        iteration: int,
        timestamp: float,
        mutation_strategy: str,
        violation: Optional[Counterexample],
        coverage: float,
        coverage_delta: float,
        energy: float,
        seed_id: str,
        # Level 1+ parameters (optional)
        input_before: Optional[torch.Tensor] = None,
        input_after: Optional[torch.Tensor] = None,
        parent_id: Optional[str] = None,
        depth: Optional[int] = None,
        # Level 2+ parameters (optional)
        activations: Optional[Dict[str, torch.Tensor]] = None,
        # Level 3+ parameters (optional)
        gradients: Optional[Dict[str, torch.Tensor]] = None,
        loss_value: Optional[float] = None,
    ):
        """
        Record iteration trace with level-appropriate detail.

        All levels call this same method - the level determines what gets stored.
        This ensures consistent API and avoids code duplication.

        Args:
            iteration: Iteration number
            timestamp: Unix timestamp
            mutation_strategy: Strategy used ("gradient", "activation", etc.)
            violation: Counterexample if found, None otherwise
            coverage: Current neuron coverage (0.0-1.0)
            coverage_delta: Coverage increase this iteration
            energy: Seed energy value
            seed_id: Unique seed identifier
            input_before: Seed tensor before mutation (Level 1+)
            input_after: Mutated tensor (Level 1+)
            parent_id: Parent seed ID (Level 1+)
            depth: Mutation depth from original seeds (Level 1+)
            activations: Layer activations dict (Level 2+)
            gradients: Gradient tensors dict (Level 3+)
            loss_value: Loss value used in mutation (Level 3+)
        """
        # Build trace entry based on level
        trace = self._build_trace_entry(
            iteration=iteration,
            timestamp=timestamp,
            mutation_strategy=mutation_strategy,
            violation=violation,
            coverage=coverage,
            coverage_delta=coverage_delta,
            energy=energy,
            seed_id=seed_id,
            input_before=input_before,
            input_after=input_after,
            parent_id=parent_id,
            depth=depth,
            activations=activations,
            gradients=gradients,
            loss_value=loss_value,
        )

        # Write to storage (async, non-blocking)
        self.storage.write(trace)
        self.traces_captured += 1

    def _build_trace_entry(self, **kwargs) -> Dict[str, Any]:
        """
        Build trace entry based on current level.

        This method implements the level-based filtering to avoid storing
        unnecessary data. Higher levels progressively add more fields.

        Args:
            **kwargs: All possible trace data

        Returns:
            Dictionary with level-appropriate fields
        """
        # Level 0: Always include basic metrics
        trace = {
            "iteration": kwargs["iteration"],
            "timestamp": kwargs["timestamp"],
            "mutation_strategy": kwargs["mutation_strategy"],
            "violation_found": kwargs["violation"] is not None,
            "coverage": kwargs["coverage"],
            "coverage_delta": kwargs["coverage_delta"],
            "energy": kwargs["energy"],
            "seed_id": kwargs["seed_id"],
        }

        # Level 1+: Add input tensors and genealogy
        if self.level >= 1:
            if kwargs["input_before"] is not None:
                # Move to CPU for storage (avoid keeping GPU memory)
                tensor = kwargs["input_before"]
                trace["input_before"] = (
                    tensor.cpu() if tensor.device.type != "cpu" else tensor
                )

            if kwargs["input_after"] is not None:
                tensor = kwargs["input_after"]
                trace["input_after"] = (
                    tensor.cpu() if tensor.device.type != "cpu" else tensor
                )

            trace["parent_id"] = kwargs.get("parent_id")
            trace["depth"] = kwargs.get("depth")

        # Level 2+: Add activations
        if self.level >= 2 and kwargs.get("activations") is not None:
            trace["activations"] = {
                k: v.cpu() if v.device.type != "cpu" else v
                for k, v in kwargs["activations"].items()
            }

        # Level 3+: Add gradients
        if self.level >= 3:
            if kwargs.get("gradients") is not None:
                trace["gradients"] = {
                    k: v.cpu() if v.device.type != "cpu" else v
                    for k, v in kwargs["gradients"].items()
                }
            trace["loss_value"] = kwargs.get("loss_value")

        return trace

    def get_stats(self) -> Dict[str, Any]:
        """
        Get tracing statistics.

        Returns:
            Dictionary with tracing statistics
        """
        elapsed = time.time() - self.start_time

        return {
            "level": self.level,
            "sample_rate": self.sample_rate,
            "traces_captured": self.traces_captured,
            "traces_skipped": self.traces_skipped,
            "output_path": str(self.output_path),
            "elapsed_seconds": elapsed,
            "traces_per_second": self.traces_captured / elapsed if elapsed > 0 else 0,
        }

    def flush(self):
        """Force write buffered traces to disk."""
        self.storage.flush()

    def close(self):
        """Finalize tracing and close storage."""
        # Print statistics
        stats = self.get_stats()
        print(f"\n{rule()}")
        print(f"📊 Tracing Statistics")
        print(f"{rule()}")
        print(f"Level: {stats['level']}")
        print(f"Traces captured: {stats['traces_captured']}")
        print(f"Sample rate: 1/{stats['sample_rate']}")
        print(f"Output: {os.path.relpath(stats['output_path'])}")
        print(f"{rule()}\n")

        # Close storage (waits for async writes to complete)
        self.storage.close()
