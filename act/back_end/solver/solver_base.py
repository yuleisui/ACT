
#===- act/back_end/solver/solver_base.py - Base Solver Interface -------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Base Solver Interface. Defines abstract base class and common interfaces
#   for constraint satisfaction problem solvers.
#
#===---------------------------------------------------------------------===#

from __future__ import annotations
import numpy as np
from typing import List, Optional

class SolveStatus:
    OPTIMAL = "OPTIMAL"
    FEASIBLE = "FEASIBLE"
    INFEASIBLE = "INFEASIBLE"
    UNKNOWN = "UNKNOWN"

class SolverCaps:
    def __init__(self, supports_gpu: bool = False):
        self.supports_gpu = supports_gpu

class Solver:
    """Abstract solver interface used by the exporter and verification pipeline."""

    # --- Capabilities / lifecycle ---
    def capabilities(self) -> SolverCaps:  # pragma: no cover - abstract
        return SolverCaps(False)

    def begin(self, name: str = "verify", device: Optional[str] = None):  # pragma: no cover - abstract
        ...

    def add_vars(self, n: int) -> None:  # pragma: no cover - abstract
        ...

    def set_bounds(self, idxs: List[int], lb: np.ndarray, ub: np.ndarray) -> None:  # pragma: no cover - abstract
        ...

    def add_binary_vars(self, n: int) -> List[int]:  # pragma: no cover - abstract
        ...

    # --- Linear constraints ---
    def add_lin_eq(self, vids: List[int], coeffs: List[float], rhs: float) -> None:  # pragma: no cover - abstract
        ...

    def add_lin_ge(self, vids: List[int], coeffs: List[float], rhs: float) -> None:  # pragma: no cover - abstract
        ...

    def add_lin_le(self, vids: List[int], coeffs: List[float], rhs: float) -> None:  # pragma: no cover - abstract
        ...

    # --- Convenience constraints ---
    def add_sum_eq(self, vids: List[int], rhs: float) -> None:  # pragma: no cover - abstract
        ...

    def add_ge_zero(self, vids: List[int]) -> None:  # pragma: no cover - abstract
        ...

    def add_sos2(self, var_ids: List[int], weights: Optional[List[float]] = None) -> None:  # pragma: no cover - abstract
        ...

    # --- Objective & solve ---
    def set_objective_linear(self, vids: List[int], coeffs: List[float], const: float = 0.0, sense: str = "min") -> None:  # pragma: no cover - abstract
        ...

    def optimize(self, timelimit: Optional[float] = None) -> None:  # pragma: no cover - abstract
        ...

    def status(self) -> str:  # pragma: no cover - abstract
        ...

    def has_solution(self) -> bool:  # pragma: no cover - abstract
        ...

    # --- Accessors ---
    def get_values(self, vids: List[int]) -> np.ndarray:  # pragma: no cover - abstract
        ...

    @property
    def n(self) -> int:  # pragma: no cover - abstract
        ...

    def get_counterexample(self, input_ids: List[int]) -> np.ndarray:
        """Return a concrete DNN input candidate ordered by input_ids.
        Default proxies to get_values(input_ids). Backends may override to
        perform rounding or clipping before returning.
        """
        return self.get_values(input_ids)
