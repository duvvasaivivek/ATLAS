"""
Abstract Base Class and telemetry structures for Tier 1 Dynamic Workload Partitioners.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

from greengpu.config import Tier1Config


@dataclass
class PartitionStep:
    """Historical record of an individual workload division step."""
    iteration: int
    cpu_ratio: float                   # Ratio r assigned to CPU [0.0 - 1.0]
    gpu_ratio: float                   # Ratio (1 - r) assigned to GPU [0.0 - 1.0]
    time_cpu: float                    # Measured execution time on CPU (seconds)
    time_gpu: float                    # Measured execution time on GPU (seconds)
    idle_time_wasted: float            # abs(tc - tg) - energy-wasting idle waiting time
    oscillation_detected: bool = False
    notes: str = ""


class BaseWorkloadPartitioner(ABC):
    """Abstract interface for dynamic workload allocation algorithms."""

    def __init__(self, config: Optional[Tier1Config] = None):
        self.config = config or Tier1Config()
        self.current_ratio: float = self.config.initial_cpu_ratio
        self.history: List[PartitionStep] = []
        self._converged: bool = False

    @property
    def is_converged(self) -> bool:
        """Indicates whether the partitioner has reached convergence."""
        return self._converged

    @abstractmethod
    def compute_next_ratio(self, tc: float, tg: float) -> float:
        """
        Compute and return the next CPU workload ratio r' given current iteration timings.
        Args:
            tc: Measured CPU execution time (seconds).
            tg: Measured GPU execution time (seconds, including PCIe transfers).
        Returns:
            Updated CPU workload ratio r' in range [min_cpu_ratio, max_cpu_ratio].
        """
        pass

    def update(self, iteration: int, tc: float, tg: float) -> float:
        """
        Record step results, compute next ratio, and update state.
        Returns the new CPU workload ratio r' for the next iteration.
        """
        idle_wasted = abs(tc - tg)
        old_ratio = self.current_ratio

        next_ratio = self.compute_next_ratio(tc, tg)
        next_ratio = max(self.config.min_cpu_ratio, min(self.config.max_cpu_ratio, next_ratio))

        step = PartitionStep(
            iteration=iteration,
            cpu_ratio=old_ratio,
            gpu_ratio=1.0 - old_ratio,
            time_cpu=tc,
            time_gpu=tg,
            idle_time_wasted=idle_wasted,
            oscillation_detected=False
        )
        self.history.append(step)
        self.current_ratio = next_ratio

        # Check precision threshold convergence
        if abs(next_ratio - old_ratio) < self.config.precision_threshold:
            self._converged = True

        return next_ratio

    def reset(self, initial_ratio: Optional[float] = None) -> None:
        """Reset history and ratio to initial state."""
        self.current_ratio = initial_ratio if initial_ratio is not None else self.config.initial_cpu_ratio
        self.history.clear()
        self._converged = False
