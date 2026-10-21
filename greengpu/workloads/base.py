"""
Abstract Base Class and result structures for Heterogeneous CPU-GPU Workloads.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import time
from typing import Any, Dict, Optional

from greengpu.telemetry.base import TelemetrySnapshot


@dataclass
class WorkloadIterationResult:
    """Execution telemetry and timing measured for a single heterogeneous iteration."""
    iteration: int
    cpu_ratio: float                   # Workload fraction r assigned to CPU [0.0 - 1.0]
    gpu_ratio: float                   # Workload fraction (1 - r) assigned to GPU [0.0 - 1.0]
    time_cpu_sec: float                # Wall-clock time spent computing on CPU (seconds)
    time_gpu_sec: float                # Wall-clock time spent on GPU (including PCIe transfers)
    time_wall_sec: float               # Total iteration duration: max(time_cpu, time_gpu)
    idle_time_wasted_sec: float        # abs(time_cpu - time_gpu): synchronization wait penalty
    telemetry: Optional[TelemetrySnapshot] = None
    extra_metrics: Dict[str, Any] = field(default_factory=dict)


class HeterogeneousWorkload(ABC):
    """
    Abstract interface for cooperatively executed CPU-GPU workloads.
    Supports dynamic reconfiguration of work fraction r between successive iterations.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def run_iteration(self, iteration: int, cpu_ratio: float) -> WorkloadIterationResult:
        """
        Execute one iteration/reduction phase split according to cpu_ratio r.
        Args:
            iteration: Iteration sequence number (1-indexed)
            cpu_ratio: Fraction r assigned to CPU [0.0, 1.0]; (1 - r) assigned to GPU.
        Returns:
            WorkloadIterationResult with exact execution timings and idle metrics.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset internal workload state or data buffers."""
        pass
