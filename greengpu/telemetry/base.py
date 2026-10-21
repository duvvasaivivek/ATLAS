"""
Abstract base classes and data transfer objects for hardware telemetry.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import time
from typing import List, Optional


@dataclass
class TelemetrySnapshot:
    """Unified telemetry record captured at an instant."""
    timestamp: float = field(default_factory=time.time)
    
    # GPU Metrics
    gpu_util_core: float = 0.0          # GPU compute core utilization [0.0 - 100.0 %]
    gpu_util_memory: float = 0.0        # GPU memory bandwidth utilization [0.0 - 100.0 %]
    gpu_power_watts: float = 0.0        # Instantaneous GPU power consumption in Watts
    gpu_clock_core_mhz: int = 0         # Current graphics/SM clock frequency in MHz
    gpu_clock_memory_mhz: int = 0       # Current memory clock frequency in MHz
    gpu_temperature_c: float = 0.0      # GPU die temperature in Celsius
    
    # CPU Metrics
    cpu_util_total: float = 0.0         # Host CPU overall utilization [0.0 - 100.0 %]
    cpu_clock_mhz: float = 0.0          # Current CPU frequency in MHz
    cpu_per_core_util: List[float] = field(default_factory=list) # Per-thread utilization


class BaseTelemetryMonitor(ABC):
    """Abstract interface for system telemetry collection."""

    @abstractmethod
    def sample(self) -> TelemetrySnapshot:
        """Sample and return the latest telemetry snapshot."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Release underlying telemetry resources."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
