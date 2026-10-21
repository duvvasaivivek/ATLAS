"""
Mock Telemetry Monitor for testing, CI, and deterministic trace playback.
Simulates hardware workload signatures from the GreenGPU research paper.
"""

import time
from typing import Callable, Optional

from greengpu.telemetry.base import BaseTelemetryMonitor, TelemetrySnapshot


class MockTelemetryMonitor(BaseTelemetryMonitor):
    """
    Simulates hardware utilization and power for reproducible unit testing.
    Can emulate different workload profiles (core-bound, memory-bound, fluctuating).
    """

    def __init__(
        self,
        gpu_util_core: float = 75.0,
        gpu_util_memory: float = 20.0,
        gpu_power_watts: float = 45.0,
        gpu_clock_core_mhz: int = 1605,
        gpu_clock_memory_mhz: int = 7001,
        cpu_util_total: float = 30.0,
        dynamic_generator: Optional[Callable[[int], TelemetrySnapshot]] = None
    ):
        self.gpu_util_core = gpu_util_core
        self.gpu_util_memory = gpu_util_memory
        self.gpu_power_watts = gpu_power_watts
        self.gpu_clock_core_mhz = gpu_clock_core_mhz
        self.gpu_clock_memory_mhz = gpu_clock_memory_mhz
        self.cpu_util_total = cpu_util_total
        self.dynamic_generator = dynamic_generator
        self.step_count = 0

    def sample(self) -> TelemetrySnapshot:
        self.step_count += 1
        if self.dynamic_generator is not None:
            return self.dynamic_generator(self.step_count)

        return TelemetrySnapshot(
            timestamp=time.time(),
            gpu_util_core=self.gpu_util_core,
            gpu_util_memory=self.gpu_util_memory,
            gpu_power_watts=self.gpu_power_watts,
            gpu_clock_core_mhz=self.gpu_clock_core_mhz,
            gpu_clock_memory_mhz=self.gpu_clock_memory_mhz,
            cpu_util_total=self.cpu_util_total,
            cpu_clock_mhz=2000.0,
            cpu_per_core_util=[self.cpu_util_total] * 8
        )

    def close(self) -> None:
        pass
