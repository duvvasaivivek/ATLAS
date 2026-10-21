"""
Host CPU Telemetry Monitor using psutil.
Provides CPU utilization, frequency, and core metrics.
"""

import time
from typing import List, Optional

try:
    import psutil
except ImportError:
    psutil = None

from greengpu.telemetry.base import BaseTelemetryMonitor, TelemetrySnapshot


class CPUSnapshotMonitor(BaseTelemetryMonitor):
    """Monitor dedicated to host CPU telemetry."""

    def __init__(self):
        # Initial call to seed psutil cpu_percent
        if psutil is not None:
            psutil.cpu_percent(interval=None)

    def sample(self) -> TelemetrySnapshot:
        cpu_util = 0.0
        cpu_clk = 0.0
        per_core: List[float] = []

        if psutil is not None:
            cpu_util = psutil.cpu_percent(interval=None)
            per_core = psutil.cpu_percent(interval=None, percpu=True)
            freq = psutil.cpu_freq()
            cpu_clk = freq.current if freq else 0.0

        return TelemetrySnapshot(
            timestamp=time.time(),
            cpu_util_total=cpu_util,
            cpu_clock_mhz=cpu_clk,
            cpu_per_core_util=per_core
        )

    def close(self) -> None:
        pass
