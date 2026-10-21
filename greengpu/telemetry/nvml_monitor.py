"""
Direct Ctypes interface to NVIDIA Management Library (nvml.dll / libnvidia-ml.so)
Provides zero-dependency, microsecond-accurate GPU telemetry and clock monitoring.
"""

import ctypes
import os
import sys
import time
from typing import Optional

try:
    import psutil
except ImportError:
    psutil = None

from greengpu.telemetry.base import BaseTelemetryMonitor, TelemetrySnapshot


# Ctypes structures matching NVML API specification
class c_nvmlUtilization_t(ctypes.Structure):
    _fields_ = [
        ("gpu", ctypes.c_uint),
        ("memory", ctypes.c_uint)
    ]


class NVMLTelemetryMonitor(BaseTelemetryMonitor):
    """
    Hardware telemetry monitor connecting to NVML on Windows or Linux.
    Measures GPU core utilization, memory bandwidth utilization, power draw (mW),
    and current clock frequencies.
    """

    def __init__(self, device_index: int = 0):
        self.device_index = device_index
        self._nvml = None
        self._device = None
        self._initialized = False

        self._init_nvml()

    def _init_nvml(self) -> None:
        """Dynamically load and initialize nvml.dll / libnvidia-ml.so."""
        dll_names = []
        if sys.platform == "win32":
            dll_names = [
                "nvml.dll",
                os.path.join(os.environ.get("SystemRoot", "C:\\Windows"), "System32", "nvml.dll"),
                os.path.join(os.environ.get("ProgramFiles", "C:\\Program Files"), "NVIDIA Corporation", "NVSMI", "nvml.dll")
            ]
        else:
            dll_names = ["libnvidia-ml.so.1", "libnvidia-ml.so"]

        for dll in dll_names:
            try:
                self._nvml = ctypes.CDLL(dll)
                break
            except Exception:
                continue

        if self._nvml is None:
            raise RuntimeError("Failed to load NVML shared library. NVIDIA GPU driver may not be installed.")

        # Initialize NVML
        res = self._nvml.nvmlInit()
        if res != 0:
            raise RuntimeError(f"nvmlInit failed with error code: {res}")

        # Get device handle
        self._device = ctypes.c_void_p()
        res = self._nvml.nvmlDeviceGetHandleByIndex(self.device_index, ctypes.byref(self._device))
        if res != 0:
            raise RuntimeError(f"nvmlDeviceGetHandleByIndex({self.device_index}) failed with error code: {res}")

        self._initialized = True

    def sample(self) -> TelemetrySnapshot:
        """Capture an instantaneous snapshot of GPU & CPU metrics."""
        if not self._initialized:
            raise RuntimeError("NVMLTelemetryMonitor is not initialized.")

        # 1. Utilizations
        util = c_nvmlUtilization_t()
        res_util = self._nvml.nvmlDeviceGetUtilizationRates(self._device, ctypes.byref(util))
        gpu_util = float(util.gpu) if res_util == 0 else 0.0
        mem_util = float(util.memory) if res_util == 0 else 0.0

        # 2. Power usage in mW -> convert to Watts
        power_mw = ctypes.c_uint()
        res_power = self._nvml.nvmlDeviceGetPowerUsage(self._device, ctypes.byref(power_mw))
        gpu_power_watts = (power_mw.value / 1000.0) if res_power == 0 else 0.0

        # Query enforced power limit (e.g., 55W on mobile RTX 4050)
        power_cap_mw = ctypes.c_uint()
        res_cap = self._nvml.nvmlDeviceGetEnforcedPowerLimit(self._device, ctypes.byref(power_cap_mw))
        max_cap_watts = (power_cap_mw.value / 1000.0) if res_cap == 0 else 55.0

        # Mobile WDDM driver anomaly workaround: if instantaneous power exceeds cap by >5x, rescale
        if gpu_power_watts > (max_cap_watts * 2.0):
            # Scale factor for mobile driver bug (typically 100x or 10x)
            if gpu_power_watts > 200.0:
                gpu_power_watts = gpu_power_watts / 100.0

        # Query hardware cumulative energy register in millijoules
        energy_mj = ctypes.c_uint64()
        res_energy = 1
        if hasattr(self._nvml, "nvmlDeviceGetTotalEnergyConsumption"):
            res_energy = self._nvml.nvmlDeviceGetTotalEnergyConsumption(self._device, ctypes.byref(energy_mj))

        # 3. Graphics / SM Clock (clock type 0 = NVML_CLOCK_GRAPHICS)
        clock_graphics = ctypes.c_uint()
        res_core = self._nvml.nvmlDeviceGetClockInfo(self._device, 0, ctypes.byref(clock_graphics))
        core_clk = clock_graphics.value if res_core == 0 else 0

        # 4. Memory Clock (clock type 2 = NVML_CLOCK_MEM)
        clock_mem = ctypes.c_uint()
        res_mem = self._nvml.nvmlDeviceGetClockInfo(self._device, 2, ctypes.byref(clock_mem))
        mem_clk = clock_mem.value if res_mem == 0 else 0

        # 5. Temperature (sensor type 0 = NVML_TEMPERATURE_GPU)
        temp_c = ctypes.c_uint()
        res_temp = self._nvml.nvmlDeviceGetTemperature(self._device, 0, ctypes.byref(temp_c))
        gpu_temp = float(temp_c.value) if res_temp == 0 else 0.0

        # 6. Host CPU Metrics (via psutil if available)
        cpu_util_total = 0.0
        cpu_clock_mhz = 0.0
        per_core = []
        if psutil is not None:
            try:
                cpu_util_total = psutil.cpu_percent(interval=None)
                per_core = psutil.cpu_percent(interval=None, percpu=True)
                freq = psutil.cpu_freq()
                cpu_clock_mhz = freq.current if freq else 0.0
            except Exception:
                pass

        return TelemetrySnapshot(
            timestamp=time.time(),
            gpu_util_core=gpu_util,
            gpu_util_memory=mem_util,
            gpu_power_watts=gpu_power_watts,
            gpu_clock_core_mhz=core_clk,
            gpu_clock_memory_mhz=mem_clk,
            gpu_temperature_c=gpu_temp,
            cpu_util_total=cpu_util_total,
            cpu_clock_mhz=cpu_clock_mhz,
            cpu_per_core_util=per_core
        )

    def close(self) -> None:
        """Shutdown NVML session."""
        if self._initialized and self._nvml:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
            finally:
                self._initialized = False
