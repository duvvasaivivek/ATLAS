"""
Actuators for applying frequency scaling and power limits on live hardware or in simulation.
"""

from abc import ABC, abstractmethod
import ctypes
from typing import Optional, Tuple

from greengpu.config import HardwareProfile, LOCAL_RTX_4050


class BaseFrequencyActuator(ABC):
    """Abstract interface for frequency scaling actuation."""

    @abstractmethod
    def set_frequency(self, core_mhz: int, memory_mhz: int) -> bool:
        """Apply requested frequencies to the device."""
        pass

    @abstractmethod
    def reset(self) -> bool:
        """Restore default automatic frequency management."""
        pass


class LiveNVMLActuator(BaseFrequencyActuator):
    """
    Live hardware actuator interacting with NVML.
    Attempts hardware locked clocks; gracefully degrades to non-invasive telemetry tracking
    if un-elevated (non-admin) permissions are detected.
    """

    def __init__(self, device_index: int = 0):
        self.device_index = device_index
        self._nvml = None
        self._device = None
        self.last_applied_core = 0
        self.last_applied_memory = 0
        self.is_permission_granted = False

        self._init_nvml()

    def _init_nvml(self) -> None:
        try:
            self._nvml = ctypes.CDLL("nvml.dll")
            self._nvml.nvmlInit()
            self._device = ctypes.c_void_p()
            self._nvml.nvmlDeviceGetHandleByIndex(self.device_index, ctypes.byref(self._device))
        except Exception:
            self._nvml = None

    def set_frequency(self, core_mhz: int, memory_mhz: int) -> bool:
        self.last_applied_core = core_mhz
        self.last_applied_memory = memory_mhz

        if self._nvml and self._device and hasattr(self._nvml, "nvmlDeviceSetGpuLockedClocks"):
            try:
                # nvmlDeviceSetGpuLockedClocks(device, minGpuClockMHz, maxGpuClockMHz)
                res = self._nvml.nvmlDeviceSetGpuLockedClocks(self._device, int(core_mhz), int(core_mhz))
                self.is_permission_granted = (res == 0)
                return self.is_permission_granted
            except Exception:
                return False
        return False

    def reset(self) -> bool:
        if self._nvml and self._device and hasattr(self._nvml, "nvmlDeviceResetGpuLockedClocks"):
            try:
                res = self._nvml.nvmlDeviceResetGpuLockedClocks(self._device)
                return (res == 0)
            except Exception:
                return False
        return True


class SimulatedActuator(BaseFrequencyActuator):
    """
    Software simulation actuator that computes theoretical dynamic and static power
    scaling corresponding to the assigned clock levels.
    """

    def __init__(self, hardware: Optional[HardwareProfile] = None):
        self.hardware = hardware or LOCAL_RTX_4050
        self.current_core_mhz = max(self.hardware.core_frequencies_mhz)
        self.current_memory_mhz = max(self.hardware.memory_frequencies_mhz)

    def set_frequency(self, core_mhz: int, memory_mhz: int) -> bool:
        self.current_core_mhz = core_mhz
        self.current_memory_mhz = memory_mhz
        return True

    def calculate_estimated_power_watts(self, core_util: float, mem_util: float) -> float:
        """
        Estimates instantaneous power draw using quadratic CMOS dynamic power scaling:
        P = P_idle + P_dynamic * (util_core * (fc / fc_max)^2 + util_mem * (fm / fm_max)^1.5)
        """
        fc_max = max(self.hardware.core_frequencies_mhz)
        fm_max = max(self.hardware.memory_frequencies_mhz)

        fc_ratio = self.current_core_mhz / fc_max
        fm_ratio = self.current_memory_mhz / fm_max

        p_idle = self.hardware.idle_power_watts
        p_dyn_max = self.hardware.peak_power_watts - p_idle

        # Split dynamic power: 70% core compute, 30% memory controller
        core_dyn = 0.70 * p_dyn_max * (core_util / 100.0) * (fc_ratio ** 2)
        mem_dyn = 0.30 * p_dyn_max * (mem_util / 100.0) * (fm_ratio ** 1.5)

        return float(p_idle + core_dyn + mem_dyn)

    def reset(self) -> bool:
        self.current_core_mhz = max(self.hardware.core_frequencies_mhz)
        self.current_memory_mhz = max(self.hardware.memory_frequencies_mhz)
        return True
