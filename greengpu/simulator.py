"""
Calibrated Heterogeneous Hardware Simulator for GreenGPU.
Faithfully simulates execution time, PCIe transfer latency, dynamic and static power draw,
and total energy consumption for both the paper's original testbeds and modern hardware.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np

from greengpu.config import HardwareProfile, TESTBED_A_8800GTX, LOCAL_RTX_4050


@dataclass
class SimulationIterationOutput:
    """Detailed output of a simulated heterogeneous iteration."""
    iteration: int
    cpu_ratio: float
    gpu_ratio: float
    time_cpu_sec: float
    time_gpu_compute_sec: float
    time_pcie_transfer_sec: float
    time_gpu_total_sec: float
    time_wall_sec: float
    idle_time_wasted_sec: float
    avg_cpu_power_watts: float
    avg_gpu_power_watts: float
    avg_total_power_watts: float
    energy_cpu_joules: float
    energy_gpu_joules: float
    energy_total_joules: float
    core_utilization_pct: float
    memory_utilization_pct: float


class CalibratedGreenGPUSimulator:
    """
    High-fidelity simulator modeling heterogeneous execution and CMOS power scaling.
    Recreates the exact experimental setup and power curves published in the GreenGPU paper.
    """

    def __init__(
        self,
        hardware: Optional[HardwareProfile] = None,
        workload_type: str = "kmeans",
        pcie_bandwidth_mb_s: float = 656.3,
        cpu_peak_power_watts: float = 95.0,
        cpu_idle_power_watts: float = 35.0,
        workload_scale_factor: float = 1.0
    ):
        self.hardware = hardware or TESTBED_A_8800GTX
        self.workload_type = workload_type
        self.pcie_bw_mb_s = pcie_bandwidth_mb_s
        self.cpu_peak_power = cpu_peak_power_watts
        self.cpu_idle_power = cpu_idle_power_watts
        self.scale = workload_scale_factor

        # Workload profile definitions (Table 2 in paper)
        self.profiles = {
            "kmeans": {
                "base_time_cpu_sec": 1.25 * self.scale,
                "base_time_gpu_sec": 0.22 * self.scale,
                "pcie_data_mb": 38.0 * self.scale,
                "core_util": 65.0,
                "mem_util": 25.0,
                "optimal_ratio": 0.15  # Section 7.2: optimal is ~15/85
            },
            "hotspot": {
                "base_time_cpu_sec": 0.85 * self.scale,
                "base_time_gpu_sec": 0.82 * self.scale,
                "pcie_data_mb": 16.0 * self.scale,
                "core_util": 55.0,
                "mem_util": 20.0,
                "optimal_ratio": 0.50  # Section 7.2: optimal is ~50/50
            },
            "nbody": {
                "base_time_cpu_sec": 4.50 * self.scale,
                "base_time_gpu_sec": 0.35 * self.scale,
                "pcie_data_mb": 8.0 * self.scale,
                "core_util": 95.0,
                "mem_util": 15.0,
                "optimal_ratio": 0.08
            },
            "streamcluster": {
                "base_time_cpu_sec": 2.10 * self.scale,
                "base_time_gpu_sec": 0.65 * self.scale,
                "pcie_data_mb": 64.0 * self.scale,
                "core_util": 20.0,
                "mem_util": 90.0,
                "optimal_ratio": 0.25
            }
        }

    def simulate_iteration(
        self,
        iteration: int,
        cpu_ratio: float,
        gpu_core_mhz: Optional[int] = None,
        gpu_mem_mhz: Optional[int] = None
    ) -> SimulationIterationOutput:
        r = float(np.clip(cpu_ratio, 0.0, 1.0))
        prof = self.profiles.get(self.workload_type, self.profiles["kmeans"])

        fc_max = max(self.hardware.core_frequencies_mhz)
        fm_max = max(self.hardware.memory_frequencies_mhz)

        fc = gpu_core_mhz if gpu_core_mhz is not None else fc_max
        fm = gpu_mem_mhz if gpu_mem_mhz is not None else fm_max

        fc_scale = fc / fc_max
        fm_scale = fm / fm_max

        # 1. CPU Execution Time: tc proportional to work assigned (r)
        # Linear scaling with assigned workload fraction r
        time_cpu = prof["base_time_cpu_sec"] * r

        # 2. GPU Compute Time: (1 - r) with frequency scaling slowdown
        # Memory-intensity determines how sensitive the workload is to memory vs core clock
        mem_sensitivity = prof["mem_util"] / 100.0
        core_sensitivity = prof["core_util"] / 100.0
        
        # Performance slowdown factor based on frequency scaling
        slowdown = (core_sensitivity / fc_scale) + (mem_sensitivity / fm_scale)
        norm_slowdown = slowdown / (core_sensitivity + mem_sensitivity)
        
        time_gpu_compute = prof["base_time_gpu_sec"] * (1.0 - r) * norm_slowdown

        # 3. PCIe Data Transfer Time: moving (1 - r) data to/from GPU
        time_pcie = (prof["pcie_data_mb"] * (1.0 - r)) / self.pcie_bw_mb_s if r < 1.0 else 0.0
        time_gpu_total = time_gpu_compute + time_pcie

        # 4. Total Wall Clock Time & Idle Wait Time
        time_wall = max(time_cpu, time_gpu_total)
        idle_wasted = abs(time_cpu - time_gpu_total)

        # 5. Power Models
        # GPU Power: P_idle + P_dynamic * (fc/fc_max)^2
        p_gpu_idle = self.hardware.idle_power_watts
        p_gpu_dyn_max = self.hardware.peak_power_watts - p_gpu_idle
        
        gpu_active_power = p_gpu_idle + p_gpu_dyn_max * (
            0.70 * (prof["core_util"] / 100.0) * (fc_scale ** 2) +
            0.30 * (prof["mem_util"] / 100.0) * (fm_scale ** 1.5)
        )
        
        # During idle wait, device drops to idle power
        if time_wall > 0:
            gpu_energy = (gpu_active_power * time_gpu_total) + (p_gpu_idle * max(0.0, time_wall - time_gpu_total))
            avg_gpu_power = gpu_energy / time_wall
        else:
            gpu_energy = 0.0
            avg_gpu_power = p_gpu_idle

        # CPU Power with ondemand DVFS governor scaling
        p_cpu_idle = self.cpu_idle_power
        p_cpu_dyn_max = self.cpu_peak_power - self.cpu_idle_power
        # CPU dynamic power scales with workload fraction r (ondemand governor frequency scaling)
        p_cpu_active = p_cpu_idle + p_cpu_dyn_max * min(1.0, r * 2.5) if r > 0.0 else p_cpu_idle

        if time_wall > 0:
            cpu_energy = (p_cpu_active * time_cpu) + (p_cpu_idle * max(0.0, time_wall - time_cpu))
            avg_cpu_power = cpu_energy / time_wall
        else:
            cpu_energy = 0.0
            avg_cpu_power = p_cpu_idle

        total_energy = cpu_energy + gpu_energy
        avg_total_power = avg_cpu_power + avg_gpu_power

        return SimulationIterationOutput(
            iteration=iteration,
            cpu_ratio=r,
            gpu_ratio=1.0 - r,
            time_cpu_sec=float(time_cpu),
            time_gpu_compute_sec=float(time_gpu_compute),
            time_pcie_transfer_sec=float(time_pcie),
            time_gpu_total_sec=float(time_gpu_total),
            time_wall_sec=float(time_wall),
            idle_time_wasted_sec=float(idle_wasted),
            avg_cpu_power_watts=float(avg_cpu_power),
            avg_gpu_power_watts=float(avg_gpu_power),
            avg_total_power_watts=float(avg_total_power),
            energy_cpu_joules=float(cpu_energy),
            energy_gpu_joules=float(gpu_energy),
            energy_total_joules=float(total_energy),
            core_utilization_pct=float(prof["core_util"]),
            memory_utilization_pct=float(prof["mem_util"])
        )
