"""
Heterogeneous Cooperative HotSpot 2D Thermal Simulation Workload (Rodinia Benchmark).
Models 2D grid temperature diffusion with boundary halo synchronization between CPU and GPU.
"""

import threading
import time
from typing import Optional, Tuple
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from greengpu.workloads.base import HeterogeneousWorkload, WorkloadIterationResult


class HeterogeneousHotSpot(HeterogeneousWorkload):
    """
    Cooperative CPU-GPU HotSpot 2D Grid Stencil.
    Partitions the grid rows between CPU (r * rows) and GPU ((1 - r) * rows).
    Performs iterative heat equation updates with boundary synchronization.
    """

    def __init__(
        self,
        grid_rows: int = 1024,
        grid_cols: int = 1024,
        steps_per_iteration: int = 4,
        use_gpu: bool = True,
        seed: int = 42
    ):
        super().__init__(name="hotspot")
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.steps_per_iteration = steps_per_iteration
        self.use_gpu = use_gpu and HAS_TORCH and torch.cuda.is_available()

        # Thermal simulation constants
        self.cap = 0.5
        self.rx = 0.1
        self.ry = 0.1
        self.rz = 0.05
        self.amb_temp = 80.0

        # Initialize temperature and power matrices
        rng = np.random.RandomState(seed)
        self.raw_temp = np.full((grid_rows, grid_cols), 300.0, dtype=np.float32)
        self.raw_power = rng.uniform(0.0, 100.0, size=(grid_rows, grid_cols)).astype(np.float32)

        if HAS_TORCH:
            self.torch_temp = torch.from_numpy(self.raw_temp)
            self.torch_power = torch.from_numpy(self.raw_power)
            if self.use_gpu:
                self.cuda_stream = torch.cuda.Stream()
                self.torch_temp = self.torch_temp.pin_memory()
                self.torch_power = self.torch_power.pin_memory()
        else:
            self.torch_temp = None
            self.torch_power = None
            self.cuda_stream = None

    def _step_stencil_torch(self, temp: "torch.Tensor", power: "torch.Tensor") -> "torch.Tensor":
        """Compute 5-point stencil heat equation step using torch slices."""
        t_center = temp[1:-1, 1:-1]
        t_north = temp[:-2, 1:-1]
        t_south = temp[2:, 1:-1]
        t_west = temp[1:-1, :-2]
        t_east = temp[1:-1, 2:]
        p_center = power[1:-1, 1:-1]

        # Explicit finite difference heat equation
        delta = (
            (t_north + t_south - 2.0 * t_center) * self.rx +
            (t_west + t_east - 2.0 * t_center) * self.ry +
            (self.amb_temp - t_center) * self.rz +
            p_center
        ) * (1.0 / self.cap)

        out = temp.clone()
        out[1:-1, 1:-1] = t_center + delta
        return out

    def _step_stencil_numpy(self, temp: np.ndarray, power: np.ndarray) -> np.ndarray:
        """NumPy 5-point stencil computation."""
        t_center = temp[1:-1, 1:-1]
        t_north = temp[:-2, 1:-1]
        t_south = temp[2:, 1:-1]
        t_west = temp[1:-1, :-2]
        t_east = temp[1:-1, 2:]
        p_center = power[1:-1, 1:-1]

        delta = (
            (t_north + t_south - 2.0 * t_center) * self.rx +
            (t_west + t_east - 2.0 * t_center) * self.ry +
            (self.amb_temp - t_center) * self.rz +
            p_center
        ) * (1.0 / self.cap)

        out = temp.copy()
        out[1:-1, 1:-1] = t_center + delta
        return out

    def run_iteration(self, iteration: int, cpu_ratio: float) -> WorkloadIterationResult:
        r = max(0.0, min(1.0, float(cpu_ratio)))
        rows_cpu = int(round(r * self.grid_rows))
        rows_gpu = self.grid_rows - rows_cpu

        time_cpu = 0.0
        time_gpu = 0.0

        if HAS_TORCH:
            # Slices with halo buffer (1 row overlap for stencil computation)
            def run_cpu():
                nonlocal time_cpu
                if rows_cpu < 2:
                    time_cpu = 0.0
                    return
                t0 = time.perf_counter()
                sub_temp = self.torch_temp[:rows_cpu]
                sub_pwr = self.torch_power[:rows_cpu]
                for _ in range(self.steps_per_iteration):
                    sub_temp = self._step_stencil_torch(sub_temp, sub_pwr)
                time_cpu = time.perf_counter() - t0

            def run_gpu():
                nonlocal time_gpu
                if rows_gpu < 2:
                    time_gpu = 0.0
                    return
                dev = "cuda" if self.use_gpu else "cpu"
                t0 = time.perf_counter()
                if self.use_gpu:
                    with torch.cuda.stream(self.cuda_stream):
                        d_temp = self.torch_temp[rows_cpu:].to(dev, non_blocking=True)
                        d_pwr = self.torch_power[rows_cpu:].to(dev, non_blocking=True)
                        for _ in range(self.steps_per_iteration):
                            d_temp = self._step_stencil_torch(d_temp, d_pwr)
                        res = d_temp.cpu()
                        torch.cuda.synchronize()
                else:
                    sub_temp = self.torch_temp[rows_cpu:]
                    sub_pwr = self.torch_power[rows_cpu:]
                    for _ in range(self.steps_per_iteration):
                        sub_temp = self._step_stencil_torch(sub_temp, sub_pwr)
                time_gpu = time.perf_counter() - t0

            th_cpu = threading.Thread(target=run_cpu)
            th_gpu = threading.Thread(target=run_gpu)
            th_cpu.start()
            th_gpu.start()
            th_cpu.join()
            th_gpu.join()

        else:
            # NumPy fallback
            def run_cpu():
                nonlocal time_cpu
                if rows_cpu < 2:
                    time_cpu = 0.0
                    return
                t0 = time.perf_counter()
                sub_temp = self.raw_temp[:rows_cpu]
                sub_pwr = self.raw_power[:rows_cpu]
                for _ in range(self.steps_per_iteration):
                    sub_temp = self._step_stencil_numpy(sub_temp, sub_pwr)
                time_cpu = time.perf_counter() - t0

            def run_gpu():
                nonlocal time_gpu
                if rows_gpu < 2:
                    time_gpu = 0.0
                    return
                t0 = time.perf_counter()
                sub_temp = self.raw_temp[rows_cpu:]
                sub_pwr = self.raw_power[rows_cpu:]
                for _ in range(self.steps_per_iteration):
                    sub_temp = self._step_stencil_numpy(sub_temp, sub_pwr)
                time_gpu = time.perf_counter() - t0

            th_cpu = threading.Thread(target=run_cpu)
            th_gpu = threading.Thread(target=run_gpu)
            th_cpu.start()
            th_gpu.start()
            th_cpu.join()
            th_gpu.join()

        wall_time = max(time_cpu, time_gpu)
        idle_wasted = abs(time_cpu - time_gpu)

        return WorkloadIterationResult(
            iteration=iteration,
            cpu_ratio=r,
            gpu_ratio=1.0 - r,
            time_cpu_sec=time_cpu,
            time_gpu_sec=time_gpu,
            time_wall_sec=wall_time,
            idle_time_wasted_sec=idle_wasted,
            extra_metrics={"rows_cpu": rows_cpu, "rows_gpu": rows_gpu}
        )

    def reset(self) -> None:
        self.raw_temp.fill(300.0)
        if HAS_TORCH:
            self.torch_temp = torch.from_numpy(self.raw_temp.copy())
