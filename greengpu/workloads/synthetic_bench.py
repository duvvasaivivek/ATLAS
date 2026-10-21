"""
Tunable Synthetic Heterogeneous Benchmark.
Allows generating controlled compute-bound (nbody-style) or memory-bound (streamcluster-style)
workload phases to test Tier 1 division and Tier 2 DVFS under calibrated conditions.
"""

import threading
import time
from typing import Optional
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from greengpu.workloads.base import HeterogeneousWorkload, WorkloadIterationResult


class SyntheticHeterogeneousBenchmark(HeterogeneousWorkload):
    """
    Configurable heterogeneous benchmark with tunable operational intensity.
    Modes:
      - 'compute_bound': high arithmetic intensity (GEMM/tensor operations, stresses core)
      - 'memory_bound': streaming vector operations (stresses memory bandwidth)
      - 'fluctuating': alternates between compute and memory bound (like QG/SC in paper)
    """

    def __init__(
        self,
        mode: str = "compute_bound",
        total_work_units: int = 100_000,
        dimension: int = 512,
        use_gpu: bool = True
    ):
        super().__init__(name=f"synthetic_{mode}")
        self.mode = mode
        self.total_work_units = total_work_units
        self.dimension = dimension
        self.use_gpu = use_gpu and HAS_TORCH and torch.cuda.is_available()

        if HAS_TORCH:
            self.cuda_stream = torch.cuda.Stream() if self.use_gpu else None
        else:
            self.cuda_stream = None

    def _execute_subwork(self, count: int, device: str) -> None:
        """Run workload unit on CPU or GPU device."""
        if count <= 0:
            return

        if HAS_TORCH:
            if self.mode == "compute_bound":
                # Compute intensive: repeated matrix multiplications
                a = torch.randn(count, self.dimension, device=device)
                w = torch.randn(self.dimension, 128, device=device)
                out = torch.matmul(a, w)
            elif self.mode == "memory_bound":
                # Memory intensive: large streaming element-wise operations
                x = torch.randn(count * 8, device=device)
                y = torch.randn(count * 8, device=device)
                out = (x * 1.5) + (y * 2.5)
            elif self.mode == "fluctuating":
                a = torch.randn(count, self.dimension // 2, device=device)
                out = torch.sin(a) * torch.cos(a)
        else:
            # NumPy fallback
            if self.mode == "compute_bound":
                a = np.random.randn(count, self.dimension).astype(np.float32)
                w = np.random.randn(self.dimension, 128).astype(np.float32)
                out = np.matmul(a, w)
            else:
                x = np.random.randn(count * 8).astype(np.float32)
                y = np.random.randn(count * 8).astype(np.float32)
                out = (x * 1.5) + (y * 2.5)

    def run_iteration(self, iteration: int, cpu_ratio: float) -> WorkloadIterationResult:
        r = max(0.0, min(1.0, float(cpu_ratio)))
        n_cpu = int(round(r * self.total_work_units))
        n_gpu = self.total_work_units - n_cpu

        time_cpu = 0.0
        time_gpu = 0.0

        def run_cpu():
            nonlocal time_cpu
            if n_cpu == 0:
                time_cpu = 0.0
                return
            t0 = time.perf_counter()
            self._execute_subwork(n_cpu, device="cpu")
            time_cpu = time.perf_counter() - t0

        def run_gpu():
            nonlocal time_gpu
            if n_gpu == 0:
                time_gpu = 0.0
                return
            dev = "cuda" if self.use_gpu else "cpu"
            t0 = time.perf_counter()
            if self.use_gpu:
                with torch.cuda.stream(self.cuda_stream):
                    self._execute_subwork(n_gpu, device=dev)
                    torch.cuda.synchronize()
            else:
                self._execute_subwork(n_gpu, device=dev)
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
            extra_metrics={"mode": self.mode, "n_cpu": n_cpu, "n_gpu": n_gpu}
        )

    def reset(self) -> None:
        pass
