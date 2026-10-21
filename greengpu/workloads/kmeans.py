"""
Heterogeneous Cooperative K-Means Clustering Workload (Rodinia Benchmark).
Implements iterative data-partitioned clustering where CPU and GPU cooperatively
compute nearest-centroid assignments in parallel and reduce cluster centers at the barrier.
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


class HeterogeneousKMeans(HeterogeneousWorkload):
    """
    Cooperative CPU-GPU K-Means clustering kernel.
    Partitions N feature vectors into r * N (CPU) and (1 - r) * N (GPU) chunks.
    Executes distance calculations concurrently using host threads and CUDA streams.
    """

    def __init__(
        self,
        num_points: int = 200_000,
        num_features: int = 34,
        num_clusters: int = 16,
        use_gpu: bool = True,
        seed: int = 42
    ):
        super().__init__(name="kmeans")
        self.num_points = num_points
        self.num_features = num_features
        self.num_clusters = num_clusters
        self.use_gpu = use_gpu and HAS_TORCH and torch.cuda.is_available()

        # Seed and generate deterministic synthetic cluster dataset
        rng = np.random.RandomState(seed)
        self.raw_data = rng.randn(num_points, num_features).astype(np.float32)
        self.raw_centroids = self.raw_data[:num_clusters].copy()

        # If torch is available, pre-stage torch tensors
        if HAS_TORCH:
            self.torch_data = torch.from_numpy(self.raw_data)
            self.torch_centroids = torch.from_numpy(self.raw_centroids)
            if self.use_gpu:
                self.cuda_stream = torch.cuda.Stream()
                # Pin host memory for accelerated DMA PCIe transfer
                self.torch_data = self.torch_data.pin_memory()
        else:
            self.torch_data = None
            self.torch_centroids = None
            self.cuda_stream = None

    def _compute_chunk_torch(
        self,
        data_chunk: "torch.Tensor",
        centroids: "torch.Tensor",
        device: str
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """Compute nearest clusters and partial sums on specified device."""
        if device == "cuda":
            with torch.cuda.stream(self.cuda_stream):
                # Transfer data chunk to GPU
                d_data = data_chunk.to(device, non_blocking=True)
                d_centroids = centroids.to(device, non_blocking=True)
                
                # Pairwise squared Euclidean distances: ||x - c||^2 = ||x||^2 - 2<x, c> + ||c||^2
                # shape: (chunk_size, K)
                dist = torch.cdist(d_data, d_centroids, p=2.0)
                labels = torch.argmin(dist, dim=1)
                
                # Compute cluster sums
                sums = torch.zeros_like(d_centroids)
                sums.index_add_(0, labels, d_data)
                counts = torch.bincount(labels, minlength=self.num_clusters).float()
                
                torch.cuda.synchronize()
                return sums.cpu(), counts.cpu()
        else:
            # CPU path
            dist = torch.cdist(data_chunk, centroids, p=2.0)
            labels = torch.argmin(dist, dim=1)
            sums = torch.zeros_like(centroids)
            sums.index_add_(0, labels, data_chunk)
            counts = torch.bincount(labels, minlength=self.num_clusters).float()
            return sums, counts

    def _compute_chunk_numpy(
        self,
        data_chunk: np.ndarray,
        centroids: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """NumPy fallback for CPU distance calculation."""
        # Pairwise distance: (N, 1, D) - (1, K, D)
        diff = data_chunk[:, np.newaxis, :] - centroids[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        labels = np.argmin(dist_sq, axis=1)

        sums = np.zeros_like(centroids)
        counts = np.bincount(labels, minlength=self.num_clusters).astype(np.float32)
        for k in range(self.num_clusters):
            mask = (labels == k)
            if np.any(mask):
                sums[k] = np.sum(data_chunk[mask], axis=0)

        return sums, counts

    def run_iteration(self, iteration: int, cpu_ratio: float) -> WorkloadIterationResult:
        r = max(0.0, min(1.0, float(cpu_ratio)))
        n_cpu = int(round(r * self.num_points))
        n_gpu = self.num_points - n_cpu

        time_cpu = 0.0
        time_gpu = 0.0

        cpu_sums, cpu_counts = None, None
        gpu_sums, gpu_counts = None, None

        if HAS_TORCH:
            cpu_data = self.torch_data[:n_cpu] if n_cpu > 0 else None
            gpu_data = self.torch_data[n_cpu:] if n_gpu > 0 else None
            centroids = self.torch_centroids

            def run_cpu():
                nonlocal time_cpu, cpu_sums, cpu_counts
                if n_cpu == 0:
                    time_cpu = 0.0
                    return
                t0 = time.perf_counter()
                cpu_sums, cpu_counts = self._compute_chunk_torch(cpu_data, centroids, device="cpu")
                time_cpu = time.perf_counter() - t0

            def run_gpu():
                nonlocal time_gpu, gpu_sums, gpu_counts
                if n_gpu == 0:
                    time_gpu = 0.0
                    return
                dev = "cuda" if self.use_gpu else "cpu"
                t0 = time.perf_counter()
                gpu_sums, gpu_counts = self._compute_chunk_torch(gpu_data, centroids, device=dev)
                time_gpu = time.perf_counter() - t0

            # Launch concurrent execution
            th_cpu = threading.Thread(target=run_cpu)
            th_gpu = threading.Thread(target=run_gpu)

            th_cpu.start()
            th_gpu.start()

            th_cpu.join()
            th_gpu.join()

            # Global Reduction Barrier: combine sums and counts to compute new centroids
            total_counts = torch.zeros(self.num_clusters)
            total_sums = torch.zeros_like(centroids)

            if cpu_counts is not None:
                total_counts += cpu_counts
                total_sums += cpu_sums
            if gpu_counts is not None:
                total_counts += gpu_counts
                total_sums += gpu_sums

            # Avoid division by zero
            non_zero = total_counts > 0
            if torch.any(non_zero):
                new_centroids = self.torch_centroids.clone()
                new_centroids[non_zero] = total_sums[non_zero] / total_counts[non_zero].unsqueeze(1)
                self.torch_centroids = new_centroids

        else:
            # Pure NumPy execution
            cpu_data = self.raw_data[:n_cpu] if n_cpu > 0 else None
            gpu_data = self.raw_data[n_cpu:] if n_gpu > 0 else None

            def run_cpu():
                nonlocal time_cpu, cpu_sums, cpu_counts
                if n_cpu == 0:
                    time_cpu = 0.0
                    return
                t0 = time.perf_counter()
                cpu_sums, cpu_counts = self._compute_chunk_numpy(cpu_data, self.raw_centroids)
                time_cpu = time.perf_counter() - t0

            def run_gpu():
                nonlocal time_gpu, gpu_sums, gpu_counts
                if n_gpu == 0:
                    time_gpu = 0.0
                    return
                t0 = time.perf_counter()
                gpu_sums, gpu_counts = self._compute_chunk_numpy(gpu_data, self.raw_centroids)
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
            extra_metrics={"num_points": self.num_points, "n_cpu": n_cpu, "n_gpu": n_gpu}
        )

    def reset(self) -> None:
        if HAS_TORCH:
            self.torch_centroids = torch.from_numpy(self.raw_centroids.copy())
        else:
            self.raw_centroids = self.raw_data[:self.num_clusters].copy()
