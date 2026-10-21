"""
Heterogeneous cooperative workloads for GreenGPU benchmarking.
"""

from greengpu.workloads.base import HeterogeneousWorkload, WorkloadIterationResult
from greengpu.workloads.kmeans import HeterogeneousKMeans
from greengpu.workloads.hotspot import HeterogeneousHotSpot
from greengpu.workloads.synthetic_bench import SyntheticHeterogeneousBenchmark

__all__ = [
    "HeterogeneousWorkload",
    "WorkloadIterationResult",
    "HeterogeneousKMeans",
    "HeterogeneousHotSpot",
    "SyntheticHeterogeneousBenchmark",
]
