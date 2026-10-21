"""
Binary Search Dynamic Workload Partitioner (Section 5.2.3).
Halves the workload interval based on binary decisions (tc < tg vs tc > tg).
Guaranteed convergence within ceil(log2(100 / Pt)) iterations.
"""

from typing import Optional

from greengpu.config import Tier1Config
from greengpu.tier1.base import BaseWorkloadPartitioner


class BinarySearchPartitioner(BaseWorkloadPartitioner):
    """
    Binary search partitioner (Section 5.2.3).
    Maintains search interval [low, high] bisecting the workload space.
    If tc < tg, sets low = r and assigns r = (low + high) / 2.
    If tc > tg, sets high = r and assigns r = (low + high) / 2.
    Guaranteed convergence within ceil(log2(100 / Pt)) iterations.
    """

    def __init__(self, config: Optional[Tier1Config] = None):
        super().__init__(config)
        self.low: float = self.config.min_cpu_ratio
        self.high: float = self.config.max_cpu_ratio

    def compute_next_ratio(self, tc: float, tg: float) -> float:
        r = self.current_ratio

        if tc < tg:
            # CPU faster than GPU -> allocate more work to CPU
            self.low = r
            next_r = (self.low + self.high) / 2.0
        elif tc > tg:
            # CPU slower than GPU -> allocate more work to GPU
            self.high = r
            next_r = (self.low + self.high) / 2.0
        else:
            self._converged = True
            return r

        # Check precision threshold convergence condition:
        # If bracket size or step size is below Pt, terminate
        if (self.high - self.low) < self.config.precision_threshold or abs(next_r - r) < self.config.precision_threshold:
            self._converged = True

        return next_r

    def reset(self, initial_ratio: Optional[float] = None) -> None:
        super().reset(initial_ratio)
        self.low = self.config.min_cpu_ratio
        self.high = self.config.max_cpu_ratio
