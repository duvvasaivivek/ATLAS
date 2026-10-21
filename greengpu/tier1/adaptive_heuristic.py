"""
Improved Heuristic Dynamic Workload Partitioner (Section 5.2.2).
Uses adaptive step size: starts coarse (e.g., 15%), detects oscillation,
and progressively halves the step size until falling below precision threshold Pt.
"""

from typing import Optional

from greengpu.config import Tier1Config
from greengpu.tier1.base import BaseWorkloadPartitioner


class AdaptiveHeuristicPartitioner(BaseWorkloadPartitioner):
    """
    Improved Heuristic partitioner with adaptive step halving.
    Rapidly narrows search space by cutting step size in half upon oscillation.
    """

    def __init__(self, config: Optional[Tier1Config] = None, initial_step: Optional[float] = None):
        super().__init__(config)
        self.current_step = initial_step if initial_step is not None else self.config.adaptive_initial_step
        self.prev_direction: Optional[int] = None  # +1 for tc < tg (need more CPU), -1 for tc > tg (need less CPU)

    def compute_next_ratio(self, tc: float, tg: float) -> float:
        r = self.current_ratio

        if tc > tg:
            direction = -1  # reduce CPU work
        elif tc < tg:
            direction = +1  # increase CPU work
        else:
            self._converged = True
            return r

        # Check for oscillation: sign reversal from previous step
        if self.prev_direction is not None and direction != self.prev_direction:
            # Oscillation occurred: halve the step size
            self.current_step = self.current_step / 2.0
            if len(self.history) > 0:
                self.history[-1].oscillation_detected = True

            # If step size is smaller than precision threshold, converge
            if self.current_step < self.config.precision_threshold:
                self._converged = True
                return r

        self.prev_direction = direction

        candidate_r = r + (direction * self.current_step)
        candidate_r = max(self.config.min_cpu_ratio, min(self.config.max_cpu_ratio, candidate_r))
        return candidate_r

    def reset(self, initial_ratio: Optional[float] = None) -> None:
        super().reset(initial_ratio)
        self.current_step = self.config.adaptive_initial_step
        self.prev_direction = None
