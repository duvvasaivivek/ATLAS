"""
Simple Heuristic Dynamic Workload Partitioner (Section 5.2.1).
Uses fixed step adjustments with linear lookahead safeguard to prevent oscillation.
"""

from typing import Optional

from greengpu.config import Tier1Config
from greengpu.tier1.base import BaseWorkloadPartitioner


class SimpleHeuristicPartitioner(BaseWorkloadPartitioner):
    """
    Fixed step size heuristic partitioner with linear lookahead safeguard.
    Shifts workload by fixed delta based on sign of (tc - tg).
    """

    def __init__(self, config: Optional[Tier1Config] = None, step_size: Optional[float] = None):
        super().__init__(config)
        self.step_size = step_size if step_size is not None else self.config.simple_step_size

    def compute_next_ratio(self, tc: float, tg: float) -> float:
        r = self.current_ratio

        # Determine direction
        if tc > tg:
            # CPU is slower -> decrease CPU workload
            candidate_r = r - self.step_size
        elif tc < tg:
            # GPU is slower -> increase CPU workload
            candidate_r = r + self.step_size
        else:
            return r

        # Clamp candidate
        candidate_r = max(self.config.min_cpu_ratio, min(self.config.max_cpu_ratio, candidate_r))

        # Lookahead Safeguard (Section 5.2.1)
        if self.config.lookahead_safeguard and r > 0 and (1.0 - r) > 0 and candidate_r > 0 and (1.0 - candidate_r) > 0:
            pred_tc = (candidate_r / r) * tc
            pred_tg = ((1.0 - candidate_r) / (1.0 - r)) * tg

            # If moving would cause an overshoot/oscillation, stay at current r
            if tc < tg and pred_tc > pred_tg:
                self._converged = True
                return r
            elif tc > tg and pred_tc < pred_tg:
                self._converged = True
                return r

        return candidate_r
