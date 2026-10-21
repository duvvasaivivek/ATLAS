"""
Improved Binary Search Dynamic Workload Partitioner (Section 5.2.4).
Quantitatively balances workload proportionally to the execution time difference:
    tc / tg = [r / (1 - r)] / [r' / (1 - r')]
Delivers fastest convergence (4-5 iterations) with minimum idle power waste and zero oscillation.
"""

from typing import Optional

from greengpu.config import Tier1Config
from greengpu.tier1.base import BaseWorkloadPartitioner


class ImprovedBinarySearchPartitioner(BaseWorkloadPartitioner):
    """
    Improved Binary Search with proportional execution time balancing.
    Calculates next ratio r' by inverting the work-time balance equation.
    """

    def __init__(self, config: Optional[Tier1Config] = None):
        super().__init__(config)

    def compute_next_ratio(self, tc: float, tg: float) -> float:
        r = self.current_ratio

        # Guard against zero or negative timings
        if tc <= 1e-9 or tg <= 1e-9:
            return r

        # Guard against edge-case division by zero
        eps = 1e-4
        clamped_r = max(eps, min(1.0 - eps, r))

        # Current odds ratio: x = r / (1 - r)
        current_odds = clamped_r / (1.0 - clamped_r)

        # Proportional adjustment:
        # tc / tg = [r / (1 - r)] / [r' / (1 - r')]
        # => r' / (1 - r') = (tg / tc) * (r / (1 - r))
        time_ratio = tg / tc
        target_odds = current_odds * time_ratio

        # Solve for r': r' = target_odds / (1 + target_odds)
        next_r = target_odds / (1.0 + target_odds)

        # Check convergence: execution time difference within 2% or ratio change < Pt
        relative_time_diff = abs(tc - tg) / max(tc, tg)
        ratio_delta = abs(next_r - r)

        if ratio_delta < self.config.precision_threshold or relative_time_diff < 0.02:
            self._converged = True

        return next_r
