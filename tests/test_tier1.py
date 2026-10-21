"""
Unit tests for Tier 1 Dynamic Workload Partitioners.
Verifies mathematical correctness, oscillation safeguards, convergence bounds, and proportional matching.
"""

import unittest
from greengpu.config import Tier1Config
from greengpu.tier1.simple_heuristic import SimpleHeuristicPartitioner
from greengpu.tier1.adaptive_heuristic import AdaptiveHeuristicPartitioner
from greengpu.tier1.binary_search import BinarySearchPartitioner
from greengpu.tier1.improved_binary import ImprovedBinarySearchPartitioner


class TestTier1Partitioners(unittest.TestCase):

    def setUp(self):
        self.config = Tier1Config(
            precision_threshold=0.01,
            simple_step_size=0.05,
            adaptive_initial_step=0.15,
            initial_cpu_ratio=0.50
        )

    def test_simple_heuristic_direction(self):
        """CPU slower than GPU should decrease CPU ratio."""
        p = SimpleHeuristicPartitioner(self.config)
        # tc = 10s, tg = 5s -> CPU slower, decrease CPU ratio
        next_r = p.update(iteration=1, tc=10.0, tg=5.0)
        self.assertAlmostEqual(next_r, 0.45, places=3)
        self.assertEqual(len(p.history), 1)

        # CPU faster than GPU should increase CPU ratio
        next_r = p.update(iteration=2, tc=3.0, tg=8.0)
        self.assertAlmostEqual(next_r, 0.50, places=3)

    def test_simple_heuristic_safeguard(self):
        """Lookahead safeguard should hold current ratio when step would cause oscillation."""
        cfg = Tier1Config(simple_step_size=0.05, lookahead_safeguard=True)
        p = SimpleHeuristicPartitioner(cfg)
        p.current_ratio = 0.50

        # Suppose tc is barely lower than tg: moving +0.05 to 0.55 would reverse the balance drastically
        next_r = p.compute_next_ratio(tc=4.9, tg=5.0)
        # Safeguard holds at 0.50 and flags convergence
        self.assertAlmostEqual(next_r, 0.50, places=3)
        self.assertTrue(p.is_converged)

    def test_adaptive_heuristic_step_halving(self):
        """Improved heuristic should halve step size upon oscillation."""
        p = AdaptiveHeuristicPartitioner(self.config, initial_step=0.20)
        
        # Step 1: CPU slower -> move -0.20 to 0.30
        r1 = p.update(1, tc=10.0, tg=5.0)
        self.assertAlmostEqual(r1, 0.30, places=3)
        self.assertEqual(p.current_step, 0.20)

        # Step 2: Now GPU slower -> reverse direction! Oscillation detected -> step halved to 0.10
        r2 = p.update(2, tc=2.0, tg=8.0)
        self.assertEqual(p.current_step, 0.10)
        self.assertAlmostEqual(r2, 0.40, places=3)

    def test_binary_search_bounds(self):
        """Binary search must converge within ceil(log2(100 / Pt)) iterations."""
        p = BinarySearchPartitioner(self.config)
        # Synthetic target where optimal CPU ratio is 0.20
        # If r > 0.20, tc > tg. If r < 0.20, tc < tg.
        for it in range(1, 15):
            tc = 10.0 * (p.current_ratio / 0.20)
            tg = 10.0 * ((1.0 - p.current_ratio) / 0.80)
            p.update(it, tc=tc, tg=tg)
            if p.is_converged:
                break

        # Must converge in <= 7 iterations
        self.assertLessEqual(len(p.history), 7)
        self.assertAlmostEqual(p.current_ratio, 0.20, delta=0.05)

    def test_improved_binary_proportional_matching(self):
        """Improved binary search must solve exact proportional ratio without oscillation."""
        p = ImprovedBinarySearchPartitioner(self.config)
        p.current_ratio = 0.50

        # Suppose work capability of GPU is 4x that of CPU (optimal r = 0.20)
        # At r = 0.50, tc = 4 * tg
        for it in range(1, 10):
            tc = 10.0 * (p.current_ratio / 0.20)
            tg = 10.0 * ((1.0 - p.current_ratio) / 0.80)
            p.update(it, tc=tc, tg=tg)
            if p.is_converged:
                break

        # Improved Binary converges in <= 5 iterations
        self.assertLessEqual(len(p.history), 5)
        self.assertAlmostEqual(p.current_ratio, 0.20, places=2)


if __name__ == "__main__":
    unittest.main()
