"""
Unit tests for Tier 2 Coordinated DVFS Controller.
Verifies asymmetric loss functions, WMA weight updates, and frequency selection.
"""

import unittest
import numpy as np

from greengpu.config import TESTBED_A_8800GTX, Tier2Config
from greengpu.tier2.loss_functions import (
    compute_component_loss,
    compute_total_joint_loss,
    generate_umean_levels,
)
from greengpu.tier2.wma_controller import WMAFrequencyController


class TestTier2DVFS(unittest.TestCase):

    def setUp(self):
        self.hardware = TESTBED_A_8800GTX
        self.config = Tier2Config(alpha_core=0.15, alpha_memory=0.02, gamma=0.3, beta=0.2)
        self.controller = WMAFrequencyController(self.hardware, self.config)

    def test_umean_generation(self):
        """umean levels must span linearly from 0.0 to 1.0."""
        umean = generate_umean_levels(6)
        self.assertEqual(len(umean), 6)
        self.assertAlmostEqual(umean[0], 0.0)
        self.assertAlmostEqual(umean[-1], 1.0)
        self.assertAlmostEqual(umean[1], 0.2)

    def test_asymmetric_loss_properties(self):
        """Loss function must penalize energy loss when under-utilized and performance loss when over-utilized."""
        umean = np.array([0.5])
        
        # Case 1: Utilization is lower than frequency target -> Energy loss > 0, Performance loss == 0
        loss, le, lp = compute_component_loss(utilization=0.2, umean_levels=umean, alpha=0.15)
        self.assertAlmostEqual(le[0], 0.3)
        self.assertAlmostEqual(lp[0], 0.0)
        self.assertAlmostEqual(loss[0], 0.15 * 0.3)

        # Case 2: Utilization is higher than frequency target -> Performance loss > 0, Energy loss == 0
        loss, le, lp = compute_component_loss(utilization=0.8, umean_levels=umean, alpha=0.15)
        self.assertAlmostEqual(le[0], 0.0)
        self.assertAlmostEqual(lp[0], 0.3)
        self.assertAlmostEqual(loss[0], (1.0 - 0.15) * 0.3)

    def test_joint_loss_shape(self):
        """Joint loss must yield an N x M matrix."""
        lc = np.array([0.1, 0.2, 0.3])
        lm = np.array([0.4, 0.5])
        joint = compute_total_joint_loss(lc, lm, gamma=0.3)
        self.assertEqual(joint.shape, (3, 2))
        self.assertAlmostEqual(joint[0, 0], 0.3 * 0.1 + 0.7 * 0.4)

    def test_wma_core_bounded_workload(self):
        """Core-bounded workload (high core util, low mem util) should throttle memory frequency."""
        # Emulate nbody workload: 95% core util, 10% mem util over multiple intervals
        for _ in range(10):
            fc, fm = self.controller.update(core_utilization=95.0, memory_utilization=10.0)

        # Core frequency should converge to high (576 MHz or 513 MHz)
        # Memory frequency should be throttled to save power (500 MHz or 580 MHz)
        self.assertIn(fc, [513, 576])
        self.assertIn(fm, [500, 580])

    def test_wma_memory_bounded_workload(self):
        """Memory-bounded workload (low core util, high mem util) should throttle core frequency."""
        # Emulate streamcluster workload: 10% core util, 90% mem util
        self.controller.reset()
        for _ in range(10):
            fc, fm = self.controller.update(core_utilization=10.0, memory_utilization=90.0)

        # Core should throttle to low (297 MHz or 356 MHz), memory should stay high (820 or 900 MHz)
        self.assertIn(fc, [297, 356])
        self.assertIn(fm, [820, 900])


if __name__ == "__main__":
    unittest.main()
