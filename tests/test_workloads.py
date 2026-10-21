"""
Unit tests for Heterogeneous Workloads and Hardware Simulator.
Verifies cooperative execution, partitioning behavior, and simulation accuracy.
"""

import unittest
from greengpu.workloads.kmeans import HeterogeneousKMeans
from greengpu.workloads.hotspot import HeterogeneousHotSpot
from greengpu.workloads.synthetic_bench import SyntheticHeterogeneousBenchmark
from greengpu.simulator import CalibratedGreenGPUSimulator


class TestHeterogeneousWorkloads(unittest.TestCase):

    def test_kmeans_partitioning(self):
        """KMeans must partition work and execute across CPU and GPU."""
        workload = HeterogeneousKMeans(
            num_points=5_000,
            num_features=16,
            num_clusters=4,
            use_gpu=False  # test host execution fallback
        )
        # Test 50/50 split
        res = workload.run_iteration(iteration=1, cpu_ratio=0.50)
        self.assertEqual(res.iteration, 1)
        self.assertEqual(res.cpu_ratio, 0.50)
        self.assertGreater(res.time_wall_sec, 0.0)
        self.assertGreaterEqual(res.time_cpu_sec, 0.0)
        self.assertGreaterEqual(res.time_gpu_sec, 0.0)

    def test_hotspot_stencil(self):
        """HotSpot must compute 2D stencil and return valid timings."""
        workload = HeterogeneousHotSpot(
            grid_rows=128,
            grid_cols=128,
            steps_per_iteration=2,
            use_gpu=False
        )
        res = workload.run_iteration(iteration=1, cpu_ratio=0.30)
        self.assertEqual(res.cpu_ratio, 0.30)
        self.assertGreater(res.time_wall_sec, 0.0)
        self.assertGreaterEqual(res.idle_time_wasted_sec, 0.0)

    def test_synthetic_benchmarks(self):
        """Synthetic benchmark must execute compute-bound and memory-bound modes."""
        for mode in ["compute_bound", "memory_bound"]:
            bench = SyntheticHeterogeneousBenchmark(
                mode=mode,
                total_work_units=1000,
                dimension=64,
                use_gpu=False
            )
            res = bench.run_iteration(iteration=1, cpu_ratio=0.40)
            self.assertEqual(res.extra_metrics["mode"], mode)
            self.assertGreater(res.time_wall_sec, 0.0)

    def test_calibrated_simulator_optimal_division(self):
        """Simulator must demonstrate that optimal division yields less energy than all-GPU."""
        sim = CalibratedGreenGPUSimulator(workload_type="hotspot")
        
        # All GPU (r = 0.0)
        out_all_gpu = sim.simulate_iteration(iteration=1, cpu_ratio=0.0)
        # Balanced division (r = 0.50 for hotspot)
        out_balanced = sim.simulate_iteration(iteration=1, cpu_ratio=0.50)

        # Balanced division must reduce wall clock time and total energy
        self.assertLess(out_balanced.time_wall_sec, out_all_gpu.time_wall_sec)
        self.assertLess(out_balanced.energy_total_joules, out_all_gpu.energy_total_joules)
        self.assertLess(out_balanced.idle_time_wasted_sec, out_all_gpu.idle_time_wasted_sec)


if __name__ == "__main__":
    unittest.main()
