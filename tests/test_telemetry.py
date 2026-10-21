"""
Unit tests for Telemetry Subsystem.
Verifies Mock, CPU, and native NVML telemetry capture.
"""

import unittest

from greengpu.telemetry.mock_monitor import MockTelemetryMonitor
from greengpu.telemetry.cpu_monitor import CPUSnapshotMonitor
from greengpu.telemetry.nvml_monitor import NVMLTelemetryMonitor


class TestTelemetry(unittest.TestCase):

    def test_mock_monitor(self):
        """Mock monitor must return accurate deterministic snapshots."""
        monitor = MockTelemetryMonitor(
            gpu_util_core=80.0,
            gpu_util_memory=30.0,
            gpu_power_watts=50.0
        )
        snap = monitor.sample()
        self.assertEqual(snap.gpu_util_core, 80.0)
        self.assertEqual(snap.gpu_util_memory, 30.0)
        self.assertEqual(snap.gpu_power_watts, 50.0)

    def test_cpu_monitor(self):
        """CPU monitor must produce valid utilization numbers."""
        monitor = CPUSnapshotMonitor()
        snap = monitor.sample()
        self.assertGreaterEqual(snap.cpu_util_total, 0.0)
        self.assertLessEqual(snap.cpu_util_total, 100.0)

    def test_nvml_monitor_graceful_init(self):
        """NVML monitor must initialize and query when NVIDIA GPU is available."""
        try:
            monitor = NVMLTelemetryMonitor(device_index=0)
            snap = monitor.sample()
            self.assertGreaterEqual(snap.gpu_power_watts, 0.0)
            self.assertGreaterEqual(snap.gpu_clock_core_mhz, 0)
            monitor.close()
        except RuntimeError as e:
            # If run in an environment without GPU driver, should fail gracefully
            self.assertIn("NVML", str(e))


if __name__ == "__main__":
    unittest.main()
