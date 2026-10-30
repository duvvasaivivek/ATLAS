import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from greengpu.capabilities import (command, cpu_interfaces, evaluate_smoke,
                                  numeric_reading, parse_gpu_xml)
from greengpu.cli import main


class CapabilityTests(unittest.TestCase):
    def test_zero_utilization_is_available_and_normalized(self):
        self.assertEqual(numeric_reading("0 %", "%", .01)["value"], 0)
        reading = numeric_reading("42 %", "%", .01)
        self.assertEqual(reading["status"], "available")
        self.assertEqual(reading["unit"], "fraction")
        self.assertAlmostEqual(reading["value"], .42)

    def test_missing_and_unsupported_do_not_become_zero(self):
        for value in (None, "N/A", "Not Supported", ""):
            with self.subTest(value=value):
                reading = numeric_reading(value, "W")
                self.assertIsNone(reading["value"])
                self.assertNotEqual(reading["status"], "available")
        self.assertEqual(numeric_reading("N/A", "%", .01)["unit"], "fraction")

    def test_unit_changes_and_invalid_values_are_rejected(self):
        for value in ("12000 mW", "NaN W", "Inf W", "-1 W"):
            with self.subTest(value=value):
                self.assertNotEqual(numeric_reading(value, "W")["status"], "available")
        self.assertEqual(numeric_reading("101 %", "%", .01)["status"], "invalid")

    def test_xml_reports_only_and_does_not_claim_clock_writes(self):
        data = parse_gpu_xml('''<nvidia_smi_log><driver_version>test</driver_version>
          <gpu id="test-bus"><product_name>Fixture GPU</product_name>
            <utilization><gpu_util>0 %</gpu_util><memory_util>N/A</memory_util></utilization>
            <applications_clocks><graphics_clock>Requested functionality has been deprecated</graphics_clock></applications_clocks>
            <supported_clocks><supported_mem_clock><value>500 MHz</value>
              <supported_graphics_clock>300 MHz</supported_graphics_clock>
              <supported_graphics_clock>N/A</supported_graphics_clock>
            </supported_mem_clock></supported_clocks>
          </gpu></nvidia_smi_log>''')
        gpu = data["devices"][0]
        self.assertEqual(gpu["readings"]["gpu_activity"]["value"], 0)
        self.assertIsNone(gpu["readings"]["memory_activity"]["value"])
        self.assertEqual(gpu["readings"]["application_graphics_clock"]["status"], "deprecated")
        self.assertEqual(gpu["reported_clock_pairs_mhz"], [[300, 500]])
        self.assertFalse(gpu["management_writes"]["attempted"])
        self.assertEqual(gpu["management_writes"]["status"], "unverified")

    def test_empty_gpu_list_is_not_success(self):
        self.assertEqual(parse_gpu_xml("<nvidia_smi_log/>")["status"], "no_devices")

    def test_absent_cpu_interfaces_are_not_fabricated(self):
        with tempfile.TemporaryDirectory() as folder:
            data = cpu_interfaces(Path(folder))
        self.assertEqual(data["package_energy"]["status"], "not_exposed")
        self.assertEqual(data["governors"]["status"], "not_exposed")

    def test_cpu_zone_and_governor_reading(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            zone = root / "class/powercap/intel-rapl:0"
            zone.mkdir(parents=True)
            for name, value in {"name": "package-0", "energy_uj": "123", "max_energy_range_uj": "1000"}.items():
                (zone/name).write_text(value)
            policy = root / "devices/system/cpu/cpufreq/policy0"
            policy.mkdir(parents=True)
            (policy/"scaling_governor").write_text("performance")
            data = cpu_interfaces(root)
            self.assertEqual(data["package_energy"]["zones"][0]["energy_uj"]["value"], "123")
            self.assertEqual(data["governors"]["policies"][0]["value"], "performance")
            self.assertFalse(data["governors"]["writes_attempted"])

    @patch("greengpu.capabilities.subprocess.run", side_effect=FileNotFoundError("missing"))
    def test_missing_command_is_recorded(self, _):
        self.assertEqual(command(["not-installed"])["status"], "not_installed")

    @patch("greengpu.capabilities.subprocess.run", side_effect=subprocess.TimeoutExpired("probe", 1, b"partial"))
    def test_timeout_retains_partial_evidence(self, _):
        result = command(["probe"], timeout=1)
        self.assertEqual(result["status"], "timeout")
        self.assertEqual(result["stdout"], "partial")

    def test_failed_process_cannot_claim_smoke_success(self):
        result = evaluate_smoke({"status": "command_failed", "stdout": '{"status":"passed"}'})
        self.assertEqual(result["status"], "failed")

    def test_incomplete_or_malformed_smoke_result_fails(self):
        for text in ('{"status":"passed"}', '[]', 'not JSON'):
            with self.subTest(text=text):
                self.assertEqual(evaluate_smoke({"status": "ok", "stdout": text})["status"], "failed")

    def test_complete_smoke_result_and_mismatch(self):
        data = dict(status="passed", sizes=[1, 255, 256, 257, 4099, 1048579],
                    elements_verified_per_backend=1053447, max_absolute_error=0,
                    openmp_workers=2, comparison="exact_against_integer_oracle")
        self.assertEqual(evaluate_smoke({"status": "ok", "stdout": json.dumps(data)})["status"], "passed")
        data["max_absolute_error"] = 1
        self.assertEqual(evaluate_smoke({"status": "ok", "stdout": json.dumps(data)})["status"], "failed")

    def test_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sentinel.txt"
            path.write_text("preserve")
            with self.assertRaises(SystemExit) as error:
                main(["discover", "--output", folder])
            self.assertEqual(error.exception.code, 2)
            self.assertEqual(path.read_text(), "preserve")


if __name__ == "__main__":
    unittest.main()
