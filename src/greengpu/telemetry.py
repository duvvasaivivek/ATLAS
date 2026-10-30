"""Read-only telemetry adapters with per-query acquisition bounds and units."""

import importlib.metadata
import time
from pathlib import Path

from .energy import finite_nonnegative


def acquire(read, raw_unit, scale=1.0):
    start = time.monotonic_ns()
    try:
        raw = read()
        value = raw * scale
        status, reason = ("ok", None) if finite_nonnegative(value) else ("invalid", "nonfinite_or_negative")
        if raw_unit == "%" and status == "ok" and raw > 100:
            status, reason = "invalid", "percentage_out_of_range"
        if isinstance(raw, float) and not finite_nonnegative(raw):
            raw = repr(raw)  # Retain invalid evidence without emitting nonstandard JSON NaN.
    except Exception as error:
        raw, value, status, reason = None, None, "unavailable", f"{type(error).__name__}: {error}"
    end = time.monotonic_ns()
    return dict(status=status, raw=raw, raw_unit=raw_unit, value=value if status == "ok" else None,
                reason=reason, acquisition_start_ns=start, acquisition_end_ns=end,
                time_ns=(start + end) // 2, timestamp_uncertainty_ns=(end - start + 1) // 2)


class NvmlTelemetry:
    """One device, one NVML session. No setters or implicit device selection."""

    def __init__(self):
        import pynvml as nv
        self.nv = nv
        nv.nvmlInit()
        try:
            if nv.nvmlDeviceGetCount() != 1:
                raise RuntimeError("Calibration currently requires exactly one visible NVML GPU")
            self.handle = nv.nvmlDeviceGetHandleByIndex(0)
            self.identity = dict(name=nv.nvmlDeviceGetName(self.handle), uuid=nv.nvmlDeviceGetUUID(self.handle),
                                 driver=nv.nvmlSystemGetDriverVersion(), binding=importlib.metadata.version("nvidia-ml-py"))
        except BaseException:
            nv.nvmlShutdown()
            raise
        try:
            capability = nv.nvmlDeviceGetCudaComputeCapability(self.handle)
            averaged = tuple(capability) >= (8, 0) and tuple(capability) != (8, 0)
            averaging_ns = 1_000_000_000 if averaged else 0
        except nv.NVMLError:
            capability, averaging_ns = None, None
        self.metadata = dict(
            domain="gpu_board", identity=self.identity,
            power=dict(api="nvmlDeviceGetPowerUsage", unit="W",
                       semantics="1 second average on Ampere except GA100 and newer; instantaneous on older/GA100",
                       averaging_window_ns=averaging_ns, compute_capability=capability,
                       sensor_update_cadence_ns=None, device_timestamp_available=False),
            counter=dict(api="nvmlDeviceGetTotalEnergyConsumption", unit="J", raw_unit="mJ",
                         semantics="cumulative since driver reload", resolution_j=0.001,
                         reset_detection="same live session plus monotonic trace; undetected reset cannot be excluded externally"),
            calibration="not externally calibrated", attribution="whole GPU domain; other workloads may contribute")

    def counter(self):
        return acquire(lambda: self.nv.nvmlDeviceGetTotalEnergyConsumption(self.handle), "mJ", 0.001)

    def sample(self):
        start = time.monotonic_ns()
        cpu_start = time.thread_time_ns()
        nv, h = self.nv, self.handle
        readings = dict(energy_j=self.counter(),
                        power_w=acquire(lambda: nv.nvmlDeviceGetPowerUsage(h), "mW", 0.001),
                        utilization=acquire(lambda: nv.nvmlDeviceGetUtilizationRates(h).gpu, "%", 0.01),
                        temperature_c=acquire(lambda: nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU), "degC"),
                        graphics_mhz=acquire(lambda: nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_GRAPHICS), "MHz"),
                        memory_mhz=acquire(lambda: nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_MEM), "MHz"))
        return dict(kind="telemetry", acquisition_start_ns=start, acquisition_end_ns=time.monotonic_ns(),
                    query_cpu_ns=time.thread_time_ns() - cpu_start, readings=readings)

    def close(self):
        if not getattr(self, "worker_stuck", False):
            self.nv.nvmlShutdown()


class RaplTelemetry:
    """Select only package-N powercap zones; never add core/DRAM child zones.

    Exposes counters and ranges. Interpretation requires reset exclusion and a
    user-justified maximum package power, not the advisory power limit setting.
    """

    def __init__(self, root=Path("/sys/class/powercap")):
        self.zones = []
        self.errors = []
        seen = set()
        # Explicit one-level glob also follows class symlinks on Linux sysfs.
        for name_file in sorted(set(root.glob("*/name")) | set(root.glob("**/name"))):
            zone = name_file.parent
            try:
                resolved = zone.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                name = name_file.read_text().strip()
                if name.startswith("package-") and (zone / "energy_uj").exists():
                    self.zones.append(dict(path=str(zone), name=name))
            except OSError as error:
                self.errors.append(str(error))
        self.metadata = dict(domain="cpu_package", status="available" if self.zones else "unavailable",
                             zones=self.zones, errors=self.errors,
                             reason=None if self.zones else "No readable package energy zones exposed",
                             aggregation="package zones only; nested core/DRAM zones excluded")

    def sample(self):
        return [dict(zone=zone, energy_j=acquire(lambda p=Path(zone["path"]): int((p / "energy_uj").read_text()), "uJ", 1e-6),
                     range_j=acquire(lambda p=Path(zone["path"]): int((p / "max_energy_range_uj").read_text()), "uJ", 1e-6))
                for zone in self.zones]


WALL_METER = dict(domain="system_wall", status="unavailable", joules=None,
                  reason="No external wall meter configured; component counters are not wall energy")
