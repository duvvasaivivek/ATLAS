"""Read-only capability snapshots; no clock, power, or governor writes.

Phase 1 uses the installed nvidia-smi XML interface without Python runtime
dependencies. Phase 2 will use a versioned telemetry adapter suitable for
continuous sampling. Raw outputs are retained because XML fields can change.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def command(argv: list[str], timeout: float = 30) -> dict:
    started = utc_now()
    begin = time.monotonic()
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout,
                                encoding="utf-8", errors="replace", check=False)
        status = "ok" if result.returncode == 0 else "command_failed"
        record = dict(status=status, returncode=result.returncode,
                      stdout=result.stdout, stderr=result.stderr)
    except FileNotFoundError as error:
        record = dict(status="not_installed", returncode=None, stdout="", stderr=str(error))
    except subprocess.TimeoutExpired as error:
        def decode(value):
            return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value or ""
        record = dict(status="timeout", returncode=None,
                      stdout=decode(error.stdout), stderr=decode(error.stderr))
    except OSError as error:
        record = dict(status="os_error", returncode=None, stdout="", stderr=str(error))
    return dict(argv=argv, started_utc=started, elapsed_s=time.monotonic()-begin, **record)


def numeric_reading(raw: str | None, unit: str, scale: float = 1.0) -> dict:
    result = dict(raw=raw, value=None, raw_unit=unit,
                  unit="fraction" if unit == "%" else unit,
                  quality="unvalidated_snapshot")
    if raw is None:
        return dict(result, status="not_reported")
    raw = raw.strip()
    if raw in ("N/A", "Not Supported", "[Not Supported]", ""):
        return dict(result, status="unavailable")
    if "deprecated" in raw.lower():
        return dict(result, status="deprecated")
    match = re.fullmatch(r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*([^\d]*)", raw)
    if not match or match.group(2).strip() != unit:
        return dict(result, status="unrecognized", reason="Unexpected numeric value or unit")
    value = float(match.group(1)) * scale
    if not math.isfinite(value) or value < 0 or (unit == "%" and value > 1):
        return dict(result, status="invalid", reason="Value outside declared range")
    return dict(result, status="available", value=value,
                unit="fraction" if unit == "%" else unit)


def parse_gpu_xml(xml: str) -> dict:
    root = ET.fromstring(xml)
    devices = []
    fields = {
        "gpu_activity": ("utilization/gpu_util", "%", 0.01),
        "memory_activity": ("utilization/memory_util", "%", 0.01),
        "temperature": ("temperature/gpu_temp", "C", 1),
        "power_average": ("gpu_power_readings/average_power_draw", "W", 1),
        "power_instant": ("gpu_power_readings/instant_power_draw", "W", 1),
        "power_legacy": ("power_readings/power_draw", "W", 1),
        "power_limit_reported": ("gpu_power_readings/current_power_limit", "W", 1),
        "graphics_clock": ("clocks/graphics_clock", "MHz", 1),
        "sm_clock": ("clocks/sm_clock", "MHz", 1),
        "memory_clock": ("clocks/mem_clock", "MHz", 1),
        "memory_total": ("fb_memory_usage/total", "MiB", 1),
        "application_graphics_clock": ("applications_clocks/graphics_clock", "MHz", 1),
        "application_memory_clock": ("applications_clocks/mem_clock", "MHz", 1),
    }
    for gpu in root.findall("gpu"):
        readings = {name: numeric_reading(gpu.findtext(path), unit, scale)
                    for name, (path, unit, scale) in fields.items()}
        pairs = []
        for memory in gpu.findall("supported_clocks/supported_mem_clock"):
            mem = numeric_reading(memory.findtext("value"), "MHz")
            for core in memory.findall("supported_graphics_clock"):
                graphics = numeric_reading(core.text, "MHz")
                if mem["status"] == graphics["status"] == "available":
                    pairs.append([graphics["value"], mem["value"]])
        devices.append(dict(
            name=gpu.findtext("product_name"), uuid=gpu.findtext("uuid"),
            pci_bus_id=gpu.get("id"), architecture=gpu.findtext("product_architecture"),
            readings=readings,
            reported_clock_pairs_mhz=pairs,
            clock_pair_columns=["graphics", "memory"],
            clock_pair_validation="reported_only_not_actuated",
            limiter_snapshot={x.tag: x.text for x in gpu.findall("clocks_event_reasons/*")},
            utilization_semantics="activity_time_fraction_not_bandwidth_or_allocated_memory",
            energy_counter=dict(status="not_probed", reason="Not exposed by this Phase 1 XML adapter"),
            management_writes=dict(status="unverified", attempted=False,
                                   reason="Read-only Phase 1; support and privileges require Phase 8 verification"),
        ))
    return dict(status="available" if devices else "no_devices", devices=devices,
                driver_version=root.findtext("driver_version"),
                driver_reported_cuda_version=root.findtext("cuda_version"))


def read_text(path: Path) -> dict:
    try:
        return dict(status="available", value=path.read_text().strip(), path=str(path))
    except PermissionError as error:
        return dict(status="permission_denied", value=None, path=str(path), reason=str(error))
    except FileNotFoundError:
        return dict(status="not_exposed", value=None, path=str(path))
    except OSError as error:
        return dict(status="read_error", value=None, path=str(path), reason=str(error))


def cpu_interfaces(sys_root: Path = Path("/sys")) -> dict:
    # The class entries themselves may be symlinks; glob each exposed zone,
    # rather than assuming recursive traversal follows sysfs symlinks.
    powercap = sys_root / "class/powercap"
    zones = []
    for entry in sorted(powercap.glob("*")):
        if (entry / "energy_uj").exists():
            zones.append(dict(name=read_text(entry / "name"),
                              energy_uj=read_text(entry / "energy_uj"),
                              max_energy_range_uj=read_text(entry / "max_energy_range_uj")))
    governors = [read_text(p) for p in sorted(
        (sys_root / "devices/system/cpu/cpufreq").glob("policy*/scaling_governor"))]
    return dict(
        package_energy=dict(status="exposed" if zones else "not_exposed", zones=zones,
                            reason="Only Linux powercap interfaces checked; absence is not a claim about Windows sensors"),
        governors=dict(status="exposed" if governors else "not_exposed", policies=governors,
                       writes_attempted=False),
        whole_system_energy=dict(status="not_configured", reason="No external meter adapter configured"),
    )


def evaluate_smoke(record: dict) -> dict:
    if record["status"] != "ok":
        return dict(status="failed", reason="Smoke process did not exit successfully")
    try:
        data = json.loads(record["stdout"])
        sizes = [1, 255, 256, 257, 4099, 1048579]
        valid = (data.get("status") == "passed" and data.get("sizes") == sizes
                 and data.get("elements_verified_per_backend") == sum(sizes)
                 and data.get("max_absolute_error") == 0
                 and data.get("openmp_workers", 0) >= 1
                 and data.get("comparison") == "exact_against_integer_oracle")
        return dict(status="passed" if valid else "failed", result=data)
    except (ValueError, TypeError, AttributeError):
        return dict(status="failed", reason="Smoke output is not a valid result record")


def collect(output: Path, cuda_compiler: str | None, smoke_binary: Path | None) -> dict:
    # Do not overwrite prior evidence. Each invocation needs a new directory.
    output.mkdir(parents=True, exist_ok=False)
    raw_dir = output / "raw"
    raw_dir.mkdir()
    commands = {}

    def run(name, argv, timeout=30):
        result = command(argv, timeout)
        commands[name] = result
        (raw_dir / f"{name}.txt").write_text(result["stdout"], encoding="utf-8")
        return result

    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi and Path("/usr/lib/wsl/lib/nvidia-smi").is_file():
        nvidia_smi = "/usr/lib/wsl/lib/nvidia-smi"
    nvcc = cuda_compiler or shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    cpu = run("lscpu", ["lscpu", "--json"])
    tools = {name: run(name, [executable, flag]) for name, executable, flag in (
        ("nvcc", nvcc, "--version"), ("gxx", "g++", "--version"),
        ("cmake", "cmake", "--version"), ("ninja", "ninja", "--version"),
        ("git", "git", "--version"))}
    packages = run("dpkg", ["dpkg-query", "-W", "build-essential", "g++", "cmake", "ninja-build",
                           "python3-dev", "python3-venv", "cuda-toolkit-13-1", "cuda-nvcc-13-1"])
    run("pip_freeze", [sys.executable, "-m", "pip", "freeze", "--all"])
    revision = run("git_revision", ["git", "rev-parse", "HEAD"])
    tree = run("git_status", ["git", "status", "--porcelain"])
    smi = run("nvidia_smi", [nvidia_smi or "nvidia-smi", "-q", "-x"])
    gpu = dict(status=smi["status"], devices=[])
    if smi["status"] == "ok":
        try:
            gpu = parse_gpu_xml(smi["stdout"])
        except ET.ParseError as error:
            gpu = dict(status="parse_error", devices=[], reason=str(error))
    interfaces = cpu_interfaces()
    smoke = dict(status="not_run")
    if smoke_binary:
        smoke_path = smoke_binary.resolve()
        result = run("smoke", [str(smoke_path)], timeout=60)
        smoke = evaluate_smoke(result)
        if smoke_path.is_file():
            smoke["binary_sha256"] = hashlib.sha256(smoke_path.read_bytes()).hexdigest()
        cache = smoke_path.parent / "CMakeCache.txt"
        if cache.is_file():
            (raw_dir / "CMakeCache.txt").write_text(cache.read_text(), encoding="utf-8")
    env = dict(
        schema_version="1.0", captured_utc=utc_now(), platform=platform.platform(),
        os_release=read_text(Path("/etc/os-release")), kernel=platform.release(),
        is_wsl="microsoft" in platform.release().lower(),
        python=dict(version=sys.version, executable=sys.executable, prefix=sys.prefix,
                    base_prefix=sys.base_prefix, virtual_environment=sys.prefix != sys.base_prefix),
        visible_logical_cpus=os.cpu_count(), cpu_inventory=cpu,
        memory=read_text(Path("/proc/meminfo")), tools=tools, packages=packages,
        process_configuration={key: os.environ.get(key) for key in (
            "OMP_NUM_THREADS", "OMP_DYNAMIC", "CUDA_VISIBLE_DEVICES", "CUDACXX")},
        git_revision=revision["stdout"].strip(), git_status=tree["stdout"],
        topology_scope="Guest-visible topology; do not infer physical hybrid core count from WSL lscpu",
    )
    capabilities = dict(
        schema_version="1.0", captured_utc=env["captured_utc"], execution_mode="hardware_discovery",
        gpu=gpu, cpu_interfaces=interfaces, smoke=smoke,
        chosen_path="hardware_workloads_with_partial_telemetry" if smoke["status"] == "passed" else "hardware_execution_unverified",
        full_energy_control_replication="unverified",
        energy_measurement_validated=False, hardware_settings_changed=False,
    )
    hashes = {}
    for directory in (Path("src"), Path("native"), Path("tests")):
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix in (".py", ".cpp", ".cu", ".hpp", ".h"):
                hashes[path.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ("CMakeLists.txt", "pyproject.toml"):
        path = Path(name)
        if path.is_file():
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name, value in (("capabilities.json", capabilities), ("environment.json", env),
                        ("commands.json", commands), ("source_hashes.json", hashes), ("smoke_result.json", smoke)):
        (output / name).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return capabilities
