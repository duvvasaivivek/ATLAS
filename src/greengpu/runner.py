"""Linux/WSL calibration runner; no frequency or power-policy writes."""

import hashlib
import json
import math
import os
import platform
import queue
import random
import struct
import subprocess
import threading
import time
from pathlib import Path

from .capabilities import collect, utc_now
from .experiment import Recorder, SCHEMA_VERSION, digest, read_records, save_new, summarize_run, summarize_trial
from .telemetry import NvmlTelemetry, RaplTelemetry, WALL_METER
from .supervision import Supervisor
from .sampling import IsolatedSampler, telemetry_factory


class Sampler:
    def __init__(self, adapter, recorder, period_s, lock):
        self.adapter, self.recorder, self.period_s, self.lock = adapter, recorder, period_s, lock
        self.stop_event = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self.run, name="atlas-telemetry", daemon=True)

    def run(self):
        try:
            while not self.stop_event.is_set():
                cycle = time.monotonic()
                with self.lock:
                    sample = self.adapter.sample()
                self.recorder.write(sample)
                self.stop_event.wait(max(0, self.period_s-(time.monotonic()-cycle)))
        except Exception as error:
            self.error = f"{type(error).__name__}: {error}"

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=4)
        if self.thread.is_alive():
            # Do not close NVML/recorder beneath a live worker.
            self.adapter.worker_stuck = True
            raise RuntimeError("NVML sampler did not stop within 4 seconds")


def trial(path, binary, adapter, rapl, settings, trial_id, block, monitored):
    recorder = Recorder(path)
    proc, sampler, supervisor, isolated = None, None, None, None
    start = time.monotonic_ns()
    recorder.write(dict(kind="trial_config", trial_id=trial_id, block=block, monitored=monitored,
                        settings=settings, lifecycle_start_ns=start, utc=utc_now()))
    errors = []
    lock = threading.Lock()
    lines = queue.Queue()
    stderr_path = Path(path).with_suffix(".stderr.txt")
    try:
        with stderr_path.open("x", encoding="utf-8") as stderr:
            supervisor = Supervisor([str(binary), *[str(settings[k]) for k in ("elements", "repetitions", "rounds", "seed")]],
                                    stderr, settings["timeout_s"]*3,
                                    failure=lambda: sampler.error if sampler else None)
            proc = supervisor.process
            isolated = IsolatedSampler(telemetry_factory) if isinstance(adapter,NvmlTelemetry) else None
            reader = isolated or adapter

            def receive(kind):
                line = supervisor.receive(settings["timeout_s"])
                recorder.write(dict(kind="native_stdout", line=line, received_ns=time.monotonic_ns()))
                if line is None:
                    raise RuntimeError(f"Native worker exited before {kind}")
                record = json.loads(line)
                if record.get("kind") != kind:
                    raise RuntimeError(f"Expected {kind}, received {record.get('kind')}")
                recorder.write(record)
                return record

            def send(value):
                proc.stdin.write(value + "\n")
                proc.stdin.flush()

            ready = receive("ready")
            # NVML and CUDA ordinals can differ; compare PCI identity explicitly.
            pci = adapter.nv.nvmlDeviceGetPciInfo(adapter.handle).busId
            if isinstance(pci, bytes):
                pci = pci.decode()
            if pci.lower().lstrip("0") != ready["pci_bus_id"].lower().lstrip("0"):
                raise RuntimeError("CUDA and NVML identify different GPUs")
            if monitored:
                sampler = Sampler(reader, recorder, settings["period_s"], lock)
                sampler.thread.start()
            # Equal settling buffers in both arms; outside primary time/energy.
            time.sleep(settings["buffer_s"])
            recorder.write(dict(kind="rapl_before", readings=rapl.sample()))
            with lock:
                before = reader.counter()
            recorder.write(dict(kind="counter_before", reading=before))
            send("run")
            receive("complete")
            with lock:
                after = reader.counter()
            recorder.write(dict(kind="counter_after", reading=after))
            recorder.write(dict(kind="rapl_after", readings=rapl.sample()))
            send("validate")
            receive("validation")
            proc.wait(timeout=settings["timeout_s"])
            time.sleep(settings["buffer_s"])
    except (Exception, KeyboardInterrupt) as error:
        errors.append(f"{type(error).__name__}: {error}")
        recorder.write(dict(kind="failure", reason=errors[-1]))
    finally:
        if supervisor is not None:
            supervisor.close()
        if sampler is not None:
            sampler.stop()
        if isolated is not None:
            isolated.close()
        recorder.write(dict(kind="finish", returncode=proc.returncode if proc else None,
                            sampler_error=sampler.error if sampler else None,
                            failures=errors, time_ns=time.monotonic_ns(), hardware_settings_changed=False))
        recorder.close()
    return summarize_trial(read_records(path))


def run_calibration(output, binary, blocks=3, target_s=6.0, period_s=0.1, seed=1729):
    if platform.system() != "Linux" or time.get_clock_info("monotonic").implementation != "clock_gettime(CLOCK_MONOTONIC)":
        raise ValueError("Use Linux/WSL with CLOCK_MONOTONIC for matching native and Python timestamps")
    if not 1 <= blocks <= 30 or not 1 <= target_s <= 30 or not 0.02 <= period_s <= 1 or not 1 <= seed <= 0xffffffff:
        raise ValueError("Expected blocks 1..30, target seconds 1..30, period .02..1, seed 1..2^32-1")
    binary = Path(binary).resolve(strict=True)
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "", "0"):
        raise ValueError("Calibration requires a single unremapped CUDA GPU")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    adapter = None
    try:
        snapshot = collect(output / "environment", "/usr/local/cuda-13.1/bin/nvcc", None)
        adapter, rapl = NvmlTelemetry(), RaplTelemetry()
        settings = dict(elements=65539, repetitions=4096, rounds=256, seed=seed,
                        period_s=period_s, max_gap_ns=round(period_s*5*1e9), buffer_s=1.1,
                        timeout_s=120, policy="B0-default", precision="uint32_modulo_2^32",
                        generator="index_xor_seed_v1", workload="measurement_xorshift_v1",
                        allocation="GPU only", gpu_clock_policy="unchanged platform automatic policy",
                        cpu_policy="unchanged", controller="none", actuation="none",
                        averaging_window_ns=adapter.metadata["power"]["averaging_window_ns"],
                        gpu_event_crosscheck=dict(relative_allowance=0.01, absolute_allowance_s=0.001),
                        counter_power_crosscheck=dict(min_duration_s=5.0, relative_allowance=0.1,
                            rule="abs(counter-power) <= 10% of larger energy + averaging_seconds * observed power span",
                            purpose="exploratory consistency screen, not an accuracy guarantee"))
        input_bytes = b"".join(struct.pack("<I", i ^ seed) for i in range(settings["elements"]))
        settings["input_sha256"] = hashlib.sha256(input_bytes).hexdigest()
        manifest = dict(schema_version=SCHEMA_VERSION, run_id=output.name, started_utc=utc_now(),
                        mode="hardware_calibration", specification="measurement_spec_v1.0",
                        clock=dict(name="CLOCK_MONOTONIC", unit="ns", resolution_s=time.get_clock_info("monotonic").resolution),
                        gpu=adapter.metadata, cpu=rapl.metadata, wall=WALL_METER,
                        binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(), binary=str(binary),
                        capability_report="environment/capabilities.json",
                        capability_sha256=hashlib.sha256((output/"environment/capabilities.json").read_bytes()).hexdigest(),
                        source_hashes=json.loads((output/"environment/source_hashes.json").read_text()),
                        pilot_settings=settings.copy(), hardware_settings_changed=False,
                        exclusions=["incorrect output", "protocol/worker failure", "unbracketed power trace", "gap beyond frozen limit", "counter discontinuity"],
                        process_attribution="not isolated from Windows/other GPU workloads; no savings claim permitted")
        cache = binary.parent/"CMakeCache.txt"
        if cache.is_file():
            (output/"environment/raw/CMakeCache.txt").write_bytes(cache.read_bytes())
        save_new(output/"manifest.json", manifest)
        pilot = trial(output/"pilot.jsonl", binary, adapter, rapl, settings, "pilot", "pilot", True)
        save_new(output/"pilot-summary.json", pilot)
        if pilot["status"] != "ok" or pilot["native_duration_s"] <= 0:
            raise RuntimeError("Calibration pilot failed; retained raw evidence")
        settings["repetitions"] = min(1000000, max(1, math.ceil(settings["repetitions"]*target_s/pilot["native_duration_s"])))
        # Guard is a declared tolerance, not a claim about hardware resolution.
        settings["max_gap_ns"] = max(settings["max_gap_ns"], math.ceil((pilot["max_poll_gap_s"] or period_s)*3*1e9))
        rng = random.Random(seed)
        order = []
        for block in range(blocks):
            pair = [False, True]
            rng.shuffle(pair)
            order.extend(dict(block=block, monitored=mode) for mode in pair)
        save_new(output/"frozen-config.json", dict(settings=settings, config_hash=digest(settings), order=order,
                  target_s=target_s, stage="exploratory_validation_after_tuning_pilot",
                  gap_rule="max(5 requested periods, 3 maximum pilot gaps)",
                  sensor_cadence="unknown; observed change intervals retained; power averaged over 1 second on Ada"))
        for index, item in enumerate(order):
            trial_id = f"trial-{index:03d}"
            result = trial(output/f"{trial_id}.jsonl", binary, adapter, rapl, settings, trial_id, **item)
            print(json.dumps(dict(trial=trial_id, status=result["status"], duration_s=result.get("duration_s"),
                                  monitored=item["monitored"], gpu_j=result["energy"].get("joules"))), flush=True)
            if result["status"] != "ok":
                raise RuntimeError(f"{trial_id} failed; aborting further GPU trials")
        result = summarize_run(output)
        save_new(output/"summary.json", result)
        checksums = {p.relative_to(output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted(output.rglob("*")) if p.is_file()}
        save_new(output/"checksums.json", checksums)
        return result
    except BaseException as error:
        save_new(output/"failure.json", dict(status="failed", reason=f"{type(error).__name__}: {error}", utc=utc_now()))
        raise
    finally:
        if adapter is not None:
            adapter.close()


def replay(root):
    root = Path(root)
    checksums = json.loads((root/"checksums.json").read_text())
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p != root/"checksums.json"}
    if set(checksums) != actual:
        raise ValueError("Evidence file inventory differs from checksum manifest")
    for name, expected in checksums.items():
        path = (root/name).resolve()
        if not path.is_relative_to(root.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Evidence checksum mismatch: {name}")
    return summarize_run(root)
