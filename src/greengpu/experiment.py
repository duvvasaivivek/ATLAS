"""Versioned, append-only raw trials and deterministic summary replay."""

import hashlib
import json
import math
import statistics
import threading
import time
from pathlib import Path

from .energy import counter_delta, integrate_power, invalid

SCHEMA_VERSION = "2.0"


def encode(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def digest(value):
    return hashlib.sha256(encode(value).encode()).hexdigest()


def save_new(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


class Recorder:
    def __init__(self, path):
        self.stream = Path(path).open("x", encoding="utf-8", newline="\n")
        self.lock = threading.Lock()
        self.sequence = 0

    def write(self, value):
        with self.lock:
            record = dict(value, sequence=self.sequence, recorded_ns=time.monotonic_ns(), schema_version=SCHEMA_VERSION)
            self.stream.write(encode(record) + "\n")
            self.stream.flush()
            self.sequence += 1

    def close(self):
        self.stream.close()


def read_records(path):
    records = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("schema_version") != SCHEMA_VERSION or record.get("sequence") != len(records):
            raise ValueError("Unsupported schema or discontinuous raw sequence")
        records.append(record)
    return records


def summarize_trial(records):
    """The same pure function is used at collection and on later replay."""
    def one(kind):
        matches = [r for r in records if r["kind"] == kind]
        if len(matches) != 1:
            raise ValueError(f"Expected one {kind} record; found {len(matches)}")
        return matches[0]

    result = dict(status="invalid", reasons=[], energy=dict(domain="gpu_board", joules=None, status="invalid"))
    try:
        config = one("trial_config")
        result.update(trial_id=config["trial_id"], block=config["block"], monitored=config["monitored"], config_hash=digest(config["settings"]))
        ready, complete, validation, finish = (one(k) for k in ("ready", "complete", "validation", "finish"))
        before, after = (one(k)["reading"] for k in ("counter_before", "counter_after"))
        start, end = before["time_ns"], after["time_ns"]
        if ready["clock"] != "CLOCK_MONOTONIC" or not (ready["warmup_end_ns"] <= start <= complete["start_ns"] <= complete["h2d_end_ns"] <= complete["compute_end_ns"] <= complete["end_ns"] <= end):
            raise ValueError("Invalid timing order or incompatible native/host clock")
        settings = config["settings"]
        if any(complete[k] != settings[k] for k in ("elements", "repetitions", "rounds", "seed")):
            raise ValueError("Executed work does not match frozen configuration")
        if validation["status"] != "passed" or validation["mismatches"] != 0 or validation["elements_verified"] != settings["elements"] or validation["oracle"] != "masked_uint64_xorshift" or validation["tolerance"] != 0:
            raise ValueError("Incorrect or unverified output")
        if finish["returncode"] != 0 or finish.get("sampler_error") or finish.get("failures"):
            raise ValueError("Worker or sampler failed")
        if not math.isfinite(complete["gpu_event_ms"]) or complete["gpu_event_ms"] < 0:
            raise ValueError("Invalid CUDA event duration")
        result.update(status="ok", duration_s=(end-start)/1e9,
                      native_duration_s=(complete["end_ns"]-complete["start_ns"])/1e9,
                      gpu_event_s=complete["gpu_event_ms"]/1000,
                      preparation_s=(ready["warmup_end_ns"]-ready["preparation_start_ns"])/1e9,
                      warmup_s=(ready["warmup_end_ns"]-ready["warmup_start_ns"])/1e9,
                      h2d_s=(complete["h2d_end_ns"]-complete["start_ns"])/1e9,
                      d2h_and_sync_s=(complete["end_ns"]-complete["compute_end_ns"])/1e9,
                      boundary_overhead_s=((complete["start_ns"]-start)+(end-complete["end_ns"]))/1e9,
                      boundary_uncertainty_ns=before["timestamp_uncertainty_ns"]+after["timestamp_uncertainty_ns"],
                      lifecycle_s=(finish["time_ns"]-config["lifecycle_start_ns"])/1e9,
                      recovery="not_needed_no_hardware_settings_changed")
        host_compute_span = (complete["compute_end_ns"]-complete["h2d_end_ns"])/1e9
        timing_rule = settings.get("gpu_event_crosscheck", dict(relative_allowance=0.01, absolute_allowance_s=0.001))
        timing_allowance = timing_rule["absolute_allowance_s"] + timing_rule["relative_allowance"]*host_compute_span
        event_reported = complete["gpu_event_ms"]/1000
        timing_consistent = event_reported <= host_compute_span + timing_allowance
        result["gpu_event_reported_s"] = event_reported
        result["gpu_event_quality"] = dict(status="consistent" if timing_consistent else "inconsistent",
            host_enclosing_compute_s=host_compute_span, allowance_s=timing_allowance,
            note="A synchronized device event span should fit within its enclosing host interval; agreement is not clock calibration")
        if not timing_consistent:
            result["gpu_event_s"] = None
            result["reasons"].append("device_event_exceeds_enclosing_host_interval")
        telemetry = [r for r in records if r["kind"] == "telemetry"]
        power = [r["readings"]["power_w"] for r in telemetry]
        power_result = integrate_power(power, start, end, settings["max_gap_ns"])
        power_result.update(domain="gpu_board", qualification="sensor-reported estimate; averaging and acquisition uncertainty retained")
        counter_result = counter_delta(before, after, same_epoch=True)
        # A loss/reset anywhere in the measured interval invalidates a delta;
        # endpoints alone cannot prove continuity through a driver reload.
        interior = [r["readings"]["energy_j"] for r in telemetry
                    if start < r["readings"]["energy_j"]["time_ns"] < end]
        counter_trace = [before] + interior + [after]
        if counter_result["status"] == "ok" and any(
            a["status"] != "ok" or b["status"] != "ok" or b["value"] < a["value"]
            for a,b in zip(counter_trace, counter_trace[1:])):
            counter_result = invalid("counter_discontinuity_or_read_failure", method="counter_delta", start_ns=start, end_ns=end)
        counter_result.update(domain="gpu_board", qualification="same NVML session; no externally guaranteed reset exclusion")
        # Fall back for unsupported counters, never conceal a detected reset.
        if counter_result["status"] == "ok":
            energy = counter_result
        elif before["status"] == "unavailable" and after["status"] == "unavailable":
            energy = power_result
        else:
            energy = counter_result
        crosscheck = dict(status="unresolved", reason="Requires both sensor methods and a sufficiently long interval")
        qc = settings.get("counter_power_crosscheck")
        averaging_ns = settings.get("averaging_window_ns")
        if qc and averaging_ns is not None and result["duration_s"] >= qc["min_duration_s"] and counter_result["status"] == power_result["status"] == "ok":
            observed = [p["value"] for p in power if p["status"] == "ok"]
            difference = abs(counter_result["joules"] - power_result["joules"])
            allowance = qc["relative_allowance"]*max(counter_result["joules"], power_result["joules"]) + averaging_ns/1e9*(max(observed)-min(observed))
            crosscheck = dict(status="consistent" if difference <= allowance else "inconsistent",
                              difference_j=difference, allowance_j=allowance,
                              note="Both methods may share sensor error; agreement is not external calibration")
            if crosscheck["status"] == "inconsistent":
                energy = invalid("counter_power_disagreement", domain="gpu_board", start_ns=start, end_ns=end)
        elif qc and config["monitored"] and counter_result["status"] == "ok":
            energy = invalid("counter_power_crosscheck_unresolved", domain="gpu_board", start_ns=start, end_ns=end)
        result.update(energy=energy, counter_energy=counter_result, sampled_power_energy=power_result,
                      counter_power_crosscheck=crosscheck,
                      energy_claim_eligible=False,
                      energy_claim_reason="Calibration fixture only; sensor accuracy and competing-process attribution not externally established",
                      telemetry_samples=len(telemetry), query_cpu_s=sum(r["query_cpu_ns"] for r in telemetry)/1e9,
                      query_wall_s=sum(r["acquisition_end_ns"]-r["acquisition_start_ns"] for r in telemetry)/1e9)
        if config["monitored"] and energy["status"] != "ok":
            result["reasons"].append("energy_unavailable: " + energy["reason"])
        changes = [b["time_ns"] for a,b in zip(power,power[1:]) if a["status"] == b["status"] == "ok" and a["value"] != b["value"]]
        result["power_change_intervals_s"] = [(b-a)/1e9 for a,b in zip(changes,changes[1:])]
        result["sensor_cadence_note"] = "Observed change spacing is not a guaranteed sensor update period; repeated readings are not automatically stale"
        result["max_poll_gap_s"] = max(((b["time_ns"]-a["time_ns"])/1e9 for a,b in zip(power,power[1:])), default=None)
    except (ValueError, KeyError, TypeError) as error:
        result["status"] = "invalid"
        result["reasons"].append(str(error))
        result["energy"] = invalid("invalid_trial") | {"domain":"gpu_board"}
    return result


def distribution(values):
    if not values:
        return dict(n=0, mean=None, stdev=None, min=None, max=None)
    return dict(n=len(values), mean=statistics.mean(values),
                stdev=statistics.stdev(values) if len(values)>1 else None, min=min(values), max=max(values))


def summarize_run(root):
    root = Path(root)
    trials = [summarize_trial(read_records(path)) for path in sorted(root.glob("trial-*.jsonl"))]
    paired = []
    for block in sorted({t.get("block") for t in trials if isinstance(t.get("block"), int)}):
        group = [t for t in trials if t.get("block") == block]
        if len(group) == 2 and all(t["status"] == "ok" for t in group) and len({t["config_hash"] for t in group}) == 1 and {t["monitored"] for t in group} == {True,False}:
            measured = next(t for t in group if t["monitored"])
            control = next(t for t in group if not t["monitored"])
            paired.append(dict(block=block, overhead_percent=100*(measured["duration_s"]/control["duration_s"]-1)))
    measured = [t for t in trials if t.get("monitored") and t["status"] == "ok"]
    return dict(schema_version=SCHEMA_VERSION, execution_mode="hardware_calibration", trials=trials,
                energy_qualification="blocked" if any(t.get("counter_power_crosscheck",{}).get("status")=="inconsistent" for t in trials) else "requires_external_validation",
                device_timing_qualification="blocked" if any(t.get("gpu_event_quality",{}).get("status")=="inconsistent" for t in trials) else "requires_external_validation",
                paired_overhead=paired, overhead_percent=distribution([p["overhead_percent"] for p in paired]),
                monitored_duration_s=distribution([t["duration_s"] for t in measured]),
                monitored_gpu_j=distribution([t["energy"]["joules"] for t in measured if t["energy"]["status"]=="ok"]),
                interpretation="Exploratory paired trials; overhead may be negative within drift/noise. Samples are not independent trials.",
                system_wall=dict(status="unavailable", joules=None, reason="No wall meter configured"))
