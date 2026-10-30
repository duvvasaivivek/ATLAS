"""Pure energy accounting. All times are host CLOCK_MONOTONIC nanoseconds.

No function repairs missing observations, extrapolates, or turns failure into zero.
"""

import math


def invalid(reason, **metadata):
    return dict(status="invalid", joules=None, reason=reason, **metadata)


def finite_nonnegative(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def valid_time(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def integrate_power(samples, start_ns, end_ns, max_gap_ns):
    """Clip a piecewise linear W trace to [start,end], preserving invalid gaps.

    Samples: {time_ns, value (W), status}. Explicit stale status is invalid;
    identical successive values alone are NOT evidence of a stale sensor.
    """
    meta = dict(method="trapezoid", start_ns=start_ns, end_ns=end_ns)
    if not all(valid_time(t) for t in (start_ns,end_ns,max_gap_ns)) or end_ns <= start_ns or max_gap_ns <= 0:
        return invalid("invalid_interval_or_gap_limit", **meta)
    if len(samples) < 2:
        return invalid("insufficient_samples", **meta)
    times = [s["time_ns"] for s in samples]
    if not all(valid_time(t) for t in times) or any(b <= a for a, b in zip(times, times[1:])):
        return invalid("timestamps_not_strictly_increasing", **meta)
    if times[0] > start_ns or times[-1] < end_ns:
        return invalid("unbracketed_boundary", **meta)
    joules = 0.0
    segments = 0
    for a, b in zip(samples, samples[1:]):
        lo, hi = max(start_ns, a["time_ns"]), min(end_ns, b["time_ns"])
        if lo >= hi:
            continue
        if any(s.get("status") != "ok" or not finite_nonnegative(s.get("value")) for s in (a, b)):
            return invalid("invalid_or_missing_sample", **meta)
        dt = b["time_ns"] - a["time_ns"]
        if dt > max_gap_ns:
            return invalid("sampling_gap_exceeded", **meta)
        slope = (b["value"] - a["value"]) / dt
        p_lo = a["value"] + slope * (lo - a["time_ns"])
        p_hi = a["value"] + slope * (hi - a["time_ns"])
        joules += 0.5 * (p_lo + p_hi) * (hi - lo) / 1e9
        segments += 1
    return dict(status="ok", joules=joules, segments=segments, **meta)


def counter_delta(before, after, *, same_epoch, range_j=None, max_power_w=None):
    """Counter samples are joules. RAPL wrap requires a physical accumulation bound.

    same_epoch is a caller assertion: resetting/reloading a driver must invalidate it.
    An unknown-width NVML decrease is never treated as a wrap.
    """
    meta = dict(method="counter_delta", start_ns=before["time_ns"], end_ns=after["time_ns"])
    if not all(valid_time(s["time_ns"]) for s in (before,after)):
        return invalid("invalid_counter_timestamp", **meta)
    if not same_epoch:
        return invalid("reset_or_epoch_unknown", **meta)
    if any(s.get("status") != "ok" or not finite_nonnegative(s.get("value")) for s in (before, after)):
        return invalid("counter_unavailable", **meta)
    dt = (after["time_ns"] - before["time_ns"]) / 1e9
    if dt <= 0:
        return invalid("invalid_counter_interval", **meta)
    delta = after["value"] - before["value"]
    if range_j is not None:
        if not finite_nonnegative(range_j) or range_j == 0 or any(s["value"] >= range_j for s in (before, after)):
            return invalid("invalid_counter_range", **meta)
        if not finite_nonnegative(max_power_w) or max_power_w * dt >= range_j:
            return invalid("multiple_wraps_not_excluded", **meta)
        delta %= range_j
    elif delta < 0:
        return invalid("counter_decreased_or_reset", **meta)
    if max_power_w is not None and (not finite_nonnegative(max_power_w) or delta > max_power_w * dt):
        return invalid("counter_exceeds_physical_bound", **meta)
    return dict(status="ok", joules=delta, **meta)


def combine_components(cpu, gpu, *, disjoint_verified=False):
    """An aligned component sum is explicitly not whole-system wall energy."""
    meta = dict(domain="cpu_package_plus_gpu_board")
    if not disjoint_verified or cpu.get("domain") != "cpu_package" or gpu.get("domain") != "gpu_board":
        return invalid("domains_not_verified_disjoint", **meta)
    if any(x.get("status") != "ok" or not finite_nonnegative(x.get("joules")) for x in (cpu, gpu)):
        return invalid("component_unavailable", **meta)
    if any(cpu.get(k) != gpu.get(k) or cpu.get(k) is None for k in ("start_ns", "end_ns")):
        return invalid("unaligned_windows", **meta)
    return dict(status="ok", joules=cpu["joules"] + gpu["joules"],
                start_ns=cpu["start_ns"], end_ns=cpu["end_ns"], **meta)
