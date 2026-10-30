# Phase 2 run contract (schema 2.0)

The implemented collector is a **hardware calibration** runner. Its fixed-work
xorshift fixture is not kmeans, HotSpot, a CPU/GPU partitioning implementation, or
an energy-saving policy. `B0-default` identifies the unchanged GPU platform policy.

## Directory contract

Every invocation creates a new directory exclusively. Files are created with
exclusive writes; raw JSONL is append-only during acquisition and flushed after
each record. Completed runs contain `checksums.json` with SHA-256 identities of
every other file. This detects accidental modification; it is not a signed archive
or protection against someone rewriting both data and hashes.

- `manifest.json`: schema, UTC/run identity, host clock, binary identity, source
  hashes, sensor semantics, hardware identity, specification and capability references,
  input generator settings, and declared exclusions.
- `environment/`: fresh Phase 1-style capability/environment capture, Git revision
  and dirty status, tool versions, dependency freeze, source hashes, and CMake cache.
  The source hashes identify uncommitted code; Git revision alone is insufficient.
- `pilot.jsonl` and `pilot-summary.json`: a separate tuning trial, excluded from
  repeated-trial statistics. Repetition count scales from its measured duration.
- `frozen-config.json`: exact repetitions, elements, inner rounds, seed, input
  SHA-256, policies, polling/gap rules, consistency-screen parameters, deterministic
  trial order, and configuration hash. The target duration is an estimate, not a
  stopping rule; every paired trial executes the same fixed work.
- `trial-NNN.jsonl` and `trial-NNN.stderr.txt`: raw observations and native output.
- `summary.json`: deterministic derivation from the trial records; no additional
  measurement is needed for replay.
- `failure.json`: retained failure reason when collection aborts. Incomplete runs
  may lack a final summary/checksum manifest; their partial JSONL is retained.

`atlas replay RUN_DIRECTORY` checks the retained hashes and prints a regenerated
summary without modifying the archive. It does not replay computation or estimate
the outcome of an unexecuted policy. Source hashes identify the implementation
used to collect older archives; the final Phase 2 validation archive is the
authoritative schema example.

## JSONL envelope and event types

Each record has `schema_version`, increasing zero-based `sequence`, `kind`, and
`recorded_ns` (host receipt/write time). Measurement timestamps are separate fields
and must not be replaced by `recorded_ns`. Native events can arrive later than
their measurement timestamps. Missing values are JSON `null` with a reason;
nonstandard JSON NaN/Infinity is forbidden.

- `trial_config`: trial/block identity, monitoring arm, lifecycle start, UTC, settings.
- `native_stdout`: exact native line and host receipt timestamp, retained even for
  malformed protocol output.
- `ready`: native preparation/warm-up endpoints, clock identity, GPU PCI identity.
- `counter_before`, `counter_after`: direct NVML endpoint readings.
- `rapl_before`, `rapl_after`: package-only sysfs observations and counter ranges.
- `telemetry`: per-query readings plus total query wall/CPU time.
- `complete`: native workload endpoints, H2D completion, compute synchronization,
  D2H/final synchronization, CUDA event span, and executed work configuration.
- `validation`: independent masked-64-bit reference comparison of every returned
  uint32 element, with exact equality and mismatch count.
- `failure`: protocol, timeout, or runtime error. Failed trials are retained.
- `finish`: native exit code, sampler error, failures, lifecycle end, restoration
  status (`no hardware settings changed`).

The reader rejects unsupported schemas and sequence discontinuities. Summary
derivation checks required event uniqueness, configuration/work agreement,
correctness, process exit, and clock/order consistency. A correctness-valid trial
can still have **invalid energy**; these are separate statuses.

## Timing contract

Python `time.monotonic_ns()` and native `clock_gettime(CLOCK_MONOTONIC)` share the
Linux/WSL timebase. The runner checks clock implementation and native event ordering.
Each sensor query records acquisition start/end and a representative midpoint with
half-span acquisition uncertainty. This bounds the call timing, **not** the
sensor's internal update age or physical accuracy. NVML calls used here expose no
device sample timestamp; actual update cadence remains unknown.

The primary interval runs from the pre-dispatch energy-query midpoint to the
post-completion energy-query midpoint. It includes host dispatch/protocol overhead,
H2D, all fixed kernel launches, synchronization, D2H, and completion receipt. GPU
work must be contained within this interval. `boundary_overhead_s` exposes the
additional host time beyond the native workload interval. Endpoint queries occur
in both monitoring arms. The comparison measures the incremental effect of
periodic telemetry/logging, not the cost of all instrumentation versus none.

Input/reference preparation, allocation, eight warm-up launches, equal 1.1-second
pre/post sampling buffers, output validation, and cleanup are outside primary
energy/time and inside the recorded lifecycle. Validation starts only after the
final energy read. `gpu_event_s` is a device-event **span** enclosing all launches;
it may include device idle gaps between launches and is not a sum of active kernel
durations. The frozen timing screen permits 1 ms plus 1% above the enclosing host
compute interval. If a device-event span exceeds that allowance, `gpu_event_s` is
null, `gpu_event_reported_s` retains the raw reported duration, and
`device_timing_qualification` is blocked. Correct output and host timings remain
separate from device timing quality. This screen detects a contradiction; it does
not determine which physical clock is inaccurate. No CPU/GPU concurrency or
migration occurs in this fixture.

Query CPU/wall totals cover the **entire sampled trace, including buffers**.
Paired runtime differences measure the in-window impact including serialization,
locking, scheduling, and log writes. No settings need restoring because the runner
never modifies clocks, power caps, or CPU policy. Worker timeouts kill and reap the
native subprocess. A stuck NVML worker stops collection and is not shut down beneath
an active query; its raw archive remains incomplete.

## Energy and quality contract

`energy.py` uses joules, watts, and integer monotonic nanoseconds. Power integration
requires bracketing samples, strictly ordered timestamps, finite nonnegative
readings, and gaps within the frozen threshold. It clips valid linear segments at
the exact boundaries; no extrapolation or interpolation over invalid observations.
An explicit `stale` reading is rejected. Repeated numbers alone do not prove staleness.

NVML cumulative energy is converted from mJ to J and preferred when available and
consistent. Counter endpoints share one live NVML session. Decreases/read failures
in the observed interval invalidate the delta; no NVML wrap is guessed. A reset
that happens between reads and exceeds the previous value cannot be ruled out by
this interface. Endpoint-only trials have weaker continuity evidence. The metadata's
`resolution_j` of 0.001 describes integer-mJ reporting granularity, not independently
measured physical resolution or accuracy.

On Ada, `nvmlDeviceGetPowerUsage` is a one-second averaged value. Its integral is a
sensor-reported estimate, affected by the averaging window and boundary alignment.
Both arithmetic estimates are retained. For intervals at least five seconds, the
frozen exploratory consistency screen allows a difference of 10% of the larger
energy plus one averaging window times the observed power span. Exceeding this
screen marks primary energy invalid; it never silently substitutes the other
method. This screen was introduced after the short pilot revealed disagreement.
Intervals too short for this screen cannot establish counter suitability and
receive unresolved primary energy. It is a diagnostic threshold, not a physical accuracy bound or a confirmatory
statistical criterion. Agreement cannot validate two methods sharing sensor error.

When **both** endpoint counters are unavailable, the runner can use bracketed
power integration. Raw query errors remain visible; a decrease or an interior
discontinuity never triggers silent fallback. No calibration fixture
is eligible for a paper energy-improvement claim, even when arithmetic is valid.

The RAPL adapter reads package zones and ranges, deduplicates resolved paths, and
excludes nested core/DRAM zones. A wrap delta requires explicit reset exclusion and
a justified maximum-power bound guaranteeing less than one full range per interval.
The advisory power-limit setting alone is not such a bound. The current runner
retains RAPL endpoints without manufacturing a bound or aligning different query
midpoints. On this WSL machine no package zones are exposed, so CPU energy remains
unavailable. No wall-meter adapter is claimed operational without an attached meter.
`combine_components` requires explicitly disjoint domains and identical windows;
it never calls their sum whole-system energy.

## Statistics and scope

Trials, not telemetry samples, are independent replication units. Seeded randomized
paired blocks use identical frozen work. Reports retain each outcome and provide
sample count, mean, sample standard deviation, and range. A negative measured
monitoring overhead is retained and can reflect ordinary drift/noise. No confidence
or power claim is made from a small exploratory calibration set. The GPU is not
isolated from Windows display activity or other users' processes.

Official semantics: [NVIDIA NVML device queries](https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceQueries.html)
and [Linux powercap](https://docs.kernel.org/power/powercap/powercap.html).
