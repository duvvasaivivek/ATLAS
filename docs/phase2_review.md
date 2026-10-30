# Phase 2 implementation review and handoff

Date: 2026-10-31. Status: **measurement infrastructure implemented and tested;
physical GPU energy and device-event timing qualification blocked** on this
Windows/WSL/driver configuration. This is not a GreenGPU savings result.

## Implemented scope

- Read-only NVML telemetry using pinned `nvidia-ml-py==13.615.71`, added to the
  retained `.venv-wsl`. Each query preserves raw units, normalized values,
  acquisition bounds, midpoint uncertainty, validity, and errors.
- Package-only Linux powercap/RAPL discovery and reads, with deduplication and
  explicit absent-sensor reporting. Whole-system wall energy remains unavailable
  because no external meter is configured.
- Pure trapezoidal power integration with clipped, bracketed boundaries; explicit
  gap/missing/stale/nonfinite-data rejection; counter deltas with reset/wrap checks;
  and guarded sums of aligned, disjoint CPU/GPU domains.
- A native fixed-work CUDA calibration fixture with independent exact integer
  reference, non-multiple-of-block input size, separate warm-up, H2D/compute/D2H
  events, synchronized completion, and a host-controlled measurement handshake.
- Immutable-by-convention raw JSONL, exclusive archive creation, configuration and
  input hashes, source/binary identities, dependency/environment snapshots,
  randomized paired trial order, deterministic summary replay, and evidence
  checksums. Git attributes preserve result-file bytes across Windows/Linux.
- Counter/power and device/host consistency guards. Rejected values remain in
  diagnostic fields while the corresponding accepted measurement becomes null.

Implementation: [energy accounting](../src/greengpu/energy.py),
[adapters](../src/greengpu/telemetry.py), [records and summaries](../src/greengpu/experiment.py),
[runner](../src/greengpu/runner.py), [CUDA fixture](../native/measurement/main.cu).
The [run contract](run_schema.md) defines timing, units, quality, failure handling,
and replay semantics. [README](../README.md) contains runnable WSL commands.

## Verification

All **33 standard-library tests passed**, covering Phase 1 regressions, analytic
integration, clipped/nonuniform intervals, gaps, missing/stale readings, unit
conversion, RAPL hierarchy, admissible and ambiguous wraps, domain alignment,
invalid correctness/work records, energy and timing consistency guards, immutable
creation, checksum tampering, deterministic replay, and failed/timed-out workers.

The native targets built with CUDA 13.1.115 and CMake/Ninja. The CPU/CUDA correctness
CTest passed. The final calibration's eight trials each validated all 65,539
returned elements against an independent masked-64-bit xorshift reference with
zero mismatches. Tests do not validate physical sensor accuracy.

The [final verification audit](../results/phase2/verification-v2/audit.json)
confirmed every archived checksum, exact summary regeneration, all 15 captured
source-file hashes, the executable hash, and the unchanged original paper hash.
The [CTest evidence](../results/phase2/verification-v2/ctest.txt) is also retained.

## Authoritative calibration evidence

Use [2026-10-31-final-v2](../results/phase2/2026-10-31-final-v2/manifest.json), its
[frozen configuration](../results/phase2/2026-10-31-final-v2/frozen-config.json), and
[summary](../results/phase2/2026-10-31-final-v2/summary.json). One separate tuning
pilot selected 247,809 repetitions of 256 rounds over 65,539 uint32 elements;
seed 1729 and the input SHA-256 are fixed. The eight subsequent trials used four
randomized paired blocks: periodic telemetry versus endpoint-only measurement.
Both arms kept the normal GPU platform policy. No hardware settings were changed.

The nominal ten-second duration target only selected fixed work before trials.
Actual monitored host intervals averaged **7.8842 seconds**, with sample standard
deviation **0.1812 seconds**, ranging from **7.6207 to 8.0287 seconds**. This pilot
scaling difference is retained; the runner does not change work to force a runtime.

Paired incremental monitoring overhead averaged **+0.9210%**, with sample standard
deviation **1.8136 percentage points**, ranging from **-1.6354% to +2.4364%** across
four blocks. These exploratory observations do not establish a precise overhead
or a statistically resolved effect. They include periodic logging/scheduling costs
relative to endpoint-only instrumentation. Earlier series had negative means;
they are retained, not discarded or pooled across different configurations.

Requested polling was 100 ms; the maximum observed gap across the monitored trials
was 144.0 ms, below the frozen 500 ms gap limit. Actual sensor update cadence is
unknown. Observed changes do not establish sensor resolution. Entire-trace query
CPU time, including the equal pre/post buffers, ranged from 0.1034 to 0.2297 seconds.
Host boundary overhead was 4.69–23.17 ms. The raw records retain acquisition
uncertainty for each reading rather than presenting these calls as instantaneous.

## Hardware qualification failures

**GPU energy:** NVML's cumulative counter is readable, but periodic-monitoring
trials reported **863.424–884.171 J** from counter deltas versus
**365.953–385.012 J** from sampled-power integration. These are conflicting
diagnostic estimates, not accepted energy consumption. Endpoint-only counter
deltas were **428.368–433.001 J**. Similar fixed work and runtime with such large
method/polling dependence make the counter unsuitable for savings claims here.
This experiment identifies the disagreement; it does not establish its root cause
or prove the power estimate is physically accurate.

All four monitored trials exceeded the frozen consistency allowance. Their primary
GPU energy is null, `energy_qualification` is `blocked`, and no accepted monitored
energy mean is reported. Raw counter and power estimates remain available. The
power fallback is not used to conceal a detected disagreement.

**Device timing:** all eight CUDA event spans exceeded their enclosing synchronized
host compute intervals beyond the allowed 1 ms plus 1%. In monitored trials,
reported device spans were **8.0342–8.4207 seconds**, while enclosing host compute
spans were **7.6157–8.0051 seconds**. `gpu_event_reported_s` preserves those values;
accepted `gpu_event_s` is null and `device_timing_qualification` is `blocked`.
The common host clock still provides internally ordered dispatch/completion
records, but this does not externally calibrate its rate or prove which clock is
wrong. Do not combine these device and host durations as if they were consistent.

**Other domains:** CPU package counters and governors remain unexposed in this
WSL environment. RAPL logic is tested with synthetic sysfs fixtures, not a live
package sensor here. No wall meter or external timing reference was used. Windows
display/background GPU activity was not isolated. No whole-machine energy result,
external accuracy claim, or battery/thermal equilibrium claim is made.

## Retained earlier evidence

- `2026-10-31-pilot`: the original short trials exposed counter/power and polling
  dependence; their roughly one-second intervals were too short for qualification.
- `2026-10-31-validation`: longer exploratory trials still fell below the frozen
  five-second consistency-screen duration.
- `2026-10-31-final`: longer trials verified the energy rejection guard; final
  auditing then identified the device-event/host timing contradiction.
- `2026-10-31-final-v2`: includes both quality guards and is authoritative.

Earlier summaries were generated by the source versions identified in their own
manifests, before all guards were added. They are preserved as diagnostic history;
their unqualified arithmetic values must not be cited as validated measurements.
The `verification-v2` audit supersedes the earlier `verification` audit for the
current implementation. No adverse outcome was deleted.

## Exit gate and next phase

The software exit gate is met: analytic traces and wrap cases pass, unavailable
domains remain explicit, repeated baselines expose variability, monitoring overhead
is measured, failures are retained, and final summaries reproduce from checked raw
data. The physical measurement qualification gate is **not** met on this setup.

Phase 3 can implement and verify concurrent CPU/GPU static splits using the common
host timebase for ordering and relative branch completion. Before any physical
energy-saving or device-timing claim, investigate the observed counter/polling and
clock inconsistencies on a separately controlled configuration and validate against
an independent reference where available. Preserve GPU-board, CPU-package, and
system-wall scopes throughout; do not compensate by inventing unavailable sensors.

No Phase 3 workload/controller code has been started. The virtual environment,
Phase 1 work, source paper, and earlier evidence are preserved.
