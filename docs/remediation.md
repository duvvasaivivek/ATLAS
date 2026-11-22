# Runtime and performance remediation

Implementation started 2026-11-02 following the repository-wide review.
Existing archives are immutable. Existing paper policies and their configuration
hashes retain their semantics; new policies and timing modes are explicit.

Execution order and acceptance:

1. Process groups and telemetry cancellation: bounded cleanup after leader exit,
   silent-worker failure, and blocked driver queries. Shared Linux supervisor and
   spawned read-only telemetry worker; regression tests in test_supervision.py.
2. Kmeans validation modes: independent final reference outside timing loop,
   full per-iteration reference in verify mode, independent oracle equivalence.
3. Timing and annotations: separate communication, execution and waiting;
   CPU annotations exclude OpenMP barriers; preserve old schema readers.
4. Adaptation: explicit cadence for long-running experiments, observed-decision
   coverage gate, and separately named per-epoch controller budgets.
5. Feedback and synchronization: batched fixed-partition execution, bounded
   cancellation, fewer synchronizations, explicit unavailable stage timings.
6. Kernels and allocation: hierarchical reduction, endpoint-aware cost policy,
   and evaluated HotSpot communication overlap; no promised speedup.
7. Hardware qualification: fresh read-only clock/telemetry and energy checks.
   Unavailable authoritative clock-policy readback or physical metering remains
   a blocker; no forced clock changes or fitted timestamp offsets.
8. Evaluation: correctness, replay, sanitizers, representative profiler runs,
   randomized matched benchmarks with frozen settings and retained regressions.

The code changes for these stages are implemented. Validation so far:

- 154 Python tests pass, including process-tree deadlines, silent native failure,
  a blocked sampler, epoch renewal, batch boundaries, clock movement, and failed
  performance-pair accounting. See [unit log](../results/remediation/unit-tests-complete.txt).
- All nine native CTests pass ([log](../results/remediation/native-tests-parallel-reference.txt)). Kmeans covers independent Python oracles, all
  existing adversarial fixtures, maximum supported dimensions/clusters, partial
  blocks, endpoints, and repeated migrations in verification and timing modes.
  HotSpot covers serial and overlapped communication with independent fields.
  The expanded tests exercise scalar and parallel final references above the
  1,024-item threshold, including 1,025 points at maximum supported k/d.
- All eight CUDA sanitizer checks pass: memcheck, racecheck, synccheck, and
  initcheck for both final workloads. See [sanitizer records](../results/remediation/sanitizers-final/summary.json).
  These are device checks; they do not establish absence of host races or timing jitter.
- All authoritative Phase 2–9 archives replay unchanged under WSL Python 3.12.
  See [compatibility log](../results/remediation/historical-replay-final.txt).

The [final integration archive](../results/remediation/2026-11-02-integration-v2/summary.json)
contains 13 trials and 21,611 measured iterations. All six long trials now make
allocation decisions in both phases and change ownership. Kmeans has 62 decisions
per trial and 19 changes; HotSpot has 27–30 decisions and 7–9 changes. This
closes the former zero-adaptation coverage gap. The new clock guard accepts no
hardware frequency updates in this run; no setters are invoked.

The strict concurrency gate remains incomplete: actual four-worker, adaptive
profiles show compute overlap in 239/256 kmeans iterations and 228/256 HotSpot
iterations. Some misses have disjoint compute intervals; others lack a kernel
satisfying the existing timeline-containment checks. These observations do not
justify changing the strict gate or treating a host launch/barrier wait as GPU/CPU
compute. No stronger concurrency claim is made.
The earlier integration attempt is retained separately; its 233/256 and 246/256
counts are not substituted for the final build's measurements.

The [fresh qualification](../results/remediation/2026-11-02-qualification-v2/summary.json)
still blocks live actuation. Prior locked-clock policy readback is unavailable;
application-clock queries report unsupported. Of 92 loaded buffer samples, 90
have future driver timestamps (minimum age −580,981 microseconds), despite a
stable host wall/monotonic offset span of 8.25 microseconds. The longer measurement
repeat has 7.43–7.68 second monitored intervals, but counter energy and sampled
power disagree, so accepted GPU energy remains null. CUDA event timing still
requires external validation. CPU policy/package-energy interfaces are not
exposed and no wall meter is configured. Software cannot establish the missing
hardware capabilities or calibrate these sensors by fitting an offset.

The first performance attempt is retained under `2026-11-02-performance`: trial
117 returned CUDA `invalid argument` before the configuration record. An immediate
standalone repeat passed; the original failure is not discarded or attributed to
a proven cause. The runner now records failed evaluation trials and completes
the remaining matrix, excludes unavailable pairs explicitly, and returns a
nonzero status if any trial fails. A failed tuning trial still prevents freezing
a selection.

The next [400-trial evaluation](../results/remediation/2026-11-02-performance-v2/summary.json)
passed correctness and replay. It exposed end-to-end regressions that loop-only
timing would hide: the new 50/50 path reduced loop time by about 57% for kmeans
and 26–40% for HotSpot relative to the retained legacy binary, but full command
time increased about 20% and 14% respectively at the larger sizes. This led to
parallel final-reference validation outside the measured loop, with the original
scalar verification mode retained. These are not weakened checks: kmeans retains
point-order long-double reduction; HotSpot retains per-cell scalar flux order.
That loop comparison includes relocating validation and is not an end-to-end
speedup claim. The evaluation was repeated after the reference change. Kernel
and transfer alternatives without consistent benefits remain explicit
experimental options.

The [final evaluation](../results/remediation/2026-11-02-performance-final/summary.json)
passes all 440 trials and checksum/decision replay: 40 tuning trials plus 400
held-out trials, with ten randomized blocks per variant and workload size.
Kmeans sizes are 32,771 and 65,539 points at d=8/k=8; HotSpot sizes are 257x259
and 513x515. Runs use four requested CPU workers, 100 kmeans iterations or 128
HotSpot steps, unchanged automatic clocks, and equally disabled telemetry.
The final binaries are retained in `build/remediation-final`; binary and source
hashes, seeds, selections, and raw commands are retained in the archive.

Measured outcomes, with negative changes meaning faster:

- Separately tuned 25% CPU kmeans splits reduce the primary workload interval
  versus matched GPU-only by 8.8% and 19.5% at the two sizes. The pointwise 95%
  paired bootstrap intervals are [-18.2%, -1.5%] and [-24.2%, -15.2%]. Full-command
  intervals still include zero improvement; these are workload-interval findings.
- HotSpot tuning selects GPU-only at both sizes. Online cost-aware allocation
  remains 188.6% and 149.0% slower than GPU-only over the primary interval on
  these short jobs. It remains an experimental policy, not a promoted default.
- Against the retained legacy 50/50 binary, the final new 50/50 path's mean full
  command changes are -13.0%/-12.1% for kmeans and -5.5%/-25.8% for HotSpot.
  Only the larger HotSpot comparison excludes zero in these full-command
  intervals: [-40.0%, -10.9%]. Startup and validation variance are substantial.
- Batching reduces primary loop time versus per-step feedback. Hierarchical
  reduction, streamed transfers, and halo overlap do not establish consistent
  full-command gains across sizes. Native defaults retain atomic/diagnostic
  kmeans and serial HotSpot communication; integration uses batching and the
  explicitly declared timing mode. The alternatives remain available for testing.

See [paired command/loop comparisons](../results/remediation/performance-comparisons.json)
and their [reproduction script](../results/remediation/paired_report.py).
Intervals use 2,000 fixed-seed bootstrap resamples of paired blocks, are
pointwise/exploratory, and are not multiple-comparison-adjusted. Automatic clocks,
shared-host scheduling, and lack of thermal isolation limit generalization.
No energy-saving claim or universally faster adaptive policy follows from them.
`integration-check` deliberately exercises adaptation and is not a recommendation
to use adaptive partitioning for every workload.

Remaining acceptance gates are the strict all-iteration concurrency requirement,
authoritative restorable clock-policy readback, and independent timing/energy
qualification. These remain explicitly incomplete/blocked. The code does not
weaken those gates, and it does not automatically enable a regressing option.

Reproduction uses the commands near the top of [README](../README.md). Additional
checks, from the repository root in WSL:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-wsl/bin/python tests/replay_archives.py
PYTHONDONTWRITEBYTECODE=1 .venv-wsl/bin/python tests/native_sanitizers.py \
  build/remediation results/remediation/new-sanitizers
```
