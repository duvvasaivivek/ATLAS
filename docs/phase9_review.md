# Phase 9 review — integrated allocation and frequency shadowing

Date: 2026-10-31. The integrated runtime is implemented. **Unit tests, real
workload correctness, and decision replay pass. The strict all-iteration
concurrency gate remains incomplete in the final repeat, and live joint control
remains blocked.** No GPU clock setters were called.

## Delivered implementation

- A deterministic coordinator combines Tier 1 allocation with Tier 2 shadow
  recommendations, with explicit configuration hashes, cadence, workload/ownership
  epochs, source observations, and settling rules.
- Real persistent kmeans and HotSpot execution reuses the validated native
  feedback protocols. Allocation changes occur at completed-work barriers;
  telemetry polling runs in a separate thread while native work executes.
- Shadow inputs are reconstructed from raw timestamp-paired utilization samples.
  Missing, stale, duplicated, future, or incompatible samples cannot update
  frequency history. Shadow recommendations never become hardware requests.
- Separate synthetic joint-control scenarios exercise the causal interaction
  between allocation, frequency, workload phases, invalid telemetry, and failures.
- The runtime retains lifecycle, worker errors, watchdog termination, polling and
  controller costs, transfer accounting, independent references, profiler traces,
  and checksum-protected replay. Live mode fails before launching a workload.

See [the integration contract](phase9_spec.md) for defaults and limitations.

## Final retained evidence

The authoritative final repeat is
[2026-10-31-final-v3](../results/phase9/2026-10-31-final-v3/summary.json).

- **135 Python tests passed**, including 19 new integration tests for epoch
  aggregation, settling, nominal 40:1 cadence, sample binding, cross-poll joins,
  missing/conflicting/future telemetry, backwards barrier clocks, retained
  frequency history, bracket invalidation, live-mode rejection, malformed workers,
  process-group timeouts, and replay tampering.
- **13 hardware trials completed 2,211 iterations/timesteps** with independent
  workload correctness checks. The matrix includes small reference fixtures,
  allocation-only execution, both frequency policies in shadow mode, separate
  profiling runs, and a kmeans convergence case that stopped after three iterations.
- **18 allocation decisions produced 12 actual ownership changes.** Final outputs
  remained equivalent across the allocation/shadow modes within the established
  tolerances. Native workload kernels were not changed in this phase.
- **32 synthetic cases exercised 1,344 barriers**, including 12 expected injected
  failure/interruption outcomes with the synthetic setting restored. These model
  results are not evidence of physical clock control or energy savings.
- **Zero hardware frequency updates were accepted**, and **zero clock setters
  were called**. The observed host/driver clock mapping failed the timestamp guard.

Verification logs: [tests](../results/phase9/verification/unit-tests.txt),
[replay](../results/phase9/verification/integration-replay.txt), and
[source/binary and compatibility audit](../results/phase9/verification/audit.json).

## Concurrency result and repeatability

The final Nsight traces observed actual GPU-kernel/CPU-compute overlap in
**6/6 kmeans iterations and 5/6 HotSpot iterations**. Thus overlap is observed
with the integrated runtime, but the existing analyzer's strict requirement of
overlap in every profiled iteration is not met by the final HotSpot repeat.
The final archive truthfully retains `software_gate=incomplete_concurrency_evidence`.

These profiling runs use one CPU worker and a 75% CPU split to expose branch
overlap. The normal integration matrix uses four workers. Profile timings are
separate diagnostics and are not representative four-worker performance results.
They use Nsight's correlated activity timeline, not host enqueue intervals.

The first retained run (`2026-10-31-final`) used four-worker, 50/50 profiling and
observed overlap in 2/6 kmeans and 3/6 HotSpot iterations. The explicitly revised
one-worker diagnostic in `2026-10-31-final-v2` observed overlap in 6/6 iterations
for both workloads. The final repeat after backwards-clock and accounting checks
showed the 6/6 and 5/6 result above. All runs are preserved. The successful middle
trace does not erase the final repeat's incomplete strict gate or establish a
guarantee of overlap on every iteration under OS scheduling variability.

## Why hardware shadow decisions held

In the final run, **290 raw utilization sample records** had driver timestamps
ahead of the selected host wall clock at acquisition. The most negative observed
sample age was **-768,627 microseconds**. This violates the adapter's assumed
host/driver clock mapping; it does not establish the root cause or prove that a
particular clock is physically wrong.

The coordinator rejected those observations without fitting an offset, assigning
zero utilization, or reusing old samples. The guarded policy path is tested with
valid fixtures and synthetic feedback, while the real workload records retain
`hold_insufficient_fresh_window` where appropriate. Some runs are also too short
to satisfy the one-second guard. Workloads were not padded with sleeps to obtain
frequency decisions. Shadowing here is correctly inhibited observation, not a
successful measured frequency-response experiment.

This is an additional live-observation limitation alongside Phase 8's inability
to capture and restore the prior locked-clock policy. Physical clock changes,
their effects, and the full B5 comparison remain unavailable.

## Timing and research limits

The reports keep active native work, feedback gaps, nonfinal controller cost,
final-barrier cost, sampler query wall/CPU time, and full command duration
separate. Polling overlaps execution; these values cannot simply be added.
Settling work stays inside active work. Transfer/communication time and ownership
traffic are available, but pure migration time is not isolated from other
transfer work. No causal monitoring-overhead estimate is claimed from differing
adaptive schedules.

Both hardware modes use unchanged automatic GPU policy; neither is the fixed-clock
B3 baseline. Driver limiter masks and observed clock changes are retained as
context. Energy remains null, CUDA-event timing remains unqualified, and CPU
governor/package-energy limitations persist. No performance or energy benefit is
attributed to unexecuted frequency recommendations.

## Gate status

Implementation and deterministic tests are complete for allocation/shadow and
synthetic joint execution. Workload correctness and replay pass. The final strict
concurrency check is incomplete; live two-tier hardware qualification is blocked
by prior-policy/actuation requirements and observation-clock validity. These
limitations must carry into any Phase 10 evaluation rather than being relabeled
as a complete hardware GreenGPU reproduction.
