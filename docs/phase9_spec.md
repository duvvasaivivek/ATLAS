# Phase 9 integrated runtime contract

## Guarded live extension (2026-11-10)

The separate `live-run` / `live-replay` interface implements guarded joint
orchestration and the armed native feedback protocol. See the
[live control contract](live_control.md). Software tests include real native
workloads with a fake clock backend; physical control remains unqualified.
This extension leaves the historical shadow matrix and replay contracts below
unchanged. The `gpu-first` dispatch option is experimental and does not replace
the existing CPU-first default or satisfy the strict concurrency gate.

## Runtime 2.0 extension (2026-11-02)

The original integration-1.0 coordinator and archive replay remain supported.
New archives identify `runtime_version=runtime-2.0`. Native schemas are kmeans
4.1 and HotSpot 6.2. Native feedback holds a partition for a fixed batch;
each iteration retains its timestamps and observation, while only explicit
batch boundaries permit actions. Interior observations use their native
validation-end timestamp and do not ingest later telemetry. Final barriers
never request further work. Replay checks every realized partition, phase and
feedback boundary. Batched processing may overlap native computation; controller
CPU time must not be added to native elapsed time or feedback gaps.

The new integration matrix uses 16-iteration batches, a declared 50 ms
allocation interval, four workers, 1,000 kmeans iterations or 6,000 HotSpot steps,
and a halfway change to one CPU worker. This is the named batched-epochs
extension, not the paper's 40:1 cadence. Coverage requires allocation decisions
in both phases and at least one actual ownership change in every integration
trial. Profiles use the requested worker count and adaptive execution;
the strict all-iteration overlap gate remains unchanged.
The default integration matrix retains the atomic kmeans kernel and diagnostic
synchronization. Hierarchical reduction, streamed transfers, and HotSpot overlap
are explicit alternatives: correctness alone does not promote a performance
candidate whose measured benefit is inconsistent.

`robust_proportional_epoch_v1` renews the allocation observation budget when a
declared capacity epoch changes. Existing policies retain their behavior.
`cost_aware_v1` probes legal endpoint/quarter split candidates, compares measured
total-work costs, and requires a deadband and estimated migration payback before
choosing a different measured candidate. It uses a communication/work proxy for
transition cost, not a physically isolated migration measurement. Short horizons
can decline exploration. Neither extension guarantees a speedup.

Native processes run in owned Linux process groups. Deadlines and sampler
errors interrupt stdout waits, and cleanup kills surviving descendants even
after their group leader exits. NVML polling runs in a spawned process with a
two-second query deadline and separate read-only NVML session. Cleanup uses
bounded joins and process termination. New shadow sampling also requires two
polls whose host wall/monotonic offsets differ by at most 1,000 microseconds;
this rejects clock movement and does not calibrate or repair driver timestamps.

The original contract below describes the retained runtime 1.0 archives.

Version: `integration-1.0`, 2026-10-31.

## Execution modes and authority

`allocation_only` applies Tier 1 decisions to real persistent workloads under
the platform's unchanged automatic clock policy. It is not the fixed-clock B3
baseline. `allocation_shadow` additionally calculates Tier 2 recommendations
from eligible real samples, but never requests or acknowledges clock changes.
`synthetic_joint` applies allocation and frequency recommendations to an explicit
analytical model and is retained separately from hardware results.

The `integration-check --mode live-joint` command fails before output-directory
creation or workload launch. Phase 8's authoritative prior-policy readback,
physical restoration/actuation, and live telemetry qualification requirements
are still unmet. There is no silent fallback from requested live control to
shadow or synthetic execution. This phase does not claim a working live B5.

## Coordinator and workload boundaries

The coordinator is deterministic and performs no I/O. Its configuration includes
run identity, legal work size, initial ownership, both policies, explicit frequency
pairs, cadence, and settling guards. Configuration hashes and complete barrier
inputs accompany decisions.

The runtime records prepare, warm-up, execute, observe, finalize, restore, and
complete/stopped lifecycle transitions. Barrier records additionally expose
decision, allocation application, synthetic clock application, and settling
events. Failures pass through restore/stopped. Hardware shadow restoration is
explicitly `not_needed_no_settings_changed`, not physically verified restoration.

The existing kmeans and HotSpot native feedback protocols are reused unchanged.
Python reads a completed, validated iteration, decides at its barrier, and sends
the next ownership count before the native runtime releases another iteration.
It never repartitions a running kernel. Kmeans reductions/residency and HotSpot
current-state migration/halos retain the established workload contracts.

Small fixtures have independent Python trajectories. Native independent references
check larger workloads. The earlier HotSpot verifier receives a materialized
realized schedule only for workload checking; integration replay separately
verifies that each actual next partition equals the coordinator's decision.
Comparisons across execution modes must preserve final clustering/thermal results
within the existing tolerances. A converged run stops without another dispatch.

## Two cadences and settling

The default `observed_cadence_v1` configuration permits a frequency evaluation
at most once per 200 ms and requires at least four completed eligible iterations
and two seconds between allocation evaluations. Frequency calculations run at
available barriers, so their achieved cadence can be slower than requested.
The independent sampler polls approximately every 100 ms while native work runs.

`paper_ratio_40_v1` sets the allocation interval to at least 40 times the configured
frequency interval, retaining the minimum-iteration guard. This is a nominal
timescale configuration, not a claim of 40 actual frequency updates per allocation
decision or a stability guarantee. Native iteration lengths and sample freshness
still constrain decisions.

Allocation observations aggregate branch completion durations only while ownership
and the declared capacity epoch remain unchanged. The observation lists the exact
source iterations. After a redistribution, one completed work iteration is
excluded from allocation learning by default, and frequency observations must pass
a one-second post-change guard. Work continues during settling; the runtime adds
no sleep to pad workloads or manufacture sampling windows. The settling work time
remains in total work time and is not automatically a wasted-work overhead.

Small correctness fixtures explicitly use a zero minimum allocation time to
exercise migration. Fixed-work runs have declared iteration budgets. Convergence
runs stop at the workload's convergence condition. A short run can validly finish
with zero frequency updates and `hold_insufficient_fresh_window` decisions.

These are documented ATLAS guards. They do not empirically qualify physical
frequency settling or integration-window alignment for a live controller.

## Observation ownership and controller history

Each native barrier must match the expected run, next iteration, realized ownership,
correctness status, and valid nonempty-branch timings. An empty branch has no
fabricated duration. Corrupted barriers stop execution rather than dispatching
another partition.

Telemetry is collected in a separate thread. Only queue manipulation holds the
queue lock; NVML queries do not hold up a native worker waiting for a telemetry
query. Each frame has a sequential ID, acquisition bounds, buffer samples,
observed clocks, temperature, power/counter diagnostics, and driver clock-event
masks. The barrier drains completed frames and records its own evaluation time.
Trailing frames are retained but cannot retroactively affect decisions.

Shadow input joins core and memory utilization on an identical driver timestamp.
Conflicting duplicates, invalid values, missing memory, timestamps already consumed,
future acquisition frames, and samples older than 500 ms are excluded. An
approximate wall-to-monotonic conversion must place the sample after the current
partition's one-second guard. Cross-poll joins retain both frame IDs. Controller
inputs are reconstructed from these raw frames during replay, not supplied as an
unverifiable caller assertion of eligibility.

This remains the explicitly limited shadow timestamp contract from Phases 7–8.
Clock readings are context, not synchronized proof that a recommended pair was
realized. External clock changes and driver limiter masks are reported. Both
hardware modes remain under automatic capacity; bracketed bisection is therefore
rejected for hardware integration.

Tier 2 retains its full literal history across allocations and declared phase
changes. `discounted_loss_v1` remains a distinct extension. Synthetic applied
frequency changes advance the capacity epoch and invalidate the aggregation
window; the Tier 1 policy then invalidates stationary brackets on the new epoch.
A shadow recommendation never changes capacity or triggers a synthetic clock
application. Independent runs start fresh states.

## Synthetic joint-control model

The diagnostic suite uses 1,000 divisible work items and sparse assumed frequency
pairs. CPU capacity is 0.6, changing to 0.2 in the phase-change scenario. GPU
capacity is `0.8*(core/1800)**0.7*(memory/6000)**0.3`. Branch durations are work
share divided by capacity; model iteration duration is their maximum. Busy
fractions are bounded demand/capacity proxies, not measured GPU utilization.
The previous recommendation determines the next iteration's model capacity.

Stationary, changing-phase, missing-memory, injected actuation-failure, and
interruption cases cover literal proportional/binary/bracketed allocation and
both frequency policies. Two additional scenarios exercise the nominal 40:1
profile. Restoration means returning the model to its initial synthetic pair;
it is never hardware evidence. Sign reversals are reported without assuming
convergence. No energy model or hardware savings are fabricated.

## Accounting, failures, and evidence

Timing fields separate native active work, feedback gaps, controller computation,
sampler query wall time, sampler thread CPU time, transfer/communication stages,
settling work, and full command duration. Nonfinal controller computation nests
in feedback gaps; final-barrier computation is accounted separately. Polling
overlaps the job and must not be added to command duration as if it
were serial. Transfer-stage timing and ownership traffic are retained; pure
migration time is not isolated from other transfer work. Hardware actuation
overhead is zero because no setters are called. These are exploratory accounting
values, not matched causal overhead estimates.

The runner uses a bounded watchdog and a dedicated process group, so malformed
output, worker failure, timeout, or handled interruption can stop the worker tree
including profiler wrappers. It closes pipes, stops the sampler, retains partial
records/errors, and records stopped cleanup. A stuck sampler aborts subsequent
work and is not shut down underneath an active NVML thread. Reported temperatures
at or above 85 C cause a failure at the next workload barrier.

Archives retain inputs, native output, raw telemetry, barrier decisions, lifecycle,
independent fixture references, final thermal fields, profiler traces, source and
binary identities, and recursive checksums. Replay checks archive integrity,
workload correctness, source-observation binding, actual next ownership, all
decisions, lifecycle, frame inventory, synthetic scenarios, and regenerated
summaries. Nsight comparisons use actual GPU kernel activity and CPU work ranges
in the profiler's correlated timebase, separately from host/native timestamps.

Energy remains null, CUDA-event timing is not used as qualified timing, and the
full live joint-control gate remains blocked. Passing software/shadow validation
does not waive the Phase 8 hardware or Phase 2 measurement requirements.
