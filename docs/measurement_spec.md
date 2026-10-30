# Measurement and evaluation specification v1.0

Prepared: 2026-10-31. This document freezes definitions before Phase 1; it does not claim that any sensor or hardware control is available. Source references and initial hypotheses are in the [paper mapping](paper_mapping.md).

## 1. Primary question and admissible comparisons

For the same correct workload, input, logical work amount, precision, and measurement domain, does a policy reduce measured energy, and what happens to end-to-end execution time?

Two workloads are primary: kmeans and HotSpot. Controlled comparisons use a fixed logical iteration/timestep count. A separate kmeans convergence-to-quality study may compare different iteration counts only when the stopping criterion and achieved quality are both reported.

Correctness failure disqualifies a performance/energy result. A valid negative saving, failure to balance, or oscillating policy remains part of the findings. No historical percentage or favorable ordering of policies is an acceptance requirement.

## 2. Timing boundaries

Use monotonic host timestamps with explicit units and clock identity. Define these distinct quantities:

- **Preparation time:** input loading, initialization, allocation, compilation, and warm-up. Retain individually; exclude from the primary warmed-work interval for every policy consistently.
- **Primary workload time:** start immediately before initial measured dispatch/policy action; end after the last required computation, transfer, merge/reduction, and synchronization. Include observations, controller execution, adaptation, migration, and actuation that occur in this interval.
- **Recovery time:** restoration of the prior hardware policy after the primary workload; report separately. Also report total experiment lifecycle time when comparing deployment costs.
- **Cold-job time:** optional additional run including initialization/preparation and final cleanup, with a separately measured energy interval. Do not add a warm-up trial's energy to a cold-job result without defining that workload.
- **CPU/GPU branch completion time:** elapsed from a common iteration-release point until that branch's required result is ready for the merge. Include launch/queue delay and branch-specific transfers. A branch-specific dispatch delay must not disappear from its end-to-end time.
- **GPU kernel time:** separate device-event measurement for diagnostics. It is not interchangeable with host elapsed time or a CUDA launch-call duration.
- **Shared merge, synchronization, and migration time:** timestamp explicitly, with overlap/dependency relationships retained. Do not sum overlapping intervals and call the sum wall-clock runtime.

An iteration trace must explain when both branches overlap. `max(t_cpu,t_gpu)` may describe the branch portion but does not automatically include shared serial work. Barrier waiting is measured where possible; the absolute difference of branch times is only an inferred waiting opportunity.

Every baseline uses equivalent boundaries. If actuation before the first sample is needed, start the primary energy/timing interval before that action so adaptation is not hidden.

## 3. Energy domains and units

Record raw units and convert explicitly: watts, joules, seconds, MHz, and normalized utilization fractions. Milliwatts divide by 1000; microjoules divide by 1,000,000.

Allowed energy labels describe what the instrument actually covers:

- `gpu_board`: sensor-reported GPU board domain, with its documented scope.
- `cpu_package`: package counter, including the subdomains documented for that platform.
- `cpu_package_plus_gpu_board`: sum only when domains do not overlap and their measurement windows match.
- `system_wall`: a meter covering the full machine at the outlet, with display/peripheral and battery behavior specified.
- `synthetic`: simulation output, with model and parameters included.

Do not label component sums `system_wall`; do not add nested package and core counters; do not equate the paper's host-side wall meter with a package sensor. If a domain is unavailable, report `null` plus a reason rather than zero. Device energy is not automatically attributable solely to the workload if unrelated processes share the device.

For sampled power, integrate timestamped readings over the common workload interval:

```text
E = sum(0.5 * (P[k] + P[k+1]) * (time[k+1] - time[k]))
```

Acquire valid samples bracketing both interval boundaries; interpolate only within a valid observed segment and cut the integral at the boundary. Do not extrapolate unobserved time. Preserve raw readings and averaging-window metadata.

For cumulative counters, compute the delta over the same interval. A one-wrap correction is `(current-previous) mod counter_range` only when reset is ruled out and the acquisition interval guarantees fewer than one full-range accumulation. Otherwise mark the delta ambiguous. Prefer sufficiently frequent sampling; log resets, discontinuities, and range information. The [Linux powercap documentation](https://docs.kernel.org/power/powercap/powercap.html) describes energy counters, ranges, and domain hierarchies.

Polling does not make a slowly updated sensor high-resolution. Sampling interval, maximum admissible gap, expected update cadence, and timing uncertainty must be determined during Phase 2 and frozen in the experiment configuration before confirmatory runs. No provisional value is passed off as measured sensor capability.

## 4. Idle-subtracted energy

Report total measured energy as primary. If dynamic energy is additionally reported:

```text
E_dynamic = E_total - P_idle_matched * T_measured
```

The idle baseline must match the same hardware domain, relevant settings, and measurement conditions. If settings change over the run, use a justified piecewise reference or mark idle subtraction unsupported. Retain the idle-baseline data, uncertainty, and method. Do not clip a negative estimate to zero; investigate and report that the estimate is unresolved within uncertainty.

Dynamic-energy saving and total-energy saving have different denominators. They must never be pooled or plotted under one unlabeled saving series.

## 5. Metric definitions

For a valid baseline energy greater than zero:

```text
energy_saving_percent = 100 * (E_baseline - E_policy) / E_baseline
runtime_change_percent = 100 * (T_policy - T_baseline) / T_baseline
speedup = T_baseline / T_policy
energy_delay_product = E_policy * T_policy
branch_imbalance = abs(t_cpu - t_gpu) / max(t_cpu, t_gpu)
```

Positive runtime change means slowdown; negative energy saving means an energy regression. Branch imbalance is a diagnostic only for valid active branches. Report joules per fixed point-assignment/iteration or grid-cell/timestep when the work-unit definition is meaningful.

**ATLAS diagnostic default:** sustained balance means imbalance at most 0.05 on three consecutive eligible observations under the same declared workload phase. Report the first such window and later departures. It does not stop a literal controller unless its specification says so. Reset this diagnostic window after a phase change or an invalid observation; do not erase controller state implicitly.

Convergence has multiple outcomes: sustained balance, exact timing equality, bracket precision, adaptive precision exhaustion, quantized stagnation, cycle/budget exhaustion, invalid observation, no bracket, and unsupported partition. Report decision count, executed iterations, elapsed time, and adaptation energy separately. A small ratio change is not automatically successful balance.

## 6. Baseline identities

- **B0:** GPU-only, verified fixed high GPU clock policy, explicitly recorded CPU policy. Unavailable if the fixed setting cannot be enforced.
- **B0-default:** GPU-only with the normal platform policy. It is not renamed B0 when clock locking is unsupported.
- **B1:** CPU-only, declared CPU policy.
- **B2:** fixed split at 50/50 and independently tuned static splits, with declared clocks.
- **B3:** allocation only, fixed GPU clock policy and the same CPU policy as its matched comparator.
- **B4:** frequency control only, GPU-only allocation and a declared CPU policy.
- **B5:** allocation plus frequency control, with every policy ID recorded.
- **B6:** best measured static allocation/frequency candidate from a documented sweep, evaluated again on independent trials.

Use `paper_simple` for the initial paper-style full-framework series. Proportional, corrected bisection, and extended controllers are separate series. If CPU governor control differs, report that factor instead of attributing all benefit to GPU scaling. The CPU policy need not be ondemand on every platform; its availability and behavior remain unverified until the relevant hardware phases.

Do not charge offline tuning only to one comparator. Report execution comparison and tuning-cost amortization separately. A sampled best configuration is not a proof of a global optimum.

## 7. Minimum run record

Every retained run includes:

- Run ID, UTC start time, monotonic clock identity, independent trial/block ID, execution mode, schema/specification versions, source revision, and working-tree change identity.
- Hardware IDs, environment/runtime/driver/compiler versions, capability-report identity, and measured energy domain.
- Input hash or deterministic generator version/seed, size, logical work count, precision, correctness oracle, tolerance, and result.
- All allocation/frequency/CPU policies, initial and realized ratios, parameter values, legal work granularity, feasible clock pairs, and configuration hash.
- Raw timestamped power/counters, metric semantics, units, validity/quality flags, and sensor cadence information.
- Per-iteration timing and work counts; controller input references, decision reasons, state transitions, migration amounts, and requested versus observed settings.
- Warm-up/preparation boundary, primary workload boundary, recovery outcome, measured overheads, and excluded/invalid-run reasons if applicable.
- Summary metrics generated from the retained raw evidence; unavailable values and their reasons remain explicit.

Immutable raw records and regeneration instructions are required. A chart image is not a substitute for the values that produced it. Results from simulation, replay, and hardware never share an unlabeled evidence series.

## 8. Experimental rules frozen before optimization

Initial hypotheses are H1-H5 in the paper mapping. Select policies on tuning inputs and evaluate them on held-out inputs. Use repeated independent trial blocks and counterbalance or randomize policy order. The unit of statistical replication is the trial/block, not each correlated telemetry sample.

Ten trials per primary configuration is an ATLAS planning default from the implementation plan, not a power-analysis conclusion. Pilot variability determines the final count and stopping rule before confirmatory evaluation. Report uncertainty, measurement resolution, and sample count. A positive claim requires evidence sufficient to resolve the sign; otherwise report it as inconclusive.

Predeclare technical invalidation criteria such as failed output validation, lost sensor data, failure to apply the requested configuration, or an unrelated competing job. Retain invalid trials and reasons. Do not discard valid regressions, normal throttling, or slow adaptation simply because they weaken the hypothesis.

## 9. Specification fixtures

- **E01:** Powers `[10,20,10]` W at times `[0,1,3]` s integrate to 45 J over [0,3].
- **E02:** A counter with range 100 microjoules moving from 95 to 4 accumulates 9 microjoules if exactly the admissible zero-or-one-wrap assumption holds and no reset occurs.
- **E03:** Baseline 100 J and 10 s versus policy 80 J and 10.5 s gives 20% energy saving, 5% slowdown, and 840 joule-seconds energy-delay product.
- **E04:** Branch times 2 s and 4 s give imbalance 0.5; they do not alone establish whole-iteration runtime or energy.

The [fixture file](spec_fixtures.json) stores these numbers for future implementation tests. Phase 0 verifies the calculations only.

## 10. Decisions deliberately assigned to later phases

Phase 1 resolves hardware, environment, privileges, and sensor availability. Phase 2 resolves sampling cadence and noise limits. Workload phases resolve equations, dataset provenance, precision, and numerical tolerances against independent references before performance evaluation. Phase 8 resolves actual actuation support. Phase 10 freezes statistical trial counts and exclusions after pilots.

These are explicit future gates, not missing Phase 0 guarantees. No runtime implementation should invent those facts to satisfy this specification.
