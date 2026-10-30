# Phase 5 allocation implementation contract

Specification 1.1, 2026-10-31. The five reference/corrected policies follow
[algorithm_spec.md](algorithm_spec.md), including its explicitly named ATLAS
interpretations. `robust_proportional_v1` is an additional ATLAS policy.
Frequency control is outside this implementation.

## Pure policies and observation identity

`allocation.Config` freezes policy parameters and legal work counts.
`transition(config, state, observation)` returns a complete decision without
mutating arguments or touching clocks/hardware. Each decision retains the policy
and specification IDs, configuration hash, observation, pre/post state, requested
ratio, legal CPU count, realized ratio, and reason. Work counts default to all
integers in `[0,N]`; sparse legal grids are also supported. Rounding chooses the
closest count, with ties favoring the smaller count. N=1 has no interior probe.

Observations identify run, nondecreasing declared workload epoch, and increasing
iteration. Duplicate/stale/cross-run input, invalid computation, incompatible
timing origin, and nonfinite or nonpositive active-branch times leave policy state
unchanged. Empty branches have null timing. Throughput-based policies request the
legal interior split nearest 50% from an endpoint; they never invent throughput.
The `static` adapter deliberately holds endpoints and provides fair baseline IPC.

The fixed step is 0.05; adaptive starts at 0.15 and halves continuously; precision
is 0.01. The eligible-decision budget is 200. Native workloads allow 1–100 measured
iterations and each process has a 120-second watchdog, so endpoint probes, invalid
observations, and frozen policies cannot run indefinitely. Real experiments use
smaller fixed work budgets recorded in their plans. An allocation stop freezes
adjustment, not useful workload execution.

Relative imbalance is `abs(t_cpu-t_gpu)/max(t_cpu,t_gpu)`. Three consecutive
eligible observations within 5%, in one declared epoch, establish diagnostic
sustained balance. A new epoch or invalid observation resets that window. Reports
retain the first balanced window, all later qualifying windows, and balance at
the end. This diagnostic does not change literal policy decisions.

Quantized holds, predictive holds, precision exhaustion, bracket resolution,
no bracket, invalid bracket, unsupported partitions, and exhausted budgets remain
distinct reasons. Four or more timing-sign reversals without final sustained
balance are reported as an **oscillation diagnostic**, not a proof of a periodic
cycle. Synthetic literal-binary fixtures separately demonstrate the recurrence's
cycle-like behavior and budget exhaustion.

Corrected bisection records every bracketing observation and midpoint update.
It checks timing-sign consistency against earlier samples, including subsequent
observations after a completed bracket search. A sign contradiction invalidates
the bracket; a declared epoch change restarts bracketing. Some noisy/non-monotone
functions can evade these checks. The logarithmic bound is tested only on a
stationary monotone analytical fixture; it is not asserted for live WSL timings.

## Robust proportional extension

All following values appear in the frozen configuration:

1. Estimate full-work branch times `C=t_cpu/r` and `G=t_gpu/(1-r)` from an eligible
   interior observation. These include fixed dispatch/transfer costs, so linear
   extrapolation is an approximation. Nonfinite normalized estimates are rejected.
2. Maintain exponentially weighted estimates with alpha=0.25:
   `filtered_next = 0.25*current + 0.75*previous`. Initialize with the first sample.
   Filter normalized estimates rather than raw times from different work shares.
3. Compare current estimates with the previous filtered estimates. A multiplicative
   deviation of at least 2 in either direction for either branch on two consecutive
   eligible samples resets the filter and dwell counter. A declared epoch change
   also resets them. The first suspicious sample still updates the filter; the
   next comparison therefore uses that updated filter. Detection is heuristic,
   can miss changes, and can misclassify transfer overhead/noise as a phase change.
4. Hold if the raw relative imbalance is within the 5% deadband. Otherwise honor
   a three-observation minimum movement spacing: after a move, the next two eligible
   observations cannot move the split. Deadband holds count toward this dwell.
5. Form `target=filtered_G/(filtered_C+filtered_G)`. Apply half of `target-r`,
   bounded to at most 0.10 share movement per eligible decision, then round to a
   legal count. Log filtered state, dwell, phase-hit count, and the decision reason.

Explicit and detected phase resets re-enable immediate adjustment. Ineligible
observations do not change this state. This extension has no claimed optimality
or guaranteed stability under arbitrary changing capacity.

## Persistent native feedback protocol

The existing executable accepts an optional final `feedback` argument:

```text
atlas_kmeans INPUT.kmdat INITIAL_CPU_COUNT MAX_ITERS WORKERS fixed|converge feedback
```

One process owns all centroids, previous memberships, CPU workers, GPU point
residency, device allocations, and warm-up state for the entire trial. After each
measured iteration except the final one, it flushes the validated JSON record
and waits for one newline-terminated unsigned CPU count on stdin. The Python
adapter calculates a decision from that observation, records it, and supplies
the next count. No input is read after warm-up or after the final iteration.
Closed input, malformed numbers, and illegal counts fail with nonzero exit.
The adapter kills and reaps the process on protocol failure or watchdog expiry.

Native assignment, independent references, and migration accounting retain the
[Phase 4 contract](kmeans_spec.md). Both branch completion times are measured from
the common iteration release: CPU team/dispatch latency is included, and GPU time
includes initial/migration uploads, computation, and downloads. Shared centroid
merge time is included in total iteration duration, not attributed to one branch.

The next iteration's `control_ns` starts after previous-iteration validation and
before previous-record serialization. It ends after receipt/parsing of the next
count. It includes record export, controller calculation, Python bookkeeping,
pipe communication, and waiting. This interval is disjoint from measured iteration
work and from native independent validation. The first measured iteration has no
feedback interval; initial configuration is part of startup.

`active_and_control_ns = active_ns + sum(control_ns)` is the adaptation-inclusive
comparison metric. Separately measured Python controller calculation is **nested**
inside feedback time and must not be added again. Native `loop_elapsed_ns` also
includes validation gaps between iterations. Parent command elapsed includes
startup, warm-up, final validation/export, and process cleanup. These are distinct
scopes; none is substituted for a valid energy measurement interval.

Static baselines use the same feedback protocol and validation instrumentation,
so they pay the common transport/recording overhead. These measurements describe
this instrumented Python/native architecture, not an uninstrumented C++ controller.

## Experiment and replay protocol

Every run starts with fresh policy state and identical workload initialization.
The plan is saved before any hardware trial. Correctness checks cover five inputs,
endpoints/interior starts, all six policies, and separate convergence-mode runs.
The fixed-work benchmark uses a new generated held-out input (seed 6505,
N=32,771, D=8, K=8), 24 iterations, and three randomized blocks by default.

Starts are 15%, 50%, 90%, and a near-balance share chosen only from retained Phase 4
tuning observations. Static comparators are CPU-only, GPU-only, 50/50, and the
Phase 4 tuning-selected share. Identical endpoint configurations retain distinct
baseline names and independent trials; they are not presented as different
algorithms. All policy parameters and starts are frozen before held-out trials.

The deterministic scenario suite covers stationary, asymmetric, noisy,
transfer-dominated/no-interior-root, and declared changing-capacity functions.
All functions are explicitly synthetic assumptions. Real kmeans timings are never
replaced by those functions or artificially delayed. Hardware clocks remain on
their unchanged automatic policy; real constant capacity is not established.

Records retain input provenance, source/binary/config hashes, plans, stdout/stderr,
every decision/state transition, realized splits, control timing, and full final
labels. Assessment checks native reference results, small-input Python oracles,
same-input trajectories across policies, timing arithmetic, and each next split
against its decision. Replay verifies archive inventory and byte hashes, regenerates
the synthetic suite, then recomputes every live decision and summary.

## Qualification boundary

Controller arithmetic, partition correctness, and exploratory host-time comparisons
can pass independently of energy qualification. All energy values remain null.
Total adaptation/migration energy and fixed-capacity hardware guarantees cannot
pass the full Phase 5 research gate on the currently qualified evidence. No
counterfactual energy, speedup guarantee, or energy-saving percentage is invented.
