# ATLAS GreenGPU: Phased Implementation Plan

Date: 2026-10-31

Status: Phases 0, 1, 3, and 6 workload validation complete. Phase 7 offline frequency policy validation is complete; see [Phase 7 review](docs/phase7_review.md). Phase 4 kmeans and Phase 5 allocation implementations are complete, with correctness and exploratory timing evidence. Phase 6 adds validated thermal-state migration and allocation trials on HotSpot. Phase 2 physical GPU energy and device-event timing qualification remain blocked; automatic clock policy does not establish fixed capacity. The full Phase 4/5 research exit gates remain incomplete. See the [Phase 0 review](docs/phase0_review.md), [Phase 1 review](docs/phase1_review.md), [Phase 2 review](docs/phase2_review.md), [Phase 3 review](docs/phase3_review.md), [Phase 4 review](docs/phase4_review.md), [Phase 5 review](docs/phase5_review.md), and [Phase 6 review](docs/phase6_review.md).

Primary source: [GreenGPU research paper](GreenGPU.pdf), Kai Ma et al., *Energy conservation for GPU-CPU architectures with dynamic workload division and frequency scaling*, 2016, journal pages 21-33.

## 1. Intended outcome and starting point

Build a reproducible implementation of GreenGPU that executes real CPU and GPU work concurrently, adjusts their workload shares, and coordinates GPU core and memory frequency settings when the hardware supports those controls. Measure whether these decisions reduce energy for the same correct computation, and quantify the execution-time tradeoff.

The initial implementation, tests, and demonstration scripts were removed before this plan. Preserve the existing virtual environment, research paper, Git history, and documentation. [ARCHITECTURE.md](ARCHITECTURE.md) now describes the proposed replacement and links the Phase 0 specifications. The former architecture's completed-phase labels, assumed hardware controls, convergence guarantees, and performance targets have been superseded; its claims are audited in the Phase 0 review. This plan governs the new implementation sequence.

The result must support three explicitly identified execution modes:

- **Hardware execution:** Real workload execution and measured telemetry. Record which controls actually work and which energy domains are measured.
- **Trace replay:** Feed recorded observations to controllers to inspect decisions. Replay cannot establish the performance or energy effects of actions that were never executed.
- **Simulation:** Controlled mathematical scenarios for correctness, sensitivity, and failure testing. Synthetic energy is never presented as measured hardware energy.

An exact reproduction of historical numerical results additionally requires comparable hardware, software, workloads, and measurement boundaries. A modern implementation is a replication of the method, not automatically a reproduction of the paper's percentages.

### Scope of the first complete version

- One CPU host and one CUDA-capable GPU.
- Two substantive workloads: kmeans and HotSpot.
- The paper's four allocation algorithms, with separately named corrections where required.
- The published GPU frequency controller, with separately named optional improvements.
- CPU policy observation and supported policy control, with the actual policy recorded.
- A command-line runner, raw experiment records, automated correctness checks, and reproducible plots.

Defer multiple GPUs, cluster scheduling, a web dashboard, power-cap optimization, learned energy predictors, and new scheduling research until this version is evaluated.

## 2. Scientific requirements that shape the implementation

### 2.1 Keep the paper's results attached to their actual experiments

The complete framework's **21.04% average energy saving** is reported for **kmeans and hotspot**, relative to GPU-only execution at peak frequencies (§7.3, journal p. 30). It is a comparison point, not a required result on this machine.

GPU frequency scaling alone reports **5.97% average GPU energy saving**, **29.2% average dynamic GPU energy saving**, and **2.95% longer execution time** against its peak-frequency baseline (§7.1, p. 28). Dynamic energy excludes idle energy and is a different quantity from total energy. The **12.48%** CPU-plus-GPU scaling result in Figure 6c is explicitly emulated. The complete framework's **1.7%** slowdown is relative to workload division alone, not the peak-frequency GPU-only baseline.

The full-system results use Testbed A. Testbed B evaluates allocation algorithms without GPU frequency scaling. Improved Binary Search's favorable iteration counts on Testbed B must not be presented as the algorithm underlying all headline full-system results.

### 2.2 Resolve published ambiguities without silently changing the method

**Allocation search.** Section 5.2.3 prints `r_next = (1 + r) / 2` when the GPU is slower and `r_next = r / 2` otherwise. These rules do not retain a search bracket. For a balance point of 0.8, starting at 0.5 gives approximately `0.5, 0.75, 0.875, 0.4375, 0.71875, 0.859375, ...`, which cycles. Implement this as `paper_binary_literal` for diagnosis, with an iteration cap. Implement a distinct `bracketed_bisection` using persistent lower and upper bounds. Its bracket-width bound holds only for a stationary monotone timing difference with a bracketed root and reliable sign observations; it is not an energy-optimality guarantee.

**Frequency weights.** Equation 4 prints:

```text
W_next[i,j] = W[i,j] + 1 - (1 - beta) * L[i,j]
```

For equal initial weights and updates to every candidate each period, after `t` updates this is `W0 + t - (1 - beta) * cumulative_loss[i,j]`. For fixed `0 < beta < 1`, beta cannot change the ranking under exact arithmetic. The equation also does not keep weights in the stated [0,1] range. Implement and label the literal rule; do not introduce clipping or decay and call it unchanged. A discounted-loss controller is an optional extension with its own identity and experiments.

**Coordination claim.** The published joint loss is additive in separate core and memory losses. With equal initialization and a full Cartesian candidate set, its maximization separates into independent cumulative core and memory choices. Hardware feedback can still couple the observed utilization signals, but the loss itself has no explicit interaction term. Measure joint behavior instead of assuming the table guarantees an optimal coupled decision.

**Timescale separation.** Section 4 suggests an allocation interval at least 40 times the frequency interval for the tested 6-by-6 settings. This is a design heuristic, not a stability proof. The 3-second frequency interval in §7.1 and the 40-times suggestion must not become a universal claim that a 120-second interval guarantees convergence.

**Balanced time versus minimum energy.** Matching CPU and GPU completion times targets waiting time. The paper itself reports kmeans settling at 20% CPU work while a static sweep finds minimum energy at 15% (§7.2). Keep a measured static energy sweep as a comparator.

### 2.3 Measurement semantics are part of fidelity

The paper describes memory utilization as achieved bandwidth divided by peak bandwidth. NVIDIA's documented NVML `memory` utilization is instead the fraction of a sample period during which device memory was active. It is also different from allocated VRAM divided by capacity. Preserve the metric's actual meaning and label a substitution as a methodological adaptation. See the [NVIDIA NVML utilization definition, archived R535 reference](https://docs.nvidia.com/deploy/archive/R535/nvml-api/structnvmlUtilization__t.html).

Normalize utilization to [0,1] inside the controller. Retain original units and metric names in raw logs. Missing telemetry is unavailable data, never zero utilization.

## 3. Architecture and implementation choices

Use Python for configuration, controllers, telemetry adapters, experiment orchestration, and analysis. Use a narrow native execution backend with C++/OpenMP CPU kernels and CUDA GPU kernels for the benchmark path. This keeps the computational structure close to the paper and makes concurrency and synchronization explicit. Use a simple CPU reference implementation as the correctness oracle.

Select and pin compatible compiler, CUDA, Python, and binding versions after capability discovery. Do not install competing GPU frameworks just to get a demo running. A high-level framework prototype may help check mathematics, but results from different workload backends must remain separate experimental series.

Planned repository organization, created incrementally rather than as empty scaffolding:

```text
ATLAS/
  GreenGPU.pdf
  ARCHITECTURE.md
  IMPLEMENTATION_PLAN.md
  pyproject.toml
  src/greengpu/
    config.py
    contracts.py
    capabilities.py
    telemetry/
    energy/
    partitioning/
    frequency/
    workloads/
    runtime/
    cli.py
  native/
    cpu/
    cuda/
    bindings/
  configs/
    workloads/
    policies/
    experiments/
  tests/
    unit/
    integration/
    hardware/
  experiments/
  analysis/
  docs/
  results/
```

Core contracts:

- `CapabilityReport`: hardware identity, runtime versions, available sensors, units, supported controls, and failure reasons.
- `Workload`: prepare, partition, execute both branches, synchronize, merge, validate, and finalize. Expose legal partition granularity and migration requirements.
- `IterationObservation`: requested and realized CPU fraction, branch completion times, transfer/reduction/barrier timings, workload phase, and validity flags.
- `PartitionPolicy`: consumes an observation and emits a proposed split plus a reason; performs no hardware I/O.
- `FrequencyPolicy`: consumes valid utilization observations and feasible clock pairs; emits a proposed pair plus controller state.
- `Actuator`: applies supported settings, verifies the result, and restores the previous policy. Requested and observed clocks are separate fields.
- `EnergyMeter`: declares its physical domain and returns timestamped readings or counter deltas with quality flags.
- `RunRecord`: binds input identity, correctness, configuration, environment, raw observations, and result calculations to one run ID.

Prefer existing supported telemetry bindings over a handwritten native interface. Hardware adapters must not contain controller mathematics, and controllers must not fabricate telemetry to continue execution.

## 4. Phase 0 - Establish the scientific specification

**Status:** Complete for specification work only. Deliverables: [paper mapping](docs/paper_mapping.md), [controller specification](docs/algorithm_spec.md), [measurement specification](docs/measurement_spec.md), [analytical fixtures](docs/spec_fixtures.json), and [review/handoff](docs/phase0_review.md). This does not mark controller code, workloads, or hardware validation complete.

**Goal:** Make the intended behavior testable before rebuilding code.

**Work:**

1. Map each planned algorithm and metric to the paper's section, equation, figure, and experimental baseline.
2. Record the ambiguities above in `docs/paper_mapping.md`, distinguishing published text, derived mathematics, and proposed corrections.
3. Define policy IDs: `paper_simple`, `paper_adaptive`, `paper_binary_literal`, `paper_proportional`, and `bracketed_bisection`; reserve a separate ID for any robust proportional extension.
4. Define the literal frequency policy as `paper_additive`; any discounted or directly energy-aware policy is a later extension.
5. Specify valid ratio units, rounding, tie-breaking, initialization, stopping, phase-change handling, and missing-observation behavior. Mark decisions absent from the paper as implementation choices.
6. Freeze initial hypotheses: allocation may reduce waiting and energy; frequency control may save energy with a runtime cost; combining them may outperform either alone. Negative results remain valid outcomes.

**Deliverables:** Paper mapping, algorithm specification, metric definitions, and a list of unsupported historical claims in the old architecture draft.

**Exit gate:** Every numerical target and algorithm has a source or is explicitly marked as an ATLAS design choice. No universal 21.04% saving, five-iteration convergence, or 3% slowdown claim remains in the active specification.

## 5. Phase 1 - Discover capabilities and select the execution path

**Status:** Complete with real CPU/CUDA correctness evidence and partial telemetry. See [environment decision and limitations](docs/phase1_review.md), [capabilities](results/phase1/2026-10-31-final/capabilities.json), and [environment manifest](results/phase1/2026-10-31-final/environment.json). CPU powercap/governor interfaces are not exposed in WSL; GPU actuation and measurement accuracy remain unverified.

**Depends on:** Phase 0.

**Goal:** Determine what this machine can execute, measure, and control.

**Work:**

1. Inventory the CPU, GPU, memory, OS, driver, compiler, CUDA runtime, Python interpreter, and existing virtual environment without recreating the environment.
2. Test GPU execution independently from monitoring and management. Record device identity rather than assuming the hardware mentioned in the historical architecture document is present.
3. Probe power/energy readings, utilization, temperature, observed clocks, clock-state queries, and CPU energy/governor interfaces. Read-only discovery does not imply clock writes work; label writes unverified until Phase 8.
4. Determine whether the intended environment is native Windows, WSL, or native Linux. The [NVIDIA WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html) documents management and query limitations; CUDA execution alone does not establish access to every needed control.
5. Select the native build route and produce a CPU/GPU smoke computation. Record unsupported features and required privileges precisely.
6. Choose an evidence path: complete control and energy measurement; allocation-only hardware evaluation; or controller simulation/replay. Hardware limitations narrow the claims, not the honesty of the output.

**Deliverables:** `capabilities.json`, environment manifest, minimal packaging/build configuration, and an environment decision note.

**Exit gate:** A real CPU result and GPU result agree; the selected environment is reproducible; sensor availability and unresolved actuation capabilities are explicit. If no GPU execution is available, the hardware branch is blocked and only a labeled simulation deliverable proceeds.

## 6. Phase 2 - Build measurement and experiment infrastructure

**Status:** Implementation complete with analytic tests, real fixed-work CUDA trials, paired monitoring-overhead measurements, and checksum-verified replay. On the current WSL setup, the cumulative counter disagrees with sampled power, and device-event spans exceed their enclosing host intervals. Consistency guards reject disputed energy and device timing. CPU package and wall energy are unavailable. Infrastructure is ready for Phase 3 correctness work using the common host clock, but physical energy/device-timing claims remain blocked pending qualification. See [review and evidence](docs/phase2_review.md) and [run contract](docs/run_schema.md).

**Depends on:** Phase 1.

**Goal:** Make energy and timing results trustworthy before optimizing them.

**Work:**

1. Use one monotonic host timebase for run events. Measure synchronized GPU kernel duration separately from end-to-end elapsed time; an asynchronous launch duration is not GPU execution time.
2. Implement GPU telemetry and available CPU or wall-meter adapters. Record sensor cadence, acquisition timestamp, documented averaging behavior, sample validity, and polling overhead.
3. Prefer a supported energy-counter delta when suitable. Otherwise integrate timestamped power readings over the defined run boundary using trapezoidal integration. Check stale samples and gaps rather than interpolating silently across an unreliable interval.
4. If using Linux powercap, handle counter wrap and domain hierarchy. `energy_uj` and `max_energy_range_uj` are documented by the [Linux powercap framework](https://docs.kernel.org/power/powercap/powercap.html). Do not add package energy to its own included subdomains.
5. Keep GPU-board energy, CPU-package energy, their non-overlapping sum, and whole-system wall energy as distinct metrics. GPU plus CPU measurements are not automatically whole-system energy. Control battery charging and background load when using a wall meter on a laptop.
6. Capture warm-up separately. Define a common measured boundary including dispatch, transfers, computation, barriers, reductions, migration, and controller/actuation overhead. Report preparation/startup separately and include it in an additional total-job metric when relevant.
7. Write immutable raw records, input hashes, seeds, trial order, software versions, and explicit invalid-run reasons.

**Deliverables:** Telemetry adapters, energy integrator, timing recorder, run schema, and a repeatable baseline runner.

**Exit gate:** Known power traces integrate correctly; counter wrap, missing sensors, and gaps are handled; repeated baseline runs expose measurement noise; monitoring overhead is measured. No energy-improvement claim proceeds without a valid energy domain.

## 7. Phase 3 - Prove correct concurrent execution with static splits

**Status:** Complete. The native OpenMP/CUDA harness passes 40 same-input partition configurations and 120 measured iterations, including endpoints and one-item branches. Nsight kernel activity overlaps CPU compute regions in all three measured profile iterations. Unit tests, native CTests, CUDA memory/race checks, and checksum-verified replay pass. See the [implementation review](docs/phase3_review.md) and [final evidence](results/phase3/2026-10-31-final/summary.json). Phase 2 energy/device-timing restrictions remain in force.

**Depends on:** Phase 2.

**Goal:** Establish a reliable execution substrate before adding feedback control.

**Work:**

1. Implement a deterministic elementwise/reduction fixture with a CPU reference, CPU backend, and GPU backend.
2. Assign disjoint logical work to CPU and GPU. Execute both branches concurrently, then synchronize and merge results. Ensure the Python boundary releases control during native execution and does not serialize the branches.
3. Validate CPU-only, GPU-only, and several interior splits, including awkward sizes and minimum-size partitions. Record the realized fraction after rounding to legal work units.
4. Expose branch timing from a common iteration boundary, including branch-specific transfer costs. Record shared merge and reduction time separately.
5. Use timelines or profiler evidence to demonstrate overlap. Bound CPU worker count so orchestration and GPU submission are not starved.
6. Establish controlled allocation and buffer reuse. Ensure data movement required by correctness is counted, while validation overhead is separately identified and applied consistently across policies.

**Deliverables:** Native execution adapter, static partition dispatcher, correctness fixture, and concurrency trace.

**Exit gate:** All splits compute the same logical result within predefined tolerances; observed overlap is demonstrated; no endpoint divides by zero; every input item is processed exactly once.

## 8. Phase 4 - Implement kmeans as the first real workload

**Status:** Software/correctness/timing implementation complete. The FP64 OpenMP/CUDA workload passes independent references, fixed and convergence modes, endpoint/interior splits, and boundary migrations. The final run retains 102 trials, coarse/refined static timing curves, held-out evaluation, and kernel/CPU overlap in three profile iterations. Tests and CUDA memory/race checks pass. The platform automatic clock policy is unchanged, not locked or verified constant; this is an exploratory timing baseline. Energy curves/minima remain unavailable, so the full exit gate below is blocked pending Phase 2 measurement qualification. See the [variant contract](docs/kmeans_spec.md), [review](docs/phase4_review.md), and [final results](results/phase4/2026-10-31-final-v2/summary.json).

**Depends on:** Phase 3.

**Goal:** Deliver the first complete heterogeneous benchmark with an iteration barrier.

**Work:**

1. Define a reference kmeans variant: initialization seed, distance metric, precision, number of clusters, empty-cluster rule, convergence rule, and maximum iterations.
2. Partition points for assignment. Each side calculates local cluster sums and counts using the same centroids; merge them once and broadcast the next centroids.
3. Keep persistent device data where beneficial. Document exactly which point data, labels, and aggregate values move when a boundary changes. Count migration and centroid communication.
4. Compare per-iteration aggregates and objectives to the reference. Resolve label permutation and floating-point reduction differences without hiding incorrect assignments.
5. Use fixed logical iteration counts for controlled policy comparisons and separately evaluate convergence-to-tolerance runs with matched output quality. Do not compare policies that perform different amounts of useful work without accounting for that difference.
6. Run CPU-only, GPU-only, 50/50, and a static split sweep at one controlled clock policy. Sweep coarsely first and refine near candidate minima using independent repeat trials.

**Deliverables:** kmeans workload, correctness dataset, input configurations, and measured runtime/energy curves against CPU share.

**Exit gate:** kmeans is correct at endpoints and interior splits; migration works; static timing and energy minima are measured separately. The workload remains usable even if dynamic control later loses to a baseline.

## 9. Phase 5 - Implement and evaluate workload allocation

**Status:** Tier 1 implementation complete: pure reference/corrected policies, the separately specified robust extension, deterministic diagnostics, and persistent native kmeans feedback. Real trials include endpoint probes, changing partitions, convergence-mode correctness, held-out static comparisons, and auditable transition replay. Adaptation/IPC and migration time are included. No claim of universal convergence or improvement is made. Energy accounting and fixed-capacity hardware qualification remain blocked, so the full research gate is not complete. See the [contract](docs/phase5_spec.md) and [validation review](docs/phase5_review.md).

**Depends on:** Phase 4.

**Goal:** Validate Tier 1 independently with the clock policy held constant.

**Work:**

1. Implement the fixed-step heuristic and its prediction safeguard, then adaptive step halving. Specify continuous versus integer-percent steps rather than inheriting ambiguous rounding from examples.
2. Implement the literal binary recurrence and bounded diagnostic termination. Demonstrate its cycle counterexample as an expected limitation.
3. Implement corrected bracketed bisection. Evaluate legal interior bracket endpoints to establish whether a balance point exists without inventing throughput for an empty branch. If timing signs are noisy or non-monotone, report that the assumptions failed rather than asserting convergence.
4. Implement the paper's proportional rule:

   ```text
   r_target = (r * t_gpu) / (r * t_gpu + (1 - r) * t_cpu)
   ```

5. Require valid positive branch observations and interior fractions for that estimate. Use an explicit interior probe or other documented fallback when one branch has no work. Never inject arbitrary tiny denominators and call the result measured throughput.
6. Add a separately named robust variant only after the literal policy works: deadband, limited ratio movement, filtered observations, minimum dwell, and reinitialization after a detected phase change.
7. Test starts at 15%, 50%, and 90%, plus cases near the measured balance point. Evaluate stationary, noisy, transfer-dominated, asymmetric, and changing-phase workloads.
8. Define convergence as balance within tolerance over consecutive eligible observations. A small ratio update alone is stagnation, not proof of balance. Report timeout and oscillation explicitly.

**Deliverables:** Pure allocation policies, deterministic policy tests, kmeans allocation experiments, and convergence traces.

**Exit gate:** Corrected bisection obeys its bracket-width bound on a stationary monotone fixture; the proportional rule gives approximately 0.17647 for `r=0.3, t_cpu=6, t_gpu=3`; real-run correctness survives changing partitions; total adaptation and migration costs are included in energy results.

## 10. Phase 6 - Add HotSpot and validate state migration

**Status:** Complete for the workload/migration exit gate. The FP64 thermal stencil passes independent references, halo/timestep checks, endpoint and repeated ownership changes, deferred-validation comparisons, and 600-step trajectories including a 2048×2048 migration case. Static tuning and held-out Tier 1 trials retain all communication/feedback costs. CUDA overlap, memory/race checks, and checksum replay are verified. The model uses a documented stability limiter and generated input rather than claiming historical numerical reproduction. Energy and fixed-capacity qualification remain blocked. See the [model contract](docs/hotspot_spec.md) and [review/evidence](docs/phase6_review.md).

**Depends on:** Phase 5.

**Goal:** Test Tier 1 on a workload whose partitions exchange boundary data every timestep.

**Work:**

1. Implement the intended Rodinia-style thermal stencil with documented equations, boundary conditions, coefficients, inputs, and precision. Verify against an independent CPU reference.
2. Partition contiguous rows. Exchange the required halo rows from the same previous timestep; use double buffering so neither branch reads partially updated temperatures.
3. Synchronize at each logical timestep. If a split moves, transfer ownership of the affected rows and reconstruct halos before the next computation.
4. Enforce legal minimum partition sizes and provide genuine CPU-only/GPU-only paths without phantom halo transfers.
5. Test irregular grids, small grids, repeated boundary movements, and many timesteps to expose accumulated error and off-by-one defects.
6. Repeat static sweeps and Tier 1 experiments. Compare the cost of stateful migration with kmeans, including cases where frequent repartitioning is counterproductive.

**Deliverables:** HotSpot workload, halo/migration tests, and allocation results for both substantive benchmarks.

**Exit gate:** Dynamic splits preserve the reference solution within declared tolerances; every boundary exchange is timed; adaptation never changes the total grid or timestep count.

## 11. Phase 7 - Implement the frequency controller offline

**Status:** Offline implementation and validation complete, 2026-10-31. The
literal policy, separate discounted-loss extension, 23 deterministic scenarios,
strict decision replay, and timestamp-paired read-only hardware trace are retained.
See the [Phase 7 review](docs/phase7_review.md) and
[frequency contract](docs/frequency_spec.md). No live actuation or energy benefit
is established by this gate.

**Depends on:** Phases 0-2; complete before live frequency experiments.

**Goal:** Validate Tier 2 mathematics without confusing controller decisions with hardware effects.

**Work:**

1. Implement the paper's asymmetric utilization losses, core/memory weighting, and additive update. Initial reference parameters are `alpha_core=0.15`, `alpha_memory=0.02`, `gamma=0.3`, and `beta=0.2`; they are paper settings, not calibrated constants for this machine.
2. Map utilization targets linearly between the lowest and highest selected frequencies as a documented interpretation of §5.1. Handle single-level domains explicitly and verify nonuniform frequency sets.
3. Use a feasible-pair list. The paper uses 6-by-6 pairs, but a modern device must not be assumed to support every Cartesian combination.
4. Specify initial frequency selection and deterministic tie-breaking. Log weights or equivalent cumulative losses, observations, and decisions.
5. Test equal initialization, full-history behavior, ranking invariance to beta in the literal rule, phase changes, missing data, and long runs. If subtracting a common offset for numerical stability, prove that it preserves rankings.
6. Replay valid hardware traces for decision inspection. Use analytical fixtures for causal closed-loop tests; label their timing and power functions as synthetic assumptions.
7. Optionally compare a discounted-loss extension that explicitly forgets old observations. Define its formula, discount parameter, and rationale separately from the paper's beta.

**Deliverables:** Frequency policies, mathematical tests, replay tool, and diagnostic synthetic scenarios.

**Exit gate:** Published equations match independent hand calculations; the beta limitation is demonstrated; missing memory telemetry disables or explicitly changes the policy; replay output contains no counterfactual hardware energy claim.

## 12. Phase 8 - Validate live actuation and frequency-only behavior

**Depends on:** Phases 1-2, 4, 6, and 7.

**Goal:** Establish that requested clock changes actually occur and evaluate Tier 2 alone.

**Work:**

1. Discover permitted settings through the installed driver interfaces. Verify each selected core/memory pair, relevant permissions, settling behavior, and actual observed clocks under load. Never hardcode the historical draft's power range.
2. Capture the prior clock/governor policy, implement restoration on ordinary exit and handled interruption, and persist recovery instructions for failures that cannot execute cleanup. Test restoration before long experiments.
3. Separate independent GPU core and memory control, CPU governor control, and GPU power caps. A power-cap experiment is a different actuator study and cannot stand in for the paper's frequency controller.
4. If a control is unsupported, explicitly mark the corresponding experiment unavailable. Continue allocation-only hardware experiments and frequency-policy diagnostics; do not silently switch to simulated clocks.
5. Perform a bounded static frequency sweep at a fixed split. Obtain a measured runtime/energy reference surface, excluding unsupported pairs. Treat the best sampled point as an empirical comparator, not a proven global optimum.
6. Run GPU-only frequency-control experiments, holding the CPU policy constant and reporting it. Add compute-heavy and memory-heavy diagnostic kernels, characterizing their behavior instead of relying on their names.
7. Test the chosen CPU policy separately. CPU utilization from host synchronization can be high while useful CPU computation is absent; do not assume an ondemand policy necessarily lowers the clock in that case.

**Deliverables:** Verified actuator, recovery procedure, capability updates, static clock sweep, and measured frequency-only experiments.

**Exit gate:** Supported changes are verified and restored; blocked features have explicit outcomes; correctness is unchanged; reported energy savings and slowdowns use matching baseline and measurement domains. Without independent core/memory control, full Tier 2 hardware replication remains unavailable.

## 13. Phase 9 - Integrate the two control loops

**Depends on:** Phases 5-8.

**Goal:** Run both tiers without corrupting observations or causing repeated repartitioning and clock oscillation.

**Work:**

1. Implement a runtime state machine: prepare, warm up, execute, observe, decide, apply, settle, finalize, and restore. Record every transition and failure.
2. Apply workload changes only at valid application barriers. Aggregate frequency observations over windows containing enough fresh samples and record which clock settings each branch experienced.
3. Support a paper-inspired timescale configuration and a separately labeled adaptation based on observed settling. Batch multiple logical workload iterations into an allocation epoch when necessary without introducing artificial sleep to satisfy a timing ratio.
4. After a workload redistribution, provide a measured settling period before another allocation decision. Identify clock-transition windows so they do not masquerade as stable throughput estimates; retain their real time and energy in totals.
5. Test the paper's retained history first. Any controller reset on a new workload phase or allocation epoch is an explicit extension, evaluated separately.
6. Add stale-telemetry handling, bounded controller time, stop conditions, interruption recovery, and thermal/power-throttling annotations. Log external limiter events rather than attributing every clock change to ATLAS.
7. Measure polling, computation, redistribution, and actuation overhead separately. Confirm host control work is included in the measured job and does not serialize CPU/GPU computation.

**Deliverables:** Integrated runner, end-to-end configuration, interaction traces, and integration tests.

**Exit gate:** Both workloads remain correct during joint control; restoration works; every result can be traced to effective settings; stable and changing-phase trials expose settling and oscillation behavior. No claim of stability follows merely from selecting a 40:1 interval ratio.

## 14. Phase 10 - Run the controlled evaluation

**Depends on:** Phase 9 for complete-framework claims; allocation-only studies may proceed after Phase 6.

**Goal:** Determine which parts help, under what conditions, and at what cost.

### Baselines and ablations

- **B0:** GPU-only with a documented fixed high clock policy, where enforceable.
- **B0-default:** GPU-only with the platform's normal automatic policy. Keep this distinct from B0.
- **B1:** CPU-only under a declared CPU policy.
- **B2:** Fixed CPU/GPU split, including 50/50 and a tuned static split.
- **B3:** Dynamic allocation only; fixed GPU clock policy.
- **B4:** Frequency control only; GPU-only allocation.
- **B5:** Complete two-tier controller.
- **B6:** Best sampled static allocation/clock combination from an independent sweep, where feasible.

Record CPU governor/policy for every baseline. Where CPU policy is an experimental factor, run matched comparisons that hold it constant as well as the complete paper-inspired configuration. B3 versus B5 should isolate the additional frequency control being claimed.

Run the paper-style full framework first with the simple allocation heuristic; evaluate proportional and corrected variants as separate configurations. Do not blend all variants into a single result named GreenGPU.

### Experimental protocol

1. Freeze configurations after pilot runs. Use separate tuning and held-out evaluation inputs; the same data must not select a policy and then establish its general superiority.
2. Cover both workloads, multiple problem sizes, CPU starting shares of 15%, 50%, and 90%, and stable/changing-phase scenarios where meaningful.
3. Use randomized or counterbalanced policy order within repeat blocks. Record temperature, observed clocks, background load, and power source. Control initial thermal conditions sufficiently to avoid always testing one policy hotter.
4. Begin with ten independent measured trials per primary configuration as a planning default. Use pilot variability to set the final trial count and stopping rule before confirmatory runs; do not stop when a desired result becomes significant.
5. Use the independent trial or paired block as the statistical unit, not correlated telemetry samples. Report sample count, central tendency, dispersion, and 95% confidence intervals. Use a documented paired estimator for energy-saving uncertainty.
6. Set exclusions before evaluation: incorrect output, sensor failure, unrelated competing workload, or failed requested control. Preserve excluded runs and reasons. Do not discard valid negative savings or normal thermal behavior to improve the result.
7. Report total-run energy including adaptation, and separately show post-settling behavior. Short-job experiments must include the cost of learning from the initial split.
8. Compare to the best static settings without charging their offline search cost to only one side. Report deployment execution comparison and optional search-amortization analysis separately.

### Metrics

```text
energy_saving_percent = 100 * (E_baseline - E_policy) / E_baseline
runtime_change_percent = 100 * (T_policy - T_baseline) / T_baseline
branch_imbalance = abs(t_cpu - t_gpu) / max(t_cpu, t_gpu)
energy_delay_product = E * T
```

Branch imbalance is meaningful for active, nonzero branches with comparable timing boundaries. Record measured barrier waiting separately. Additional metrics include joules per fixed work unit, convergence/stagnation/timeout outcome, migration bytes and time, controller overhead, number of clock changes, and restoration outcome.

For dynamic energy, define and measure the idle reference under matching settings and use the corresponding elapsed duration. Never compare idle-subtracted savings directly to total-energy savings. Use end-to-end wall energy for whole-system claims; otherwise name the measured component domain.

**Deliverables:** Frozen experiment manifest, retained raw trials, confidence intervals, ablation analysis, and a findings report including regressions.

**Exit gate:** Every primary configuration has valid correctness and measurement evidence, or a documented capability limitation. A positive saving claim requires uncertainty and measurement quality sufficient to support its sign; an inconclusive or negative result is reported as such.

## 15. Phase 11 - Package the reproducible research result

**Depends on:** Phase 10.

**Goal:** Make the implementation and evidence understandable and repeatable by another person.

**Work:**

1. Generate allocation/timing traces analogous to Figure 7; per-iteration energy comparisons analogous to Figure 8; allocation trajectories analogous to Figures 9-13; convergence comparisons analogous to Figures 14-15; and total energy comparisons analogous to Figure 16.
2. Add static energy/runtime curves and a runtime-versus-energy tradeoff plot. Label axes, units, hardware, baseline, repeat counts, and uncertainty.
3. Render only collected evidence. Synthetic curves and historical numbers have separate labels. Similar visual layout is not evidence of numerical reproduction.
4. Provide commands for environment checking, correctness tests, one static run, one allocation run, one frequency-only run, one complete-framework run, and report regeneration.
5. Document algorithm deviations, unsupported controls, measurement domains, dependency versions, workload provenance/licenses, and reproducibility limits.
6. Keep small representative raw fixtures and manifests in the repository. Store larger traces in a documented location with checksums. Do not place essential research evidence only under an ignored temporary or `runs/` directory.
7. Reconcile the proposed architecture document with the actual final implementation. Phase 0 already removed obsolete historical completion claims; do not mark planned modules implemented until evidence exists.

**Deliverables:** Runnable release, updated architecture and README, reproducibility guide, plots, and final research report.

**Exit gate:** A fresh compatible environment can reproduce correctness and regenerate the published analysis from retained data. The report distinguishes implementation success, measured benefit, historical-paper comparison, and hardware limitations.

## 16. Milestones and dependency order

1. **Measurement-ready foundation:** Phases 0-2. Paper interpretations, runtime environment, and energy boundaries are explicit.
2. **First useful hardware result:** Phases 3-5. Correct concurrent kmeans, static baselines, and independently evaluated allocation policies.
3. **Two-workload allocation result:** Phase 6. HotSpot validates barriers, halos, and migration beyond a simple reduction workload.
4. **Frequency-control result:** Phases 7-8. Controller mathematics are verified and actual actuator capability is established.
5. **Complete two-tier implementation:** Phase 9. Both workloads run with traceable interacting policies.
6. **Research-quality delivery:** Phases 10-11. Controlled comparisons, retained evidence, and a reproducible report.

Phase 7's offline mathematics can be developed independently after the specification and contracts are settled, but live integration waits for the workload and measurement gates. A blocked live-control phase does not prevent allocation research; it does prevent claiming a complete hardware reproduction.

Estimate calendar effort only after Phases 1-3 expose toolchain, measurement, and concurrency costs. There is no honest fixed completion date before those uncertainties are resolved. Implement one demonstrable slice per milestone rather than a large set of unverified modules.

## 17. Decisions for the first implementation session

1. Create the paper mapping and explicit policy specifications from Phase 0.
2. Inspect the retained environment and run read-only capability discovery.
3. Select the execution environment and native build route from observed capabilities.
4. Decide which energy domain can be measured reliably and whether an external meter is required for whole-system claims.
5. Establish the run-record schema and build the first measured CPU/GPU correctness fixture.

Do not begin with the dashboard, a historical-hardware simulator tuned to match 21.04%, or aggressive live clock changes. The first deliverable is a correct, measured computation whose behavior can be explained.

## 18. Completion checklist

- [ ] CPU-only, GPU-only, static split, and dynamic split execution preserve workload correctness.
- [ ] Real CPU and GPU overlap is demonstrated rather than inferred from API calls.
- [ ] Every allocation policy has an explicit identity, valid stopping behavior, and documented assumptions.
- [ ] The literal paper algorithms and corrected/extended algorithms remain distinguishable.
- [ ] Frequency decisions use valid metrics and feasible settings; requested actions and observed effects are recorded separately.
- [ ] Missing hardware controls lead to an explicit partial-scope result, never hidden simulation.
- [ ] Energy domains, timing boundaries, transfers, reductions, adaptation, and measurement uncertainty are included in the relevant claims.
- [ ] kmeans and HotSpot results include independent repeats, appropriate baselines, and negative outcomes.
- [ ] Prior hardware policies are restored after controlled experiments, with a documented recovery path for abrupt failure.
- [ ] Raw data and manifests regenerate every reported plot and summary.
- [ ] The final report states what was reproduced, what was adapted, and what remains unverified.

## References

- [Local GreenGPU paper](GreenGPU.pdf), especially §§4-7; journal pp. 24-32 correspond to PDF pp. 4-12. Primary authority for the historical method and results.
- [Proposed repository architecture](ARCHITECTURE.md). Updated in Phase 0; the review records the superseded historical roadmap and unsupported guarantees.
- [NVIDIA NVML utilization definition, archived R535](https://docs.nvidia.com/deploy/archive/R535/nvml-api/structnvmlUtilization__t.html). Metric semantics; actual device availability is established by capability checks.
- [NVIDIA CUDA on WSL User Guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html). Environment and management limitations; checked 2026-10-31.
- [Linux Power Capping Framework](https://docs.kernel.org/power/powercap/powercap.html). Energy counters, counter ranges, and domain hierarchy; checked 2026-10-31.
