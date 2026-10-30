# ATLAS GreenGPU: Proposed System Architecture

Specification version: 1.0

Updated: 2026-10-31

Status: Scientific specification prepared in Phase 0; implementation and hardware capabilities remain unverified.

## Document authority and project state

The [implementation plan](IMPLEMENTATION_PLAN.md) defines phase order and acceptance gates. The [controller specification](docs/algorithm_spec.md) defines policy behavior, and the [measurement specification](docs/measurement_spec.md) defines evidence and metrics. The [paper mapping](docs/paper_mapping.md) distinguishes published claims, mathematical deductions, and ATLAS decisions.

The earlier implementation was deliberately removed. The virtual environment, source paper, and Git history remain. This architecture describes the intended replacement; module names below are planned responsibilities, not existing working code.

The previous architecture is retained in Git history at revision `ce1e09a942dad5b9b9900b30ce873736d9c46d28`. Its unsupported assumptions and their resolutions are recorded in the [Phase 0 review](docs/phase0_review.md).

## Intended system

An iterative workload is divided between a CPU and a GPU. Both branches execute concurrently and produce results at an application barrier. The allocation controller uses their measured completion times to propose the next split. A faster GPU frequency controller uses valid core and memory observations to propose a feasible frequency pair. CPU policy is an independently recorded experimental factor.

The runtime owns dispatch, synchronization, data migration, timing, control application, and restoration. Controllers perform mathematical state transitions without hardware I/O. No component infers that a successful calculation or driver request proves a physical energy benefit.

## Execution and data flow

```text
Input + frozen experiment configuration
  -> capability validation
  -> preparation and separately recorded warm-up
  -> measured workload interval
       -> legal CPU/GPU partition
       -> concurrent CPU execution and GPU execution
       -> required transfers, synchronization, and merge
       -> correctness and timing observations
       -> allocation decision at a valid barrier
       -> frequency observations/decisions on eligible sampling windows
       -> verified setting changes and recorded settling
  -> final workload validation
  -> prior-policy restoration
  -> retained raw records and regenerated analysis
```

CPU and GPU completion times include their branch-specific transfer and dispatch costs. Shared serial work remains visible. Dynamic partitioning must preserve kmeans reduction semantics and HotSpot halo/timestep consistency.

## Module boundaries

- **Configuration and contracts:** versioned policy parameters, workload definitions, capability reports, observations, decisions, and run records.
- **Workload backend:** independent CPU reference plus native CPU/OpenMP and CUDA execution implementations, exposed through a narrow Python interface after toolchain validation.
- **Partitioning:** `paper_simple`, `paper_adaptive`, `paper_binary_literal`, `paper_proportional`, and corrected `bracketed_bisection`.
- **Frequency selection:** `paper_additive`, using the printed loss/update equations and documented initialization/tie choices.
- **Telemetry and energy:** timestamped observations with physical domain, metric semantics, units, and quality information.
- **Actuation:** support discovery, requested-versus-observed setting verification, and prior-policy restoration.
- **Runtime:** concurrency, barriers, controller cadence, invalid-observation handling, and experiment lifecycle.
- **Analysis:** baselines, uncertainty, adaptation cost, and plots regenerated from retained records.

Planned files and their staged creation are listed in the implementation plan. Do not create placeholders that imply an unfinished module is operational.

## Capability-dependent execution

**Hardware execution** uses real computation and available sensors. Full GPU frequency replication additionally requires verified independent core/memory control. Without it, allocation-only hardware evaluation remains a useful but explicitly limited result.

**Trace replay** inspects controller decisions on recorded observations. It cannot measure the counterfactual energy of unexecuted actions.

**Simulation** tests mathematical scenarios and failure cases with disclosed assumptions. It is not historical-hardware calibration unless separately supported by calibration data and validation.

No specific current CPU/GPU model, frequency range, power range, operating system path, or sensor availability is assumed. Phase 1 resolves the environment and capability questions; Phase 8 verifies live actuation.

## Important design constraints

- The printed binary recurrence is diagnostic; only the corrected bracketed variant carries a conditional bracket-width bound.
- The printed additive weight update has no implicit forgetting or weight clipping. Its beta parameter does not tune ranking for fixed observations and equal initial weights under exact arithmetic.
- Timescale separation is configured and evaluated empirically. Neither 40 frequency periods nor 120 seconds guarantees convergence.
- Balanced branch times are a scheduling objective, not proof of minimum energy.
- The paper's 21.04% result is historical evidence for two workloads, not an ATLAS acceptance threshold.
- Monitoring and control overhead are measured rather than assumed zero.
- Hardware settings come from supported capabilities; no hardcoded GPU power range is adopted from the former draft.
- Whole-system, CPU-package, GPU-board, and synthetic energy remain distinct.

## Repository workflow

Keep the retained virtual environment and paper. Record specification/configuration changes with their reasons. Preserve source and workload license information. No remote push is performed without explicit user instruction. Seeds support repeatability of fixtures and generated inputs, but do not eliminate hardware timing variability.

The next work is [Phase 1 capability discovery](docs/phase0_review.md), not a benchmark claim or automatic hardware tuning.
