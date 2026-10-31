# ATLAS GreenGPU: Proposed System Architecture

Specification version: 1.0

Updated: 2026-10-31

Status: Phases 0, 1, 3, and 6 workload validation complete; Phase 2 measurement infrastructure implemented. Kmeans, HotSpot, and Tier 1 workload allocation have independent correctness checks and exploratory timing evaluation. HotSpot includes real evolving-state migration and halo exchange. GPU energy, device-event timing, and fixed-capacity hardware qualification remain blocked, including the full Phase 4/5 research gates. Tier 2 policies are implemented and validated offline in Phase 7; Phase 8 adds tested transaction infrastructure and real loaded diagnostics, but live actuation is blocked by unavailable prior-policy readback.

Phase 9 adds an integrated allocation/shadow coordinator, independent telemetry sampling, barrier-safe workload feedback, causal synthetic joint scenarios, and replay. The final strict concurrency gate remains incomplete and live joint control remains blocked; see [Phase 9 review](docs/phase9_review.md).

## Document authority and project state

The [implementation plan](IMPLEMENTATION_PLAN.md) defines phase order and acceptance gates. The [controller specification](docs/algorithm_spec.md) defines policy behavior, and the [measurement specification](docs/measurement_spec.md) defines evidence and metrics. The [paper mapping](docs/paper_mapping.md) distinguishes published claims, mathematical deductions, and ATLAS decisions.

The earlier implementation was deliberately removed. The virtual environment, source paper, and Git history remain. The replacement includes capability discovery, CPU/CUDA correctness fixtures, measurement infrastructure, and native partitioned execution. Kmeans provides persistent point residency, cluster reduction/merge, and memberships. HotSpot provides evolving temperature buffers, current-state migration, and halo exchange. Independent references validate both workloads, while pure Tier 1 policies drive persistent feedback execution. Python runs reproducible split sweeps and held-out comparisons; Nsight checks actual kernel overlap. Tier 2 offline policies, synthetic diagnostics, and read-only sample replay are implemented; see the [Phase 7 review](docs/phase7_review.md) and [frequency contract](docs/frequency_spec.md). See the [Phase 6 review](docs/phase6_review.md), [workload contract](docs/hotspot_spec.md), and [run instructions](README.md).

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
- **Frequency selection:** implemented offline in Phase 7 as `paper_additive`,
  using the printed loss/update equations and documented initialization/tie
  choices; `discounted_loss_v1` is a separately defined extension. Pure transitions
  are in `frequency.py`; `frequency_scenarios.py` declares synthetic assumptions;
  `frequency_trace.py` captures and joins read-only NVML samples;
  `frequency_experiment.py` archives and replays decisions. No actuator is present.
- **Telemetry and energy:** timestamped observations with physical domain, metric semantics, units, and quality information.
- **Actuation:** Phase 8 implements transaction ownership, durable requests, restoration/failure handling, and generic sweep/Tier 2-only orchestration. The default NVML adapter blocks without authoritative prior-policy capture. Its hardware gate remains blocked on WSL; see [Phase 8 review](docs/phase8_review.md). Real CUDA diagnostic workloads characterize loaded telemetry without changing clocks.
- **Runtime:** Phase 9 implements the pure coordinator in `integration.py`, independent sampling/native feedback in `integration_runner.py`, analytical scenarios in `integration_scenarios.py`, and retained hardware validation/replay in `phase9.py`. Real execution applies allocations and shadows frequency decisions. Live joint mode is explicitly gated.
- **Analysis:** baselines, uncertainty, adaptation cost, and plots regenerated from retained records.

Planned files and their staged creation are listed in the implementation plan. Do not create placeholders that imply an unfinished module is operational.

## Capability-dependent execution

**Hardware execution** uses real computation and available sensors. Full GPU frequency replication additionally requires verified independent core/memory control. Without it, allocation-only hardware evaluation remains a useful but explicitly limited result.

**Trace replay** inspects controller decisions on recorded observations. It cannot measure the counterfactual energy of unexecuted actions.

**Simulation** tests mathematical scenarios and failure cases with disclosed assumptions. It is not historical-hardware calibration unless separately supported by calibration data and validation.

Phase 1 verified Ubuntu 24.04 on WSL 2, an i5-12450H, an RTX 4050 Laptop GPU, and working CPU/CUDA compilation and execution. Phase 2 confirmed NVML power/counter reads but detected inconsistent energy estimates; read availability does not establish suitability. CPU powercap and governor interfaces are not exposed in this environment. No frequency or power range is assumed enforceable from a query alone. Phase 8 verifies live actuation.

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

Phase 5 separates pure allocation state transitions from native dispatch. One Python adapter consumes validated iteration records and returns the next legal CPU count through a bounded feedback protocol. The native process retains point residency and centroids throughout each trial. Decisions carry input identity, configuration hash, pre/post state, and reason; replay reconstructs every transition and checks the realized next partition. Timing records separate iteration work, feedback/export/IPC, nested controller calculation, and full loop elapsed time. See the [Phase 5 contract](docs/phase5_spec.md).

HotSpot now partitions contiguous rows with persistent previous/next buffers on both processors. The owner alone has authoritative evolving temperatures; boundary movement transfers current state, then refreshes the two halo rows. Immutable power can be cached. Verification snapshots are separate from simulation buffers, and timing loops defer full-field checks until the end. CPU/GPU compute follows a measured shared communication prefix. See the [HotSpot contract](docs/hotspot_spec.md) and [Phase 6 review](docs/phase6_review.md).

The next implementation is Phase 7's offline Tier 2 frequency policy, equations, and diagnostic replay. Live clock changes belong to Phase 8 after actuation/restoration qualification. Energy and device-event timings remain invalid until separately qualified; automatic clock policy is not a fixed-capacity guarantee.
