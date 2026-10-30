# Phase 0 review and Phase 1 handoff

Date: 2026-10-31

Specification version: 1.0

Scope: Complete the scientific specification and remove contradictions before hardware discovery.

## Deliverables

- [Paper mapping and hypotheses](paper_mapping.md): M01-M10 link the architecture, algorithms, experimental results, and limitations to paper sections/pages; H1-H5 state falsifiable initial hypotheses.
- [Controller specification](algorithm_spec.md): exact policy IDs, update rules, defaults, granularity, tie-breaking, invalid-input behavior, initialization, stopping, and history/reset handling.
- [Measurement specification](measurement_spec.md): correct-work comparisons, time/energy boundaries, units, domains, baselines, trial evidence, and statistical rules.
- [Reference fixtures](spec_fixtures.json): ten analytical allocation, frequency, and measurement examples for later implementation tests.
- [Proposed architecture](../ARCHITECTURE.md): revised to describe planned modules and capability-dependent execution without claiming they already work.
- [Updated implementation plan](../IMPLEMENTATION_PLAN.md): Phase 0 status and links to the specifications; later phase gates remain intact.

## Audit of the superseded architecture

The original `ARCHITECTURE.md` is available in Git revision `ce1e09a942dad5b9b9900b30ce873736d9c46d28`. The following claims no longer govern implementation:

1. **Core implementation phases marked done / workload phase active:** obsolete after the requested code removal. Replacement: Phase 0 specification complete; implementation phases await their own evidence.
2. **RTX 4050 and i5-12450H treated as verified local hardware:** the document alone does not establish current hardware. Replacement: discover and record device identity in Phase 1.
3. **A certified GPU power range of 5-55 W:** unsupported as a control contract. Replacement: query supported capabilities and verify actuation; power caps are a separate actuator study.
4. **Printed binary formulas described as retaining a bracket:** the text contains no persistent bracket. Replacement: distinct literal diagnostic and corrected bracketed policy IDs, with an explicit counterexample.
5. **Universal binary-search convergence within seven iterations:** conditional on valid stationary monotone bracketing and an unrounded interval; initial measurements also cost work. Replacement: scoped bracket-width bound and measured hardware outcomes.
6. **Proportional allocation always removes oscillation or converges within 4-5 iterations:** those are selected paper results, not a guarantee. Replacement: measured behavior under transfer costs, phase changes, and quantization.
7. **Beta controls forgetting in the printed update:** not supported by the additive equation with equal initialization. Replacement: literal cumulative rule; any forgetting extension has another identity.
8. **A 40:1 loop ratio or 120-second interval guarantees the optimal clock setting:** no such proof follows from the candidate count. Replacement: empirical settling and stability evaluation.
9. **Zero-overhead telemetry:** unjustified. Replacement: measure monitoring/control cost and include in relevant job energy/time.
10. **A high-fidelity calibrated historical-hardware simulator:** no calibration evidence was established. Replacement: disclosed synthetic fixtures first; calibration is a separate evidence requirement.
11. **Expected whole-system energy saving of approximately 21.04%:** incorrectly turns one historical result into a project target. Replacement: preserve its two-workload scope and actual baseline; accept valid negative results.
12. **29.2% described as an upper limit:** the paper reports it as average dynamic GPU energy saving. Replacement: preserve mean-versus-maximum and dynamic-versus-total distinctions.
13. **Runtime below 3% of an optimal fastest configuration:** not the paper's stated comparison. Replacement: distinguish 2.95% against best-performance for the frequency study and 1.7% against allocation-only for the complete framework.
14. **Figures required to prove Improved Binary superiority:** prejudges the experiment. Replacement: regenerate figures from retained evidence, including inconclusive or unfavorable outcomes.
15. **An assumed implementation backend and sensor interface:** replacement architecture chooses a native CPU/CUDA workload path with Python orchestration, conditional on Phase 1 toolchain validation; no particular framework or handwritten binding is installed by Phase 0.

## Decisions fixed for implementation

- One CPU host, one GPU; kmeans and HotSpot are the primary substantive workloads.
- Literal, corrected, and extended policies keep distinct IDs and recorded configuration hashes.
- Continuous ratio arithmetic is separated from legal integer workload allocation; realized splits drive timing-based updates.
- `paper_adaptive` uses observed sign reversals and continuous halving, an explicitly recorded interpretation of underspecified prose.
- Published performance results are contextual evidence, not required outputs of a simulator or acceptance targets.
- Hardware, replay, and simulation have distinct evidence labels.
- Whole-system energy claims require an appropriate system measurement domain; component readings remain component readings.
- No specification default replaces capability discovery or actual measurement.

## Phase 0 validation evidence

Validation is limited to source traceability, document consistency, and independent analytical calculations. It does not test a rebuilt controller or run CPU/GPU benchmarks.

- Verified the source PDF against its recorded SHA-256.
- Parsed all ten analytical fixture cases and independently recomputed their expected values using rational arithmetic where applicable; all passed.
- Checked the binary recurrence counterexample, seven-update continuous bisection bracket, beta-ranking invariance under equal initialization, and weights exceeding 1; all matched the documented derivations.
- Checked all six active Markdown documents for local link targets and paired code fences; fixture IDs are unique and all Phase 0 deliverables are linked.
- Confirmed the implementation files remain removed and the retained virtual environment and paper remain present. No Phase 0 command modified the virtual environment.

These checks ran successfully on 2026-10-31. They were one-time specification checks, not tests of production controller code or hardware. Implementation tests remain required in their assigned phases.

## Exit gate

- [x] Each controller and historical quantitative result has a paper reference or is labeled an ATLAS decision/derivation.
- [x] Missing semantics needed for implementation have explicit rules or documented later-phase owners.
- [x] Defaults and reporting thresholds are distinguished from scientific guarantees.
- [x] Corrected algorithms cannot be confused with literal published rules.
- [x] Initial hypotheses allow negative and inconclusive outcomes.
- [x] Stale architecture claims have been replaced in the active architecture.
- [x] Analytical and documentation validation completed.

## Phase 1 handoff

**Next action:** Read-only discovery of CPU/GPU identity, OS/runtime/driver/compiler versions, retained environment compatibility, available sensors, and management interfaces. Then select a compatible execution environment and establish CPU/GPU smoke-computation correctness according to the implementation plan.

**Open questions owned by Phase 1:** actual hardware; Windows/WSL/native-Linux route; CUDA/compiler compatibility; usable interpreter; available energy domains; sensor/query support; and privilege requirements. Clock-setting support remains provisional until verified in Phase 8.

**Later owned questions:** sensor cadence/noise in Phase 2; numerical workload tolerances and data provenance in workload phases; actual clock effects and restoration in Phase 8; confirmatory trial count after pilots in Phase 10.

There is no unresolved Phase 0 approval question. Do not report unsupported hardware features as available, install speculative dependencies, recreate the virtual environment, or restore the deleted implementation merely to begin Phase 1.

**Status:** Phase 0 complete. This document records the specification gate before implementation. Phase 1 was subsequently completed; see the [Phase 1 review](phase1_review.md) for the hardware evidence and current handoff.
