# Phase 7 review — offline Tier 2 frequency control

Date: 2026-10-31. **Phase 7 offline implementation and validation gate: passed.**
Live actuation remains Phase 8 work. GPU energy and device-event timing
qualification from Phase 2 remain blocked.

## Implemented behavior

`src/greengpu/frequency.py` implements the paper's asymmetric domain losses,
weighted pair loss, and literal additive update. Candidates are explicit sparse
pairs, targets use actual MHz spacing, all candidates start at zero and update
together, and exact ties follow the specified deterministic rule. Configuration
hashes isolate state; allocation/workload epochs retain full history. Invalid,
missing, stale, duplicate, or incompatible observations cannot update scores.
Single-level domains and empty candidate sets have explicit outcomes.

`discounted_loss_v1` is implemented as a separate extension with
`score_next=0.95*score-loss` by default. It is never substituted for the literal
paper rule. The complete policy, observation, and simulation contracts are in
[frequency_spec.md](frequency_spec.md).

Three CLI commands provide collection, replay, and figure export:
`frequency-check`, `frequency-replay`, and `frequency-plot`.
`--hardware-trace` adds read-only NVML discovery and capture. Controllers do not
import an actuator, request clocks, or dispatch GPU workloads.

## Retained validation

Authoritative archive: [2026-10-31-final](../results/phase7/2026-10-31-final/summary.json).
All evidence files are covered by a SHA-256 inventory; source hashes identify
the tested implementation independently of a future Git commit.

- **92 Python tests passed**, including 16 new Phase 7 tests. Checks cover the
  hand calculation F01, a separate Decimal loss/score oracle, 10,000 additive
  updates, beta variants, exact/near ties, state purity, sparse/nonuniform pairs,
  singleton domains, invalid/missing memory, stale identities/windows, phase
  history, feedback-model causality, trace joins, missing telemetry, archive
  corruption, and altered decisions even after recomputing their checksum.
- **23 deterministic scenarios, 4,144 synthetic observations**, retained with
  complete pre/post decision state. Both policies exercise stationary extremes,
  asymmetric utilization, phase changes, seeded noise, gaps, and declared
  closed-loop assumptions. Three beta runs and four candidate/domain edge cases
  complete the matrix.
- **Six real timestamp-paired idle observations replayed successfully.** Both
  utilizations were zero. There were no NVML capture errors and no clock-setting
  calls. This checks trace handling, not workload response or energy savings.
- **790 reported hardware pairs**, with an explicitly selected 15-pair subset
  used for offline recommendations. The final logical recommendation was
  `(210,405)` MHz. This was never requested from the driver. Recorded clocks were
  acquired separately and are not represented as acknowledgments of that pair.
- The archive regenerated identical decisions and summary. PNG and PDF
  [diagnostics](../results/phase7/figures/frequency-diagnostics.png) were exported
  separately; the PNG was visually inspected for legibility.

The [test log](../results/phase7/verification/unit-tests.txt),
[replay output](../results/phase7/verification/frequency-replay.txt), and
[audit](../results/phase7/verification/audit.json) retain verification evidence.
The audit found no source-hash mismatches. All five retained Phase 2–6 archives
also replayed to their original summaries.
No native workload sources changed, so this phase did not require rebuilding
CUDA binaries or rerunning native workload benchmarks.

The retained Phase 7 evidence and figures total approximately 8.4 MiB; the
largest individual evidence file is under 1 MiB. No large binary dataset was
added, and earlier datasets and virtual environments were preserved.

## Findings and limits

F01 produces core loss `0.03`, memory loss `0.392`, combined loss `0.2834`, and
first additive increment `0.77328`. After 10,000 identical updates, the relevant
score is `7732.8` within floating-point tolerance; clipping at one would violate
the rule.

With equal initialization, `W_i(t)=t-(1-beta)*sum(loss_i)`. The beta values
`0.01`, `0.2`, and `0.99` gave identical selections in the retained diagnostic
and in the varied independent-oracle test. Beta therefore does not implement
forgetting. This exact-arithmetic deduction does not guarantee identical
floating-point behavior for arbitrarily close ties or arbitrarily long runs.

In the retained high-to-low phase change, the literal policy kept the highest
synthetic pair through the end of the run. The discounted extension responded
to recent observations and ended at `(900,3000)` in that scenario. The comparison
illustrates different history rules; it establishes no hardware advantage.

Real trace eligibility uses exact matching NVML buffer timestamps as an explicit
offline co-sampling assumption. The APIs do not expose the integration start/end
bounds used here, so physical window alignment and live control cadence remain
unqualified. Ordinary timestamp-free utilization rates are retained but rejected
for replay. Memory busy time is not allocated-memory usage or achieved bandwidth.
Unsupported/missing memory produces no update rather than a fabricated zero.

The hardware trace is idle and uncontrolled. Reported clock combinations do not
establish permission to set them, simultaneous realized settings, or recovery
behavior. Phase 2–6 traces lack paired memory utilization and are not silently
reused as eligible Tier 2 traces. Synthetic runtime/power functions are declared
assumptions, with dimensionless outputs kept separate from hardware records.

## Next boundary

Phase 8 can investigate clock-setting capability, verification, restoration,
and frequency-only workload experiments. Phase 7 has not implemented these
actions or qualified energy effects. Full energy-control replication remains
unavailable until the corresponding measurement and actuation gates pass.
