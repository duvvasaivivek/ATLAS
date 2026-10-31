# Phase 8 review — actuation infrastructure and hardware qualification

Date: 2026-10-31. **Software infrastructure implemented and tested; hardware
exit gate blocked.** The real GPU diagnostic gate passed. No GPU or CPU settings
were changed, and no physical frequency-control or energy-saving claim is made.

## Hardware finding

The RTX 4050 Laptop GPU under this WSL driver reports 790 candidate clock pairs.
The bounded selection algorithm retained five pairs for prospective tests.
However, the application-clock, default application-clock, and related permission
queries returned `Not Supported`. The installed NVML interface exposes locked-clock
setters and resets but no getters for the prior locked ranges or unlocked mode.

Observed frequencies cannot reconstruct the prior control policy. Therefore the
transaction gate stopped before any setter. This establishes a **restoration
precondition failure**, not proof that every setting API would reject a request.
Setter permission and independent core/memory actuation remain untested.

The default adapter deliberately stays blocked without authoritative policy
readback. It supports a platform readback-provider interface, but no provider is
available for this configuration, and the CLI does not accept an assumed baseline.
Resetting to defaults was not substituted for restoring the unknown prior policy.

CPU governors and package energy remain unexposed through the checked Linux
interfaces. No whole-system meter is configured. A power-cap change is not used
as a replacement for independent frequency control.

## Implemented components

- `actuation.py`: device-specific process lock, durable request/recovery journal,
  explicit feasible pairs, guarded NVML setters, partial-failure cleanup,
  restoration readback, and normal/exception/SIGTERM handling.
- `frequency_hardware.py`: fresh loaded-setting verification, restoration
  preflight, bounded static-sweep orchestration, repeated counterbalanced
  measurements, and a bounded Tier 2-only harness using the Phase 7 policies.
- `native/frequency/main.cu`: persistent compute and streaming diagnostics with
  independent masked-integer CPU references, full output checking, and a bounded
  host protocol. `frequency_probe.py` handles native execution and loaded sampling.
- `phase8.py`: capability-gated hardware collection, raw evidence, source/binary
  identity, strict archive replay, and explicit blocked research outcomes.
  Public commands are `actuation-check` and `actuation-replay`.

The generic actuation and control paths are validated with failure-injection
devices. They are **not hardware-validated live controllers**. Details, assumptions,
and platform requirements are in [phase8_spec.md](phase8_spec.md).

## Validation evidence

The authoritative archive is
[2026-10-31-final-v2](../results/phase8/2026-10-31-final-v2/summary.json).

- **116 Python tests passed**, including 24 new Phase 8 tests. These cover
  concurrency exclusion, unknown-policy blocking, sparse candidates, durable
  journaling, partial setter failures, restoration errors, disk-write failures,
  KeyboardInterrupt, actual handled SIGTERM, stale/duplicate/missing telemetry,
  observed-clock mismatch, thermal/clock-event rejection, sweep ordering,
  frequency-only decisions, archive tampering, and invalid trial paths.
- **Seven native CTests passed.** The new diagnostic test covers both kernels,
  six size/mode combinations, 12 batches, independent Python checksums, partial
  CUDA blocks, and invalid-input rejection. Earlier workload tests also passed.
- **CUDA Compute Sanitizer reported zero memory errors** for both new kernels
  using partial-block inputs. These checks ran after the final collection ended.
- **Eight real native batches passed full correctness checks:** two calibration
  pilots and six measured trials, three per diagnostic mode. Measured repetition
  counts were frozen after calibration; mode order alternated by block.
- **47 matched loaded utilization samples** passed the diagnostic timestamp/age
  and one-second integration guard. All accepted samples were real observations;
  missing data remained unavailable.

Retained verification: [Python tests](../results/phase8/verification/unit-tests.txt),
[native tests](../results/phase8/verification/ctest.txt),
[compute memcheck](../results/phase8/verification/memcheck-compute.txt),
[stream memcheck](../results/phase8/verification/memcheck-stream.txt),
[replay](../results/phase8/verification/actuation-replay.txt), and
[source/binary audit](../results/phase8/verification/audit.json).

The final archive regenerated its summary exactly, source and binary hashes
matched, and the retained Phase 7 archive still replayed to its original summary.
All Phase 8 evidence, including development runs and logs, occupies less than
1 MiB; no generated binary dataset was added to the repository.

## What the real diagnostics show

The compute mode used 65,536 elements and 4,096 recurrence rounds; measured
batches used 4,741 launches. Its synchronized host compute spans were
2.141–2.216 seconds. Each measured trial had median GPU busy time 99% and median
memory busy time 0% among the guarded paired samples.

The streaming mode used 16,777,216 elements with two 64 MiB device arrays;
measured batches used 3,824 launches. Host compute spans were 2.872–2.950 seconds.
Each trial had median GPU busy time 99% and memory busy time 100%. These busy-time
metrics distinguish the observed diagnostics, but do not independently prove a
roofline bottleneck or measure achieved DRAM traffic.

Median paired-sample gaps were about 200 ms. Observed median core clocks varied
across workloads/trials; nonzero driver clock-event masks were common. The
highest recorded loaded temperature was 76 C. These are observations under the
unchanged automatic platform policy, not verified fixed settings. Their runtime
numbers must not be treated as a static frequency surface or evidence that a
controller improved performance.

Sensor integration boundaries and the wall-to-monotonic mapping remain
unqualified for live feedback. Raw power and cumulative-energy readings are
retained only as diagnostics: accepted energy and savings are null under the
earlier Phase 2 qualification failures. CUDA-event duration is not used.

## Retained development runs

`2026-10-31-pilot` exposed inadequate observation duration: its compute batch
lasted only about 0.349 seconds, too short for the one-second integration guard.
The final workload increased compute work per launch and used 64-repetition
pilots to reduce launch-overhead bias in sizing. This was calibration development,
not selection for a favorable performance result.

`2026-10-31-final` retained the first longer diagnostic matrix. A subsequent
review added SIGTERM handling to the generic transaction path. The finalized
source was tested and collected again in `2026-10-31-final-v2`, which is
authoritative. Earlier runs are preserved and are not pooled with the final trials.

## Exit gate and next step

The implementation provides tested transaction/control infrastructure and real
diagnostic evidence. Physical restoration, independent clock changes, a verified
static frequency sweep, frequency-only hardware experiments, and energy benefits
remain **blocked or unverified**. There were **zero clock-setting calls** in the
retained hardware runs, so physical restoration was neither necessary nor tested.

Enabling those experiments requires a platform with authoritative prior-policy
capture and supported controls, followed by the restoration preflight and loaded
verification. Merely switching operating systems or observing a reported clock
list would not establish those capabilities. Phase 9 must not label this setup
as a validated combined allocation-and-frequency controller.
