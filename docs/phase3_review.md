# Phase 3: static CPU/GPU execution and concurrency evidence

Date: 2026-10-31. Status: **complete for the Phase 3 correctness/concurrency gate**.
Phase 3 implements a reusable native execution harness for fixed
CPU/GPU partitions. It establishes correctness and profiler-observed concurrency;
it does not implement kmeans, adaptive allocation, clock control, or energy savings.

## Execution and partition contract

The Python [adapter](../src/greengpu/static_execution.py) converts a requested CPU
share to an exact rational and assigns `floor(N * share)` items to the CPU. The GPU
receives the remainder. Both ranges are contiguous, disjoint, and exhaustive:
CPU `[0, cpu_items)`, GPU `[cpu_items, N)`. Rounding is explicit through requested
and realized shares. The minimum legal work unit is one element. Empty branches
have null timings and no work; whole-input sizes below one are rejected.

The [native implementation](../native/static/main.cu) owns CUDA submission on its
main thread and a persistent CPU worker thread that invokes the
[OpenMP backend](../native/static/cpu.cpp). Python starts the native process and
waits for its records; its interpreter cannot serialize these native branches.
Four CPU workers are requested by default. The Python harness caps the request at
eight and reserves two visible logical CPUs when possible. Actual OpenMP team size
is recorded; there is no CPU affinity or OS scheduling-isolation claim.

The fixture transforms deterministic uint32 inputs with xorshift operations and
computes a uint64 sum. CPU workers reduce their assigned ranges. Each CUDA block
performs a shared-memory reduction; the host combines its partial sums with the
CPU result at the merge barrier. The maximum supported input (1,048,579 elements)
keeps the mathematical sum within uint64 range. All outputs are checked against
an independently written scalar reference using masked uint64 arithmetic.
Small cases additionally check the reference sum with independent Python code.

Every input has a visit counter, cleared before each iteration and checked for
exactly one visit afterward. Output buffers are also reset to a sentinel, so a
stale result from a previous iteration cannot satisfy the check. Integer equality
is exact; there is no floating-point tolerance to hide assignment or reduction
errors. Input SHA-256 is over little-endian uint32 values `index xor seed`.

## Allocation, transfers, and timing

Pinned host input/GPU-result buffers, device input/output/visit/reduction buffers,
CPU work buffers, and a nonblocking CUDA stream are allocated once per native
invocation and reused across one warm-up plus the measured iterations. Small
tracing/result metadata can still allocate; this is not an allocation-free runtime.
No workload-buffer allocation or implicit Python array conversion occurs inside
the measured iteration. CPU-only endpoints still initialize the common CUDA-capable
harness during preparation; they are not a CPU-only installation fallback.

The primary host interval begins at the common release timestamp and ends after
both branches finish and the result/reduction merge completes. It includes CPU
dispatch/team overhead, GPU H2D, kernel submission/execution/synchronization, D2H
of outputs/visit counters/block sums, the final host wait, and merge work. CPU/GPU
completion times are measured from that same release, not from unrelated local
starts. They must not be added together to obtain wall time.

GPU H2D, compute completion, and D2H completion are recorded after explicit stream
synchronization using `CLOCK_MONOTONIC`. They are host observations, not accepted
CUDA event durations. The final host wait is the main thread's wait after GPU
completion, including CPU-result handoff; it is not a measurement of every CPU
worker's idle time. Shared merge and validation durations are separate.

Allocation/reference generation, diagnostic sentinel/visit resets, and warm-up are
outside the primary interval and retained in raw records. The per-element and
reduction validation pass follows the measured merge for every split. These are
instrumented correctness fixtures, not tuned benchmark implementations. Tiny jobs
may finish before the other branch starts; speedup or overlap at every size is not
an acceptance requirement.

## Validation matrix and gates

The final matrix contains **40 distinct partitions** over sizes
`1, 2, 255, 256, 257, 4099, 65539`. For each size it covers CPU-only, GPU-only,
quarter/half/three-quarter shares after rounding, and one-item CPU/GPU partitions,
deduplicating shares that round to the same count. Each configuration executes one
warm-up and three measured iterations: **120 measured iterations and 40 warm-ups**.

All splits at a given size use identical inputs, seed 1729, and 128 rounds. The
archive-level gate requires identical input/work identity and equal validated
results across splits. Individual cases must also pass every output, reduction,
visit-count, configuration, iteration-sequence, active-worker, and timing-order
check. A failed command, missing record, wrong result, or malformed timing makes
the case fail rather than becoming a zero-time success.

Validation includes **42 passing Python tests**, two passing native CTests, CUDA
Compute Sanitizer memcheck with **zero errors**, and racecheck with **zero hazards,
errors, or warnings** on the awkward-size mixed reduction. The sanitizers cover
the tested CUDA execution; they are not an exhaustive proof of all host scheduling
behaviors. The pure profiler tests explicitly reject a queued launch with no kernel
activity and a real kernel interval that does not overlap CPU compute regions.

## Concurrency evidence

The final runner separately profiles three measured iterations at N=1,048,579,
CPU share 25%, 1,024 rounds, and four requested CPU workers using Nsight Systems
2025.5.2 with CUDA and NVTX tracing. Warm-up is excluded from the overlap gate.

The [profile analyzer](../src/greengpu/concurrency.py) selects measured NVTX
iteration ranges, correlates CUDA runtime launches to **CUPTI kernel activity**
using process/correlation IDs, and intersects that activity with CPU worker compute
regions from the same process and iteration. A launch interval alone cannot pass.
Overlapping CPU worker intervals are unioned to avoid quadruple-counting the same
GPU time. Every measured profile iteration must contain its expected kernel and
positive observed overlap with CPU computation.

Nsight uses its own correlated timeline. Its timestamps are never subtracted from
the application's `CLOCK_MONOTONIC` timestamps. This is qualitative evidence of
kernel activity overlapping annotated CPU compute regions, not an independent
calibration of absolute duration or instruction-by-instruction simultaneous
execution. OS scheduling/IP sampling is disabled; CPU regions can include scheduler
preemption. Phase 2's device-event timing and energy qualifications remain blocked.

The official [Nsight analysis guide](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html)
documents kernel/runtime correlation and NVTX data in the exported database.
[NVTX](https://nvidia.github.io/NVTX/) headers come from the installed CUDA toolkit;
no additional Python dependency was introduced.

## Retained evidence and replay

The authoritative run is [2026-10-31-final](../results/phase3/2026-10-31-final/manifest.json).
Its [summary](../results/phase3/2026-10-31-final/summary.json) retains every split,
every measured iteration, the same-input equivalence checks, and profiler evidence.
The native stdout/stderr and command outcomes are retained in `case-NNN.json` and
`profile-command.json`. Environment, dependency versions, Git state, source/header
hashes, executable hash, and CMake configuration accompany the run.

The [Nsight capture](../results/phase3/2026-10-31-final/profile.nsys-rep),
[SQLite export](../results/phase3/2026-10-31-final/profile.sqlite), and
[Chrome trace JSON](../results/phase3/2026-10-31-final/timeline.json) retain the
timeline. Open the capture in Nsight Systems, or import the trace JSON into a
compatible trace viewer. Trace durations retain the qualification above.

The [final integrity audit](../results/phase3/verification/final-audit.json) verifies
the final replay, source/header and binary hashes, same-input equivalence, and the
unchanged Phase 2 replay. [Memcheck](../results/phase3/verification/memcheck.json),
[racecheck](../results/phase3/verification/racecheck.json), and
[CTest output](../results/phase3/verification/ctest.txt) retain the native validation
evidence. The native binary did not change between sanitizer checks and the final
same-input matrix; their binary identity is verified in the audit.

New run directories are created exclusively. Checksum verification rejects changed,
missing, or additional archive files before regenerating the summary. Results are
stored without Git line-ending conversion. Failure records and partial output are
preserved. Profiling failure or absent overlap produces an incomplete phase gate
and a nonzero `static-check` exit status; passing correctness alone is insufficient.

The earlier `profiler-probe` and `2026-10-31-validation` directories remain as
development evidence. The first validation matrix varied seeds across split cases;
the final matrix strengthens the comparison by fixing the input for each size.
Earlier summaries correspond to the source versions recorded in their manifests.
The current authoritative replay target is the final archive.

## Handoff

The final archive reports `passed`: equal correct work across all splits,
exactly-once coverage, safe endpoints, and profiler-observed overlap.
The next implementation is Phase 4's kmeans workload: shared centroids, partitioned
point assignment, local cluster sums/counts, and one validated merge per iteration.

Do not carry xorshift's integer tolerance into kmeans floating-point reductions.
Do not interpret these fixtures as a static-optimal split or a speedup claim.
Energy measurements and device-event duration claims remain blocked by the Phase 2
findings until separately qualified. The original paper, virtual environment,
Phase 1/2 work, and retained evidence are preserved.
