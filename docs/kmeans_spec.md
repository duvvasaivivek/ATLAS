# ATLAS kmeans variant and Phase 4 evidence contract

## Native schema 4.1 (2026-11-02)

Optional arguments after `fixed|converge`: `feedback`, `verify|timing`,
`batch=N`, `phase=ITER:WORKERS`, `reduced`, and `streamed`. Defaults retain
per-iteration verification, atomic reduction and diagnostic synchronization.
The maximum iteration budget is now 100,000, independently bounded by the host
watchdog. Phase changes may reduce the initial worker count at a fixed iteration.

Timing mode defers the independent CPU Lloyd trajectory until after the
measured loop and checks final memberships, counts, sums and centers against it.
Finite-value, membership-range and objective checks still run during execution;
their time is explicit. Full command duration includes independent validation.
Verification mode still checks every iteration against the independent reference.
For deferred final validation of at least 1,024 points, independent assignments
use the requested CPU worker count; cluster sums and objective still accumulate
in original point order with long double. The scalar per-iteration oracle is
unchanged. `final_reference_workers_requested` records this choice. Parallel
assignment changes neither tie handling nor the reduction order, and is tested
against the scalar and Python oracles beyond the parallelization threshold.
`validation_ns`, `final_validation_ns` and `independent_validation` distinguish
these costs and outcomes. The replay reader accepts schemas 4.0 and 4.1.

`reduced` uses block-local cluster accumulation, block objective reduction, and
a separate ordered cross-block reduction. `streamed` uses one final stream
synchronization and reusable pinned staging buffers. Intermediate host stage
timestamps are null in this mode; no unmeasured stage or CUDA-event time is
fabricated. Complete GPU branch readiness still includes transfers and staging.
Both paths use FP64 and the existing tolerances and are evaluated separately.

Specification version 4.0, 2026-10-31. These choices define the implemented
workload; they do not claim to reconstruct unspecified settings or datasets from
the historical GreenGPU experiment.

## Algorithm

The workload is Lloyd kmeans with squared Euclidean distance. Points, centroids,
distance arithmetic, and backend sums use float64. The CPU assignment/reference
translation unit disables floating-point contraction; CUDA compiles with
`--fmad=false`. Exact distance ties choose the
lowest centroid index. A cluster receiving zero points retains its old centroid.
No random reinitialization or hidden reassignment occurs for empty clusters.

Generated inputs use seeded Forgy initialization: select K distinct row indices
from a deterministic generated/shuffled point set. Coordinates may repeat. The
initial indices, seed, generator version, actual points/centroids, and file SHA-256
are retained. Handwritten fixtures specify their centroids directly. Every split
for a given dataset uses identical input bytes and centroid identities, so cluster
labels are compared directly without permutation matching.

`fixed` mode always executes the declared iteration count, including after apparent
convergence. `converge` mode stops when labels are unchanged **or** the maximum
per-centroid Euclidean displacement is at most 1e-6, subject to the maximum-iteration
cap. The first iteration compares labels against an unassigned sentinel. The
termination reason, iteration count, label changes, displacement, and achieved
objective are recorded. Convergence-mode trials are never included in fixed-work
timing curves.

Returned labels are the last assignment, and returned centroids are their updated
means (or retained empty centroids). Fixed-budget output does not imply that another
assignment against those final means would leave all labels unchanged. The reported
final objective is squared error for the returned labels and centroids. There is
no extra uncounted final assignment.

## Independent correctness checks

An independent scalar native reference advances its **own centroid trajectory**
from the common initialization. It uses ordered FP64 distances and long-double
accumulation for sums/objectives. Its state never overwrites backend state. Every
iteration, including warm-up, checks:

- Every label exactly, and each cluster count exactly.
- All coordinate sums and updated centroids within `abs(error) <= 1e-8 + 1e-10 * abs(reference)`.
- Assignment objective against the reference with the same tolerance.
- The centroid update must not increase squared error beyond that tolerance.

An independent Python oracle uses scalar distances and `math.fsum` aggregates.
Small fixtures additionally compare every iteration's labels, sums, counts,
centroids, objective, empty clusters, and stopping observations against that oracle.
No near-tie label mismatch is excused by a numeric tolerance: unstable assignments
fail validation. Backend sums may differ in their last bits because CPU grouping
and GPU atomic-add order differ. Bitwise equality of floating-point aggregates is
not required.

Each point has one assignment owner. CPU/GPU intervals are disjoint and exhaustive;
cluster counts must sum to N, all labels must be valid, and the full label vector
must match the independent reference. Output labels are reset to a sentinel before
each iteration to expose missed assignments. Aggregate buffers are reset every
iteration. The FNV-1a label hash is a compact comparison aid, not the sole correctness
check. Full final labels and centroids are exported in the native summary.

Fixtures cover a single point, K=N, K=1, signed fractional values, exact ties,
duplicate centroids/empty clusters, N=257 partial CUDA blocks, and N=513 with seven
dimensions. The binary CTest fixture is documented in [tests/data](../tests/data/README.md).

## Concurrent runtime and data ownership

Python coordinates experiments through a native subprocess. A persistent native
CPU dispatcher runs bounded OpenMP workers, while the native main thread submits
CUDA operations. Both branches receive the same old centroid array. Each produces
local sums, counts, and assignment objective. After both complete, the host merges
once, updates centroids once, computes stopping observations, and retains the full
membership vector. The next GPU iteration receives the resulting centroids.

All point data is immutable and available in a pinned host master copy. The device
has capacity for the full dataset but starts with no **logically valid** point
residency. For a CPU prefix `[0, cut)` and GPU suffix `[cut, N)`, `resident_begin`
tracks the lowest point index already uploaded. Expanding GPU ownership copies
only `[new_cut, resident_begin)` when needed. Shrinking GPU ownership does not evict
the cache. Regrowing into cached points therefore requires no new point transfer.

GPU-to-CPU ownership changes need no point D2H because the immutable host copy is
already current. GPU labels are returned after **every** iteration, so host labels
are current before a boundary change. The algorithm recomputes assignment each
iteration; it never relies on stale device memberships. Logical ownership changes
and actual physical transfer bytes are reported separately.

Per active GPU iteration, communication includes:

- Newly required points: `newly_resident_points * D * 8` H2D bytes.
- Current centroids: `K * D * 8` H2D bytes.
- Current GPU memberships: `gpu_points * 4` D2H bytes.
- GPU sums, counts, and objective: `K * D * 8 + K * 4 + 8` D2H bytes.

The CPU can access the host master without a copy. Label/sum/count outputs on the
host are pageable; CUDA may stage or block those transfers. Their full observed
cost is included. All device allocations, CPU reduction scratch, point storage,
and the CUDA stream are reused. No controller or clock setter is present. CPU worker
count defaults to four and is bounded by the adapter's available-CPU budget.

## Timing boundaries

Each iteration has one `CLOCK_MONOTONIC` release before branch dispatch and one
end after both branches, all required transfers, reduction merge, centroid update,
label-change counting, and state handoff. GPU reset/memset, point migration,
centroid communication, CPU team overhead, and synchronization are inside this
interval. Active branch completion is measured relative to the common release.
Inactive branch timestamps are null, never fake throughput samples.

Context creation, input/reference setup, allocations, and one full warm-up precede
measurement. Warm-up state is discarded: centroids and previous labels return to
their initial values, and logical GPU residency is reset. Thus the first measured
iteration is charged for its initial GPU point upload even though warm-up touched
the same memory. Subsequent iterations exploit residency as declared.

Independent validation and JSON logging occur between measured iterations. The
primary timing curve is therefore explicitly the **sum of complete iteration
intervals**, not uninterrupted job wall time. The separate `loop_elapsed_ns`
includes intervening validation/logging gaps from first release through final
iteration completion. Command elapsed time additionally includes startup, final
validation/export, and cleanup. These quantities are never substituted for one
another or used to invent an energy interval.

CUDA event durations are not used. Nsight CUDA/NVTX activity supplies a separate
qualitative concurrency check, using its own correlated timeline. It does not
resolve Phase 2's clock inconsistencies or externally calibrate host duration.
The profile uses N=65,539, D=32, K=32 and a 75% CPU prefix to keep CPU computation
active across the initial GPU upload. Smaller or imbalanced workloads can finish
one branch before the other starts computing; overlap is not guaranteed for every
input or partition. The profile is excluded from timing curves.

## Static experiment protocol

Synthetic benchmark inputs have N=32,771, D=8, K=8. Tuning uses generator seed 2027;
held-out evaluation uses seed 2402 and its own recorded initialization. Each trial
executes six logical iterations. The default is three independent trials per share,
with seeded randomized order within blocks.

1. Coarse tuning evaluates CPU shares near 0%, 25%, 50%, 75%, and 100% after integer rounding.
2. New tuning trials refine around the lowest observed mean, approximately 12.5 percentage points either side, clipped to endpoints.
3. The selected share is frozen **before** evaluation on the held-out input.
4. Fresh held-out trials evaluate the union of coarse and refined candidates.

Plans, trial order, selection rule, and selected count are saved before their
dependent evaluation. Reports distinguish the preselected share from the minimum
observed on the held-out trials. A sampled minimum is not a global optimum or a
statistically resolved winner. Error bars are sample standard deviations across
trials, not confidence intervals or variation across correlated iteration samples.
Valid regressions and slow trials are retained.

The GPU policy remains the platform's automatic default, and CPU policy is
unchanged. Frequencies are not locked or verified constant. Thermal equilibrium,
exclusive system use, and CPU affinity are not claimed. These are exploratory
instrumented measurements, not the paper's historical benchmark numbers.

## Input and evidence formats

`ATLSKM01` files contain eight magic bytes, little-endian uint32 N/D/K, then
row-major little-endian float64 points and initial centroids. Bounds are
`1 <= N <= 262147`, `1 <= D <= 32`, `1 <= K <= min(N,32)`; coordinates must be finite
and within ±1e6. Extra/truncated bytes are rejected. Generated data is original
project test data; no external dataset is downloaded or represented as paper data.

Native invocation:

```text
atlas_kmeans INPUT.kmdat CPU_COUNTS_COMMA MAX_ITERS WORKERS fixed|converge
```

One CPU count is held fixed; a list must contain exactly MAX_ITERS counts. Counts
must lie in `[0,N]`. This scripted schedule tests migration; it is not an adaptive
allocation policy. The summary exports all final labels and flattened row-major
centroids. Intermediate records include full labels for N≤513 and hashes for
larger inputs, plus all cluster aggregates and centroid states.

The Python runner creates new archives exclusively and retains every command's
stdout/stderr, input files/hashes, provenance, golden trajectories, source/binary
identity, build/environment snapshots, plans, per-trial records, and summary.
Completed archives carry byte-level checksums. Replay checks inventory and hashes
before regenerating the summary. Plot artifacts are generated outside the immutable
archive, with their own provenance.

## Qualification boundary

Runtime curves and clustering correctness can be evaluated now. GPU energy,
CPU-package energy, whole-system energy, and an energy-minimizing split are not
qualified on this setup. Every energy curve value remains null; no power model or
zero replaces missing energy. Consequently Phase 4's implementation/correctness
gate can pass while its full research exit gate remains blocked by Phase 2's
measurement findings. No GreenGPU saving percentage is claimed.
