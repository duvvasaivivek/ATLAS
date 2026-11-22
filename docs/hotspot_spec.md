# HotSpot model, ownership, and measurement contract

## Native schema 6.2 (2026-11-02)

Optional feedback arguments include `batch=N`, `phase=ITER:WORKERS`, and
`overlap`. The step limit is 100,000, with an independent host watchdog.
The serial communication path remains the default. In overlap mode, migration
finishes before CPU interior rows are released; those rows can execute during
halo exchange. The final CPU boundary row executes only after its halo arrives.
`compute_release_ns` identifies the serial prefix; `communication_end_ns` can
overlap compute and must not be added again to total time. State ownership,
double buffering and reference tolerances are unchanged. Readers retain support
for schema 6.0 and 6.1 evidence.

CPU profiler ranges now cover nonempty row computations only. OpenMP barrier
waiting has a separate `hotspot_cpu_wait` annotation and cannot establish
CPU-compute/GPU-kernel overlap. Thread preemption can still occur inside a range.

Deferred final validation of at least 1,024 cells can parallelize independent
reference rows using the initial requested worker count. Per-cell scalar flux
order, long-double state, and the join before each state swap remain unchanged.
Verification mode uses the scalar reference. `final_reference_workers_requested`
records the choice; tests compare both with the independent Python field.

Specification 6.0, 2026-10-31. HotSpot is a 2D explicit thermal stencil with a
fixed power input and evolving temperature field. CPU/GPU work units are complete
rows; every cell advances exactly one timestep per iteration.

## Reference model and declared differences

The pinned [Rodinia CUDA source](../references/rodinia/cuda-hotspot.cu) supplies the
thermal model. Its [OpenMP counterpart](../references/rodinia/openmp-hotspot.cpp)
uses a nominal timestep smaller by 1000, so ATLAS deliberately selects the CUDA
expression. Source revision, original license, and byte hashes are retained in
[reference provenance](../references/rodinia/provenance.json).

For H rows and W columns, chip height and width are both 0.016 m, thickness is
0.0005 m, thermal conductivity is 100 W/(m K), volumetric specific heat is
1.75e6 J/(m³ K), capacitance fitting factor is 0.5, and ambient temperature is 80 °C.
Let `dy=0.016/H`, `dx=0.016/W`:

```text
Cap = 0.5 * 1.75e6 * 0.0005 * dx * dy
Rx  = dx / (2 * 100 * 0.0005 * dy)
Ry  = dy / (2 * 100 * 0.0005 * dx)
Rz  = 0.0005 / (100 * dx * dy)
nominal_dt = 0.001 / (3e6 / (0.5 * 0.0005 * 1.75e6))
dt = min(nominal_dt, 0.9 * Cap / (2/Rx + 2/Ry + 1/Rz))

T_next = T + (dt/Cap) * (
    power + (south + north - 2*T)/Ry
          + (east  + west  - 2*T)/Rx
          + (80 - T)/Rz)
```

The stability limiter is an **ATLAS addition**. It bounds the sum of explicit
neighbor/environment coefficients by 0.9, keeping the central coefficient
nonnegative. It activates for the 2048×2048 test; the nominal CUDA expression alone
does not meet this bound there. Coefficients, actual dt, and `steps*dt` simulated
duration are recorded. Equal step counts on different grid shapes need not cover
the same physical time. Comparisons use the same shape, coefficients, and step
count within each experiment.

Exterior lateral boundaries are insulated: a missing neighbor contributes zero
flux, implemented as a clamped self-neighbor. Cooling to the ambient reservoir
through Rz remains active at every cell. The same rule covers 1×1, 1×W, and H×1
without out-of-bounds neighbors. There are no periodic boundaries or in-place
temperature updates.

ATLAS uses FP64 temperatures, power, and coefficients rather than the reference's
FP32. CPU assignment arithmetic disables contraction; CUDA uses `--fmad=false`.
The CPU/OpenMP and CUDA kernels perform one timestep at a time rather than the
reference CUDA temporal pyramid. These choices favor a directly verifiable
migration baseline and are not a claim of equivalent performance to Rodinia.

The scalar native oracle evolves its own long-double field and accumulates flux
from individual existing neighbors. It never overwrites backend state. The Python
oracle independently sums neighbor fluxes with `math.fsum`. The frozen comparison
rule is `abs(actual-reference) <= 1e-8 + 1e-10*abs(reference)` °C. Analytical tests
cover a single-cell update, equilibrium, symmetry, lateral-flux cancellation, and
the stable diffusion maximum principle; real trajectories report maximum error.

## Input and output

`ATLSHS01` input files contain eight magic bytes, little-endian uint32 H/W, then
H*W row-major little-endian FP64 temperatures followed by H*W per-cell powers.
Dimensions lie in `[1,2048]`; temperatures must be finite in `[0,500]` °C and power
finite in `[0,3e6*0.016*0.016/(H*W)]` watts per cell. Truncation, trailing bytes, and
invalid values are rejected. `generated` and `write_input` in
[hotspot_reference.py](../src/greengpu/hotspot_reference.py) create local inputs
and record seeds, patterns, shapes, coefficients, and hashes.

Outputs use `ATLSTP01`, H/W, and the full final row-major FP64 field. Output creation
is exclusive; existing files are not overwritten. Generated inputs are original
project fixtures, not the paper's temperature/power datasets.

## Double buffering and ownership

Each processor has persistent full-capacity previous/next temperature buffers.
Only its owned rows are authoritative. CPU owns `[0,cut)` and GPU owns `[cut,H)`.
Buffers swap only after both branches finish. Unowned storage may contain NaNs or
old temperatures; it is never treated as an up-to-date cache.

Before computation at timestep t:

1. If GPU ownership expands, copy current temperatures for `[new_cut,old_cut)`
   from host previous to device previous. If CPU ownership expands, copy
   `[old_cut,new_cut)` in the opposite direction. The first iteration separately
   uploads initial GPU-owned temperatures; it is not mislabeled as migration.
2. Upload any not-yet-resident immutable power rows. Power may remain cached even
   after ownership changes. Temperature must migrate again whenever ownership
   returns, because the state evolved while the other processor owned it.
3. For an interior split, copy host row `cut-1` to the device and device row `cut`
   to the host. These are the two current-epoch halos. Finish migration before
   copying a newly acquired host halo; finish halo exchange before either branch
   reads neighbors.
4. Release the persistent OpenMP worker and dispatch the CUDA suffix. Each writes
   only its next-buffer owned rows, using previous-buffer neighbors. Join both,
   swap buffers, and record input/output epochs t and t+1.

This baseline serializes migration and halo communication before concurrent
compute; it does not claim to overlap communication with computation. Migration
can duplicate a newly acquired boundary row in a subsequent halo copy. The
implementation intentionally records both actual transfers without deduplicating
their measured cost. CPU-only/GPU-only splits have zero halo traffic. One row per
active branch is legal; H=1 has no interior split for throughput policies.

Per-step records distinguish initial-temperature H2D, migration H2D/D2H, halo
H2D/D2H, immutable-power H2D, validation D2H, and final-result D2H. This separation
prevents a correct-looking field obtained through extra validation transfers from
being mistaken for an efficient halo protocol.

## Verification and timing modes

`verify` assembles the complete next field into a **separate snapshot** after every
timestep, copies the GPU suffix for verification, and advances the independent
oracle. Small inputs also export the full per-step field for Python comparison.
Snapshot downloads never refresh the persistent host simulation buffers.

`timing` makes no full-field validation downloads during measured timesteps.
Observations marked `deferred` report successful execution provisionally, allowing
the controller to choose the next split. After the loop, the final GPU suffix is
gathered and the independent native reference runs the same number of timesteps.
A trial is accepted only if this full final comparison passes. Small timing runs
also compare the final field with the Python oracle. Same-input final fields are
compared across policies, static shares, migration schedules, and validation modes.

Both modes use one discarded warm-up and restore the complete initial state before
measurement. Logical power residency resets too, so initial GPU temperature/power
uploads are charged again in the first measured timestep. Context setup, allocation,
warm-up, and restoring initial state are outside timestep timing.

The common release timestamp precedes all migration and halo exchange. Both branch
completion signals include the shared communication prefix. Record that prefix
separately, but do not add it again to total duration. This shared fixed cost can
weaken the proportional controller's linear-capacity assumption.

Feedback timing follows Phase 5: after previous-step validation and before record
export through receipt of the next row count. It includes export/IPC, controller
calculation, and waiting. Nested controller compute time is not added a second time.
Primary timing is **sum of timestep work + feedback intervals + final field gather**.
Validation-only copies and reference computation are excluded and reported
separately. Native loop elapsed includes inter-step gaps; parent command elapsed
additionally includes startup, warm-up, reference checks, final file I/O, and cleanup.

Timing curves include only `timing` mode. Static baselines use the same feedback
adapter. A stopped or unsupported policy holds its split while the fixed thermal
work continues. No policy may shorten the grid or timestep budget.

## Execution and evidence

```text
atlas_hotspot INPUT CPU_ROWS_COMMA STEPS WORKERS verify|timing OUTPUT [feedback]
```

Without feedback, CPU counts are a single constant or exactly STEPS counts. With
feedback, provide one initial count; after each non-final measured record, send the
next count as an unsigned integer followed by a newline. No warm-up decision is
requested. EOF/malformed/illegal input fails. The Python adapter imposes a process
watchdog (120 seconds ordinarily, 600 seconds for long trajectories).

Static tuning uses a 257×259 generated grid, coarse endpoint/quarter shares, fresh
refinement trials, and a frozen selected count before a distinct held-out input.
All six Phase 5 allocation policies run from 15%, nearest 50%, and 90%, in randomized
blocks with the static candidates. Fixed 24-step budgets isolate timing comparisons.
Long checks include 600 steps on a small irregular grid and a 2048×2048 grid with
ownership changes every 100 steps. Matching the paper's dimensions/count alone is
not a historical numerical reproduction.

The separate three-step Nsight capture uses the 2048×2048 input with a 75% CPU
prefix. Actual CUDA activity must overlap annotated CPU computation. Its correlated
timeline is qualitative concurrency evidence, not calibration of host duration or
energy. All source/binary/input identities, plans, commands, outputs, decisions,
and byte checksums are retained. Replay reconstructs decisions, communication
arithmetic, final-field equivalence, and the summary.

Energy and device-event timing remain unqualified under the Phase 2 findings.
Clock policy stays automatic and unchanged; fixed frequency, thermal equilibrium,
and exclusive system use are not claimed. Results support workload correctness and
exploratory allocation timing, not energy-saving claims.
