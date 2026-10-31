# ATLAS GreenGPU

ATLAS implements and evaluates the GreenGPU research method in stages. The
[implementation plan](IMPLEMENTATION_PLAN.md) and [Phase 0 specification](docs/phase0_review.md)
define the scope. Phases 1–3 add environment discovery, CPU/CUDA correctness,
measurement infrastructure, and concurrent static partitions with profiler evidence.
Application workloads and adaptive controllers are later phases.

## Run Phase 1 in Ubuntu / WSL 2

Use the retained environment. The commands below do not require additional Python
packages: discovery and its tests use the standard library.

```bash
cd /mnt/c/Users/saivi/Downloads/ATLAS
source .venv-wsl/bin/activate
export PYTHONPATH="$PWD/src"

cmake -S . -B build/phase1 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc
cmake --build build/phase1 --parallel 2

python -m unittest discover -s tests -v

OMP_NUM_THREADS=4 python -m greengpu discover \
  --cuda-compiler /usr/local/cuda-13.1/bin/nvcc \
  --smoke-binary build/phase1/atlas_smoke \
  --output results/phase1/local-check
```

Choose a new output directory on every run. Existing evidence is never overwritten.
CUDA architecture defaults to CMake's `native`; an explicit
`-DCMAKE_CUDA_ARCHITECTURES=89` can be used for the verified local RTX 4050 target.
Changing compiler or architecture requires a fresh build directory.

`pyproject.toml` defines optional Python packaging using a pinned build backend;
the source-tree commands above do not install it. Native compilation is an explicit
CMake step, not an implicit package-install side effect. No runtime Python
dependencies are required by Phase 1.

## What the checks establish

The native fixture performs CPU/OpenMP and CUDA computations against an independent
integer oracle at six sizes, including partial CUDA blocks. It checks allocation,
host/device transfers, kernel launch, synchronization, and returned results. It is
not a concurrency, performance, or energy benchmark.

Discovery records OS, guest-visible CPU/memory, tool versions, GPU monitoring
queries, CPU powercap/governor exposure, and smoke-test evidence. Reported clock
pairs do not prove they can be applied. No clock, governor, or power-limit settings
are changed. Missing sensors remain explicitly unavailable or unprobed.

See [Phase 1 results and environment decision](docs/phase1_review.md) for the
validated machine and remaining limitations. The [architecture](ARCHITECTURE.md)
distinguishes the implemented foundation from planned later modules.

## Run Phase 2 measurement calibration

Inside Ubuntu/WSL, from the repository root and retained virtual environment:

```bash
export PYTHONPATH="$PWD/src"
python -m pip install nvidia-ml-py==13.615.71

cmake -S . -B build/phase2 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc
cmake --build build/phase2 --parallel 2
python -m unittest discover -s tests -v
OMP_NUM_THREADS=4 ctest --test-dir build/phase2 --output-on-failure

python -m greengpu calibrate \
  --binary build/phase2/atlas_measure \
  --output results/phase2/local-check \
  --blocks 4 --target-seconds 6 --period 0.1

python -m greengpu replay results/phase2/local-check
```

The runner performs a separate tuning pilot, then randomized paired trials with
periodic monitoring enabled or endpoint reads only. Every paired trial executes
identical fixed work. The duration target only chooses the repetition count before
the trial series; actual durations can differ. Existing output directories are
rejected. One CUDA/NVML GPU is required and PCI identities must match.

The only new runtime package is NVIDIA's pinned NVML binding. Phase 1 discovery
still runs without it. Neither the virtual environment nor the research paper is
replaced. GPU/CPU policy settings are never changed by calibration.

See the [run schema and timing contract](docs/run_schema.md) and
[Phase 2 review](docs/phase2_review.md) before interpreting results. In particular,
a completed trial does not guarantee valid energy or device timing: counter/power
disagreement blocks primary energy reporting, device-event/host disagreement
blocks device timing, and CPU/wall energy remains unavailable on the
current setup. This fixture measures infrastructure behavior, not GreenGPU savings.

## Run Phase 3 static CPU/GPU validation

From the repository root inside the retained WSL environment:

```bash
export PYTHONPATH="$PWD/src"
cmake -S . -B build/phase3 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc
cmake --build build/phase3 --parallel 2
python -m unittest discover -s tests -v
OMP_NUM_THREADS=4 ctest --test-dir build/phase3 --output-on-failure

python -m greengpu static-check \
  --binary build/phase3/atlas_static \
  --output results/phase3/local-check --workers 4
python -m greengpu static-replay results/phase3/local-check
```

`static-check` runs 40 partition configurations with three measured iterations each,
then captures a separate CUDA/NVTX trace with the installed `nsys` CLI. It requires
exact output/coverage/reduction checks and observed kernel/CPU-region overlap.
Each new run needs a fresh output directory. For a single native experiment:

```bash
# Arguments: elements, CPU items, rounds, measured iterations, CPU workers, seed
build/phase3/atlas_static 65539 32769 128 5 4 1729
```

The GPU receives `elements - CPU items`. Empty branches are supported; N must be
positive. The harness uses one warm-up and reuses work buffers across iterations.
No CUDA event-duration or energy claim is enabled by these checks. See the
[Phase 3 review and execution contract](docs/phase3_review.md) for evidence and
limitations, including the distinction between host submission and kernel activity.

## Run Phase 4 kmeans

From the repository root in WSL, with `.venv-wsl` activated:

```bash
export PYTHONPATH="$PWD/src"
cmake -S . -B build/phase4 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc
cmake --build build/phase4 --parallel 2
python -m unittest discover -s tests -v
OMP_NUM_THREADS=4 ctest --test-dir build/phase4 --output-on-failure

python -m greengpu kmeans-check \
  --binary build/phase4/atlas_kmeans \
  --output results/phase4/local-check --workers 4 --blocks 3
python -m greengpu kmeans-replay results/phase4/local-check
```

This runs independent correctness fixtures, fixed and convergence modes, changing
ownership boundaries, a randomized static split sweep, refinement, held-out
evaluation, and a separate Nsight concurrency trace. `nsys` must be on PATH.
Each invocation needs a fresh output directory. The native program exports full
final memberships and centroids; records retain transfer costs and iteration
aggregates. A small migration example is:

```bash
# Input, CPU counts per iteration, maximum iterations, CPU workers, mode
build/phase4/atlas_kmeans tests/data/kmeans_ties.kmdat 3,1,0,3,0 5 4 fixed
```

The binary input format and Python `write_dataset` helper are described in the
[kmeans specification](docs/kmeans_spec.md). Optional scientific plots require
Matplotlib; this extra is separate from the benchmark runtime:

```bash
python -m pip install -e '.[plots]'
python -m greengpu kmeans-plot results/phase4/local-check \
  --output results/phase4/local-figures
```

Plots show the sum of full iteration intervals, including migration, transfers,
and merging. Intervening validation/logging time is reported separately.
`implementation_status: passed` does **not** mean the full research gate passed:
energy curves/minima remain blocked by Phase 2 qualification. See the
[Phase 4 review and retained evidence](docs/phase4_review.md).

## Run Phase 5 workload allocation

Phase 5 implements the five reference/corrected allocation policies and the
separately specified robust proportional extension. Run from the repository root
inside the retained WSL environment:

```bash
export PYTHONPATH="$PWD/src"
cmake -S . -B build/phase5 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc
cmake --build build/phase5 --parallel 2
python -m unittest discover -s tests -v
OMP_NUM_THREADS=4 ctest --test-dir build/phase5 --output-on-failure

python -m greengpu allocation-check \
  --binary build/phase5/atlas_kmeans \
  --output results/phase5/local-check --workers 4 --blocks 3
python -m greengpu allocation-replay results/phase5/local-check
# Requires the optional Matplotlib extra documented above.
python -m greengpu allocation-plot results/phase5/local-check \
  --output results/phase5/local-figures
```

The runner verifies the retained Phase 4 final-v2 archive, selects its static
comparator and a near-balance starting share from tuning evidence, then freezes
the plan before new held-out trials. It retains synthetic diagnostics, native
feedback records, independent correctness checks, and every policy transition.
Each trial keeps one native process alive, preserving centroids and GPU residency.
Malformed feedback, missing decisions, inconsistent next splits, altered evidence,
and incorrect output fail validation. Every run needs a new output directory.

Timing comparisons include migration and feedback/IPC costs. Static baselines use
the same protocol. Frozen adjustment, oscillation, failed bracketing, and lack of
balance are reported honestly; passing correctness does not imply faster runtime.
Energy and fixed-capacity hardware qualification remain blocked. See the
[Phase 5 contract](docs/phase5_spec.md) and [review](docs/phase5_review.md).

## Run Phase 6 HotSpot

HotSpot adds evolving thermal state, row halos, and bidirectional ownership
migration. It uses the documented Rodinia CUDA thermal model with FP64 and an
explicit stability limit. The pinned model sources and license are retained under
`references/rodinia`; no external dataset is needed.

```bash
# WSL, repository root, retained .venv-wsl activated
export PYTHONPATH="$PWD/src"
cmake -S . -B build/phase6 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc
cmake --build build/phase6 --parallel 2
python -m unittest discover -s tests -v
OMP_NUM_THREADS=4 ctest --test-dir build/phase6 --output-on-failure

python -m greengpu hotspot-check \
  --binary build/phase6/atlas_hotspot \
  --output results/phase6/local-check --workers 4 --blocks 3
python -m greengpu hotspot-replay results/phase6/local-check
# Optional Matplotlib extra, already used for earlier phase figures
python -m greengpu hotspot-plot results/phase6/local-check \
  --output results/phase6/local-figures
```

The complete matrix includes six small fixtures, static tuning/refinement, held-out
allocation trials, a 600-step irregular trajectory, a 2048×2048/600-step migration
check, and a separate Nsight trace. The large independent scalar reference check
can take several minutes. `nsys` must be on PATH. New archives and output fields
are created exclusively; use a fresh output directory for each run.

For a single small migration run:

```bash
# Input, CPU row counts, steps, workers, validation mode, new output field
build/phase6/atlas_hotspot tests/data/hotspot_edge.hsdat \
  3,1,0,3,2,0 6 4 verify build/hotspot-example.hstemp
```

`verify` checks every timestep using a separate full-field snapshot. `timing`
avoids those validation downloads during the loop and checks the full final field
afterward. Required halo/migration transfers and final gathering remain counted.
Pure Phase 5 policies are reused with rows as their work units.

See the [model and execution contract](docs/hotspot_spec.md) and
[Phase 6 review](docs/phase6_review.md). Energy remains unavailable, and an unchanged
automatic clock policy does not establish fixed hardware capacity.

## Phase 7: offline Tier 2 frequency policies

`paper_additive` implements the paper's asymmetric losses and additive score
updates. `discounted_loss_v1` provides a separately specified forgetting
extension. Both are pure offline policies with explicit candidate pairs, input
validation, history, deterministic ties, and complete decision logs.

```bash
# WSL, repository root, existing .venv-wsl activated
export PYTHONPATH="$PWD/src"
python -m unittest discover -s tests -v
python -m greengpu frequency-check --output results/phase7/local-check
# Optional: also retain timestamp-paired read-only NVML observations
python -m greengpu frequency-check \
  --output results/phase7/local-hardware-check --hardware-trace
python -m greengpu frequency-replay results/phase7/local-hardware-check
python -m greengpu frequency-plot results/phase7/local-hardware-check \
  --output results/phase7/local-figures
```

Use a new directory for each collection and figure export. Replay requires no
GPU. The 23 synthetic diagnostics are assumptions for checking controller
behavior; their runtime/power units are not hardware measurements. Optional
hardware replay uses matching NVML driver timestamps as an explicit offline
eligibility assumption; sensor integration boundaries remain unqualified.

The retained [Phase 7 review](docs/phase7_review.md) covers 92 passing tests,
4,144 synthetic observations, six idle hardware observations, and the beta
limitation. See the [policy contract](docs/frequency_spec.md) and
[diagnostic figure](results/phase7/figures/frequency-diagnostics.png).
This phase does not request or acknowledge GPU clock changes. Live actuation
and measured effects are Phase 8 work; energy qualification remains blocked.
