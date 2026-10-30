# ATLAS GreenGPU

ATLAS implements and evaluates the GreenGPU research method in stages. The
[implementation plan](IMPLEMENTATION_PLAN.md) and [Phase 0 specification](docs/phase0_review.md)
define the scope. Phases 1 and 2 add environment discovery, CPU/CUDA correctness
checks, telemetry, energy accounting, and replayable calibration trials. Allocation
and energy controllers are later phases.

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
