# ATLAS GreenGPU

ATLAS implements and evaluates the GreenGPU research method in stages. The
[implementation plan](IMPLEMENTATION_PLAN.md) and [Phase 0 specification](docs/phase0_review.md)
define the scope. Kmeans, HotSpot, allocation policies, offline frequency policies,
and integrated allocation/shadow execution are implemented through Phase 9.
Live frequency control and physical energy qualification remain capability-gated.

## Start here: teammates running the demo

The demo groups synthetic delivery locations and compares the same task on the
CPU alone, GPU alone, and ATLAS sharing work between them. It checks that all
three approaches produce matching answers and displays fresh measurements.
No dataset download, model training, API key, or cloud service is needed.

**Live execution requires an NVIDIA CUDA-capable GPU.** The tested setup is
Windows with Ubuntu in WSL 2, CUDA Toolkit 13.1 and an RTX 4050. Use a GPU
supported by your installed CUDA Toolkit. Machines without a compatible GPU can
open a teammate's exported `demo.html` in a browser, but cannot run this live demo.

### 1. Install the prerequisites once

On Windows, install Git and an NVIDIA Windows driver compatible with your CUDA
Toolkit. Install Ubuntu 24.04 in WSL 2 if you do not already have it. Run this
in **PowerShell as Administrator**, then restart if requested and finish Ubuntu's
first-launch username/password setup:

```powershell
wsl --install -d Ubuntu-24.04
```

See [Microsoft's WSL installation guide](https://learn.microsoft.com/en-us/windows/wsl/install)
if installation needs additional steps. In normal **PowerShell**, check the
distribution name and select the one you will use for ATLAS:

```powershell
wsl --list --verbose
wsl --set-default Ubuntu-24.04
```

Substitute your actual distribution name if different; its version must be `2`.
The desktop launcher always uses your **default WSL distribution**, so install
the following tools in that same distribution.

In **Ubuntu / WSL**, install the build tools and Python:

```bash
sudo apt update
sudo apt install -y build-essential cmake ninja-build python3 python3-venv
python3 --version
cmake --version
```

Python must be **3.12 or newer** and CMake **3.24 or newer**. Ubuntu 24.04 provides
suitable versions. Install **CUDA Toolkit 13.1 inside WSL** using the
[CUDA 13.1 download instructions](https://developer.nvidia.com/cuda-13-1-0-download-archive)
and [NVIDIA's WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/).
Choose the WSL toolkit instructions; do not install a Linux NVIDIA display
driver inside WSL. The Windows driver supplies GPU access.

Check from **Ubuntu / WSL** before continuing:

```bash
nvidia-smi
/usr/local/cuda-13.1/bin/nvcc --version
```

Both commands must succeed. Windows CUDA installation alone does not supply the
Linux compiler required here. The live demo does not require Nsight Systems,
NVML Python packages, or plotting packages; full research validation has additional
requirements in the [reproducibility guide](docs/reproducibility.md).

### 2. Get the code on Windows

For a **first checkout**, run in normal **PowerShell**:

```powershell
New-Item -ItemType Directory -Force "$HOME\Projects" | Out-Null
Set-Location "$HOME\Projects"
git clone https://github.com/duvvasaivivek/ATLAS.git
Set-Location ATLAS
```

If the repository is private, your GitHub account needs access. If you already
have a checkout, open **PowerShell in that ATLAS folder** and update it instead:

```powershell
git pull --ff-only
```

Keep any local edits safe before updating. If Git refuses the pull, resolve the
reported local changes or branch divergence before proceeding.

### 3. Build and check in WSL

From **PowerShell in the ATLAS folder**, enter Ubuntu at the same directory:

```powershell
wsl
```

The prompt should now be a Linux prompt such as `name@machine:/mnt/c/.../ATLAS$`.
Run the following **inside WSL**, from the repository root:

```bash
cmake -S . -B build/demo -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc -DPython3_EXECUTABLE="$(command -v python3)"
cmake --build build/demo --parallel 2
PYTHONPATH=src python3 -m unittest discover -s tests -v
OMP_NUM_THREADS=4 ctest --test-dir build/demo --output-on-failure --no-tests=error
```

Continue only when the build and both test suites pass. Native tests need the
GPU and may take a few minutes. The source-tree demo and these tests need no
virtual environment or `pip install`; the launcher calls WSL's `python3` directly.
Build products are local and must be created on each teammate's machine.

If CUDA is installed elsewhere, replace the compiler path with the output of
`command -v nvcc`. CMake selects the local GPU architecture by default. After
changing CUDA compiler, GPU architecture, or moving the checkout, use a **new
build directory**, such as `build/demo-v2`, instead of reusing an old CMake cache.

### 4. Open and present the demo

Return to **PowerShell**:

```bash
exit
```

Then launch from **PowerShell in the ATLAS folder**, or double-click `demo.bat`
in File Explorer:

```powershell
.\demo.bat
```

1. Select **build/demo** under **Demo engine**.
2. Click **Check setup** and confirm the native checks pass.
3. Choose **Medium: 16,000 stops**, then click **Start live demo**.
4. In **Watch it work**, look at the grouping and the count of rows assigned
   to each processor. Open **Machine readings** to see what the machine reports.
5. Select **See the result** when the answers match. Compare measured times,
   then choose a try, step and row under **Where did a row go?**
6. Repeat with **Large: 64,000 stops** to show how the workload size affects the
   comparison. Sharing work can add overhead and is not guaranteed to win.

Each click starts new computation. Results are saved in
`results/demo/<timestamp>/`. Use **Open results** to inspect them. The generated
`demo.html` can be shared and opened offline; keep the whole run folder for raw
trials and checksum verification. **Watch saved demo** requires an existing demo
run. Use the research workflow below to generate the full research report.

### Use your own locations

Select **Use my CSV** and choose a UTF-8 file with the header `id,x,y`.
For example:

```csv
id,x,y
A,18,20
B,78,22
C,25,76
D,79,78
```

Use unique IDs, at least four distinct locations, at most 64,000 rows, and
at most 8 MiB. Choose coordinates in a consistent planar unit, such as meters.
The program groups them by straight-line distance. The result folder contains
`groups.csv` with row numbers and group assignments. Share the report and data
only with recipients allowed to see the locations. Exported IDs starting with
`=`, `+`, `-`, or `@` are prefixed with an apostrophe for spreadsheet safety.
Git ignores new local runs in `results/demo/`.

### Read the measurements

- **Rows assigned** describes the work split, independent of hardware utilization.
- **Machine readings** show sampled Linux CPU counters and whole-device GPU
  measurements from `nvidia-smi`. The native worker's CPU usage is relative to one
  CPU core and can exceed 100%. A short trial can finish before it is sampled.
  Select **Watch longer** in **Machine readings** before starting to run 512
  actual grouping steps per try. Unsupported measurements are marked unavailable.
- **See the result** shows recorded per-step assignments, elapsed times and
  allocation decisions. Graphics dispatch to returned results includes more
  than kernel execution. The detailed record also contains transfer byte counts.
- **Open results** provides the raw trial JSON, timestamped machine samples
  (`machine.jsonl` when collected), CSV and checksums. **Stop** requests
  cancellation and retains available records. Turn off **Show machine readings**
  to compare timing without machine polling. Display updates still add overhead.
  For matched research baselines and profiling, use the research workflow below.

For the presentation, explain: "ATLAS decides how to split work between the main
processor and graphics processor while checking that the answer stays correct."
This is a research prototype: the demo measures task time, and energy savings
and full hardware acceptance remain unproven.

### After the next pull

Close the demo, run `git pull --ff-only` in PowerShell from the checkout, then
repeat **step 3** to rebuild and run the checks. Reopen `demo.bat` for a fresh
comparison. Do not copy another machine's build directory or virtual environment.

### Common setup problems

- **PowerShell says "Missing expression after unary operator '--'":** Bash
  commands were pasted into PowerShell. Enter WSL first. Bash `source`, `export`,
  and `\` line continuation belong in WSL. The quick-start commands above use
  single lines to avoid continuation mistakes.
- **`python` not found or an activated environment points to another folder:**
  use `python3` for the demo setup. Do not reuse a copied virtual environment.
  Research packaging instructions below require a newly created local environment.
- **`nvidia-smi` fails in WSL:** check the Windows NVIDIA driver and WSL 2 setup
  using the NVIDIA guide above before attempting the build.
- **`nvcc` not found:** install the Linux/WSL CUDA Toolkit and check its actual
  compiler path; the Windows compiler cannot build these Linux binaries.
- **No demo engine appears:** finish building in this checkout. The launcher
  looks directly under `build/<name>/` for `atlas_kmeans` and `CTestTestfile.cmake`.
  Close and reopen the launcher after building.
- **Terminal tests pass but the launcher fails:** check `wsl --list --verbose`
  in PowerShell. The default distribution must contain the same Python, CMake,
  CUDA tools and GPU access used to build the project.
- **A live run fails:** inspect **Technical details** and the fresh run folder for
  `failure.json` and raw trial records. Preserve that evidence before starting a
  new run; a failed run is not a successful timing comparison.

## Windows demo launcher

Double-click **`demo.bat`** to open the minimal desktop UI, similar to SafeSliceSQL.
The UI needs Windows PowerShell and an existing CUDA build in WSL; Windows Python
and extra UI packages are not required.

- **Start live demo:** start fresh computation on real CPU/CUDA
  hardware. The live dashboard shows the current execution mode, native group
  centers, actual CPU work-share trace and validated trial times while the job
  runs. **Why this split?** shows the current explanation and retains ATLAS
  decisions that occurred between screen refreshes. Select a decision to see the
  timing window, the reason for a change or hold, and the next CPU/GPU assignment.
  At completion the dashboard shows the lowest median grouping and whole-command
  times separately. Preview centers remain provisional until correctness validation passes.
  An offline-capable final visual presentation shows the groups, the same
  task's measured times, and ATLAS's recorded work-allocation changes. All modes
  must pass independent native validation and produce matching answers.
- **Watch saved demo:** open a checksum-verified presentation without WSL
  or GPU execution. Recorded measurements are explicitly labelled and dated.
- **Check setup:** the complete native correctness suite.
- **See the result:** open the current completed run and inspect its work by step.
- **Open results:** demo logs, or the results folder before the first run.

Select a build and Small (2,000), Medium (16,000), or Large (64,000) locations.
Each comparison uses identical input, 32 grouping rounds by default, four CPU workers and
three shuffled repetitions per mode. Workload and full-command times are both
retained, including live-observation overhead. Screen refreshes can skip fast
rounds; artificial delays are not added. Each native trial has a 90-second deadline; failures are retained and
are not silently retried. This illustrates scaling on one host, not distributed
execution. A short demo does not establish a general speedup or energy saving.

Each run saves raw trials, input, summary, checksums and a self-contained
`demo.html` under `results/demo/<timestamp>/`. Share that HTML for a portable
presentation, or keep the whole folder for verifiable evidence. The launcher
adds `native-checks.txt`. For a fresh checkout, follow the WSL setup and build
instructions below first.

For a two-minute presentation: show the ungrouped dots, click **Show the groups**,
compare the three timing bars, then explain the recorded CPU work-share trace.
The page includes a plain-language speaking script and the method's limits.
The example groups nearby locations; it does not optimize delivery routes.

The latest continuation adds a [guarded live runner](docs/live_control.md),
armed native feedback and replay. Validation passes 172 Python tests, 11 native
tests and 16 CUDA sanitizer checks across the default and experimental dispatch
paths. Native live tests use a fake clock actuator; full physical acceptance and
the strict compute-overlap gate remain incomplete.

## Reproduce and generate the research report

The [November 10 completion review](docs/completion_review.md) records the
telemetry-startup fix, 163 passing Python tests, nine native tests, eight sanitizer
checks and a new 400-trial allocation study. The hardware and strict concurrency
gates remain incomplete; this is a capability-limited research result.

The [reproducibility guide](docs/reproducibility.md) covers fresh installation,
native builds, testing, static-partition tuning, held-out evaluation and report
generation. ATLAS uses explicit control policies; no learned model needs training.

```bash
# From the source root in Linux/WSL, using a fresh build/output path:
python3 -m venv .venv-wsl
.venv-wsl/bin/python -m pip install .
.venv-wsl/bin/python scripts/validate_release.py \
  --build build/reproduction --output results/reproduction --blocks 10
```

The workflow produces a standalone HTML report and machine-readable acceptance
record. Exit code 2 means evidence was generated but full-project gates remain
incomplete; it must not be interpreted as a completed hardware replication.
`research-report` can also regenerate the analysis from existing verified archives
without executing GPU workloads. See the guide for commands and exit codes.

## Runtime and performance remediation (2026-11-02)

The [remediation record](docs/remediation.md) tracks bounded process supervision,
isolated telemetry, kmeans timing mode, batched feedback, per-epoch allocation,
hierarchical reduction, optional HotSpot overlap, and controlled evaluation.
Existing evidence and paper-policy behavior remain available.

The November 2 validation passed 154 Python tests, nine native tests, eight CUDA sanitizer
checks, and a 440-trial matched evaluation with replay. All six long integration
trials now change allocation and make decisions in both phases. Strict per-iteration concurrency,
live clock control, and physical energy qualification remain incomplete or
blocked. See the remediation record for measured gains, uncertainty, and retained
regressions; experimental options are not universal performance defaults.

```bash
# Repository root in Ubuntu/WSL; use fresh result directories.
source .venv-wsl/bin/activate
export PYTHONPATH="$PWD/src"
cmake -S . -B build/remediation -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc
cmake --build build/remediation --parallel 2
python -m unittest discover -s tests -v
OMP_NUM_THREADS=4 ctest --test-dir build/remediation --output-on-failure
python -m greengpu qualification-check --build build/remediation --output results/remediation/new-qualification
python -m greengpu qualification-replay results/remediation/new-qualification
python -m greengpu integration-check --kmeans-binary build/remediation/atlas_kmeans \
  --hotspot-binary build/remediation/atlas_hotspot --output results/remediation/new-integration
python -m greengpu performance-check --build build/remediation \
  --baseline-build build/phase8 --blocks 10 --output results/remediation/new-performance
python -m greengpu performance-replay results/remediation/new-performance
```

The optional baseline build must be an actual retained earlier build; omit it on
a fresh checkout. Benchmark results include regressions, matched GPU-only/CPU-only
baselines and a static split chosen on separate tuning inputs. The primary timing
is a contiguous workload envelope; complete command time includes final independent
validation. An incomplete strict concurrency gate still returns exit code 1.

## Run Phase 1 in Ubuntu / WSL 2

Use the retained environment. The commands below do not require additional Python
packages: discovery and its tests use the standard library.

```bash
# Open WSL from your own checkout, as described in the teammate setup above.
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

## Phase 8: actuation qualification and loaded diagnostics

Phase 8 adds transactional clock-control infrastructure, restoration/failure
tests, loaded-setting verification, and two independently checked CUDA diagnostics.
On the current WSL setup, the prior locked-clock policy cannot be read back;
the hardware gate blocks all setters. The real diagnostics run under the unchanged
platform policy. Static frequency sweeps and live Tier 2 experiments remain blocked.

```bash
# WSL, repository root, existing .venv-wsl activated
export PYTHONPATH="$PWD/src"
cmake -S . -B build/phase8 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc
cmake --build build/phase8 --parallel 2
python -m unittest discover -s tests -v
OMP_NUM_THREADS=4 ctest --test-dir build/phase8 --output-on-failure

python -m greengpu actuation-check \
  --binary build/phase8/atlas_frequency_probe \
  --output results/phase8/local-check --blocks 3 --target-seconds 3
python -m greengpu actuation-replay results/phase8/local-check
```

Use a fresh output directory. A successful command exit means diagnostic
collection/replay completed; inspect `hardware_exit_gate`, which is `blocked`
on the retained machine. No baseline-override or force-actuation flag is provided.
Replay needs no GPU. Accepted energy savings remain unavailable.

See the [Phase 8 review](docs/phase8_review.md),
[control and diagnostic contract](docs/phase8_spec.md), and
[authoritative results](results/phase8/2026-10-31-final-v2/summary.json).

## Phase 9: integrated allocation and frequency shadowing

The integrated runner applies real allocation decisions to persistent kmeans and
HotSpot, samples telemetry independently, and evaluates eligible frequency inputs
in shadow mode. Separate analytical scenarios exercise both control loops. New
runs require the remediation build described above; historical Phase 9 archives
remain replayable with their original policy and timing semantics.

```bash
export PYTHONPATH="$PWD/src"
python -m unittest discover -s tests -v
python -m greengpu integration-check \
  --kmeans-binary build/remediation/atlas_kmeans \
  --hotspot-binary build/remediation/atlas_hotspot \
  --output results/phase9/local-check --workers 4
python -m greengpu integration-replay results/phase9/local-check
```

`nsys` must be on PATH. Use a fresh output directory. The matrix retains
independent references, raw telemetry, full decisions, final fields, and profiler
activity. Replay requires no GPU. An exit code of 1 with a generated summary can
mean the strict concurrency gate remains incomplete; it does not by itself mean
checksum or decision replay failed. Read `software_gate` and the concurrency
details. Integrity failures instead report an explicit replay error.

`--mode live-joint` exits with code 2 before launching work because the hardware
prerequisites remain unmet. No fallback or clock-setting override is provided.
The historical October 31 run passed 135 tests and workload/replay checks, observed
overlap in 6/6 kmeans and 5/6 HotSpot profile iterations, and rejected hardware
frequency inputs whose timestamps failed the host/driver mapping.

See [Phase 9 review](docs/phase9_review.md),
[runtime contract](docs/phase9_spec.md), and
[final repeat](results/phase9/2026-10-31-final-v3/summary.json).

## Guarded live runner

The new `live-run` command implements barrier-based joint control with device
identity checks, loaded clock verification, settling guards, independent native
correctness checks and transactional restoration. `live-replay` verifies the
retained decisions and results without a GPU. See the
[provider contract and commands](docs/live_control.md).

Real CPU/CUDA integration tests use an explicitly fake clock actuator. They do
not qualify physical control or energy savings. The default provider remains
blocked on this laptop; a separately qualified platform provider is required.
The optional GPU-first dispatch implementation also leaves the strict
all-iteration compute-overlap gate incomplete, so CPU-first remains the default.
