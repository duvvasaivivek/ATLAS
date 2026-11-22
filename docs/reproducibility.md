# Reproducing ATLAS

ATLAS is a CPU/CUDA research implementation. It has no neural network or learned
energy predictor to train. Kmeans fits cluster centroids during each workload run.
The benchmark runner tunes a static CPU share on separate generated inputs, freezes
the choice, then evaluates it on held-out inputs. Controllers follow the explicit
equations and policy identities in `algorithm_spec.md` and `frequency_spec.md`.

## Supported execution environment

Use Linux or Ubuntu/WSL, Python 3.12 or newer, a C++17 compiler with OpenMP,
CMake 3.24 or newer, Ninja, and one CUDA GPU supporting FP64 atomics (compute
capability 6.0 or newer). The retained environment uses CUDA 13.1 and Nsight
Systems. `nsys` must be on PATH. CUDA Compute Sanitizer is used for the native
checks. Current test scripts use `/usr/local/cuda-13.1/bin/compute-sanitizer`.

The Windows host does not run the native execution harness directly: supervision,
clock transactions and native timestamps use Linux interfaces. WSL can execute
CUDA without exposing the clock controls and measurements required for full
GreenGPU hardware evaluation. Capability checks determine the actual scope.

From the repository or an extracted source distribution, create an environment:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install .
```

Python packaging installs the controllers and CLI. Native CUDA binaries are built
separately with CMake. The source distribution includes CMake inputs, native
sources, tests, small fixtures, specifications and upstream provenance/licenses.
Large results, local environments and build products are excluded. A wheel alone
does not contain the complete native research workspace. Run the CLI from the
repository/source-distribution root, as its experiment commands expect that layout.

To build the source distribution and wheel and test the installed package in a
separate fresh environment (requires package-index access):

```bash
.venv/bin/python scripts/package_release.py --output build/package-release
```

The artifacts are written to `build/package-release/distributions`. The command
checks source contents, confirms imports use the installed wheel, and runs the
Python tests from the extracted source distribution. It retains package hashes,
installation output and test logs. Native build/hardware checks remain separate.

## Complete validation workflow

Run this once with fresh output and build directories:

```bash
.venv/bin/python scripts/validate_release.py \
  --build build/reproduction \
  --output results/reproduction \
  --blocks 10
```

The workflow runs Python tests, configures/builds native binaries, runs CTest and
all eight CUDA sanitizer checks, validates offline frequency policies, probes
hardware qualification, runs integrated allocation/shadow trials with Nsight
profiling, and runs the randomized matched performance experiment. It then
verifies and replays the three evidence archives before generating
`report/report.html` and `report/report.json`.

The default benchmark includes 40 tuning trials and 360 held-out trials (ten
blocks across both sizes of both workloads). A separately retained legacy build
can be added using `performance-check --baseline-build`; it is not required for
a fresh reproduction. Preserve failures and slower variants. Do not compare
different timing boundaries or select a policy from held-out outcomes.

Exit codes are intentional:

- `0`: all full-project acceptance gates satisfied.
- `1`: a command or validation failed; inspect `failure.json` and stage logs.
- `2`: the report was generated, but full-project acceptance remains incomplete.

The current archive contracts describe allocation/shadow experiments and cannot
establish full hardware acceptance. Consequently a successful current workflow
ends with `2`, even when every correctness test and benchmark trial passes.
Each stage has a bounded timeout and a retained command/return code. Output
directories cannot be reused. Do not put the report inside an immutable archive.

## Regenerate analysis without a GPU

Use existing complete archives on Linux/WSL with Python 3.12:

```bash
PYTHONPATH=src .venv/bin/python -m greengpu research-report \
  --performance results/remediation/2026-11-02-performance-final \
  --integration results/remediation/2026-11-02-integration-v2 \
  --qualification results/remediation/2026-11-02-qualification-v2 \
  --output results/regenerated-report
```

This command checks inventory/hashes and recomputes correctness, policy decisions
and summaries from raw evidence. It does not require the archived executables or
change GPU settings. Add `--require-full` to return `2` when acceptance is
incomplete; without that option, `0` means report generation succeeded.

The report includes full-command comparisons as well as the workload interval.
Both use matched independent blocks and pointwise 95% bootstrap intervals. An
interval spanning zero does not establish the sign of an improvement. The JSON
contains source checksum identities and all remaining qualification requirements.

Historical compatibility is checked separately with:

```bash
PYTHONPATH=src .venv/bin/python tests/replay_archives.py
```

Those historical archives live in the repository's results collection, not the
source package. Preserve their checksums and never add generated plots to them.

## Hardware work still required for full acceptance

Obtain authoritative prior clock-policy readback, verify applying and restoring
each supported clock pair, and qualify loaded telemetry timestamps independently.
Qualify GPU energy and device timing against independent measurements. CPU-package
or wall energy must be available for claims about those domains; no external
wall meter is currently configured.

The separate [guarded live runner](live_control.md) is implemented and has
software tests with a fake actuator and real native workloads. Supply and
validate a qualified platform provider, then run physical frequency-only and
joint-control experiments and collect the remaining fixed-clock/static-clock
baselines. The default provider rejects live control before workload launch.
A different OS by itself is not proof that the
required hardware features work. Do not bypass these gates or substitute
synthetic energy. The project also requires strict per-iteration compute overlap
evidence; a profiler launch or CPU barrier wait is not compute overlap.
