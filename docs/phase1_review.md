# Phase 1 review: environment and hardware execution

Status: Complete against the Phase 1 gate, with limited measurement/control scope.

Date: 2026-10-31, Asia/Calcutta. The final snapshot timestamp is
`2026-10-30T21:32:46.406162+00:00` in UTC.

## Outcome and selected environment

Use the retained Ubuntu 24.04 WSL 2 environment and `.venv-wsl` for development.
Native CPU/OpenMP and CUDA compilation and actual GPU execution both succeeded.
The environment supports implementing the real workloads and allocation policies.
It does not yet support a claim of complete measured GreenGPU energy control.

Verified inventory:

- CPU model reported to WSL: 12th Gen Intel Core i5-12450H.
- WSL-visible resources: 12 logical processors and approximately 7.58 GiB RAM.
  These are guest-visible resources; the virtual topology is not a physical
  P-core/E-core inventory.
- GPU: NVIDIA GeForce RTX 4050 Laptop GPU; CUDA compute capability 8.9.
- GPU memory reported by CUDA: 6,438,780,928 bytes; nvidia-smi reports 6,141 MiB.
- Windows GPU driver: 591.44. CUDA driver/runtime API versions: 13010 (13.1).
- CUDA compiler: 13.1.115; installed toolkit metapackage: 13.1.2-1.
- GCC/G++: 13.3.0; OpenMP found by CMake: 4.5.
- CMake: 3.28.3; Ninja: 1.11.1; Python: 3.12.3 in the existing virtual environment.
- Ubuntu: 24.04.5; kernel: 5.15.167.4-microsoft-standard-WSL2.

Exact package versions, tool output, paths, and process settings are retained in
the [environment manifest](../results/phase1/2026-10-31-final/environment.json).
Only selected non-secret process settings are captured; the program does not dump
the complete environment.

## Implementation delivered

- [Python discovery CLI](../src/greengpu/cli.py) and
  [capability collector](../src/greengpu/capabilities.py): read-only queries,
  explicit unavailable/deprecated/error states, unit normalization, command
  timeouts, raw outputs, source hashes, and immutable output directories.
- [CMake configuration](../CMakeLists.txt) and the
  [CUDA fixture](../native/smoke/main.cu) with a separate
  [OpenMP CPU implementation](../native/smoke/cpu.cpp).
- [Minimal Python packaging configuration](../pyproject.toml): no runtime Python
  dependencies. Source-tree execution is the verified route. The optional pinned
  setuptools packaging backend has not been installed or wheel-tested in Phase 1.
- [Discovery tests](../tests/test_capabilities.py) and
  [reproduction commands](../README.md).

Phase 1 uses installed nvidia-smi XML for a dependency-free capability snapshot.
NVIDIA documents that its output can change between releases; raw XML is retained
and unrecognized units are rejected. A versioned NVML adapter for continuous
sampling belongs to Phase 2. See the
[NVIDIA interface documentation](https://docs.nvidia.com/deploy/nvidia-smi/index.html).

## Correctness and validation evidence

The native fixture evaluated `a[i] + 3*b[i]` independently on the CPU and GPU,
then compared both outputs element-by-element to a separately constructed integer
oracle. Inputs remain in a range exactly representable as floats, so exact
comparison is intentional.

Sizes: 1, 255, 256, 257, 4,099, and 1,048,579 elements. This exercises single-element,
block-boundary, and partial-block cases. Output buffers use sentinels, so uncomputed
values cannot silently appear correct. Kernel-launch and synchronization errors
cause a nonzero process exit.

**Result:** all six cases passed, totaling **1,053,447 elements per backend**, with
**zero absolute error**. Four OpenMP workers were observed. CUDA allocation,
host-to-device transfer, kernel execution, completion, and device-to-host transfer
were executed successfully. This establishes neither CPU/GPU concurrency nor a
performance or energy result; those are later gates.

The discovery suite passed **13 tests**. It covers zero versus missing readings,
unsupported/deprecated values, unexpected units, invalid utilization, absent and
exposed CPU interfaces, process timeouts, missing commands, malformed or failed
smoke results, and preservation of an existing evidence directory.

The final run returned success after building with CMake/Ninja and running the
fixture through the discovery CLI. No Python packages were added to the virtual
environment. No GPU clock, power cap, CPU governor, or system configuration was
changed by Phase 1.

## Capability limitations and their consequences

**GPU monitoring:** GPU/memory activity, temperature, GPU average/instantaneous
power, observed graphics/SM/memory clocks, and reported clock pairs are returned
by this environment. These are unvalidated snapshots. Zero activity in the
snapshot is a valid reading, not missing data. Acquisition cadence, fidelity,
load response, and integration accuracy still require Phase 2 experiments.

**GPU clock management:** the driver reports clock pairs, but their enforcement
has not been attempted. Application-clock queries report deprecated functionality.
This is not proof that every other clock-control API is unavailable. Writes,
required privileges, settling, and restoration remain explicitly unverified until
Phase 8. Hardware or OS automatic policies can change the observed power limit;
Phase 1 does not enforce a constant one.

**CPU measurement/control:** Linux powercap energy zones and cpufreq governor
policies were not exposed in this WSL environment. That finding is scoped to these
Linux interfaces; Windows-side measurement possibilities were not assessed.
CPU utilization/frequency measurement quality has not been validated.

**Energy evidence:** no external wall meter is configured. The GPU cumulative
energy counter was not probed by this XML adapter. Phase 2 must probe available
counter APIs or validate sampled GPU power. Until additional CPU/system
measurement is established, GPU-board data cannot support whole-system saving
claims. CPU DVFS replication is also not available through the queried interfaces.

**Decision:** proceed with real hardware workloads and partial GPU telemetry.
Build allocation control after the measurement and execution-substrate phases.
Evaluate complete two-tier hardware control only after the required interfaces
and measurement boundaries are verified. No switch to native Linux or purchase
of hardware is required just to continue the next implementation phases.

## Retained evidence

The final snapshot is authoritative for the delivered code:

- [capabilities.json](../results/phase1/2026-10-31-final/capabilities.json)
- [environment.json](../results/phase1/2026-10-31-final/environment.json)
- [smoke_result.json](../results/phase1/2026-10-31-final/smoke_result.json)
- [commands.json](../results/phase1/2026-10-31-final/commands.json)
- [source_hashes.json](../results/phase1/2026-10-31-final/source_hashes.json)
- [raw GPU XML](../results/phase1/2026-10-31-final/raw/nvidia_smi.txt)
- [CMake configuration cache](../results/phase1/2026-10-31-final/raw/CMakeCache.txt)

An earlier [initial snapshot](../results/phase1/2026-10-31/capabilities.json) is
retained without modification. The final metadata adds normalized output units,
raw units, selected process configuration, and test-source hashes. Neither snapshot
is a controlled energy benchmark. The working tree was uncommitted during capture;
the recorded Git revision plus per-file hashes identify the code actually run.

## Phase 1 exit gate

- [x] Hardware, OS, toolchain, virtual environment, and exposed interfaces recorded.
- [x] Real CPU and GPU results agree with an independent oracle.
- [x] Reproduction commands and exact observed versions retained.
- [x] Monitoring availability is distinguished from measurement validation.
- [x] Clock writes and their privileges remain explicitly unverified.
- [x] A supported development route and limited evidence path are selected.
- [x] Existing virtual environment and prior specification are preserved.

## Handoff to Phase 2

Implement the timestamped telemetry/energy interfaces and experiment recorder.
Validate GPU sensor cadence, repeated values, gaps, units, and polling overhead
under known load. Probe cumulative energy through a supported API before choosing
counter deltas versus power integration. Decide how to measure CPU/system energy,
or keep the initial evaluation explicitly restricted to the GPU domain. Add
timing boundaries and retained-run validation before implementing an optimizer.

Phase 2 has not started. The smoke fixture is an environment check, not the Phase 3
concurrent workload dispatcher.

## Technical references

- [CUDA 13.1 Linux installation guide](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-installation-guide-linux/index.html): toolkit and host-toolchain requirements.
- [CMake CUDA architectures](https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html): native architecture selection.
- [NVIDIA CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html): environment limitations; observed local results remain the capability evidence.
