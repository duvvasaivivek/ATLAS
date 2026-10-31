# Phase 8 actuation and diagnostic contract

Version: `frequency-hardware-1.0`, 2026-10-31.

Phase 8 separates a software control transaction from verified physical control.
The transaction engine, bounded static-sweep harness, and frequency-only harness
are implemented and tested with failure-injection devices. The default NVML
adapter on the current WSL system cannot capture the prior locked-clock policy,
so its hardware path blocks before setters. No successful physical actuation or
restoration is claimed by the software tests.

## Policy ownership and restoration

`ClockTransaction` accepts an explicit candidate list and an authoritative
snapshot bound to the device UUID. A snapshot must distinguish a known unlocked
domain (`null`) from unknown state (the snapshot is not restorable). Numeric
policy ranges must be positive, ordered MHz values. Current clock readings,
reported supported frequencies, and default application clocks cannot establish
the prior locked-clock policy.

Transactions run on the Linux/WSL main thread. An OS file lock keyed by the GPU
UUID excludes concurrent ATLAS transactions across processes. It does not exclude
Windows power management, other users' management tools, or unrelated workloads.
Directory creation is exclusive. Before mutation, the transaction persists a
prior-policy snapshot and recovery instructions, then flushes and fsyncs each
request to an append-only journal. Memory and core setters are acknowledged
separately. API success is not recorded as physical frequency verification.

Both domains are considered dirty before the first setter. A partial setter
failure therefore triggers restoration of both prior policies. Restoration
attempts the other domain even when the first domain fails, and a journal-write
failure must not suppress hardware cleanup. Only matching authoritative policy
readback confirms restoration. A restoration failure aborts subsequent work and
retains recovery evidence; it never becomes a successful trial.

Normal exit, exceptions, KeyboardInterrupt, and handled SIGTERM enter cleanup.
SIGTERM handlers are restored afterward. SIGKILL, process crashes, power loss,
and a lost device cannot guarantee cleanup. The recovery JSON contains the prior
policy, GPU UUID, and explicit `nvidia-smi` argument lists where a restorable
snapshot existed. These are instructions, not automatically executed commands.
An archive blocked before mutation instead says no reset is needed; resetting
an unknown pre-existing policy would itself change the machine.

`NvmlClockBackend` implements guarded modern locked-clock setters/resets. It
accepts an optional platform adapter for authoritative policy readback. **No such
provider is shipped for this WSL configuration.** The public CLI has no flag to
assert an unknown baseline is safe. Without a provider, snapshot status remains
non-restorable and direct setter calls are also blocked. This is an explicit
hardware-enablement dependency, not an operational live-control claim.

Application-clock queries are retained as diagnostic evidence, but are not used
to infer locked-clock policy or silently select a fallback actuator. NVIDIA
[deprecates these application-clock interfaces](https://docs.nvidia.com/deploy/nvml-api/deprecations.html).
Power limits, GPU core locks, GPU memory locks, and CPU governors remain separate
controls; no power-cap change substitutes for independent frequency control.

## Physical verification and control harnesses

`verify_setting` requires three distinct, fresh, paired utilization timestamps
under load spanning at least 400 ms. The acquisition bounds and estimated sample
times must be at least one second after the request, sample age at most 500 ms,
GPU busy fraction at least 0.1, both utilization values valid, temperature below
85 C, and driver clock-event mask zero. Fixed reporting tolerances are 15 MHz for
core and 2 MHz for memory. Duplicate valid timestamps do not inflate the count;
invalid readings break the verification streak.

The one-second guard is a conservative observation contract, not external sensor
calibration. NVML buffer timestamps do not expose integration start/end bounds.
Physical interval alignment and wall-to-monotonic conversion still require
qualification before a platform adapter enables live experiments.

`controlled_sweep` first runs a short apply/restore transaction. Longer work
begins only after successful restoration readback. It probes explicit pairs
under load, excludes pairs that fail observed verification, then alternates
candidate order across repeated blocks. Each setting is reverified before and
during measurement. Incorrect workload output aborts and restores. Settling
trials remain distinct from steady measurements. These generic paths are tested
with synthetic devices; no static frequency surface was collected on this GPU.

`frequency_only` uses the Phase 7 policies with a bounded observation budget and
an explicit verified-pair list supplied by the sweep. It verifies the initial
setting and every later request. Observations failing the loaded-frame check
cannot update the policy. It retains controller history, logs input qualification
and requests separately from acknowledgments, and makes no Tier 1 allocation
changes. Setting failures abort and restore rather than substituting simulated
hardware effects. The current CLI does not expose this path while the hardware
restoration gate is blocked.

## Real diagnostic workload contract

`atlas_frequency_probe` supplies two integer CUDA diagnostics with persistent
input/output allocations and a bounded stdin protocol (`run N`, `quit`):

- `compute`: independent xorshift32 recurrences held in registers. The final
  retained configuration uses 65,536 elements and 4,096 rounds per launch.
- `stream`: a coalesced 32-bit affine transform, one input read and one output
  write per element. The final configuration uses 16,777,216 elements: 64 MiB
  per array and 128 MiB total device input/output storage.

The CPU oracle uses masked 64-bit arithmetic. Every batch downloads and checks
every output, reports a checksum, and requires zero mismatches. Separate Python
oracles test small and partial-block cases. Each launch has bounded work; each
batch has 1–10,000 repetitions. Native protocol waits have timeouts and worker
cleanup. Sampling aborts a diagnostic if a reported GPU temperature reaches
85 C. A missing sensor is retained as unavailable, never synthesized.

Each mode has a 64-repetition pilot. The measured repetition count is selected
from pilot host compute time and then frozen for all blocks. Mode order alternates
by block. Preparation, initial upload, and warm-up precede the batches. Synchronized
host compute duration, final download, CPU validation, and host envelope are
reported separately; these are not interchangeable with whole-job duration.
Unqualified CUDA-event time is not used. Logical read-plus-write GB/s is an
algorithmic byte count divided by host compute time, not a DRAM counter.

`actuation-check` records the current capability block and then runs these
diagnostics under the unchanged platform policy. This is real workload
characterization, **not a fixed-frequency sweep or a closed-loop experiment**.
CPU governor and package-energy exposure are recorded without writes.

## Telemetry and archives

The sampler retains per-query acquisition bounds, current core/memory rates,
temperature, raw power/counter readings, driver clock-event masks, and new core
and memory utilization-buffer samples. Loaded summaries use queries wholly
inside the native compute interval. Timestamp pairs must match exactly, have
valid percentages and age, and pass the one-second integration guard using an
explicitly approximate wall-to-monotonic mapping. Conflicting duplicate samples
are discarded; missing memory stays null. Sample-gap and query-wall-time
statistics characterize the interface; they do not establish feedback stability
or the application's total monitoring overhead.

Raw power and energy values remain unqualified under the Phase 2 findings.
Accepted energy and energy savings are null. No device timing, CPU package, or
whole-machine energy claim is inferred from the new diagnostics.

Each archive retains capabilities, the blocked transaction journal, recovery
status, native readiness records, raw trials, worker stderr, runtime/binary/source
identity, summary, and recursive checksums. Replay verifies checksums and file
inventory, diagnostic matrix/order, native identity and sequence, repeated-input
checksums, correctness, timing order, and regenerated summaries. Source hashes
identify code but checksums are not digital signatures. Incomplete runs retain a
failure record and available partial trial evidence; they are not passing archives.

The hardware gate stays blocked until authoritative prior-policy capture,
verified restoration, independent observed controls, and qualified live
observations are available. Software validation does not waive these gates.
