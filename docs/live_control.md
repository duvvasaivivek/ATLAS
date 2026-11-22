# Guarded live control

The separate `live-joint-1.0` runner implements joint allocation and frequency
control for the native k-means and HotSpot workloads. It does not alter the
historical allocation/shadow archive contracts. The current laptop remains
unqualified for physical clock control; the default provider refuses to actuate.

## Execution contract

`live-run` checks a fresh, device-bound platform qualification and authoritative
prior clock policy before launching a workload. The native `feedback armed`
protocol reports its CUDA PCI identity and waits for `start`. The runner compares
that identity with the qualified device before requesting clocks or releasing
work. Every later iteration ends at a feedback barrier.

The transaction journals intent before each setter, owns an exclusive device
lock, and restores both prior clock domains after stopping the workload. Handled
termination, native errors, telemetry errors, failed setters and failed final
correctness checks follow the same cleanup path. Restoration readback failure is
an error, with the prior policy and recovery commands retained. An uncatchable
process or machine failure still requires recovery from that journal.

Allocation learning excludes unverified clock intervals. A request must verify
under load after the one-second guard, using three fresh timestamped samples
spanning at least 400 ms. Each barrier excludes samples older than 500 ms. Clock
changes start a new capacity epoch; settling and allocation guards prevent
mixing measurements across capacity changes. Setting verification and subsequent
loss of verification each have a ten-second deadline. Temperature at or above
85 C, mismatched device identity, expired qualification and invalid telemetry
stop execution. A short job or unverified final setting returns an incomplete
status rather than physical-control success.

The final native result is checked against the independent workload reference.
`live-replay` verifies the archive inventory and hashes, recomputes every control
decision, binds observations to native output, reruns final correctness checks,
and checks the setting and restoration journal. It does not require a GPU.

## Platform provider contract

`--provider package.module:factory` loads trusted installed Python code. Its
factory returns an object with `backend`, `qualification()`, `samples()` and
`close()`. This is a platform integration interface, not a configuration switch
that qualifies unsupported hardware. No qualified physical provider ships for
the current machine.

For physical execution, `backend` must be an `NvmlClockBackend` with an
independently implemented authoritative `policy_reader`. Current observed
frequencies are not a prior policy. `qualification()` supplies:

- `device_uuid`, `pci_bus_id`, and `evidence_kind="hardware"`;
- `telemetry_clock`, `restoration_probe`, and `device_timing`, each `qualified`;
- `evidence_id` identifying the separately retained calibration evidence;
- `qualified_at_ns` and `valid_until_ns` in the current host monotonic domain;
- `verified_pairs`, explicit core/memory MHz pairs verified under load.

The provider is responsible for substantiating these attestations, calibrating
timestamp mapping, bounding all platform calls, collecting samples while native
work executes, and returning each sample once in increasing driver timestamp
order. `samples()` returns normalized frames containing device UUID,
`sample_timestamp_us`, `sample_valid`, `sample_monotonic_estimate_ns`,
`sample_age_us`, acquisition start/end nanoseconds, core/memory utilization
fractions, core/memory MHz, temperature C and `clock_event_reasons`. A declaration
without independent evidence does not establish hardware qualification.

Software tests use `evidence_kind="software_test"` and a dictionary-backed
actuator. That qualification is rejected with a physical NVML actuator. These
archives always report `physical_control_evidence=false`, even if real CUDA
workloads ran successfully. Neither evidence kind establishes energy savings;
the live runner reports `energy_savings=null`.

## Commands

Build the current native sources first. On a separately qualified platform with
its trusted provider installed, run from the repository root:

```bash
PYTHONPATH=src python -m greengpu live-run \
  --workload kmeans --binary build/live-development/atlas_kmeans \
  --dataset tests/data/kmeans_ties.kmdat --steps 1000 --workers 4 \
  --provider qualified_platform:make_provider \
  --output results/live/new-run
PYTHONPATH=src python -m greengpu live-replay results/live/new-run
```

The provider name above is illustrative; it is not an included dependency.
Omitting it uses the default read-only provider and returns exit code 2 on this
machine before workload launch or output creation. Exit 0 means the individual
run passed its checks, 1 means execution/validation failed, and 2 means blocked
or incomplete. Inspect `evidence_kind` before interpreting any successful run.
The historical `integration-check --mode live-joint` remains blocked; use the
separate interface above for provider-based execution.

The `guarded_live_protocol` CTest runs real CPU/CUDA work with the test actuator.
For retained test archives:

```bash
OMP_NUM_THREADS=4 python tests/native_live.py \
  build/live-development/atlas_kmeans build/live-development/atlas_hotspot \
  results/live/software-test
```

## Remaining research acceptance

A software test is insufficient for full project acceptance. Physical actuation
and restoration, independent energy/timing qualification, frequency-only and
joint-control experiments, and the specified fixed-clock baselines still require
qualified hardware. The strict all-iteration CPU/GPU compute-overlap gate also
remains unmet. The experimental `gpu-first` native dispatch option is tested but
did not consistently improve overlap or runtime in the retained comparison, so
the default remains `cpu_first`. It cannot be combined with HotSpot halo
`overlap`. Do not count barrier waits or kernel submission as compute overlap.
