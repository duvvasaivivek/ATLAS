# Completion and release validation — 2026-11-10

This review records the release before the subsequent
[guarded live-control extension](live_control.md). Its experiment counts and
archives remain a historical snapshot; see the README for current test counts.

Full project acceptance is still incomplete. This pass adds reproducible delivery
and fixes a telemetry startup failure; it does not establish live frequency
control or physical energy savings on the current RTX 4050 Laptop GPU/WSL setup.

## Delivered changes

- Telemetry worker initialization now has its own bounded startup handshake.
  The worker is ready before native work or Nsight launches. The existing
  two-second query deadline remains in force; initialization has a separate
  ten-second deadline. Startup time remains inside the full command duration.
- `research-report` verifies and replays the performance, integration and
  qualification archives and generates standalone HTML plus machine-readable
  JSON. It includes full-command and workload estimates, paired uncertainty,
  source identities, missing evidence and explicit incomplete acceptance gates.
- `scripts/validate_release.py` runs the build/test/qualification/evaluation
  workflow with bounded commands, retained logs and explicit exit codes.
- `MANIFEST.in` packages native sources, tests, fixtures, specifications and
  upstream provenance. `scripts/package_release.py` builds the source package
  and wheel, installs the wheel into a separate environment, checks import
  location and CLI execution, and runs the extracted source's Python tests.
- [Reproducibility instructions](reproducibility.md) distinguish software
  installation, native builds, experiment tuning and hardware acceptance.

No neural-network training is required. Kmeans fits centroids during its workload
runs. Static-partition tuning uses separate inputs; the online controllers retain
their published or explicitly named extension equations.

## Validation evidence

The release workflow's [evidence directory](../results/completion/2026-11-10-release/)
retains commands, return codes, raw observations, dataset identities, profiler
captures, summaries and archive checksums.

- 163 Python tests pass, including startup deadlines, failure cleanup, report
  claim boundaries, missing benchmark pairs and HTML escaping.
- Nine native CTests and eight CUDA sanitizer checks pass.
- Offline frequency-controller validation passes.
- All 13 integrated trials preserve workload correctness, covering 21,611
  measured iterations, 326 allocation decisions and 102 ownership changes.
- The performance study has 400 successful trials: 40 tuning and 360 held-out
  trials, with ten randomized blocks per variant and workload size. It does not
  require a retained legacy executable, unlike the earlier 440-trial comparison.

The first integration attempt under `results/completion/2026-11-10/integration`
failed during the final workload's profile with a telemetry query timeout before
the first sample. That failure is retained. Initialization was previously charged
to the first query deadline. The startup-handshake change precedes the successful
complete integration run.

The initial isolated package test exposed a process-exit assertion race: SIGKILL
delivery to an orphan was checked immediately after waiting for its already-exited
leader. The test now checks for exit within a bounded half-second and handles
concurrent removal of `/proc/<pid>/stat`. This is a test timing correction; the
production process-group termination behavior and its deadlines are unchanged.
The failed package attempt remains under `build/package-release`.

## Measured allocation findings

Static tuning selects 8,193/32,771 and 16,385/65,539 CPU points for kmeans
(approximately 25%), and zero CPU rows for both HotSpot sizes.

Against matched GPU-only trials, the tuned kmeans workload intervals change by
-13.55% and -11.52%. Their pointwise 95% paired bootstrap intervals are
[-19.16%, -8.34%] and [-14.05%, -8.96%]. These are workload-interval findings;
full-command estimates, including startup and independent final validation,
are reported separately by `research-report`.

HotSpot cost-aware allocation is 176.21% and 150.10% slower over the workload
interval at the two sizes. GPU-only remains the tuning selection. These results
do not promote an adaptive policy to a universal default. Generated inputs,
automatic clocks, shared-host scheduling and lack of thermal isolation limit
generalization. No physical energy value is inferred from runtime.

## Remaining gates

1. Strict concurrency remains incomplete. The final profiles demonstrate compute
   overlap in 235/256 kmeans and 140/256 HotSpot iterations. All iterations have
   CPU regions and a correlated kernel; the misses have disjoint compute
   intervals. In 21 kmeans and 113 HotSpot misses the CPU computation ends before
   the GPU kernel starts; in three HotSpot misses it starts after the kernel ends.
   Launches and barrier waits have not been relabeled as useful computation.
2. Authoritative prior clock-policy readback and verified restoration remain
   unavailable. No clock setters are called. Live joint mode remains explicitly
   disabled and requires a qualified hardware adapter and validated live runner.
3. Loaded telemetry timing is unqualified. The release qualification observes
   two future samples among 102 samples and a roughly 675 ms host wall/monotonic
   offset span. No fitted offset is used to force acceptance.
4. GPU energy and device-event timing remain blocked. CPU package/governor
   interfaces are not exposed, and no wall meter is configured.
5. Fixed-clock allocation, frequency-only, complete-framework and joint static
   allocation/clock comparisons still need qualified hardware experiments.

The validation workflow therefore reports a capability-limited result and exit
code 2 after successful evidence generation. Packaging and passing correctness
tests do not close these hardware or concurrency gates. A different machine must
be checked for the actual required capabilities rather than assumed sufficient.
