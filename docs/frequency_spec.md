# Phase 7 offline frequency contract

Version: `frequency-1.0`, 2026-10-31. Implements M06 in the
[paper mapping](paper_mapping.md) and Section 7 of the
[controller specification](algorithm_spec.md). This version has no actuator.

## Literal policy

`paper_additive` implements Equations 1–4 with defaults
`alpha_core=0.15`, `alpha_memory=0.02`, `gamma=0.3`, `beta=0.2`.
For each domain, the ATLAS target interpretation is
`target(f)=(f-min_frequency)/(max_frequency-min_frequency)`, using MHz values,
not the ordinal position of a frequency in a list.

The domain loss is `alpha*max(0,target-u)+(1-alpha)*max(0,u-target)`.
The pair loss is `L=gamma*core_loss+(1-gamma)*memory_loss`.
Every eligible observation updates every candidate:

```text
W_next = W + 1 - (1-beta)*L
W_initial = 0
```

Configurations contain explicit positive integer `(core_MHz,memory_MHz)` pairs.
Pairs are sorted lexicographically; duplicates are rejected. No Cartesian
completion is performed. Empty candidates produce `unsupported_no_candidates`.
A single-level domain contributes zero loss, and decisions carry
`reduced_actuation=true`. Both utilization fields are still required, even for
a single candidate or zero domain mixing coefficient.

The initial logical choice is the lowest core, then lowest memory pair. Select
the greatest weight; exact ties retain the current choice when possible,
otherwise choose the highest core, then highest memory pair. This initialization
and tie convention are ATLAS choices. They make no hardware request.

Scores are Python double precision floats. They are never clipped, normalized,
or reset within a run; this implementation uses no common-offset transformation.
All history survives workload/allocation epochs. A new independent run starts
with zero scores. A configuration hash binds state to a policy, ordered candidate
set, parameters, provenance, and specification version.

In exact arithmetic, after `t` eligible observations:

```text
W_i(t) = t - (1-beta) * sum_s L_i(s)
```

Thus every fixed `0<beta<1` ranks cumulative losses identically under equal
initialization. Beta is not a forgetting factor. Large offsets, nearly equal
loss sums, and beta extremely close to one can make floating-point rankings
ambiguous; the diagnostic verifies well-separated fixtures, not a universal
floating-point invariance claim. A 10,000-update test checks the unbounded-score
rule against a hand result; an independent Decimal oracle checks varied losses.

## Optional forgetting extension

`discounted_loss_v1` is an explicitly different ATLAS policy:

```text
score_next_i = discount * score_i - L_i
score_initial_i = 0
discount = 0.95, strictly between zero and one
```

Select the maximum score using the same tie rule. A loss from `k` eligible
observations ago contributes `discount**k` of its original magnitude. Invalid
observations do not advance this observation-count discount. Beta is retained
in the shared configuration for traceability but is unused by this extension.
No timing/energy advantage on hardware is claimed by comparing these policies.

## Observation and state contract

Observations have a run ID, globally increasing nonnegative integer sample ID,
nondecreasing integer epoch, source reference, validity flag, metric semantics,
and finite core/memory fractions in `[0,1]`. Booleans are not numeric inputs.
Epoch changes retain history. Cross-run, stale epoch, missing memory, invalid
values, incompatible windows, unknown semantics, duplicate/stale sample IDs,
and overlapping/stale windows cause a hold with unchanged state. Rejected
nonfinite input is represented by an explicit tagged value in strict JSON.

The two permitted window contracts are:

- `synthetic_interval`: matching core/memory `[start,end)` intervals in integer
  model ticks, `0<=start<end`, with nonoverlapping accepted intervals.
- `nvml_buffer_timestamp_us`: an exact shared timestamp from the core and
  memory NVML sample buffers. `start=null` because integration interval bounds
  are not exposed. This is an **offline co-sampling assumption**, not proof of
  sensor integration alignment, latency, live cadence, or measurement quality.

The NVML adapter normalizes percentages by 100. Core utilization is kernel busy
time; memory utilization is time spent reading/writing device memory, not memory
allocation or fraction of peak bandwidth. See NVIDIA's
[utilization documentation](https://docs.nvidia.com/deploy/nvidia-smi/index.html#utilization)
and [sample-buffer API](https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceQueries.html).

Capture retains raw samples, errors, ordinary rates, clock observations, device
identity, and discovery results. Only equal driver timestamps present in both
buffers and inside the host wall-clock capture bounds are joined. Repeated
identical samples are deduplicated; conflicting values for a timestamp remove
that domain's sample. Wall-clock discontinuities can exclude data; no interpolation
or inferred timestamps are used. Newer API acquisition time never makes a stale
buffer sample fresh. Ordinary rates without sample timestamps are ineligible.

Supported graphics clocks are queried separately for each supported memory
clock. All returned pairs are retained. The offline diagnostic selects the
lowest, middle, and highest reported core level per memory level, deduplicating
indices. Its provenance is `hardware_reported_unverified`; permission and realized
actuation are untested. The trace runs no workload and may contain only idle
samples. Separately acquired observed clocks are not attached to utilization
samples as if their windows were synchronized.

Every decision records version, policy, configuration hash, complete input,
validity/errors, pre/post state, candidate losses, reason, recommendation, and
reduced-domain status. `requested_pair` and `acknowledged_pair` remain null;
`actuation=offline_none`. Observed clocks, if supplied by a future qualified
adapter, remain distinct from recommendations.

## Diagnostics, evidence, and replay

`frequency-check` runs 23 deterministic cases: eight scenarios for each policy,
three beta variants, and four domain/candidate edge cases. The scenarios include
idle, saturated, core-heavy, memory-heavy, phase change, seeded noise, invalid
data gaps, and a causal synthetic feedback model. None assumes RTX 4050 clocks.

The feedback model applies the previous recommendation in simulation. Its
dimensionless capacities are `c=core/1800`, `m=memory/6000`; demands are `(0.65,0.15)`
then `(0.15,0.65)`. Observed busy fractions are `min(1,demand/capacity)`.
Assumed runtime is `max(core_demand/c,memory_demand/m)` and assumed power is
`10+20*c**3+5*m**3`; energy is their product. These are illustrative functions,
not calibrated physical units or performance/energy predictions for hardware.

Archives contain the manifest, source hashes, raw observations, full decisions,
summary, and SHA-256 inventory. Creation is exclusive. Replay verifies the exact
inventory and checksums, regenerates the versioned synthetic matrix, reconstructs
hardware joins when present, and compares every decision and summary. Checksums
detect accidental changes; they are not digital signatures. Replay does not
require current hardware. Figures are exported outside the immutable archive.

The Phase 2–6 traces lack paired memory utilization and cannot be repurposed as
valid Tier 2 inputs. The optional new read-only capture addresses offline decision
inspection only. Live cadence, setters, restoration, fixed-frequency sweeps, and
hardware effects remain Phase 8 work. Earlier energy qualification limits remain.
