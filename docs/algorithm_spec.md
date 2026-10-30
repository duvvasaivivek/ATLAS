# Controller specification v1.0

Status: Phase 0 specification, 2026-10-31. Algorithms are not implemented by this document.

Traceability: [paper mapping](paper_mapping.md). Every default or edge-case choice identified as **ATLAS** is an implementation decision, not an experimental result from the paper.

## 1. Shared allocation contract

Inputs are `run_id`, `workload_epoch`, `iteration_id`, actual CPU fraction `r`, CPU/GPU branch times in seconds, legal CPU work counts, total logical work count, and observation-validity flags. The fraction describes actual assigned work after rounding, not the previous requested fraction.

The dispatcher converts a requested ratio to the nearest legal CPU count; equal-distance ties select the smaller count. GPU work is the exact complement. Counts can represent points for kmeans or rows for HotSpot. Legal counts and halo requirements come from the workload, never the allocation policy. Log both requested and realized ratios.

An interior observation requires `0 < r < 1`, positive finite times, a common timing origin, equivalent workload amount/phase, and valid completion of both branches. `NaN`, infinite values, negative times, failed computation, and stale or duplicate observations are ineligible. Do not update policy state on an ineligible observation. Emit `hold_invalid_observation`; sustained invalid observations invalidate the experiment according to its predeclared run budget.

An empty branch has no measured throughput. If a throughput-based policy starts at an endpoint, emit `needs_interior_probe`; the dispatcher uses the legal interior count nearest 50% for one real probe. Count its work, time, and energy. If no legal interior split exists, return `unsupported_partition`; endpoint-only baselines remain available. Never replace a missing branch time with an epsilon.

Defaults below are an **ATLAS development profile** and must appear in each run manifest:

- Initial requested CPU share: `0.50`.
- Fixed step: `0.05` (five percentage points); the paper's later 3% comparison is a distinct profile with `0.03`.
- Adaptive initial step: `0.15`; ratio precision: `0.01` (one percentage point).
- Diagnostic balance tolerance: `0.05` relative imbalance, sustained for three consecutive eligible interior observations.
- Allocation-decision budget: 200 eligible decisions per run. This limits pathological diagnostic runs; it is not a convergence claim. Each experiment must also set a finite total-iteration or wall-time budget to bound missing-observation cases.

Literal policy decisions use the sign of `D=t_cpu-t_gpu`, with exact equality meaning hold. The 5% diagnostic tolerance never silently changes a literal update. Floating-point comparisons use finite double-precision values; only exact ties trigger the tie rule. No hidden smoothing or near-tie threshold is allowed in the literal policies.

A repeated realized ratio due to quantization emits `hold_quantized`. The runtime continues workload execution, records that the policy is stationary, and can still measure imbalance. Stationary, balanced, precision-limited, and budget-exhausted are separate outcomes.

At a new independent run, reset all policy state. A phase change inside a run is metadata; the reference policies retain state and history. Automatic reset or filtering on phase changes is an extension. Corrected bisection is an exception: a declared phase change invalidates its stationary bracket and requires a newly logged bracketing procedure.

## 2. `paper_simple`

Source: M02, §5.2.1.

1. If `D=0`, hold with `hold_equal_time`.
2. Propose `r-step` if `D>0`, otherwise `r+step`; clamp to [0,1], then map to the legal work count.
3. For an eligible interior observation, predict `t_cpu_next=t_cpu*r_next/r` and `t_gpu_next=t_gpu*(1-r_next)/(1-r)`.
4. If the predicted difference has the opposite strict sign to `D`, retain the current split and emit `hold_predicted_oscillation`. A predicted difference of zero is accepted.
5. Otherwise emit the candidate. An endpoint has predicted zero work on its empty branch, not a fabricated endpoint throughput measurement.

The policy has no claim of reaching a globally optimal split. It can hold while still imbalanced because the available step is coarse. Balance monitoring and the experiment budget do not relabel this as convergence.

## 3. `paper_adaptive`

Source: M03, §5.2.2. **ATLAS interpretation:** detect observed timing-sign reversal, not predicted reversal; do not apply the simple heuristic's predictive hold.

State: current step, previous nonzero timing-error sign, and adjustment-enabled flag.

1. On each eligible observation with nonzero `D`, compare its sign with the previous eligible nonzero sign.
2. If a previous nonzero sign exists and the signs differ, halve the step exactly as a ratio. For example, 0.15 becomes 0.075, not an integer-rounded percent. After this comparison always store the current nonzero sign, including on the first observation.
3. If the step is now strictly below precision, freeze adjustment and emit `stop_precision`. This is precision exhaustion, not evidence of balance.
4. Otherwise propose `r-step` for positive `D` or `r+step` for negative `D`, clamp, and map to a legal count. For zero `D`, hold and retain the previous nonzero sign.
5. A quantized hold does not itself count as a timing reversal. Frozen adjustment persists for the remainder of the independent run; a reactive restart is a separately named extension.

## 4. `paper_binary_literal`

Source: M04, §5.2.3.

For `D<0`, propose `(1+r)/2`; for `D>0`, propose `r/2`; for zero, hold. It does not maintain a bracket.

**ATLAS interpretation of the stopping text:** after an applied candidate has produced an eligible observation, compare its realized ratio to the previous realized ratio. If the nonzero change is strictly smaller than precision, stop adjustment with `stop_small_change`. Zero movement due to quantization is `hold_quantized`, not proof of balanced timing. Endpoints requiring a throughput observation use the shared probe rule. The decision budget stops persistent cycles with `budget_exhausted`.

Do not add an undisclosed bracket or claim logarithmic convergence for this ID. Record the full sequence so its published-text limitation can be studied.

## 5. `bracketed_bisection`

Source: M04; an **ATLAS correction**, not a literal paper policy.

The function is `D(r)=t_cpu(r)-t_gpu(r)`. Establish `lo<hi` using measured eligible interior candidates with `D(lo)<=0` and `D(hi)>=0` under the same stationary workload and fixed clock policy. Start with the lowest and highest legal interior candidates; zero at either candidate is an exact root. If their signs do not bracket a root, return `no_bracket` with their measurements. Do not infer an optimum or fabricate an endpoint branch time.

For a valid bracket:

1. Propose its midpoint mapped to a legal interior candidate strictly between the bracket endpoints. If none exists, stop with `stop_resolution`.
2. Measure at that candidate; positive `D` replaces `hi`, negative `D` replaces `lo`, and zero gives `stop_exact_root`.
3. If bracket width is at most precision, stop with `stop_bracket_width`. For the continued workload use the measured bracket endpoint with smaller absolute `D`; ties choose the smaller ratio. An unmeasured midpoint is only an estimate, not a measured final result.
4. Restart bracket establishment after a declared workload phase change or an effective capacity change, including a changed GPU clock setting or CPU policy. A constant frequency-controller ID does not make changing actual frequencies stationary. Detect observed sign inconsistencies and return `invalid_bracket`; noise can invalidate the premise even when not detected.

The continuous/noiseless bound counts midpoint updates, excluding initial bracketing measurements and additional executions. With work-count rounding, report actual bracket widths and decisions; do not assert the unrounded bound for every grid. Use this correction primarily for fixed-setting allocation studies; a continuously adapting frequency loop does not inherit the stationary-search guarantee.

## 6. `paper_proportional`

Source: M05, Equation 5.

For an eligible interior observation:

```text
r_target = (r * t_gpu) / (r * t_gpu + (1-r) * t_cpu)
```

The denominator is positive by contract. Map the result to a legal ratio; record the unrounded target. Equal times reproduce `r`. The shared probe rule handles endpoints. No damping, moving average, maximum ratio change, or predicted-oscillation hold is applied under this ID.

**ATLAS stopping choice:** continue to evaluate each eligible observation until the experiment ends or its budget is reached; record sustained balance without freezing adaptation. This preserves the distinction between the paper's proportional update and an added stability heuristic.

Reserve `robust_proportional_v1` for a later fully specified filtered/damped extension. It has no default implementation or claims in v1.0.

## 7. `paper_additive` frequency policy

Source: M06, Equations 1-4. Input core and memory observations must be finite normalized fractions in [0,1], from the same eligible observation window, with their actual metric semantics recorded. Missing values, unsupported sensors, or incompatible time windows cause `hold_invalid_observation` and no history update.

Enumerate unique selected frequencies in ascending MHz. For a domain with two or more levels:

```text
u_mean(f) = (f - f_min) / (f_max - f_min)
energy_loss(u, f) = max(0, u_mean(f) - u)
performance_loss(u, f) = max(0, u - u_mean(f))
loss(u, f, alpha) = alpha*energy_loss + (1-alpha)*performance_loss
L[i,j] = gamma*loss_core[i] + (1-gamma)*loss_memory[j]
W_next[i,j] = W[i,j] + 1 - (1-beta)*L[i,j]
```

Parameters: `alpha_core=0.15`, `alpha_memory=0.02`, `gamma=0.3`, `beta=0.2`. Require alpha and gamma in [0,1] and beta strictly in (0,1). The parameters are historical settings; changes create a recorded configuration, not an unreported recalibration.

**ATLAS choices:**

- Initialize every feasible pair's weight to zero. Update every feasible candidate on each valid new observation, not only the active pair.
- Use only explicitly enumerated feasible pairs. An empty set is unsupported. A domain with one level has a constant loss contribution set to zero; label the run a reduced-actuation adaptation, not full two-domain replication.
- Paper-style initialization requests the feasible pair with lowest core frequency, then lowest memory frequency, after the actuator is verified. Record the initial request and measured frequencies. A platform-default initialization is another profile.
- Select the maximum-weight pair. On exact weight ties retain the current selected pair if tied; otherwise select the tied pair with highest core frequency, then highest memory frequency. This deterministic tie preference has no claimed optimality.
- Store weights in double precision without [0,1] clipping. Subtracting the same constant from all weights is permitted only as a logged numerical transformation with equivalence tests; no unequal rescaling, decay, or reset is implicit.
- Record requested, acknowledged, and observed frequencies separately. No feasible setting or successful API call alone proves the hardware ran at that setting.
- Keep full history across allocation updates and phase changes within the same run. A new independent run resets history.

Reserve `discounted_loss_v1` for a later explicit forgetting rule; do not describe beta in the printed additive rule as a forgetting coefficient.

## 8. Observation and decision identity

Every decision records specification version, policy ID, configuration hash, run/iteration or sample ID, input-validity status, input references, pre/post state, requested setting, realized setting when available, and reason. Duplicate observations are ignored without another state update. Allocation policy state uses only observations from its own run and workload epoch identity; cross-run data mixing is an error.

The runtime owns actuation, hardware restoration, and workload dispatch. Controllers are pure state transitions. A successful controller calculation does not establish measurement validity or hardware success.

## 9. Reference calculations for later tests

Machine-readable numbers are in [specification fixtures](spec_fixtures.json). They are specification examples, not implemented-controller test results.

- **A01:** At `r=0.1`, times 1 and 1.4, step 0.05, the proposed 0.15 predicts CPU 1.5 and GPU 1.322222... . The sign reverses, so `paper_simple` holds 0.1.
- **A02:** At `r=0.3`, times 6 and 3, `paper_proportional` gives `3/17 = 0.17647058823529413`.
- **A03:** At `r=0.15`, times 0.6 and 0.8, previous error sign positive, and step 0.15, `paper_adaptive` halves to 0.075 and proposes 0.225.
- **A04:** With `D(r)=r/0.8-(1-r)/0.2`, the literal binary sequence begins `[0.5, 0.75, 0.875, 0.4375, 0.71875, 0.859375, 0.4296875]`.
- **A05:** For the same analytic oracle on continuous bracket [0,1], seven midpoint updates leave `[0.796875, 0.8046875]`, width 0.0078125. This synthetic endpoint oracle is not a hardware throughput observation. Bracketing work is excluded from this update count.
- **F01:** Core observation 0.4 and target 0.6 give loss 0.03. Memory observation 0.9 and target 0.5 give loss 0.392. Combined loss is 0.2834; the first additive weight increment is 0.77328.
- **F02:** With equal initial weights, any fixed beta in (0,1) ranks the same fixed cumulative losses identically. Repeated F01 updates make the weight exceed 1; clipping would alter the specified rule.
- **I01:** Invalid timing, an empty branch, or missing memory observation cannot enter the corresponding formula as zero. State remains unchanged except for the runtime's separately recorded validity/error counters.

Phase 0 checks arithmetic and logical consequences. Unit, integration, concurrency, and hardware acceptance tests are required in their implementation phases.
