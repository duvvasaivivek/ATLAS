# GreenGPU paper mapping and interpretation register

Specification version: 1.0

Prepared: 2026-10-31

Source: [GreenGPU.pdf](../GreenGPU.pdf), DOI `10.1016/j.suscom.2016.05.002`.

The source PDF SHA-256 is `064f486a8000a60b1b93483461d793222c5f53227f080c3dfd476d680d2ca341`.
Journal page numbers run from 21 to 33; PDF page number = journal page number minus 20.

This register distinguishes **Paper** (a published description or result), **Derived** (a mathematical consequence checked during Phase 0), and **ATLAS decision** (an explicit implementation or evaluation choice). The [algorithm specification](algorithm_spec.md) and [measurement specification](measurement_spec.md) make these decisions executable in later phases.

## M01 - Two-tier system

**Paper:** Section 4, journal pp. 24-25, Figure 3. Allocate work at application iteration barriers; scale frequencies on a faster loop. Prior iteration behavior is used to predict the next iteration. CPU scaling uses Linux's ondemand governor. CPU-side memory DVFS is future work.

**ATLAS decision:** Use one CPU host and one GPU, with independently recorded CPU policy, allocation policy, and GPU frequency policy. An allocation epoch can contain multiple logical iterations, but their work and barriers must be preserved. Clock/voltage controls are capability-dependent; GPU frequency scaling does not establish that voltage was directly controlled.

**Validation destination:** Phase 3 concurrency trace; Phase 9 integrated state transitions and interaction experiments.

## M02 - Simple allocation heuristic

**Paper:** Section 5.2.1, p. 26; §7.2 and Figure 7, p. 29. The CPU share decreases if its branch is slower, and increases if the GPU branch is slower. Branch times include data transfer. Predictive linear scaling suppresses an adjustment if it would reverse the timing order. A 5 percentage-point step is used in the primary allocation study; the later comparison uses other step sizes, including 3 percentage points.

**ATLAS decision:** `paper_simple` preserves this core rule. Ratios, granularity, equality handling, endpoint probes, and diagnostic stopping are specified separately. A held split is not necessarily an energy optimum or balanced split.

**Validation destination:** Phase 5 safeguard and migration tests; fixture A01.

## M03 - Adaptive allocation heuristic

**Paper:** Section 5.2.2, p. 26; Figures 10 and 14, pp. 30 and 32. Begin with a larger step, then halve it when allocation oscillates, stopping adjustment when it falls below a precision threshold. Examples differ in rounding of half steps.

**Ambiguity:** Applying the simple heuristic's predictive hold unchanged can prevent the observed oscillation that this algorithm uses to halve its step. Exact rounding and whether oscillation is predicted or observed are not fully specified.

**ATLAS decision:** `paper_adaptive` uses observed changes in the timing-error sign, permits overshoot, and halves continuous ratio steps without integer-percent rounding. This interpretation is logged, not presented as recovered author source code. A predictive-halving alternative would require another policy ID.

**Validation destination:** Phase 5 sign-reversal, precision, and quantization tests; fixture A03.

## M04 - Printed binary search and corrected bisection

**Paper:** Section 5.2.3, p. 26; Figures 11, 13-15, pp. 31-32. The printed recurrence moves toward 1 or 0 by halving the distance to that endpoint and claims a logarithmic convergence bound. The stopping description uses consecutive allocation changes.

**Derived:** For stationary times `t_cpu=r/0.8` and `t_gpu=(1-r)/0.2`, the printed rule cycles around the balance point. It does not retain a bracket. The advertised bound does not follow from those two update formulas.

**ATLAS decision:** Keep `paper_binary_literal` as a diagnostic implementation of the text, with bounded evaluation. Add `bracketed_bisection` as a correction. Its width bound is `ceil(log2((hi-lo)/precision))` midpoint updates for an initially valid continuous bracket; endpoint establishment, a custom starting probe, and final execution are additional costs. Quantization, noise, and non-monotone timing differences require separate outcomes.

**Validation destination:** Phase 5 counterexample and stationary-bracket checks; fixtures A04 and A05. This is a finding about the printed description, not proof of how the original authors' code ran.

## M05 - Improved Binary Search / proportional allocation

**Paper:** Section 5.2.4, p. 26, Equation 5; Figure 12, p. 31. Adjust using the magnitude of the timing difference. The reported 4-5 iteration examples are results on particular configurations.

**Derived:** Rearranging Equation 5 gives `r_next = r*t_gpu / (r*t_gpu + (1-r)*t_cpu)`. If times are proportional to work with constant rates and no fixed overhead, this gives the balanced share in one ideal update.

**ATLAS decision:** Call this `paper_proportional` to avoid implying it is bracketed binary search. Reject unidentifiable throughput observations. Filtering, damping, and automatic phase resets belong to a separately named extension, not this rule.

**Validation destination:** Phase 5 analytic checks and real convergence experiments; fixture A02. No universal iteration bound is asserted.

## M06 - GPU loss and frequency selection

**Paper:** Section 5.1, p. 25, Table 1, Algorithm 1, Equations 1-4. Linearly relate selected frequencies to target utilizations; use asymmetric losses; combine core and memory loss; update all candidate weights; select a maximum-weight pair. Parameters: `alpha_core=0.15`, `alpha_memory=0.02`, core-loss mixing coefficient `0.3`, and `beta=0.2`.

**Derived:** With equal initialization, the additive Equation 4 ranks cumulative losses. Fixed beta in (0,1) does not change those rankings under exact arithmetic. Weights can grow above 1. With a full Cartesian candidate set, equal initial weights, and additive loss, the cumulative objective separates into core and memory terms; there is no explicit interaction-loss term.

**ATLAS decision:** `paper_additive` uses the printed additive rule with unbounded floating-point weights, no clipping, and no implicit decay. The mixing coefficient is named `gamma` in ATLAS. Linear mapping by actual frequency value is an explicit interpretation when spacing is uneven. A feasible-pair subset and utilization substitutions are recorded adaptations.

**Validation destination:** Phase 7 hand-calculated loss, beta-ranking, tie, history, and invalid-observation checks; fixture F01. Phase 8 establishes physical effects separately.

## M07 - Timescales and historical frequency levels

**Paper:** Section 4, pp. 24-25, discusses at least 40 frequency periods per allocation interval for 36 core/memory pairs. Section 7.1, p. 28, uses a 3-second frequency interval in a trace. Section 6, p. 26, lists core MHz `[576, 513, 466, 411, 356, 297]` and memory MHz `[900, 820, 740, 660, 580, 500]`.

**Derived:** Neither a finite candidate count nor these examples proves convergence of a cumulative-loss feedback controller in 36 samples. The additive policy does not physically explore every pair merely because it updates every weight.

**ATLAS decision:** Preserve these as historical reference configurations, never as current hardware commands. Choose actual sample periods, feasible pairs, and settling rules after capability/measurement phases. Do not sleep to make a short workload artificially satisfy a 120-second allocation interval.

## M08 - Workloads and implementation evidence

**Paper:** Section 6, pp. 26-27, Tables 2-3 and Figure 4. CUDA and CPU implementations run in pthreads; the GPU frequency daemon is Python. Transfers are included. Table 2 specifies 988,040 kmeans points and a 2048-by-2048 HotSpot grid with 600 iterations, among other workloads. Testbed A uses GeForce 8800 GTX and AMD Phenom II; Testbed B uses Tesla 2070 and Intel Xeon E3-1200. Frequency control is unavailable on Testbed B in their setup.

**ATLAS decision:** Python orchestration with a native CPU/OpenMP and CUDA execution backend is the intended route, conditional on Phase 1 toolchain evidence. These workload sizes are reference sizes, not complete datasets: dimensions, initialization, coefficients, source version, precision, and licenses must be recorded during workload implementation. Do not assert exact historical reproduction from size alone.

## M09 - Energy and slowdown results

**Paper:** Section 7.1, p. 28, Figure 6: GPU frequency scaling saves 5.97% mean GPU energy, up to 14.53%; 29.2% is mean dynamic GPU energy saving after idle subtraction. The reported slowdown against best-performance is 2.95%. Figure 6c's 12.48% average CPU/GPU energy saving is emulated, not a fully measured joint-control result.

**Paper:** Sections 7.2-7.3, pp. 29-30, Figures 7-8: static kmeans energy minimum at 15% CPU versus heuristic convergence at 20%; hotspot reaches the sampled 50/50 energy-minimum split. The complete framework saves 21.04% on average across kmeans and hotspot versus GPU-only at peak frequencies. Its 1.7% slowdown is relative to allocation-only. The reported allocation-only 5.45% runtime difference uses the optimal-division comparator in §7.2.

**ATLAS decision:** Preserve every baseline and energy boundary in result metadata. None of these percentages is a pass/fail target for ATLAS. The measured static sweep distinguishes balanced execution time from minimum energy. Testbed B's improved allocation results must not be merged into Testbed A's headline result.

## M10 - Measurement domains and utilization semantics

**Paper:** Section 6 and Figure 4, p. 27: separate wall meters cover the host-side box and the GPU-side supply on Testbed A; Testbed B uses a whole-desktop meter. Section 3.1, p. 23, describes memory utilization as achieved bandwidth divided by peak bandwidth.

**External reference:** [NVIDIA's archived NVML utilization definition](https://docs.nvidia.com/deploy/archive/R535/nvml-api/structnvmlUtilization__t.html) describes time spent executing kernels or accessing device memory. Memory activity, bandwidth saturation, and allocated VRAM fraction must not share an unlabeled `utilization` field.

**ATLAS decision:** Raw metric semantics and physical energy scope are required fields. CPU-package plus GPU-board energy is not the same domain as wall energy; it is also not the paper's host-side wall measurement. Current sensor availability remains a Phase 1 question.

## Initial hypotheses

- **H1:** At fixed clock and CPU policies, dynamic division can reduce waiting and energy relative to GPU-only for some workloads and sizes. Compare runtime and total measured energy, including adaptation.
- **H2:** At fixed placement, the frequency policy can reduce energy, with a measurable runtime tradeoff. Compare only supported physical clock controls; a replay does not test this hypothesis.
- **H3:** Joint control can outperform allocation-only and frequency-only on some configurations. Hold other policy factors constant and report the two comparisons separately.
- **H4:** The proportional policy may settle faster than a fixed-step heuristic under approximately linear timing, but transfers, quantization, or phase changes can defeat that advantage.
- **H5:** A matched static energy sweep may identify a lower-energy split than time balancing. A negative result for dynamic control is evidence, not a failing correctness test.

These hypotheses are frozen before implementation. Their experimental evaluation belongs to later phases; Phase 0 establishes no hardware benefit.
