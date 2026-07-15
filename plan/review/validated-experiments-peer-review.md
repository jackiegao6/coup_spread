# Validated Experiments Peer Review

## Overall assessment

The validated-v2 study is suitable as simulation evidence for the current
paper. It is reproducible, repeated across five selection seeds, and tied to
the proved fixed-realization coupon model through exact tiny-instance tests.
It must be described as experiments on real graph topologies with controlled
synthetic behavior probabilities, not as a field study.

## Spec-compliance review

- PASS: 360 configuration files, 2,610 method-level raw rows, and 522
  five-run summaries match the locked protocol.
- PASS: every accepted row is marked `REAL_EXPERIMENT` and
  `validated-v2.2`; no mock or synthetic planning file is read by the runner
  or plotting script.
- PASS: all stochastic selections use five seeds; every allocation is
  evaluated with 10,000 forward realizations.
- PASS: CIM-RIS and IC-RIS use the same preregistered sample budget.
- PASS: full Douban runtime and conditioned-sampler ablation were rerun from
  validated-v2 allocations.

## Quality review

- The strongest result is proximity to MC-Greedy on Netscience (1.1--4.4%)
  and consistent gains over degree/PageRank at all 72 configurations.
- The data do not support universal dominance over IC-RIS. IC-RIS is tied in
  balanced, 0.7% higher in forwarding-heavy, and 1.4% lower in
  adoption-heavy. The manuscript reports this negative result.
- Adoption-heavy local heuristics are competitive; the manuscript reports
  that 1Hop-Sort is 0.6% higher on average.
- The conditioned sampler reduces zero-contribution work and has lower
  observed variance-time products, but its 30-batch variance estimates should
  not be overinterpreted at `k=200`, where conditioning is nearly vacuous.
- The runtime curve has a designed discontinuity because the sample budget is
  halved after `k=50`; the caption and prose state this explicitly.

## Residual risks

- Node behavior probabilities are generated from degree rather than learned
  from campaign logs, limiting external validity.
- MC-Greedy is computationally feasible only on Netscience.
- Experiments use the distinct-seed policy `c_v=1`; repeated placement is
  supported by the formulation but is not empirically studied.
- The target venue and its page limit remain unspecified.
