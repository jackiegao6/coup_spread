# Adaptive Parameter-Sweep Peer Review

## Scope and verdict

This review covers the validated-v2.5 adaptive-grid runner, raw and aggregate
artifacts, the eight-panel appendix figure, and the revised interpretation.
The appendix passes the specification and reproducibility gates for a
descriptive robustness experiment. It does not establish universal dominance
or instantiate the theorem-level epsilon-delta certificate.

## Specification compliance

- The full prespecified grid retains both graphs, five transfer probabilities,
  four redemption shares, the complete `k=100` cross product, and the
  `r=0.60`, `k in {25,100,200}` budget slice.
- All six methods use unit seed capacity. CIM-RIS starts at 100,000 nested
  conditioned RR samples, reaches the prespecified observation floor, and then
  applies the uniform 0.5% stability rule with a 3-million hard cap. IC-RIS
  uses the same final sample count.
- Stability uses 5,000 fresh paired forward streams at each stage. Final
  method evaluation uses 10,000 streams disjoint from selection and all
  stability batches.
- All positive and negative cells, 11 hard-cap runs, and five unresolved runs
  are retained. MC-Greedy is absent and no exact-optimality claim is made.

## Independent checks

- The verifier reconstructs 300 atomic jobs, 1,800 raw rows, 360 method
  summaries, and 60 comparison summaries.
- Every stage uses the same RR stream prefix within a job; sample counts follow
  the required doubling/floor sequence; no stable post-floor stage is followed
  by another stage.
- Final evaluation stream IDs are disjoint from CIM-RIS, IC-RIS, and every
  stability stream. All allocations satisfy the unit-capacity constraint.
- Method summaries, strongest-baseline identities, relative differences, and
  SHA-256 entries are independently recomputed. Nine adaptive/core unit tests
  pass.
- Manuscript values match the aggregate: 33/40 higher-mean `k=100` cells,
  graph means of +0.75% and +2.20%, extrema of -0.69% and +4.75%, a 300,000
  median final sample count, and 295/300 stable runs.

## Quality review

- The prose compares against the strongest aggregate non-oracle baseline in
  every complete cell and reports the remaining unfavorable regions.
- "Higher mean" is used instead of significance language. Paired intervals
  are descriptive because the strongest baseline is selected after
  aggregation and no multiplicity-adjusted confirmatory test was prespecified.
- The figure exposes sample demand and unresolved stability instead of hiding
  hard cells. Common scales are used within each heatmap family, and both
  positive and negative spread differences remain visible.
- Duplicate-reduction correlation is reported descriptively as 0.43 and is
  interpreted as a partial mechanism, not a causal explanation.

## Residual risks

- Five of 300 selections remain unstable at the hard cap and add uncertainty
  to their aggregate cells, although all are retained and visibly flagged.
- The empirical stopping check compares adjacent allocations on 5,000 forward
  realizations; it is not the conservative theorem-level sample bound.
- The study uses two real graph topologies with controlled behavior
  probabilities, not observed coupon-campaign parameters.
- Five selection runs limit precision, and strongest-baseline intervals are
  unsuitable for unqualified statistical-significance claims.
