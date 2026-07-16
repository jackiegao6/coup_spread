# Appendix Parameter-Sweep Peer Review

## Scope and verdict

This review covers the validated-v2.4 parameter-grid runner, raw and aggregate
artifacts, Figure 8, and the appendix interpretation. The appendix passes the
specification and quality gates for a descriptive robustness experiment. It
does not support a universal dominance or optimality claim.

## Specification compliance

- The full prespecified grid contains both graph instances, five transfer
  probabilities, four redemption shares, and the complete `k=100` cross
  product. The `r=0.60` slice additionally contains `k=25` and `k=200`.
- All six methods use the distinct-seed policy. CIM-RIS and IC-RIS each use
  20,000 RR samples, and all allocations use 2,000 fresh forward evaluations
  under common cell-level streams.
- The five selection seeds are 20260715--20260719. Every accepted row is marked
  `REAL_EXPERIMENT` under `validated-v2.4-appendix-grid`.
- All favorable, tied, and unfavorable cells remain in the CSVs and figure.
  MC-Greedy is absent and the appendix makes no exact-optimality claim.

## Independent checks

- The verifier reconstructed 1,800 raw rows, 360 method summaries, and 60
  comparison summaries from 40 completed job files.
- The checksum manifest validates `raw.csv`, `method_summary.csv`, and
  `comparison_summary.csv`.
- Five-run paired intervals use the Student-t critical value for four degrees
  of freedom. The plot reads only rows with the expected status and protocol.
- Manuscript values were independently matched to the summaries: 16/20 versus
  1/20 higher-mean cells at `k=100`, the reported extrema, and the three-budget
  NetFacebookEgo pattern.
- The duplicate-reduction correlation is 0.03 across all 40 `k=100` cells, so
  the appendix correctly rejects overlap reduction as a general explanation.

## Quality review

- The prose reports the strongest baseline in each complete cell, not a
  favorable fixed comparator, and explicitly discusses the largest negative
  NetFacebookEgo cell and the broadly negative Netscience result.
- "Higher mean" is used instead of statistical-significance language. The
  confidence intervals are descriptive; no multiplicity-adjusted hypothesis
  test is claimed after selecting the strongest aggregate baseline.
- Figure 8 is legible in the compiled two-column PDF, preserves a common
  diverging scale within each heatmap family, and appears before the balanced
  references. All fonts are embedded.

## Residual risks

- The study covers two real graph topologies with controlled synthetic
  behavior probabilities, not observed campaign parameters.
- Five selection runs give limited precision, and the strongest-baseline
  choice makes the displayed intervals unsuitable for confirmatory inference.
- The fixed 20,000-sample budget is an empirical setting, not the theorem's
  epsilon-delta stopping certificate.
- Results are network dependent: Netscience is predominantly unfavorable, so
  claims must remain conditional on the observed parameter regions.

