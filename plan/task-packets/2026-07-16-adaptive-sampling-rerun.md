## Task Packet

- Scope: replace underpowered fixed-sample appendix selections with a
  prespecified, resumable doubling/stability protocol and rerun the complete
  two-graph parameter grid without filtering outcomes.
- Files to read: validated simulator and joint-RR implementation; v2.4 grid
  runner/results; weak-gain audit and diagnostics; current manuscript.
- Files allowed to edit: new v2.5 runner, verifier, result directory, plotting
  script/output, protocol/review records, and manuscript only after the full
  rerun and verification pass.
- Required skills: paper-orchestration, experiment-results-planning,
  statistical-analysis, figures-python, peer-review, latex-output, and
  verification.
- Evidence inputs: Netscience and NetFacebookEgo; the complete locked v2.4
  transfer/redemption grid and budget slice; five selection seeds; all existing
  non-oracle baselines.
- Sampling rule: start at 100,000 joint samples and double. The minimum terminal
  budget is the smaller of 1.5 million and the budget needed for 10 expected
  nonempty current-index RR sets per candidate, rounded upward to 50,000.
  After reaching that budget, stop only if two consecutive allocations differ
  by at most 0.5% under a fresh 5,000-realization paired validation batch;
  otherwise continue doubling to a hard cap of 3 million. The cap and any
  unresolved instability must be reported.
- Final evaluation: 10,000 fresh forward realizations, independent of all
  selection and stability validation. IC-RIS receives the same final RR sample
  count as CIM-RIS in each run.
- Rejection checks: no cell/seed may be omitted after observation; no result may
  reuse v2.4 final means; training RR spread cannot be reported as final spread;
  no significance or optimality claim; all cap hits retained.
- Validation commands: existing exact and claim tests; v2.5 row/protocol/stage
  verifier; deterministic aggregation and plotting; checksum manifest; full
  LaTeX compilation and visual inspection after manuscript integration.
