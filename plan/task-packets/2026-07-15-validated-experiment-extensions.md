## Task Packet

- Scope: Close the three empirical risks identified after the validated-v2.2
  rebuild: limited strong-reference coverage, no capacity/repeated-placement
  experiment, and no confirmatory RR-sample sensitivity result.
- Files to read: `paper-v2 copy.tex`; `experiments/run_real_submission.py`;
  validated-v2.2 raw/summary files; experiment protocol, traceability map, and
  peer-review record.
- Files allowed to edit: `experiments/`; `figures/experiments/`;
  `paper-v2 copy.tex`; and experiment/review records under `plan/`, `tables/`,
  and `figures/`.
- Required skills: paper-orchestration, experiment-results-planning,
  statistical-analysis, figures-python, peer-review, latex-output, and
  verification.
- Evidence/data inputs: validated-v2.2 allocations; real graph topologies;
  the allocation-form objective and transition model in the manuscript.
- Required artifacts: high-precision MC-Greedy benchmark with independent
  evaluation streams; capacity ablation; sample-sensitivity raw and summary
  CSVs; tests; one compact publication figure; revised result claims; updated
  manifests and review.
- Rejection checks: do not choose networks, scenarios, capacities, budgets, or
  sample counts after seeing outcomes; do not call MC-Greedy the global
  optimum; do not hide cases where repeated placement is neutral or harmful;
  keep reference-construction and final-evaluation random streams independent;
  do not alter or relabel validated-v2.2 rows.
- Validation commands: tiny-instance realization tests; oracle-cache metadata
  and trajectory-count checks; row/protocol/status/count checks; independent
  summary recomputation; figure-source validation; full ACM LaTeX compilation
  and log scan.
