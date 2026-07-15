## Task Packet

- Scope: Rebuild the empirical evaluation from the paper's proved coupon
  model and sampling algorithm, diagnose the poor historical results, and
  replace manuscript claims only after independent correctness checks.
- Files to read: `paper-v2 copy.tex`; the supplied theory PDFs under
  `sigmod-v1/`; `experiments/*.py`; `gzc-impl/*.py`; current result CSV/JSON
  files; experiment and traceability plans.
- Files allowed to edit: `experiments/`; `figures/experiments/`;
  `paper-v2 copy.tex`; experiment/review records under `plan/`, `tables/`,
  and `figures/`.
- Required skills: paper-orchestration, experiment-results-planning,
  statistical-analysis, figures-python, peer-review, verification.
- Evidence/data inputs: the graph files listed in
  `experiments/results/dataset_manifest.csv`; the model, estimator, and
  theorem stated in the current manuscript; supplied proof and RIS PDFs.
- Required artifacts: implementation audit; exact/small-instance validation;
  reproducible Python environment; corrected experiment runner and tests;
  preregistered real-run protocol; raw repeated-run logs; aggregate tables;
  publication figures; manuscript revisions; final audit.
- Rejection checks: do not use or relabel mock/synthetic outputs; do not tune
  scenarios or omit negative comparisons to make CIM-RIS look better; do not
  call fixed empirical sample counts theorem-calibrated; do not claim real
  campaign behavior from synthetic diffusion probabilities; do not retain a
  result whose code path fails the small-instance validation.
- Validation commands: unit/exact-enumeration tests; deterministic smoke run;
  repeated-seed experiment checks; independent CSV aggregation; figure source
  validation; full ACM LaTeX compilation and log scan.

