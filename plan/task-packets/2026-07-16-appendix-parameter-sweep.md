## Task Packet

- Scope: Run a prespecified two-network parameter sweep and add the complete
  positive and negative results to a manuscript appendix.
- Files to read: paper-v2 copy.tex; validated experiment runners and core
  simulator under experiments/; current protocol, traceability, table schema,
  and figure manifest.
- Files allowed to edit: paper-v2 copy.tex; new appendix experiment runner,
  verifier, aggregate data, plotting script, and generated figure under
  experiments/ and figures/experiments/; planning/review records.
- Required skills: paper-orchestration, experiment-results-planning,
  statistical-analysis, figures-python, peer-review, latex-output, and
  verification.
- Evidence/data inputs: Netscience and NetFacebookEgo graph instances; the
  fixed-action coupon simulator; conditioned CIM-RIS; IC-RIS and non-oracle
  heuristic baselines.
- Required artifacts: locked protocol; raw and summary CSVs with
  REAL_EXPERIMENT status; metadata and checksum manifest; independent claim
  checker; publication PNG/SVG; appendix text that reports favorable and
  unfavorable regions; compiled PDF.
- Rejection checks: no parameter may be removed after results are observed; no
  MC-Greedy or optimality claim is permitted for this sweep; all methods use
  the same capacity, RR budget where applicable, and forward streams; no mock
  value may enter the appendix.
- Validation commands: core unit tests; appendix verifier; existing v2.2/v2.3
  claim checkers; deterministic plot regeneration; CSV/status/count checks;
  git diff --check; forced ACM compilation; reference, overflow, font, page,
  and visual checks.
