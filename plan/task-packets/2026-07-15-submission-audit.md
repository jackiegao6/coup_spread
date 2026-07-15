## Task Packet

- Scope: Final submission audit of the ACM LaTeX manuscript, real experimental figures, numerical claims, theorem-to-implementation boundaries, and anonymous-review metadata.
- Files to read: `paper-v2 copy.tex`, `references.bib`, real experiment CSVs, real plotting scripts, corrected experiment runners, and existing planning/review records.
- Files allowed to edit: `paper-v2 copy.tex`, `figures/experiments/real_submission_figures.py`, regenerated real figure outputs, and `plan/` audit records.
- Required skills: paper-orchestration, peer-review, latex-output, verification; experiment-results-planning and figures-python only if a result figure needs correction.
- Evidence/data inputs: CSV rows marked `REAL_EXPERIMENT` under `experiments/results/`; no `synthetic_*` artifact may be cited or relabeled.
- Required artifacts: compiled PDF, clean reference/label checks, corrected submission metadata, readable figures, and a ranked residual-risk report.
- Rejection checks: no fabricated result; no placeholder DOI/ISBN or stale 2025 venue metadata; no unresolved citation/reference; no unsupported superiority claim; no mismatch hidden between the theorem and the evaluated implementation.
- Validation commands: `latexmk -pdf`; LaTeX log scan; PDF page and font inspection; claim-to-CSV check; Python syntax check; figure regeneration; citation-key and label/reference checks.

