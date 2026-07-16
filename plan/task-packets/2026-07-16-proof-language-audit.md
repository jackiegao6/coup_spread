## Task Packet

- Scope: Perform an independent submission-level proof audit and a restrained
  line-by-line English edit of the complete manuscript.
- Files to read: `paper-v2 copy.tex`; supplied theory/RIS PDFs under
  `sigmod-v1/`; current experiment protocols, claim checkers, and review notes.
- Files allowed to edit: `paper-v2 copy.tex`; proof/language review records and
  progress records under `plan/`; tests under `experiments/` only if a proof
  invariant needs executable verification.
- Required skills: paper-orchestration, peer-review, writing-core,
  prompts-collection, latex-output, and verification.
- Evidence inputs: current model definitions, exact tiny-instance tests,
  validated experiment outputs, cited theorem sources, and compiled PDF.
- Required artifacts: theorem-by-theorem audit with assumptions and failure
  checks; corrected manuscript; English consistency/style pass; final compiled
  PDF; capability-use audit.
- Rejection checks: do not preserve a claim merely to avoid rewriting; do not
  change mathematics during language polishing without recording it; do not
  strengthen novelty or empirical claims; do not edit numeric claims unless a
  claim checker is updated; do not hide residual theoretical assumptions.
- Validation commands: exact tiny-instance tests; proof-specific numerical or
  symbolic checks where useful; both experiment claim checkers; label/citation
  and terminology scans; forced ACM compilation; log/font/page/visual checks;
  `git diff --check`.
