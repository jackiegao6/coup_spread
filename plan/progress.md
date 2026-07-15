# Progress

## 2026-07-15 - Final Submission Audit

- Stage: S5 Review.
- Status: manuscript, real figures, and compiled PDF verified; target-specific conference metadata remains intentionally generic.
- Submission boundary: only CSV rows marked `REAL_EXPERIMENT` support empirical claims. Synthetic planning artifacts remain visibly labeled and are not referenced by the manuscript.

### Artifacts

- Compiled `paper-v2 copy.pdf` in ACM `sigconf` format (13 pages).
- Removed the speculative `Other Variants` section so that the paper moves
  directly from the proved method to its empirical evaluation.
- Added an aggregate three-scenario comparison table computed from the real
  result CSVs and strengthened the abstract, Introduction, Results, and
  Conclusion with traceable numerical claims.
- Replaced stale SIGMOD 2025/placeholder DOI metadata with anonymous generic ACM metadata.
- Added KONECT dataset provenance and cubic-vertex-cover hardness citations.
- Added ACM image descriptions to all seven figures.
- Corrected the RR-set formula overflow and the linear/near-linear runtime wording.
- Replaced the weak cross-graph runtime fit with a focused single-column
  budget-scaling plot and reformatted the sampler ablation for single-column
  readability.
- Recorded the master random seed, the conditional interpretation of simulation confidence intervals, and fixed-sample/theorem boundary.
- Added `experiments/results/dataset_manifest.csv` with graph-file SHA-256 checksums.

### Review gates

- Spec compliance: passed. No mock/synthetic artifact is cited; all numerical manuscript claims were recomputed from real CSVs.
- Quality review: passed with residual risks below. The PDF has no unresolved references, missing citations, missing graphics, missing figure descriptions, or overfull boxes. All seven figures appear before the Conclusion and References.

### Capability-use audit

- Required skills: paper-orchestration, peer-review, verification, latex-output, evidence-driven-writing, literature-review, experiment-results-planning, figures-python.
- Skills actually used: all required skills were read and applied to task scoping, evidence mapping, result-boundary review, figure correction, LaTeX compilation, and final verification.
- Inputs consumed: current manuscript and bibliography; corrected experiment runners; all real result CSV/JSON files; KONECT provenance in the supplied NeurIPS paper; current ACM class; compiled PDF and LaTeX log.
- Inputs not used and why: synthetic/mock outputs were excluded from evidence; no closest-work citation was added because external scholarly search endpoints were unavailable and no unverified reference was introduced.
- Artifacts produced: revised manuscript and bibliography, aggregate quality table, corrected real runtime and sampler figures, compiled PDF, evidence map/blueprint/review, dataset checksum manifest, and final audit records.
- Verification run: full `latexmk` compile with `acmart` 2026 and ACM fonts; static label/citation/environment/graphic checks; independent claim-to-CSV recomputation; Python syntax parsing; 450-DPI image inspection; dataset checksum verification; PDF page/order/font inspection; manual first-page and result-page review.
- Remaining risk: the exact target venue and page limit are unspecified; the header therefore says anonymous submission. The quality experiments use one reproducible stochastic selection seed, so their confidence intervals cover forward-evaluation uncertainty but not algorithm-seed variability. Related Work lacks a verified closest non-replicable-coupon citation. IC-RIS and local heuristics remain competitive on some graph--budget pairs, so the manuscript deliberately avoids a universal empirical-dominance claim.

## 2026-07-15 - Proof and Figure Revision

- Stage: S2 Method, followed by S5 Review.
- Status: implementation and static review complete; awaiting user review.
- User direction: supplement the proof and draw publication-appropriate figures.
- Manuscript policy: preserve the monolithic ACM LaTeX file; do not invent experimental results.

### Artifacts

- Added a cubic-vertex-cover reduction gadget in Section 4.
- Added a coupon-index matching diagram for joint RR samples in Section 5.
- Formalized the conditioned root distribution and its weighted unbiasedness lemma.
- Replaced the uniform-root marginal estimator with the conditioned estimator.
- Replaced the graph-quadratic Hoeffding sample bound with a Bernstein bound.
- Proved the expected conditioned-sample cost and restored graph-linear expected runtime for fixed parameters.
- Aligned the abstract and conclusion with the theorem's spread-lower-bound condition.

### Review gates

- Spec compliance: passed. All task-packet artifacts are present, and no empirical result was invented.
- Quality review: passed for notation and argument flow. The conditioned estimator was also checked by exact enumeration on a three-node, two-coupon instance.

### Capability-use audit

- Required skills: paper-orchestration, writing-chapters, latex-output, figures-diagram, peer-review, verification.
- Skills actually used: all required skills were read and applied to task scoping, method flow, TikZ output, review, and validation.
- Inputs consumed: current paper-v2 copy.tex, its model assumptions, current algorithm description, and the existing corrected implementation semantics summarized in the workspace context.
- Inputs not used and why: no old result images or mock data were used because the corrected experiments have not been rerun.
- Artifacts produced: revised manuscript, two TikZ figures, task packet, and project progress records.
- Verification run: git diff check; environment nesting; brace balance; label/reference resolution; BibTeX key resolution; targeted stale-formula search; exact conditioned-estimator enumeration.
- Remaining risk: no TeX engine or local TeX container is available, so PDF compilation, float placement, and final overfull-box inspection remain unverified. Experimental result figures still require corrected, traceable runs.

## 2026-07-15 - Synthetic Experiment Figure Planning

- Stage: S3 Experiments.
- Status: planning artifacts generated; no synthetic value was inserted into the manuscript.
- Outputs: quality-versus-budget, runtime/scalability, and conditioned-sampler-ablation figures in PNG and SVG.
- Data boundary: every CSV, script, and output uses the synthetic_ prefix; rendered figures contain a visible NOT FOR SUBMISSION notice.
- Environment: Matplotlib 3.7.5 and NumPy 1.24.4 were installed under /tmp/coup-plot-libs for rendering only.

### Review gates

- Spec compliance: passed. Protocol, traceability map, table schema, data manifest, generator, CSVs, scripts, and outputs exist.
- Quality review: passed after replacing repeated long x-axis labels in the sampler figure with one shared label.

### Capability-use audit

- Required skills: experiment-results-planning, figures-python, environment-setup, verification.
- Skills actually used: all required skills.
- Inputs consumed: current experiment setup and model constraints in paper-v2 copy.tex.
- Inputs not used and why: old mock result files were not reused because their estimator semantics are not trusted.
- Artifacts produced: three synthetic CSV files, three plotting scripts, six rendered outputs, protocol, traceability map, and data manifest.
- Verification run: deterministic regeneration; Python syntax compilation; CSV schema/status/range checks; 450-DPI PNG checks; SVG notice checks; manual image inspection.
- Remaining risk: these figures provide layout only. Their numerical trends, confidence intervals, and runtime values must all be replaced by corrected real runs before submission.
