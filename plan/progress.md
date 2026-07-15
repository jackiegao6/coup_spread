# Progress

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
