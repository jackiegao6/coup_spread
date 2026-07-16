# Progress

## 2026-07-16 - Appendix Parameter-Sweep Experiments

- Stage: S3 Experiments, followed by S5 review.
- Status: complete.
- Objective: map favorable and unfavorable parameter regimes without
  post-result filtering, quantify duplicate-redemption overlap, and add the
  complete evidence to a manuscript appendix.
- Task packet:
  plan/task-packets/2026-07-16-appendix-parameter-sweep.md.

### Artifacts

- Added a reproducible runner and independent verifier for the complete
  two-graph transfer-by-redemption-share grid and three-budget slice.
- Produced 40 job records, 1,800 raw method rows, 360 method summaries, and 60
  comparison summaries under `validated-v2.4-appendix-grid`, with checksums.
- Added a six-panel 450-DPI PNG/SVG figure and a manuscript appendix that
  reports both favorable and unfavorable parameter regions.
- Used Student-t intervals for the five paired runs and retained the negative
  duplicate-overlap mechanism result.
- Compiled the revised ACM manuscript to a 15-page PDF; the appendix figure is
  before the references, and the final bibliography columns are balanced.

### Review gates

- Spec compliance: passed. Every prespecified cell, seed, method, and budget is
  present; all accepted rows are marked `REAL_EXPERIMENT`; no mock value,
  MC-Greedy claim, or post-result cell filtering enters the appendix.
- Quality review: passed. The text identifies higher means without claiming
  statistical significance or universal superiority, reports the principal
  failure regions, and records the two-graph and controlled-parameter limits.

### Capability-use audit

- Required skills: paper-orchestration, experiment-results-planning,
  statistical-analysis, figures-python, peer-review, latex-output, and
  verification.
- Skills actually used: all required skills for protocol locking, repeated-run
  aggregation, Student-t intervals, data-bound plotting, restrained result
  prose, independent review, and final compilation checks.
- Inputs consumed: the current manuscript; fixed-action simulator and validated
  CIM-RIS/IC-RIS implementations; Netscience and NetFacebookEgo graph files;
  all 40 job records; raw and summary CSVs; checksum manifest; LaTeX log; and
  rendered appendix/reference pages.
- Inputs not used and why: MC-Greedy was excluded by the locked scope because a
  high-precision destination matrix for every grid cell was impractical;
  historical mock/synthetic outputs and unprespecified graph/parameter cells
  were excluded from evidence.
- Artifacts produced: runner, verifier, job records, CSV summaries, metadata,
  checksum manifest, plotting script, PNG/SVG, appendix prose, peer-review
  record, and recompiled PDF.
- Verification run: four coupon-core tests; validated-v2.2 and v2.3 claim
  checkers; v2.4 row/protocol/value verifier; checksum verification;
  deterministic figure regeneration; forced ACM compilation; unresolved
  reference, overflow, font, page-order, visual, and `git diff --check` scans.
- Remaining risk: behavior parameters are controlled rather than observed;
  the sweep covers two graphs and five selection runs; displayed intervals are
  descriptive after strongest-baseline selection; fixed RR budgets do not
  instantiate the theorem-level stopping certificate; venue-specific metadata
  and page limits remain unknown.

## 2026-07-16 - Independent Proof and Language Audit

- Stage: S5 Review, with S2 corrections if the proof audit finds a substantive
  issue.
- Status: complete.
- Objective: independently re-derive every theorem and estimator argument,
  then perform a conservative sentence-level English edit without changing
  supported claims.
- Task packet: `plan/task-packets/2026-07-16-proof-language-audit.md`.

### Artifacts

- Added a theorem-by-theorem independent audit covering the model identities,
  NP-hardness reduction, DR-submodularity, conditioned RR estimators,
  concentration argument, recurrence, and expected running time.
- Made the repeated-source exchange loss explicit and stated the final
  vertex-cover threshold equivalence.
- Expanded the greedy residual bound through the componentwise join and
  expanded the approximation recurrence to its final ratio.
- Clarified why conditioning on all k root gates remains unbiased for every
  greedy prefix and marginal event.
- Added constant-time node-action sampling via alias preprocessing to close the
  implementation assumption in the expected-time proof.
- Completed a conservative sentence-level English pass and a terminology
  audit without changing experimental values or strengthening claims.
- Corrected the model figure's hidden adopter nodes and moved the full-width
  result figures so that the final 14-page PDF no longer contains consecutive
  half-empty float pages.

### Review gates

- Spec compliance: passed. The proof audit, corrected manuscript, language
  record, compiled PDF, and residual-risk notes are present; mathematical and
  empirical claims were not silently changed.
- Quality review: passed with one recorded typesetting residual. Independent
  derivations found no fatal proof error, the experiment claim checkers and
  core tests pass, and the final PDF has no unresolved references, missing
  graphics, or visible overlap. The ACM bibliography balancing step reports a
  non-visible 1.166 pt vertical overfull box on the final reference page.

### Capability-use audit

- Required skills: paper-orchestration, peer-review, writing-core,
  prompts-collection, latex-output, and verification.
- Skills actually used: all required skills for task control, theorem-level
  adversarial review, conservative English editing, LaTeX preservation, and
  evidence-based completion checks.
- Inputs consumed: the current manuscript; model and proof invariants from the
  supplied theory/RIS material; exact tiny-instance tests; validated-v2.2 and
  validated-v2.3 claim checkers; experiment figures; LaTeX log; and rendered
  PDF pages.
- Inputs not used and why: historical mock results and ordinary
  absorbing-Markov evaluation were excluded because they do not support the
  fixed-action model; no new literature or empirical claim was needed.
- Artifacts produced: revised manuscript/PDF, independent proof audit,
  independent language audit, and updated project notes/progress.
- Verification run: K4 hardness enumeration; four coupon-core tests; both
  experiment claim checkers; terminology and spelling scans; repeated forced
  ACM compilation; reference/error/font/page scans; and visual inspection of
  theory and result pages.
- Remaining risk: the theorem requires a valid positive spread lower bound;
  fixed empirical sample budgets do not automatically inherit its certificate;
  behavior probabilities remain controlled synthetic parameters; and the final
  reference page has the minor vertical box warning noted above.

## 2026-07-15 - Validated Experiment Extensions

- Stage: S3 Experiments, followed by S5 review.
- Status: complete under the amended-before-results validated-v2.3 extension
  protocol.
- Objective: add a high-precision MC-Greedy benchmark on a larger graph, test
  repeated placement under explicit capacities, and measure sensitivity to RR
  sample count without changing the validated-v2.2 evidence.
- Protocol correction: a preregistration-time tiny-graph test rejected the
  proposed absorbing-Markov linear solve because it resamples actions after a
  revisit, whereas the paper fixes each node action per coupon and terminates
  transfer cycles. The locked extension therefore uses model-consistent forward
  trajectories with independent evaluation streams.
- Pre-result feasibility amendment: a timing smoke test measured 21.8 seconds
  for one Netscience `k=200`, 50,000-sample selection. Before any extension CSV
  was produced, capacity testing was focused on `k=200` and sample sensitivity
  on the forwarding-heavy stress test; the two networks, all three capacity
  policies, four sample counts, and five seeds were retained.
- Task packet:
  `plan/task-packets/2026-07-15-validated-experiment-extensions.md`.

### Artifacts

- Added uniform-capacity support to the production CIM-RIS selector while
  preserving `c_v=1` as the default used by validated-v2.2.
- Added a parallel, model-consistent MC-Greedy reference on NetFacebookEgo
  using 50,000 trajectories per source in all three scenarios; retained the
  100,000-trajectory Netscience reference.
- Produced 180 strong-reference rows, 90 capacity rows, and 120
  sample-sensitivity rows, with five selection/evaluation repeats throughout.
- Added an independent extension claim checker and a four-panel submission
  figure; updated the main quality figure to show MC-Greedy on NetFacebookEgo.
- Removed the incorrect claim that fixed-action cycle semantics can be handled
  by an ordinary absorbing-Markov linear system.
- Closed the sampling-budget explanation gap: the manuscript now separates the
  theorem-calibrated $T(\epsilon,\delta,\mathrm{LB})$ from the fixed empirical
  schedule, explains the `k=100` budget change, and reports the measured
  cost--accuracy curve with claim-checker coverage.

### Review gates

- Spec compliance: passed. All extension rows use the locked protocol/status,
  reference construction is independent of final evaluation, and no
  validated-v2.2 artifact was overwritten.
- Quality review: passed with explicit negative results. The manuscript reports
  the 11.1% worst-case gap, monotone mean improvement with sample count, and the
  small/conditional effect of repeated placement.

### Capability-use audit

- Required skills: paper-orchestration, experiment-results-planning,
  statistical-analysis, figures-python, peer-review, latex-output, and
  verification.
- Skills actually used: all required skills for protocol amendment, repeated
  evaluation, descriptive aggregation, publication plotting, manuscript
  revision, adversarial review, and compile verification.
- Inputs consumed: validated-v2.2 seed allocations and evaluation seeds; two
  real network topologies; all three controlled diffusion scenarios; current
  model/cycle semantics; existing oracle caches; and the ACM manuscript.
- Inputs not used and why: the proposed sparse linear-system benchmark was
  rejected by a tiny cyclic counterexample because it resamples actions after
  revisits; mock/synthetic planning data and historical result files were not
  used.
- Artifacts produced: capacity-aware selector/tests; extension runner, oracle
  caches, raw/summary CSVs, metadata, verifier, figure script and PNG/SVG;
  revised manuscript/PDF; traceability, manifest, and peer-review updates.
- Verification run: four tiny-instance tests; v2.2 and v2.3 claim checkers;
  protocol/status/count checks; visual inspection of both revised figures;
  Python syntax checks; forced ACM LaTeX rebuild; reference/graphic/font and
  page-order scans.
- Remaining risk: behavior probabilities are synthetic; MC-Greedy covers only
  the two smaller reference graphs; sample sensitivity is limited to
  forwarding-heavy diffusion; and capacity effects are measured at `k=200`.

## 2026-07-15 - Real Experiment Rebuild

- Stage: S3 Experiments, beginning with an S2 method-to-code audit and ending
  with an S5 review.
- Status: complete for the validated simulation scope.
- Objective: replace the historical mock-result workflow and the provisional
  reference runner with a validated implementation and genuinely reproducible
  simulation experiments on the stored real network topologies.
- Scientific boundary: network structures are observed datasets, while node
  behavior probabilities remain controlled synthetic parameters; the paper
  must describe these as simulations on real networks, not field experiments.
- Current task packet:
  `plan/task-packets/2026-07-15-real-experiment-rebuild.md`.

### Artifacts

- Audited the historical pipeline and documented the invalid root-event mode,
  unseeded evaluation, stale-cache/append behavior, and inadequate historical
  Monte Carlo budgets in `plan/review/implementation-audit.md`.
- Added exact-realization tests for the forward simulator and conditioned RR
  estimator, plus a tiny-instance optimum-recovery test.
- Added the locked `validated-v2.2` runner, aggregation, scalability, claim
  verification, and reproducible dependency files under `experiments/`.
- Ran 360 dataset--scenario--budget--seed configurations, producing 2,610
  method-level raw rows and 522 five-run summaries. All accepted rows carry
  `REAL_EXPERIMENT` status and the locked protocol key.
- Generated submission figures from the validated CSVs only and revised the
  manuscript setup, result discussion, abstract, Introduction, and Conclusion.
- Recorded the negative findings: IC-RIS is tied or slightly stronger in two
  scenarios, and 1Hop-Sort is slightly stronger in the adoption-heavy setting.

### Review gates

- Spec compliance: passed. No historical mock/synthetic result supports a
  manuscript claim; all five selection seeds and 10,000 forward evaluations
  per allocation are present, and CIM-RIS and IC-RIS use the same sample
  budgets.
- Quality review: passed for simulation evidence. Exact tests connect the code
  to the fixed-realization model, the claim checker independently recomputes
  manuscript values, and the limitations on external validity and baseline
  coverage are explicit.

### Capability-use audit

- Required skills: paper-orchestration, experiment-results-planning,
  statistical-analysis, figures-python, peer-review, latex-output, and
  verification.
- Skills actually used: all required skills for task scoping, protocol
  locking, repeated-run summaries, data-bound plotting, manuscript integration,
  independent review, and final compilation checks.
- Inputs consumed: the current manuscript; supplied theory and RIS PDFs; graph
  files and checksums; historical `gzc-impl` and reference experiment code;
  validated raw JSON/CSV outputs; and the ACM LaTeX build log.
- Inputs not used and why: historical mock figures, synthetic planning values,
  stale caches, and old append-mode CSVs were excluded because their provenance
  or estimator semantics cannot support scientific claims.
- Artifacts produced: implementation audit, exact tests, locked protocol,
  corrected runners, raw and aggregate results, checksum manifest, publication
  figures, revised manuscript/PDF, traceability map, and peer-review record.
- Verification run: exact/unit tests; 360-job and 2,610-row integrity checks;
  protocol/status/seed checks; independent claim recomputation; figure-source
  validation; Python syntax checks; and full ACM LaTeX compilation/log scan.
- Remaining risk: behavior probabilities are controlled synthetic settings
  rather than estimates from campaign logs; MC-Greedy is limited to
  Netscience; experiments impose distinct seeds (`c_v=1`); and venue-specific
  metadata/page limits remain unknown.

## 2026-07-15 - Residual Risk Closure

- Stage: S3 Experiments, S1 Evidence, then S5 Review.
- Status: selection-seed and closest-work risks closed; venue-specific
  metadata/page-limit work awaits the target conference name and year.

### Artifacts

- Added `experiments/run_selection_seed_stability.py` and real stability
  artifacts under `experiments/results/stability/`.
- Evaluated five independent CIM-RIS selection seeds over all 72
  graph--scenario--budget configurations. Across configurations, the mean
  cross-seed coefficient of variation is 0.57%, its maximum is 2.64%, and
  the mean relative range is 1.42%.
- Added the verified closest coupon-motivated seed-allocation work, Liao et
  al.'s PRM paper (DOI `10.1145/3589309`), and explicitly distinguished its
  copyable PA-IC cascade from non-replicable coupon transfer.
- Updated the experiment protocol, evidence map, coverage review, Related
  Work blueprint, bibliography, and reproducibility paragraph.

### Review gates

- Spec compliance: passed for completed scope. All 360 stability rows and 72
  summaries are marked `REAL_EXPERIMENT`; every manuscript stability number
  was independently recomputed.
- Quality review: passed. The closest-work paragraph states a narrow,
  verifiable contrast and avoids a universal novelty claim.

### Capability-use audit

- Required skills: paper-orchestration, experiment-results-planning,
  statistical-analysis, evidence-driven-writing, literature-review,
  peer-review, latex-output, verification.
- Skills actually used: all listed skills for task scoping, repeated-run
  protocol, descriptive stability analysis, evidence mapping, manuscript
  revision, and compile verification.
- Inputs consumed: real graph files and checksum manifest; existing saved
  CIM-RIS allocations; five master selection seeds; PRM author repository,
  full paper, title/authors/abstract, and DOI.
- Inputs not used and why: generic web search endpoints were inaccessible;
  no unverified coupon-diffusion citation was introduced.
- Artifacts produced: stability runner/raw CSV/summary/metadata, PRM BibTeX
  entry and Related Work paragraph, updated evidence and protocol records,
  and a recompiled 13-page PDF.
- Verification run: runner smoke test and full run; row/status/seed/count and
  numerical-summary checks; BibTeX/citation compile with `acmart` 2026;
  unresolved-reference and overfull-box scan.
- Remaining risk: target-specific page limit and official ACM conference
  metadata cannot be verified or applied until the venue and year are known.

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
