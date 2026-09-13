# Figure Data Manifest

## Chinese working draft: evidence repaired on 2026-09-13

`paper-zh.tex` uses the following supplemental outputs; archived English figures and `validated-v2` data remain unchanged. The main quality means are retained, with Student-t intervals replacing normal intervals. New sampler times include root generation and come from a separate Windows run.

| Figure | Data | Generator | Output directory |
|---|---|---|---|
| Main quality, corrected intervals | `experiments/results/evidence-20260913/ci/validated_summary.csv` plus archived strong-reference summary | `figures/experiments/evidence_20260913.py --part quality` | `figures/experiments/evidence-20260913/validated_quality_combined` |
| Sampler, retained batch evidence | `experiments/results/evidence-20260913/sampler/summary.csv` and `summary_batches.csv` | `figures/experiments/evidence_20260913.py --part sampler` | `figures/experiments/evidence-20260913/validated_sampler_ablation` |
| Full Douban independent quality | `experiments/results/evidence-20260913/large-graph/summary.csv`, `raw.csv`, and `jobs/` | `figures/experiments/evidence_20260913.py --part large` | `figures/experiments/evidence-20260913/large_graph_quality` |

The reference-strength/capacity/sensitivity, historical runtime, and adaptive-grid figures retain their earlier data sources below. Original download provenance is still incomplete; see `plan/review/dataset-provenance-2026-09-13.md`.

## Archived planning and English manuscript evidence

> The first table records synthetic planning artifacts that are not valid
> submission evidence. The second table records the real experiment
> artifacts used by the manuscript.

| Figure | Data file | Status | Source | Script | Outputs |
|---|---|---|---|---|---|
| Quality versus budget | data/synthetic_quality_vs_budget.csv | Synthetic planning only | Deterministic generator | experiments/synthetic_quality_vs_budget.py | PNG, SVG |
| Runtime and scalability | data/synthetic_runtime_scalability.csv | Synthetic planning only | Deterministic generator | experiments/synthetic_runtime_scalability.py | PNG, SVG |
| Sampler ablation | data/synthetic_importance_sampling.csv | Synthetic planning only | Deterministic generator | experiments/synthetic_importance_sampling.py | PNG, SVG |

All outputs contain the visible label: SYNTHETIC PLANNING DATA - NOT FOR SUBMISSION.

## Real experiment artifacts

### Validated-v2 submission evidence

| Figure | Data file | Status | Source | Script | Outputs |
|---|---|---|---|---|---|
| Balanced quality | ../experiments/results/validated-v2/validated_summary.csv | Real experiment, five selection runs | run_validated_study.py + aggregate_validated_study.py | experiments/validated_study_figures.py | validated_quality_balanced PNG/SVG |
| Forwarding-heavy quality | ../experiments/results/validated-v2/validated_summary.csv | Real experiment, five selection runs | run_validated_study.py + aggregate_validated_study.py | experiments/validated_study_figures.py | validated_quality_forwarding_heavy PNG/SVG |
| Combined quality (manuscript) | ../experiments/results/validated-v2/validated_summary.csv | Real experiment, five selection runs | run_validated_study.py + aggregate_validated_study.py | experiments/validated_study_figures.py | validated_quality_combined PNG/SVG |
| Runtime budget scaling | validated_summary.csv plus validated_scalability.csv | Real experiment, five selection runs | run_validated_study.py + run_validated_scalability.py | experiments/validated_study_figures.py | validated_runtime_scalability PNG/SVG |
| Conditioned-sampler ablation | ../experiments/results/validated-v2/validated_sampler_ablation.csv | Real experiment, 30 independent batches | run_real_sampler_ablation.py with validated-v2 allocations | experiments/validated_study_figures.py | validated_sampler_ablation PNG/SVG |
| Strong reference, sensitivity, and capacity | ../experiments/results/validated-v2/extensions/*_summary.csv | Real experiment, five selection runs | run_validated_extensions.py | experiments/validated_extension_evidence.py | validated_extension_evidence PNG/SVG |
| Appendix adaptive parameter regimes, sampling, and overlap | ../experiments/results/validated-v2/adaptive-grid/comparison_summary.csv | Real experiment, five selection runs; adaptive RR stopping | run_adaptive_parameter_sweep.py | figures/experiments/appendix_parameter_sweep.py | appendix_parameter_sweep PNG/SVG |

The files below are the earlier traceable reference run. They remain archived
for audit purposes but are superseded by validated-v2 and must not support the
revised manuscript.

| Figure | Data file | Status | Source | Script | Outputs |
|---|---|---|---|---|---|
| Balanced quality | ../experiments/results/balanced/real_quality_balanced.csv | Real experiment | run_real_submission.py | experiments/real_submission_figures.py | real_quality_balanced PNG/SVG |
| Forwarding-heavy quality | ../experiments/results/forwarding-heavy/real_quality_forwarding-heavy.csv | Real experiment | run_real_submission.py | experiments/real_submission_figures.py | real_quality_forwarding_heavy PNG/SVG |
| Runtime budget scaling | Balanced quality plus ../experiments/results/scalability/real_quality_balanced.csv | Real experiment | run_real_submission.py | experiments/real_submission_figures.py | real_runtime_scalability PNG/SVG |
| Conditioned-sampler ablation | ../experiments/results/sampler/real_sampler_ablation.csv | Real experiment | run_real_sampler_ablation.py | experiments/real_submission_figures.py | real_sampler_ablation PNG/SVG |

The selection-seed stability analysis is a non-figure result sourced from
`../experiments/results/stability/real_selection_seed_stability.csv` and
`real_selection_seed_stability_summary.csv`, generated by
`../experiments/run_selection_seed_stability.py`.

The manuscript references only the real experiment outputs in this section.
