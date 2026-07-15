# Figure Data Manifest

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

| Figure | Data file | Status | Source | Script | Outputs |
|---|---|---|---|---|---|
| Balanced quality | ../experiments/results/balanced/real_quality_balanced.csv | Real experiment | run_real_submission.py | experiments/real_submission_figures.py | real_quality_balanced PNG/SVG |
| Forwarding-heavy quality | ../experiments/results/forwarding-heavy/real_quality_forwarding-heavy.csv | Real experiment | run_real_submission.py | experiments/real_submission_figures.py | real_quality_forwarding_heavy PNG/SVG |
| Runtime budget scaling | Balanced quality plus ../experiments/results/scalability/real_quality_balanced.csv | Real experiment | run_real_submission.py | experiments/real_submission_figures.py | real_runtime_scalability PNG/SVG |
| Conditioned-sampler ablation | ../experiments/results/sampler/real_sampler_ablation.csv | Real experiment | run_real_sampler_ablation.py | experiments/real_submission_figures.py | real_sampler_ablation PNG/SVG |

The manuscript references only the real experiment outputs in this section.
