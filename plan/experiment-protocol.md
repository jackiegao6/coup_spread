# Experiment Protocol

> STATUS: REAL RUN COMPLETE. Submission claims use only CSV rows marked `REAL_EXPERIMENT`; synthetic planning files remain excluded from the manuscript.

## Evaluation questions

- Q1 Quality: compare adoption spread as the coupon budget increases.
- Q2 Model fidelity: compare CIM-RIS with coupon-aware heuristics and the IC-based IMM baseline.
- Q3 Efficiency: measure joint-sample generation plus greedy seed-selection time.
- Q4 Sampler ablation: compare uniform-root sampling with the conditioned adoption-root sampler using estimator variance, useful-sample rate, and wall-clock time.

## Datasets

Use Netscience, NetFacebookEgo, DoubanRandom, EmailEnron, and network.douban as listed in the manuscript. Graphs are not split. Undirected contacts are stored in both directions, and isolated nodes retain zero transfer probability.

## Methods

CIM-RIS, MC-Greedy where computationally feasible, 1Hop-Sort, Alpha-Sort, DegreeTopM, PageRank, Random, and IC-RIS. All methods use the same eligible set, capacity policy, budget, diffusion parameters, and Monte Carlo evaluator.

## Real-run protocol

- Budgets: k in {10, 25, 50, 100, 150, 200}, clipped below the eligible-node count.
- Scenarios: Balanced, Adoption-heavy, and Forwarding-heavy.
- Evaluation: 10,000 independent forward simulations per selected allocation on the four quality graphs; the full Douban scalability run uses 1,000 evaluations, which are not used for quality claims.
- Randomness: one reproducible master seed (`20260715`) determines stochastic selection and evaluation streams. Reported 95% confidence intervals quantify forward-evaluation uncertainty conditional on the selected allocation; algorithm-seed variability was not estimated.
- CIM-RIS: 10,000 joint samples in the balanced scenario and 20,000 in adoption- and forwarding-heavy scenarios. These fixed empirical budgets are not claimed to instantiate the conservative theorem-level sample count for a specified epsilon-delta pair.
- Sampler ablation: 30 independent batches of 20,000 samples for each reported condition.
- Hardware: record CPU model, core count, RAM, operating system, Python version, and whether parallelism is used.

## Planned figures

1. Adoption spread versus k on four quality datasets.
2. CIM-RIS runtime versus k and method runtime versus graph size.
3. Uniform-root versus conditioned-root sampler ablation.

Real CSVs, seed lists, metadata, runners, and plotting scripts are stored under `experiments/` and `figures/experiments/`. Synthetic files are retained only as visibly marked planning artifacts.
