# Experiment Protocol

> STATUS: PLANNING ONLY. Synthetic files and figures defined here are not experimental evidence and must be replaced before submission.

## Evaluation questions

- Q1 Quality: compare adoption spread as the coupon budget increases.
- Q2 Model fidelity: compare CIM-RIS with coupon-aware heuristics and the IC-based IMM baseline.
- Q3 Efficiency: measure preprocessing, sampling, seed selection, peak memory, and total wall-clock time.
- Q4 Sampler ablation: compare uniform-root sampling with the conditioned adoption-root sampler using estimator variance, useful-sample rate, and wall-clock time.

## Datasets

Use Netscience, NetFacebookEgo, DoubanRandom, EmailEnron, and network.douban as listed in the manuscript. Graphs are not split. Record the exact file checksum, preprocessing, direction convention, isolated-node handling, and graph statistics for every real run.

## Methods

CIM-RIS, MC-CELF where computationally feasible, 1Hop-Sort, Alpha-Sort, DegreeTopM, PageRank, Random, and IMM. All methods use the same eligible set, capacity policy, budget, diffusion parameters, and Monte Carlo evaluator.

## Real-run protocol

- Budgets: k in {10, 25, 50, 100, 150, 200}, clipped below the eligible-node count.
- Scenarios: Balanced, Adoption-heavy, and Forwarding-heavy.
- Evaluation: 10,000 independent forward simulations per selected allocation.
- Repetitions: at least five independent algorithm/evaluation seeds; report mean and 95% confidence interval.
- CIM-RIS: record epsilon, delta, lower-bound method, W, T, accepted samples, generated RR memberships, and peak memory.
- Hardware: record CPU model, core count, RAM, operating system, Python version, and whether parallelism is used.

## Planned figures

1. Adoption spread versus k on four quality datasets.
2. CIM-RIS runtime versus k and method runtime versus graph size.
3. Uniform-root versus conditioned-root sampler ablation.

The current synthetic versions only validate layout, labels, and data contracts.

