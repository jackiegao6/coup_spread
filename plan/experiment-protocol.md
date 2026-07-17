# Experiment Protocol

## Validated-v2.5 adaptive-sampling parameter sweep

> STATUS: LOCKED BEFORE ANY V2.5 RESULT FILE IS PRODUCED; EXECUTION COMPLETE.

> PRE-RUN IMPLEMENTATION CLARIFICATION: before the formal adaptive-grid run
> produced any result, smoke testing clarified that doubling uses nested RR
> sample prefixes from one stream. Regenerating an independent sample at every
> stage would not preserve the previous sample when the budget is doubled and
> would repeat all earlier generation work. The full run used nested prefixes
> uniformly in all 300 jobs; the temporary smoke outputs are outside the result
> directory and are not manuscript evidence.

This rerun preserves the complete v2.4 grid but replaces the underpowered
fixed 20,000-sample selection with a uniform doubling/stability rule. It is
designed to distinguish converged allocation quality from finite-sample
selection overfitting.

- Graphs, transfer values, redemption shares, budget slice, methods,
  distinct-seed capacity, and five selection seeds are identical to v2.4.
- Initial joint-sample budget: 100,000.
- Coverage floor: for graph size n, root normalizer W, and adoption sum A,
  require at least 10 expected nonempty current-index RR sets per candidate,
  i.e. T*A/(W*n) >= 10. The resulting floor is rounded upward to 50,000 and
  capped at 1.5 million.
- Doubling: extend one nested RR sample prefix at 100,000, 200,000, 400,000,
  ... until the coverage floor is reached; if doubling would cross the floor,
  evaluate the exact rounded floor first. Compare each new allocation with the
  preceding allocation on a fresh paired batch of 5,000 forward realizations.
  Once the floor is reached, stop when the absolute relative mean difference
  is at most 0.5%. Otherwise continue doubling, capped at 3 million samples.
- Cap handling: reaching 3 million and remaining unstable at 3 million are
  recorded separately. Every capped run is retained and is never removed or
  rerun with an outcome-dependent rule.
- IC-RIS sample parity: for each final allocation, IC-RIS uses the same number
  of RR samples as the selected final CIM-RIS stage.
- Final evaluation: 10,000 fresh forward realizations per allocation, using
  common realization-level streams within a cell and selection seed. These
  streams are disjoint from all stability batches.
- Reporting: retain all positive, tied, and negative cells. Report selected
  sample counts, stability/cap status, final spread, and strongest-baseline
  comparisons. Training RR estimates are diagnostic only.
- Output boundary: only rows marked REAL_EXPERIMENT under protocol
  validated-v2.5-adaptive-grid may replace v2.4 appendix claims.

## Validated-v2.4 appendix parameter sweep

> STATUS: LOCKED BEFORE ANY V2.4 RESULT FILE IS PRODUCED.
> EXECUTION: COMPLETE AND ARCHIVED; SUPERSEDED BY V2.5 FOR MANUSCRIPT CLAIMS.

This appendix experiment maps where coupon-aware allocation is more or less
effective than non-oracle alternatives. It is a robustness and mechanism
study, not an optimality comparison.

- Graphs: Netscience and NetFacebookEgo.
- Parameterization: for each node, let d_hat be normalized log degree and set
  the redemption share among stopping actions to
  clip(r + 0.20(0.5-d_hat), 0.05, 0.95). For non-isolated nodes,
  p_t=t, p_a=(1-t) redemption_share, and
  p_d=(1-t)(1-redemption_share). Isolated-node transfer mass is reassigned to
  discard.
- Full grid: transfer probability
  t in {0.30, 0.50, 0.70, 0.85, 0.93} and central redemption share
  r in {0.20, 0.40, 0.60, 0.80}, all evaluated at k=100.
- Budget interaction: retain all five transfer probabilities, fix r=0.60,
  and additionally evaluate k in {25, 200}; the k=100 rows are reused from
  the full grid.
- Methods: CIM-RIS, IC-RIS, 1Hop-Sort, Alpha-Sort, DegreeTopM, and PageRank.
  The best competing baseline is selected from all five non-CIM methods only
  after each complete cell is aggregated. Random is excluded from the
  best-baseline comparison, and MC-Greedy is not used because reconstructing a
  high-precision destination matrix for every grid cell would dominate the
  experiment.
- Distinct-seed policy: V_s=V and c_v=1 for every method.
- Independent selection seeds: 20260715--20260719.
- Sampling: 20,000 conditioned joint samples for CIM-RIS and 20,000 standard
  RR samples for IC-RIS in every run.
- Forward evaluation: 2,000 fresh realizations per selected allocation.
  Methods in the same dataset--parameter--budget--selection-seed cell use the
  same realization-level random streams.
- Metrics: distinct-adopter spread; total redemptions; duplicate-redemption
  fraction (mean redemptions - mean adopters) / mean redemptions; relative
  spread difference from the strongest non-oracle baseline; and win counts
  over all prespecified cells.
- Aggregation: method means and standard deviations over five selection runs.
  Heatmaps report the relative difference between the aggregated CIM-RIS mean
  and the largest aggregated baseline mean. Mechanism maps report the
  best-baseline duplicate fraction minus the CIM-RIS duplicate fraction, in
  percentage points.
- Reporting: retain every grid cell and budget slice, including ties and
  negative values. Conclusions are descriptive because the grid is a
  controlled simulation study over two graph instances.
- Output boundary: files remain archived for auditability but no longer
  support the current appendix claims, which use validated-v2.5.

> STATUS: VALIDATED-V2.2 COMPLETE. The repeated-seed study is the sole source
> of revised submission claims. Earlier `real_*` outputs remain reproducible
> reference artifacts but are superseded; synthetic planning files remain
> excluded from the manuscript.

## Validated-v2.3 extension protocol

> STATUS: LOCKED BEFORE THE EXTENSION RUN.

The extension addresses three prespecified evidence gaps without modifying the
validated-v2.2 outputs.

- Strong-reference datasets: Netscience and NetFacebookEgo. For each of the
  three existing scenarios, estimate the single-coupon destination matrix from
  100,000 trajectories per source on Netscience and 50,000 per source on
  NetFacebookEgo, then run deterministic greedy under the distinct-seed policy.
  Evaluate the resulting MC-Greedy allocation and each validated-v2.2 CIM-RIS
  allocation with common evaluation streams that are independent of reference
  construction.
- Capacity ablation datasets: Netscience and NetFacebookEgo; all three
  scenarios; `k=200`, where the distinct-seed constraint is most likely to
  bind; capacities `c_v in {1, 2, k}`; five selection seeds
  `20260715`--`20260719`; the same sample-budget rule as
  validated-v2.2. Report independently evaluated spread, gap to MC-Greedy under
  the same capacity, and the number of repeated placements.
- Sample sensitivity datasets: Netscience and NetFacebookEgo; the
  forwarding-heavy stress test; `k in {10, 50, 200}`; sample counts
  `{5,000, 20,000, 50,000, 100,000}`; five selection seeds. Evaluate all sample
  counts and the distinct-seed MC-Greedy reference with common forward streams.
- Reporting: retain every prespecified result, including flat, non-monotone,
  and unfavorable outcomes. MC-Greedy is a high-precision reference built from
  an estimated objective; it is not labeled as exact or globally optimal.

This scope was amended before any extension result was produced after a timing
smoke test showed that the original cross-product would spend most of its time
rerunning low-risk capacity settings. No network, scenario, or budget was
removed in response to an observed quality result.

## Validated-v2 locked protocol

- Quality datasets: Netscience, NetFacebookEgo, DoubanRandom, and EmailEnron.
- Scenarios: balanced, adoption-heavy, and forwarding-heavy, with parameters
  fixed before method comparison.
- Budgets: `k in {10, 25, 50, 100, 150, 200}`.
- Independent selection seeds: `20260715`--`20260719`.
- Joint RR samples: 100,000 for `k<=50` and 50,000 for `k>=100`. This rule was
  fixed from a Netscience forwarding-heavy convergence diagnostic before the
  repeated cross-method run; it is not changed by later outcomes.
- Forward evaluation: 10,000 realizations per allocation. Methods in the same
  configuration start each realization from a common random-generator state;
  this is a partial common-random-number coupling because path lengths differ.
- MC-Greedy reference: Netscience only, using a separately cached matrix with
  100,000 trajectories per source.
- Raw-data unit: one JSON file per dataset--scenario--budget--selection-seed
  configuration. Aggregation reads only files with a matching protocol key and
  `REAL_EXPERIMENT` status.
- Reporting: means and standard deviations across five selection runs; retain
  all negative, tied, and positive method comparisons.
- Scientific scope: real graph topologies with controlled synthetic behavior
  probabilities, not observed coupon-campaign outcomes.

## Evaluation questions

- Q1 Quality: compare adoption spread as the coupon budget increases.
- Q2 Model fidelity: compare CIM-RIS with coupon-aware heuristics and the IC-based IMM baseline.
- Q3 Efficiency: measure joint-sample generation plus greedy seed-selection time.
- Q4 Sampler ablation: compare uniform-root sampling with the conditioned adoption-root sampler using estimator variance, useful-sample rate, and wall-clock time.

## Datasets

Use Netscience, NetFacebookEgo, DoubanRandom, EmailEnron, and network.douban as listed in the manuscript. Graphs are not split. Undirected contacts are stored in both directions, and isolated nodes retain zero transfer probability.

## Methods

CIM-RIS, MC-Greedy where computationally feasible, 1Hop-Sort, Alpha-Sort, DegreeTopM, PageRank, Random, and IC-RIS. All methods use the same eligible set, capacity policy, budget, diffusion parameters, and Monte Carlo evaluator.

## Superseded reference-run protocol

The settings below document the earlier single-seed reference run. They are
retained for auditability and do not override the validated-v2.2 protocol or
support the revised manuscript claims.

- Budgets: k in {10, 25, 50, 100, 150, 200}, clipped below the eligible-node count.
- Scenarios: Balanced, Adoption-heavy, and Forwarding-heavy.
- Evaluation: 10,000 independent forward simulations per selected allocation on the four quality graphs; the full Douban scalability run uses 1,000 evaluations, which are not used for quality claims.
- Randomness: the main curves use reproducible master seed `20260715`.
  Reported 95% confidence intervals quantify forward-evaluation uncertainty
  conditional on that allocation. A separate stability study reruns CIM-RIS
  under five independent selection seeds (`20260715`--`20260719`) and
  evaluates each allocation with common random numbers over 10,000 fresh
  forward realizations.
- CIM-RIS: 10,000 joint samples in the balanced scenario and 20,000 in adoption- and forwarding-heavy scenarios. These fixed empirical budgets are not claimed to instantiate the conservative theorem-level sample count for a specified epsilon-delta pair.
- Sampler ablation: 30 independent batches of 20,000 samples for each reported condition.
- Hardware: record CPU model, core count, RAM, operating system, Python version, and whether parallelism is used.

## Planned figures

1. Adoption spread versus k on four quality datasets.
2. CIM-RIS runtime versus k and method runtime versus graph size.
3. Uniform-root versus conditioned-root sampler ablation.

Real CSVs, seed lists, metadata, runners, and plotting scripts are stored under `experiments/` and `figures/experiments/`. Synthetic files are retained only as visibly marked planning artifacts.
