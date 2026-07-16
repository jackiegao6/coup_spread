# Weak-Gain Code Audit

## Verdict

No correctness defect was reproduced in the current validated forward
simulator, fixed-action cycle handling, conditioned joint-RR generator, greedy
capacity logic, graph direction, or independent forward evaluator. Historical
code had known semantic and reproducibility problems, but those outputs are not
used by the validated manuscript.

The weak empirical advantage is nevertheless partly an implementation-level
sampling problem: the fixed RR budgets are too small for adaptive greedy
selection, especially when adoption is rare and the graph has many candidates.
The samples used for selection are strongly overfit. This does not bias the
independent forward results, but it lowers the quality of the selected
allocation and makes the stored in-sample `cim_estimated_spread` unusable as an
honest estimate of the selected allocation.

## Correctness checks

- The custom CSR loader returns the same shape, indices, and index pointers as
  SciPy on every directly loadable graph. All stored edges are reciprocal, with
  no duplicate edges or self-loops.
- Node probabilities are nonnegative and sum to one. Isolated-node transfer
  mass is reassigned to discard.
- Exact tiny-world enumeration agrees with forward simulation, and fixed
  allocations agree with the conditioned RR estimator.
- On the real NetFacebookEgo balanced instance at `k=100`, a fixed allocation
  selected without the holdout samples has independent matrix spread 57.16.
  Ten independent 20,000-sample RR batches estimate 55.45 on average with
  batch standard deviation 2.59. This rules out the twofold estimator bias
  suggested by the training estimate.
- Independent matrix evaluation and forward simulation also agree closely for
  stored allocations; e.g., NetFacebookEgo balanced `k=100` is approximately
  56 under both.

## Sampling diagnosis

The validated-v2 rule uses 100,000 samples for `k<=50` but only 50,000 for
`k>=100`; the appendix uses 20,000 in every cell. These are empirical budgets,
not the theorem-level sample count.

For NetFacebookEgo forwarding-heavy at `k=100`, 50,000 joint samples produce
about 2,740 nonempty RR sets per coupon index in expectation for 2,888
candidates, or 0.95 observations per index-candidate pair. In the appendix
cell `(t,r)=(0.93,0.20)`, 20,000 samples provide only about 0.16 such
observations. Maximizing noisy marginals over thousands of candidates and 100
adaptive rounds therefore creates severe selection overfitting.

Across the two reference graphs and six budgets, the mean in-sample RR
optimism is 60.0% in the balanced scenario, 47.1% in adoption-heavy, and
134.2% in forwarding-heavy. This is selection-induced optimism, not bias for
a fixed allocation.

A post-hoc high-sample diagnostic on NetFacebookEgo forwarding-heavy at
`k=100` gives:

| Joint samples | Mean gap to matrix MC-Greedy | Mean training optimism |
|---:|---:|---:|
| 50,000 | 7.46% | 279.8% |
| 100,000 | 7.24% | 189.2% |
| 250,000 | 5.92% | 109.2% |
| 500,000 | 5.03% | 73.2% |

At 500,000 samples, mean matrix spread is 12.07 versus 11.73 for the strongest
stored non-oracle baseline in that configuration, an advantage of about 2.9%.
Thus insufficient sampling can change the baseline comparison, although the
remaining 5.0% gap shows that it is not the only explanation.

## Why simple baselines remain strong

- NetFacebookEgo is highly leaf dominated: 2,790 of 2,888 nodes have degree
  one. Many candidates therefore have nearly identical local behavior.
- For MC-Greedy reference allocations, immediate redemption accounts on
  average for 90.1% of eventual redemptions in adoption-heavy settings and
  58.2% in balanced settings. Alpha-Sort and 1Hop-Sort directly target this
  local signal.
- Relative to matrix MC-Greedy, the strongest baseline is on average 1.07%
  behind in adoption-heavy, 1.70% in balanced, and 3.48% in forwarding-heavy
  settings. There is limited room for any method to show a large gain in the
  first two regimes.

## Consequences and required action

1. Keep the independent forward results, but do not use
   `cim_estimated_spread` as a quality estimate after adaptive selection.
2. Treat the existing appendix as a fixed-budget stress test, not a converged
   map of the algorithm's attainable advantage.
3. Before submission, replace the fixed budget with a prespecified
   doubling/stability protocol and an independent validation batch. Final
   forward evaluation must remain separate from selection and validation.
4. Rerun the complete grid under the new protocol, retaining every negative
   cell. Do not increase samples only in unfavorable cells.
5. Consider a marginal sampler that forces the current coupon's root-adoption
   gate; the present all-`k` conditioning wastes most samples for a particular
   index when adoption is rare. Such a change requires a new proof and should
   not be presented as the current algorithm without re-analysis.

## Artifacts

- `experiments/audit_weak_gain.py`
- `experiments/audit_sample_scaling.py`
- `experiments/verify_weak_gain_audit.py`
- `experiments/results/diagnostics/weak_gain_decomposition.csv`
- `experiments/results/diagnostics/weak_gain_holdout.csv`
- `experiments/results/diagnostics/high_sample_scaling.csv`

