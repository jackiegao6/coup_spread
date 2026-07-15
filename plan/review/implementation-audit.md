# Coupon Experiment Implementation Audit

## Scope

This audit compares the historical implementation under `gzc-impl/` and the
traceable runner under `experiments/` with the model and estimator stated in
`paper-v2 copy.tex`. Historical result files are not accepted as submission
evidence.

## Historical pipeline

The old pipeline is unsuitable for a clean rerun for four independent reasons.

1. `SSR_method.py` contains a `path_aware` root-event mode that replaces the
   root adoption event with downstream redemption probability. That event does
   not identify the final adopter represented by an RR root and therefore does
   not estimate the paper's distinct-adopter objective. Old outputs named
   `ris_path_aware` cannot be assumed to follow the current estimator.
2. Forward evaluation uses the global NumPy random generator without recording
   or resetting a per-run seed. Repeating the same command therefore need not
   reproduce the same means or selected ordering.
3. Seed caches and result CSV files are reused or opened in append mode. A run
   can silently consume stale allocations or mix rows generated under different
   code revisions.
4. Several historical comparisons use only 100--600 forward simulations for
   stochastic marginal estimates, while mock scripts coexist with plotting and
   result directories. This is too noisy and too difficult to trace for final
   evidence.

## Traceable reference runner

`experiments/run_real_submission.py` follows the current independent-coupon
model: one action is sampled on the first visit to a node, a repeated visit ends
in a transfer cycle, adopters are deduplicated across coupons, and reverse
samples use the root adoption event. It writes the selected nodes, master seed,
sample count, evaluation count, and software versions.

An exact rerun of Netscience, balanced scenario, and `k=10` reproduced every
stored result field other than wall-clock time. This establishes procedural
reproducibility for that configuration.

The reference runner is nevertheless not yet the final experimental pipeline:

- it has no exact small-instance correctness tests;
- its main curves use one stochastic selection run;
- fixed budgets of 10,000--20,000 joint samples are too small in low Blocks of
  adoption probability;
- the Monte Carlo greedy reference is available only on Netscience;
- its confidence intervals cover forward evaluation conditional on one selected
  allocation rather than full-pipeline selection variability.

## Sampling-budget diagnostic

On Netscience under the forwarding-heavy scenario, three independent CIM-RIS
selections were compared with a 30,000-trajectory-per-source MC-Greedy reference
and evaluated with 20,000 forward simulations. Increasing the number of joint
RR samples consistently reduced the mean quality gap:

| Budget | 5,000 samples | 20,000 samples | 50,000 samples | 100,000 samples |
|---:|---:|---:|---:|---:|
| 10 | 11.70% | 7.64% | 4.82% | 2.40% |
| 50 | 9.17% | 6.42% | 5.29% | 3.35% |
| 100 | 6.78% | 4.52% | 3.17% | 2.79% |
| 200 | 4.25% | 2.47% | 1.38% | 0.98% |

The historical weak quality is therefore substantially explained by estimator
variance at the fixed sample budget. This diagnostic is exploratory and will
not be reported as a confirmatory result unless it is rerun under the locked
protocol.

## Required corrections

1. Add exact enumeration and estimator tests before any full run.
2. Pin the execution environment and record hardware automatically.
3. Lock sample budgets using a convergence criterion before comparing methods.
4. Run multiple independent selection seeds and retain raw per-run records.
5. Use common evaluation seeds within each configuration and report both
   selection variability and forward-simulation uncertainty.
6. Preserve negative or tied comparisons and describe diffusion probabilities
   as controlled synthetic settings on real graph topologies.

