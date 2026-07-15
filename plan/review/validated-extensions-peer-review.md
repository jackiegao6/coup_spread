# Validated Experiment Extensions Peer Review

## Overall assessment

The validated-v2.3 extension closes the three empirical gaps identified after
the main rebuild: MC-Greedy is now available on a graph with 2,888 nodes,
sample-count sensitivity is measured under the hardest diffusion regime, and
the capacity formulation is exercised with repeated placements. The evidence
supports a more credible but less uniformly favorable account of CIM-RIS.

## Spec-compliance review

- PASS: six model-consistent MC destination matrices cover two graphs and all
  three scenarios. Netscience uses 100,000 trajectories per source and
  NetFacebookEgo uses 50,000.
- PASS: 180 strong-reference rows, 90 capacity rows, and 120 sensitivity rows
  carry `REAL_EXPERIMENT` and `validated-v2.3-extension` labels.
- PASS: every summary has five selection/evaluation repeats and all final
  evaluations use 10,000 realizations independent of reference construction.
- PASS: all prespecified negative and non-monotone outcomes are retained.
- PASS: the proposed absorbing-Markov shortcut was rejected before the run
  when a tiny cyclic graph showed that it violates fixed-action cycle
  semantics; the manuscript's corresponding linear-system claim was removed.

## Quality review

- The NetFacebookEgo reference is materially stronger than a structural or IC
  baseline. CIM-RIS is within 0.3--2.1% in balanced/adoption-heavy settings but
  4.4--11.1% behind in forwarding-heavy diffusion.
- Increasing the sample count from 5,000 to 100,000 lowers the mean gap in all
  six network--budget configurations. The average falls from 8.90% to 5.09%,
  so variance is important but does not fully explain the low-budget large-graph
  gap.
- Repeated placement is rarely selected when adoption is common. Under
  forwarding-heavy diffusion, it gives 1.60--2.09% on Netscience and about
  0.25% on NetFacebookEgo, supporting the capacity model without overstating
  its practical effect.
- The revised manuscript explicitly reports the 11.1% worst case and does not
  describe MC-Greedy as exact or globally optimal.

## Residual risks

- MC-Greedy remains unavailable on DoubanRandom and EmailEnron because its
  dense destination matrix and per-source trajectories scale poorly.
- Sample sensitivity is evaluated only in forwarding-heavy diffusion, chosen
  before the extension run as the stress-test regime.
- Capacity is evaluated at `k=200`, where the distinct-seed constraint is most
  likely to bind; smaller-budget capacity effects are not reported.
- Node probabilities remain controlled synthetic stress-test parameters rather
  than estimates from observed campaigns.
