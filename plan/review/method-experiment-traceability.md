# Method-Experiment Traceability

| Contribution | Method module | Experiment | Table/Figure | Allowed claim | Evidence status |
|---|---|---|---|---|---|
| Non-replicable coupon model | Forward simulator | Exact tiny-graph validation and scenario study | Future exact-validation table; quality figure | The implementation follows the defined coupon process | Pending real run |
| DR-submodular allocation | Greedy allocation | CIM-RIS versus MC-CELF | Quality versus budget | CIM-RIS approaches the Monte Carlo greedy reference | Synthetic layout only |
| Coupon-aware RR sampling | k-joint RR generator | IMM and heuristic comparison | Quality versus budget | Coupon-aware optimization improves over mismatched baselines | Synthetic layout only |
| Conditioned importance sampler | Adoption-root sampler | Uniform versus conditioned ablation | Sampler-ablation figure | Conditioning reduces zero-contribution work without changing expectation | Theorem proved; empirical evidence pending |
| Scalability | Inverted-list implementation | Runtime and memory scaling | Runtime/scalability figure | Runtime scales with graph size under the tested settings | Synthetic layout only |

