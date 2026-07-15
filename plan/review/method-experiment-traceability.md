# Method-Experiment Traceability

| Contribution | Method module | Experiment | Table/Figure | Allowed claim | Evidence status |
|---|---|---|---|---|---|
| Non-replicable coupon model | Forward simulator | Forward/RR estimator check, scenario study, and five-selection-seed stability analysis | Quality figures plus stability CSV | The implementation follows the defined coupon process and conclusions are not an artifact of one selection seed | Static check plus real experiment |
| DR-submodular allocation | Greedy allocation | CIM-RIS versus MC-Greedy | Quality versus budget | CIM-RIS remains within the reported gap from the Monte Carlo greedy reference | Real experiment |
| Coupon-aware RR sampling | k-joint RR generator | IC-RIS and heuristic comparison | Quality versus budget | CIM-RIS is competitive but does not uniformly dominate | Real experiment |
| Conditioned importance sampler | Adoption-root sampler | Uniform versus conditioned ablation | Sampler-ablation figure | Conditioning reduces zero-contribution work and improves variance-time efficiency for rare adoption | Theorem plus real experiment |
| Scalability | Inverted-list implementation | Runtime scaling | Runtime/scalability figure | Selection completes within the reported times; cross-graph order depends on RR sizes | Real experiment |
