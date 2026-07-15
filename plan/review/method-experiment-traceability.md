# Method-Experiment Traceability

| Contribution | Method module | Experiment | Table/Figure | Allowed claim | Evidence status |
|---|---|---|---|---|---|
| Non-replicable coupon model | Forward simulator | Exact tiny-world enumeration, scenario study, and five-selection-seed analysis | Unit tests, combined quality figure, validated raw CSV | The forward implementation follows the fixed-realization coupon process | Exact check plus validated-v2 experiment |
| DR-submodular allocation | Greedy allocation | CIM-RIS versus 100,000-trajectory-per-source MC-Greedy | Aggregate table and quality figure | CIM-RIS remains within the reported 1.1--4.4% range on Netscience | Validated-v2 experiment |
| Coupon-aware RR sampling | k-joint RR generator | IC-RIS and heuristic comparison | Aggregate table and quality figure | CIM-RIS is competitive but does not uniformly dominate IC-RIS | Validated-v2 experiment |
| Conditioned importance sampler | Adoption-root sampler | Uniform versus conditioned ablation | Validated sampler-ablation figure | Conditioning removes zero-contribution samples and improves measured variance-time efficiency | Theorem plus validated-v2 experiment |
| Scalability | Inverted-list implementation | Five-run runtime scaling | Validated runtime figure | Selection completes within the reported times; cross-graph order depends on RR sizes | Validated-v2 experiment |
