# Method-Experiment Traceability

| Contribution | Method module | Experiment | Table/Figure | Allowed claim | Evidence status |
|---|---|---|---|---|---|
| Non-replicable coupon model | Forward simulator | Exact tiny-world enumeration, scenario study, and five-selection-seed analysis | Unit tests, combined quality figure, validated raw CSV | The forward implementation follows the fixed-realization coupon process | Exact check plus validated-v2 experiment |
| DR-submodular allocation | Greedy allocation | CIM-RIS versus 100,000-trajectory-per-source MC-Greedy | Aggregate table and quality figure | CIM-RIS remains within the reported 1.1--4.4% range on Netscience | Validated-v2 experiment |
| Coupon-aware RR sampling | k-joint RR generator | IC-RIS and heuristic comparison | Aggregate table and quality figure | CIM-RIS is competitive but does not uniformly dominate IC-RIS | Validated-v2 experiment |
| Sampling accuracy | Sequential marginal estimator | Four RR budgets on two graphs and three coupon budgets | Extension evidence figure (a--c) | More samples reduce the observed MC-Greedy gap in every prespecified forwarding-heavy configuration, but do not remove it | Validated-v2.3 extension |
| Capacity-aware allocation | Per-node placement capacities | $c_v\in\{1,2,k\}$ at $k=200$ | Extension evidence figure (d) | Repeated placement gives a modest gain in forwarding-heavy diffusion and is rarely selected otherwise | Validated-v2.3 extension |
| Conditioned importance sampler | Adoption-root sampler | Uniform versus conditioned ablation | Validated sampler-ablation figure | Conditioning removes zero-contribution samples and improves measured variance-time efficiency | Theorem plus validated-v2 experiment |
| Scalability | Inverted-list implementation | Five-run runtime scaling | Validated runtime figure | Selection completes within the reported times; cross-graph order depends on RR sizes | Validated-v2 experiment |
| Parameter-regime robustness | Coupon-aware marginal coverage | Complete transfer-by-redemption-share grid on two graphs, plus a three-budget slice | Appendix parameter-sweep figure and summary CSV | CIM-RIS may be identified as stronger only in the observed prespecified regions; unfavorable cells remain visible | Validated-v2.4 appendix experiment |
| Overlap mechanism | Distinct-adopter coverage | Duplicate-redemption fraction on every parameter-grid cell | Appendix mechanism heatmaps | Duplicate-redemption reduction does not consistently explain the observed spread differences | Validated-v2.4 appendix experiment |
