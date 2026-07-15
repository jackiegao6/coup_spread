# Evidence Map

| Source ID | Citation | Source type | Abstract/full-text finding | Usable fact | Supported claim | Citation slot | Risk |
|---|---|---|---|---|---|---|---|
| E-KONECT | Kunegis, 2013, *KONECT: The Koblenz Network Collection* | Dataset paper | KONECT is a collection of network datasets; the supplied NeurIPS paper also states that its graph datasets, including Douban, come from KONECT. | The experimental graph instances are derived from KONECT network data. | Dataset provenance in Section 7.1. | Experiments-Datasets | Low |
| E-CUBIC-VC | Garey, Johnson, and Stockmeyer, 1976, *Some Simplified NP-Complete Graph Problems* | Complexity paper | Establishes NP-completeness for restricted graph problems including bounded-degree vertex cover. | Vertex cover on cubic graphs is a valid NP-hard source problem. | Section 4.1 reduction opening | Low |
| E-PRM | Liao et al., 2023, *Popularity Ratio Maximization: Surpassing Competitors through Influence Propagation*, DOI 10.1145/3589309 | Full paper and author repository | PRM allocates promotional seeds across rounds and uses coupon distribution as motivation, but its PA-IC model propagates influence through the standard copyable IC cascade. | PRM is the closest verified coupon-motivated seed-allocation work, while its diffusion object and objective differ from CIM. | Related Work-Diffusion variants | Low |
