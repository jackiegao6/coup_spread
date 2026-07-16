## Task Packet

- Scope: determine whether the weak empirical advantage is caused by an
  implementation defect, finite sampling, evaluation noise, or intrinsically
  strong baselines under the tested diffusion regimes.
- Files to read: current manuscript model/algorithm; coupon simulator and RR
  implementation; validated-v2/v2.3/v2.4 runners, tests, raw results, and prior
  implementation audits.
- Files allowed to edit: new diagnostic/audit scripts and records under
  `experiments/` and `plan/review/`; fixes to implementation only if a defect is
  independently reproduced.
- Required skills: paper-orchestration, experiment-results-planning,
  peer-review, and verification.
- Required checks: probability normalization and graph direction; fixed-world
  forward semantics; conditioned RR unbiasedness; allocation/capacity logic;
  baseline parameter parity; common evaluation streams; tiny-instance exact
  agreement; high-sample reference decomposition.
- Rejection checks: do not infer correctness solely from passing tests; do not
  change parameters after seeing outcomes; do not label MC-Greedy exact; do not
  rewrite manuscript claims before the cause is identified.
- Validation commands: existing coupon-core tests and claim checkers; new
  exact/diagnostic checks; raw-result consistency checks; `git diff --check`.
