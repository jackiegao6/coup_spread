# Historical source snapshot

`run_real_submission-240ede44730726a4.py` is the byte-exact source that produced
the evidence-20260913 records. SHA-256:

`240ede44730726a4ed8170e8b7cdf97b33bec6ea94a5fa2b5dd4dc96e9227f20`

The working core now handles empty budgets, zero adoption, infeasible capacity,
and cancellation of extremely small root weights. Positive-weight historical
branches retain their RNG calls. Regression tests compare selections, training
estimates and membership counts, excluding elapsed time.

The evidence verifier accepts this snapshot only for the registered historical
core path and digest, verifies its bytes, and explicitly reports that current
working code differs. Every other unexpected mismatch still fails. It does not
rewrite metadata or certify new code using historical hashes.

To reproduce an old run, stage this file as `experiments/run_real_submission.py`
in an isolated copy of the project with the recorded data and other source
files. Do not run the nested snapshot directly: its relative project paths
assume the original location. `.gitattributes` prevents newline conversion of
the snapshot.
