# Forward baseline integration pilot

`netscience-balanced-k10.json` records a real execution of five selectors on
Netscience, followed by 2,000 independent forward realizations per allocation.
It is an integration pilot, not publication comparison evidence.

Methods: CIM-RIS, IC-RIS, Forward-Lazy, Forward-Stochastic, Random. Forward
methods use 100 batches per source and share a frozen training bank; RIS methods
use 2,000 RR samples. These units do not represent equal work. All methods use
the same independent evaluation streams. Selection times include each method's
training generation, exclude graph loading and final evaluation, and come from
one run only. Array storage counts exclude Python overhead and are not peak
memory. Do not use these measurements to claim speedups or significant gains.

Reproduce into a new file (existing outputs are never overwritten):

```powershell
py -3.12 experiments/run_forward_baseline_pilot.py --output experiments/results/ccfa-baseline-pilot/replay.json
```

Compare configurations, selected seeds, training values and evaluation means;
timings are expected to differ. The JSON records source/data SHA-256 hashes,
Python/NumPy versions, selection streams and evaluation seed.

After the pilot, the working core's mixed CRLF/LF endings were normalized to
LF for clean version-control diffs. `source-receipt.json` links the unchanged
pilot fingerprint to its byte-exact snapshot under `sources/`. The current core
equals that snapshot after CRLF-to-LF conversion only. No result fingerprint
was rewritten; verify the old pilot against the snapshot and its recorded
digest. New runs fingerprint the current LF source.

Full paper comparison still requires a prespecified multi-graph/scenario and
repeated-seed protocol, strong existing baselines, resource trade-off curves,
and complete memory and preprocessing accounting. Existing manuscript results
have not been replaced with this pilot.
