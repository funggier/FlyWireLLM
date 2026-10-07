# Current Development Task

Current task: L004 — Base-50M Research-Only Pretraining Execution
Status: ACTIVE
GitHub Issue: #6
Branch: research/l004-base50m-pretraining

## Verified production progress

Latest verified checkpoint candidate: optimizer step 120.

- supervised tokens seen: 7,864,320 / 500,000,000
- fraction of primary budget: 1.572864%
- checkpoint SHA-256:
  `f8b6f447124c9f08250da126fcecca827ffbbb8b23d93338934a01031ea5086d`
- external state SHA-256:
  `25cf31506c855ef85aed5f4e0d3de9563a2c7206f9e368767ecea71fc4e4bb7b`
- ninth-tranche result SHA-256:
  `fd25888ce3b055de25737a2c0a6a6479f75ad3f01191b6bbfd8a359a02c3bbf1`
- pretraining complete: false
- public release eligibility: not qualified

## Rolling checkpoint retention

- retained ninth-tranche full checkpoints: 1
- historical ninth-tranche checkpoints pruned: 15
- step105-120 metrics retained: 16 / 16
- final checkpoint reload: PASS

## Post-step120 validation gate

Byte-identical 300k validation pack:

- English: 6.776078678 -> 6.567051706 (-3.084778%)
- Thai: 7.331277076 -> 7.236243482 (-1.296276%)
- technical: 6.907709166 -> 6.737619036 (-2.462323%)
- combined: 7.005021640 -> 6.846971408 (-2.256242%)
- final holdout touched: false
- gate: PASS

Tracked hashes:

- validation step120:
  `854aa31870e7af6c7454a3a7608130c67623f071d60980d3208bc29474896c43`
- step104-to-step120 comparison:
  `6d7a984b80fe5aeec65c9569d26f16192606f65dce55f9274e0ba54fddb761f3`

## Active work

Record and exact-qualify the complete step120 evidence before any new training
authorization.

Do not execute step121 until the evidence commit is exact-qualified and
remote-synchronized. Continue warmup-aware cadence because optimizer warmup
ends at step153.
