# Current Development Task

Current task: L004 — Base-50M Research-Only Pretraining Execution
Status: ACTIVE
GitHub Issue: #6
Branch: research/l004-base50m-pretraining

## Verified production progress

Latest verified checkpoint candidate: optimizer step 136.

- supervised tokens seen: 8,912,896 / 500,000,000
- fraction of primary budget: 1.7825792%
- checkpoint SHA-256:
  `8ee3cb135acd1058321c3838919dcd6e61b0fa98d37edd8cd2b32fb17d08f3a1`
- external state SHA-256:
  `3c2d16cd47bbff977e52ec7b5a022dcbc5125eb0646227d14e4121c3fa85a5cb`
- tenth-tranche result SHA-256:
  `fc762526743473932b628ea50456eb668dc6ed9592b7baaff67edb331c4d2b22`
- pretraining complete: false
- public release eligibility: not qualified

## Rolling checkpoint retention

- retained tenth-tranche full checkpoints: 1
- historical tenth-tranche checkpoints pruned: 15
- step121-136 metrics retained: 16 / 16
- final checkpoint reload: PASS

## Post-step136 validation gate

Byte-identical 300k validation pack:

- English: 6.567051706 -> 6.374071930 (-2.938606%)
- Thai: 7.236243482 -> 7.094012272 (-1.965539%)
- technical: 6.737619036 -> 6.578042986 (-2.368434%)
- combined: 6.846971408 -> 6.682042396 (-2.408788%)
- final holdout touched: false
- gate: PASS

Tracked hashes:

- validation step136:
  `a0710642d686aa6cbed38d513eabfc540312f8e8c7fc242a98d65e9e5d90604b`
- step120-to-step136 comparison:
  `2790a9f846a3c11349c4b9e0bc4a7d8981654c9e36ca9ed6ec6d50404ba0d64b`

## Active work

Record and exact-qualify the complete step136 evidence before any new training
authorization.

Do not execute step137 until the evidence commit is exact-qualified and
remote-synchronized. The next bounded tranche should end exactly at optimizer
warmup boundary step153.
