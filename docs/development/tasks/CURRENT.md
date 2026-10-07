# Current Development Task

Current task: L004 — Base-50M Research-Only Pretraining Execution
Status: ACTIVE
GitHub Issue: #6
Branch: research/l004-base50m-pretraining

## Verified production progress

Latest exact-qualified checkpoint: optimizer step 136.

- supervised tokens seen: 8,912,896 / 500,000,000
- fraction of primary budget: 1.7825792%
- checkpoint SHA-256:
  `8ee3cb135acd1058321c3838919dcd6e61b0fa98d37edd8cd2b32fb17d08f3a1`
- step136 evidence commit:
  `ead9cbe456a1ac5ce0e92f0d6a8281d93aafe9d0`
- post-step136 validation gate: PASS
- final holdout touched: false

## Exact warmup-boundary tranche

Qualified runner:
`ee8c36a082cab5e07d0346ff530902e4b6a4136e`.

Authorization candidate:
`configs/pretraining-tranche-l004-v11.json`.

Authorization SHA-256:
`b8482919369a8cc402f8addaa7c1a122d59a1fd28ce52e9dfe9b78fb78eb1f3b`.

Bound:

- source step: 136
- maximum additional updates: 17
- end step: 153
- additional supervised tokens: 1,114,112
- cumulative supervised tokens at bound: 10,027,008
- create checkpoint every update
- retain latest warmup-boundary checkpoint only
- prune old checkpoint only after new metric + atomic state commit
- same 300k validation pack required after step153
- final holdout must remain untouched
- public release remains not qualified

Do not execute step137 until v11 is committed, exact-qualified,
remote-synchronized, and `--validate-only` passes from the clean commit.
Step154 is outside this authorization.
