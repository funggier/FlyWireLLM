# Current Development Task

Current task: L004 — Base-50M Research-Only Pretraining Execution
Status: ACTIVE
GitHub Issue: #6
Branch: research/l004-base50m-pretraining

## Verified production progress

Latest exact-qualified checkpoint: optimizer step 104.

- supervised tokens seen: 6,815,744 / 500,000,000
- fraction of primary budget: 1.3631488%
- checkpoint SHA-256:
  `0f61a87d497543af516f3818d2d47ce21412fec009ac7a08492ee4396952726b`
- step104 evidence commit:
  `7997a1747cd822b2e36e44a8c6069bab22efe8e7`
- post-step104 validation gate: PASS
- final holdout touched: false

## Ninth warmup-aware tranche

Qualified runner:
`7af23fc1f62a69cf08b03cb1bf16ed4720915ba1`.

Authorization candidate:
`configs/pretraining-tranche-l004-v9.json`.

Authorization SHA-256:
`90bd19426caa899732f71fcd0ee6511f89e038f6d0cb678ee3b8b54f4ae1980b`.

Bound:

- source step: 104
- maximum additional updates: 16
- end step: 120
- additional supervised tokens: 1,048,576
- cumulative supervised tokens at bound: 7,864,320
- create checkpoint every update
- retain latest ninth-tranche checkpoint only
- prune old checkpoint only after new metric + atomic state commit
- same 300k validation pack required after step120
- final holdout must remain untouched
- public release remains not qualified

Do not execute step105 until v9 is committed, exact-qualified,
remote-synchronized, and `--validate-only` passes from the clean commit.
