# Current Development Task

Current task: L004 — Base-50M Research-Only Pretraining Execution
Status: ACTIVE
GitHub Issue: #6
Branch: research/l004-base50m-pretraining

## Verified production progress

Latest exact-qualified checkpoint: optimizer step 120.

- supervised tokens seen: 7,864,320 / 500,000,000
- fraction of primary budget: 1.572864%
- checkpoint SHA-256:
  `f8b6f447124c9f08250da126fcecca827ffbbb8b23d93338934a01031ea5086d`
- step120 evidence commit:
  `9f951c0c33b00006e7ffb376626acfcf4c60b30d`
- post-step120 validation gate: PASS
- final holdout touched: false

## Tenth warmup-aware tranche

Qualified runner:
`44586b6f643e152776129297be474e487111d694`.

Authorization candidate:
`configs/pretraining-tranche-l004-v10.json`.

Authorization SHA-256:
`1fc3a96cf3d64e86d3bfd533f8ab5e7509b9dd69d0bb7773d1cdd9b95a75182a`.

Bound:

- source step: 120
- maximum additional updates: 16
- end step: 136
- additional supervised tokens: 1,048,576
- cumulative supervised tokens at bound: 8,912,896
- create checkpoint every update
- retain latest tenth-tranche checkpoint only
- prune old checkpoint only after new metric + atomic state commit
- same 300k validation pack required after step136
- final holdout must remain untouched
- public release remains not qualified

Do not execute step121 until v10 is committed, exact-qualified,
remote-synchronized, and `--validate-only` passes from the clean commit.
