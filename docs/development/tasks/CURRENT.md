# Current Development Task

Current task: L004 — Base-50M Research-Only Pretraining Execution
Status: ACTIVE
GitHub Issue: #6
Branch: research/l004-base50m-pretraining

## Verified production progress

Latest exact-qualified checkpoint: optimizer step 169.

- supervised tokens seen: 11,075,584 / 500,000,000
- fraction of primary budget: 2.2151168%
- checkpoint SHA-256:
  `ccce01d29686a76edfe52fbaa531cf198a82bfd8aa17ded34625b8c6f14b979e`
- step169 evidence commit:
  `068a2a1eaec7affa84d719d4c1d847fe90ef5a88`
- post-step169 validation gate: PASS
- final holdout touched: false

## Second post-warmup scaling tranche

Qualified runner:
`d30273244407936a93f4bf7bb341828d2a357cbb`.

Authorization candidate:
`configs/pretraining-tranche-l004-v13.json`.

Authorization SHA-256:
`b72536d3968a89043b3d2f453ccd5b2258da9e2f1e0ec956733fc1f536857f02`.

Bound:

- source step: 169
- maximum additional updates: 32
- end step: 201
- additional supervised tokens: 2,097,152
- cumulative supervised tokens at bound: 13,172,736
- qualified cosine-decay schedule unchanged
- LR step170: 0.0005999931
- LR step201: 0.0005999451
- create checkpoint every update
- retain latest post-warmup-2 checkpoint only
- prune old checkpoint only after new metric + atomic state commit
- same 300k validation pack required after step201
- final holdout must remain untouched
- public release remains not qualified

Do not execute step170 until v13 is committed, exact-qualified,
remote-synchronized, and `--validate-only` passes from the clean commit.
