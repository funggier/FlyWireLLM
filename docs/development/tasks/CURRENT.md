# Current Development Task

Current task: L004 — Base-50M Research-Only Pretraining Execution
Status: ACTIVE
GitHub Issue: #6
Branch: research/l004-base50m-pretraining

## Verified production progress

Latest exact-qualified checkpoint: optimizer step 233.

- supervised tokens seen: 15,269,888 / 500,000,000
- fraction of primary budget: 3.0539776%
- checkpoint SHA-256:
  `1513602d920fe57b7bb12ee7ee69cdf77ea23f39f0253cdabb718e55413dd964`
- step233 evidence commit:
  `2c4650b574cd9a86989313b0d43ecdee3f0f52c3`
- post-step233 validation gate: PASS
- final holdout touched: false

## Fourth post-warmup expanded tranche

Qualified runner:
`db93205e6a105b2697449f6517ffa2f8617ea7ae`.

Authorization candidate:
`configs/pretraining-tranche-l004-v15.json`.

Authorization SHA-256:
`84b8fdd089c1211e78d6359589886d4ed83b38b328d5222d5f418cae4efbd7a9`.

Bound:

- source step: 233
- maximum additional updates: 64
- end step: 297
- additional supervised tokens: 4,194,304
- cumulative supervised tokens at bound: 19,464,192
- qualified cosine-decay schedule unchanged
- LR step234: 0.000599843647
- LR step297: 0.000599505950
- create checkpoint every update
- retain latest post-warmup-4 checkpoint only
- prune old checkpoint only after new metric + atomic state commit
- same 300k validation pack required after step297
- final holdout must remain untouched
- public release remains not qualified

Do not execute step234 until v15 is committed, exact-qualified,
remote-synchronized, and `--validate-only` passes from the clean commit.
