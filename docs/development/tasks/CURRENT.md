# Current Development Task

Current task: L004 — Base-50M Research-Only Pretraining Execution
Status: ACTIVE
GitHub Issue: #6
Branch: research/l004-base50m-pretraining

## Verified production progress

Latest exact-qualified checkpoint: optimizer step 153.

- optimizer warmup complete: true
- supervised tokens seen: 10,027,008 / 500,000,000
- fraction of primary budget: 2.0054016%
- checkpoint SHA-256:
  `f00d36e288094aab5bc0bec52a62dd134e92505ee6a87f3797e8e70d05cc3700`
- step153 evidence commit:
  `988e10fd0c419dee91abb00224f0d96e0e02e1d1`
- post-step153 validation gate: PASS
- final holdout touched: false

## First post-warmup tranche

Qualified runner:
`ed93936c426a3b6212659ddf7e30a835ea8b86bd`.

Authorization candidate:
`configs/pretraining-tranche-l004-v12.json`.

Authorization SHA-256:
`d8e1ffe91ab4af0aed58fc5635674e9b87923b2cc96347344283706743d94846`.

Bound:

- source step: 153
- maximum additional updates: 16
- end step: 169
- additional supervised tokens: 1,048,576
- cumulative supervised tokens at bound: 11,075,584
- existing qualified cosine-decay schedule only
- LR step154: 0.0006000000
- LR step169: 0.0005999939
- create checkpoint every update
- retain latest post-warmup checkpoint only
- prune old checkpoint only after new metric + atomic state commit
- same 300k validation pack required after step169
- final holdout must remain untouched
- public release remains not qualified

Do not execute step154 until v12 is committed, exact-qualified,
remote-synchronized, and `--validate-only` passes.
