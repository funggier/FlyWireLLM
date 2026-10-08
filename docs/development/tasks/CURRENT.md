# Current Development Task

Current task: L004 — Base-50M Research-Only Pretraining Execution
Status: ACTIVE
GitHub Issue: #6
Branch: research/l004-base50m-pretraining

## Verified production progress

Latest exact-qualified checkpoint: optimizer step 297.

- supervised tokens seen: 19,464,192 / 500,000,000
- fraction of primary budget: 3.8928384%
- checkpoint SHA-256:
  `f0917d81130d3fb8ff6c59d28cfd0b662633b68ae96b36c5d75caceb179870fc`
- step297 evidence commit:
  `4b8775c324f004d406a5340324c85ac37e8c8642`
- post-step297 validation gate: PASS
- final holdout touched: false

## Fifth post-warmup accelerated tranche

Qualified runner:
`39b696a3495db40fffb2cc9ca69838058cc0c408`.

Authorization candidate:
`configs/pretraining-tranche-l004-v16.json`.

Authorization SHA-256:
`71ce46ad2366ef60135abfc99ecb0b097782e1e322e35b906049b8c5cfe990ea`.

Bound:

- source step: 297
- maximum additional updates: 128
- end step: 425
- additional supervised tokens: 8,388,608
- cumulative supervised tokens at bound: 27,852,800
- cosine LR step298: 0.000599499066
- cosine LR step425: 0.000598238658
- create checkpoint every update
- retain latest post-warmup-5 checkpoint only
- prune old checkpoint only after new metric + atomic state commit
- same 300k validation pack required after step425
- final holdout must remain untouched
- public release remains not qualified

Do not execute step298 until v16 is committed, exact-qualified,
remote-synchronized, and `--validate-only` passes from the clean commit.
