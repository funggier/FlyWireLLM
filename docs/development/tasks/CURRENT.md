# Current Development Task

Current task: L004 — Base-50M Research-Only Pretraining Execution
Status: ACTIVE
GitHub Issue: #6
Branch: research/l004-base50m-pretraining

## Verified production progress

Latest exact-qualified checkpoint: optimizer step 201.

- supervised tokens seen: 13,172,736 / 500,000,000
- fraction of primary budget: 2.6345472%
- checkpoint SHA-256:
  `11fe2a45b65d43bec6a6710d64c2110a1be4cc11042e1fcbf1b6197a9feaa2a0`
- step201 evidence commit:
  `789edfeff672a0bfe56fdf909742746b46ab3bf6`
- post-step201 validation gate: PASS
- final holdout touched: false

## Third post-warmup tranche

Qualified runner:
`585aad92e6140725b8ccf7e75e0f107c974a378a`.

Authorization candidate:
`configs/pretraining-tranche-l004-v14.json`.

Authorization SHA-256:
`0e776c62ab1fde305e9524a422a512ffe895c50f438c4901e9b83a857fb04680`.

Bound:

- source step: 201
- maximum additional updates: 32
- end step: 233
- additional supervised tokens: 2,097,152
- cumulative supervised tokens at bound: 15,269,888
- qualified cosine-decay schedule unchanged
- LR step202: 0.0005999428
- LR step233: 0.0005998475
- create checkpoint every update
- retain latest post-warmup-3 checkpoint only
- prune old checkpoint only after new metric + atomic state commit
- same 300k validation pack required after step233
- final holdout must remain untouched
- public release remains not qualified

Do not execute step202 until v14 is committed, exact-qualified,
remote-synchronized, and `--validate-only` passes from the clean commit.
