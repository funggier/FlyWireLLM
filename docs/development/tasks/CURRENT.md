# Current Development Task

Current task: L004 — Base-50M Research-Only Pretraining Execution
Status: ACTIVE
GitHub Issue: #6
Branch: research/l004-base50m-pretraining

## Verified production progress

Latest verified checkpoint candidate: optimizer step 201.

- supervised tokens seen: 13,172,736 / 500,000,000
- fraction of primary budget: 2.6345472%
- checkpoint SHA-256:
  `11fe2a45b65d43bec6a6710d64c2110a1be4cc11042e1fcbf1b6197a9feaa2a0`
- external state SHA-256:
  `267f3e3bdcbb01743cd92d800518fda502d9b9642ae00186638e9b5ecd9183ac`
- post-warmup-2 result SHA-256:
  `75ac56438958999c683518dbc04f06c98672d649fc3a9074328d2aac6dd39dfd`
- optimizer warmup complete: true
- pretraining complete: false
- public release eligibility: not qualified

## Rolling checkpoint retention

- retained post-warmup-2 full checkpoints: 1
- historical post-warmup-2 checkpoints pruned: 31
- step170-201 metrics retained: 32 / 32
- final checkpoint reload: PASS

## Post-step201 validation gate

Byte-identical 300k validation pack:

- English: 6.117310139 -> 5.931827876 (-3.032089%)
- Thai: 6.745732500 -> 6.452203094 (-4.351335%)
- technical: 6.337900238 -> 6.140651583 (-3.112208%)
- combined: 6.400314292 -> 6.174894184 (-3.522016%)
- final holdout touched: false
- gate: PASS

Tracked hashes:

- validation step201:
  `1b84b6ff741701ada4a1980030a668ed217bf1fff368aa6c86adff0fb0ad3b2d`
- step169-to-step201 comparison:
  `a791e099ab10ba971d6313bfa1b79578746c0f5223076bbcda56e8a2ede97385`

## Active work

Record and exact-qualify the complete step201 evidence before any new
training authorization.

Do not execute step202 until the evidence commit is exact-qualified and
remote-synchronized. Continue the already-qualified cosine-decay schedule,
keep final holdout untouched, and remain research-only.
