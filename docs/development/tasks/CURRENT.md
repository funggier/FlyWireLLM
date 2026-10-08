# Current Development Task

Current task: L004 — Base-50M Research-Only Pretraining Execution
Status: ACTIVE
GitHub Issue: #6
Branch: research/l004-base50m-pretraining

## Verified production progress

Latest verified checkpoint candidate: optimizer step 297.

- supervised tokens seen: 19,464,192 / 500,000,000
- fraction of primary budget: 3.8928384%
- checkpoint SHA-256:
  `f0917d81130d3fb8ff6c59d28cfd0b662633b68ae96b36c5d75caceb179870fc`
- external state SHA-256:
  `bcb97673f3ce6a02f6149e03ee8c7ddecdc8ab19a647e9ec8c41c9f9fb1e28bf`
- post-warmup-4 result SHA-256:
  `83722298982d9deaa929a7fd89baf527d589f8b16a188b637198723100792450`
- pretraining complete: false
- public release eligibility: not qualified

## Rolling checkpoint retention

- retained post-warmup-4 full checkpoints: 1
- historical post-warmup-4 checkpoints pruned: 63
- step234-297 metrics retained: 64 / 64
- final checkpoint reload: PASS

## Post-step297 validation gate

Byte-identical 300k validation pack:

- English: 5.791585690 -> 5.577049066 (-3.704281%)
- Thai: 6.260309313 -> 5.905298695 (-5.670816%)
- technical: 6.005837822 -> 5.766543058 (-3.984369%)
- combined: 6.019244275 -> 5.749630273 (-4.479200%)
- final holdout touched: false
- gate: PASS

Tracked hashes:

- validation step297:
  `f1e08c42cbe62e381837370a3d3dc7000e79f0c9622311dc706140fd19906e57`
- step233-to-step297 comparison:
  `b7e048dbc355171e6fa46017dc57f1fbc8ef9475db98281732d98fa0b419e0ed`

## Active work

Record and exact-qualify the complete step297 evidence before any new training
authorization.

Do not execute step298 until the evidence commit is exact-qualified and
remote-synchronized.
