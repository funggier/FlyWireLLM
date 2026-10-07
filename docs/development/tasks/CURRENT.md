# Current Development Task

Current task:
L004 — Base-50M Research-Only Pretraining Execution

Status: ACTIVE

GitHub Issue: #6

Branch:
research/l004-base50m-pretraining

## Production progress

Verified source checkpoint: optimizer step 8.

- supervised tokens seen: 524,288 / 500,000,000;
- fraction of primary budget: 0.1048576%;
- checkpoint SHA-256:
  `2552f503438bf635f0e54a4fc1fc4bdcef3da57ad734c9ea7b0eebd379500e21`;
- post-step8 validation gate: PASS;
- final holdout touched: false.

## Third bounded expansion

Qualified runner:
`4e0ef81406da2adc2db44a0fc08b13a9ebde57b4`.

Authorization candidate:
`configs/pretraining-tranche-l004-v3.json`.

Authorization SHA-256:
`a8be24f47fa5a98be39ebd3ba46c6a12591c2e4b5744d42a6ef5f5da8419def0`.

Bound:

- source step: 8;
- maximum additional updates: 8;
- end step: 16;
- additional supervised tokens: 524,288;
- cumulative supervised tokens at bound: 1,048,576;
- checkpoint every update;
- same 300k validation pack required after step16;
- final holdout must remain untouched;
- public release remains not qualified.

Do not execute step 9 until the authorization is committed, exact-qualified,
remote-synchronized, and `--validate-only` passes on the clean commit.
