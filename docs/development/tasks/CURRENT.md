# Current Development Task

Current task:
L004 — Base-50M Research-Only Pretraining Execution

Status: ACTIVE

GitHub Issue: #6

Branch:
research/l004-base50m-pretraining

## Verified production progress

Latest verified checkpoint candidate: optimizer step 56.

- supervised tokens seen: 3,670,016 / 500,000,000;
- fraction of primary budget: 0.7340032%;
- checkpoint SHA-256:
  `4095c686eeff521c906a73ef34d890135cd5c4900addab54b7477f6f72a4d116`;
- external state SHA-256:
  `52ebd96c161855febc8bf1cf45c8b0b81780b1be264a8bfa37634f045cecf86e`;
- sixth-tranche result SHA-256:
  `ddab037da4623f706ae5d162cb2dd088c045b3972467c4497be2f68c87cb253f`;
- pretraining complete: false;
- public release eligibility: not qualified.

## Rolling checkpoint retention

- retained sixth-tranche full checkpoints: 1;
- historical sixth-tranche checkpoints pruned: 15;
- step41-56 metrics retained: 16 / 16;
- final checkpoint reload: PASS.

## Post-step56 validation gate

Byte-identical 300k validation pack:

- English: 8.340751974 -> 7.532882587 (-9.685810%);
- Thai: 8.614949760 -> 7.848998205 (-8.890958%);
- technical: 8.388065623 -> 7.615953266 (-9.204892%);
- combined: 8.447922452 -> 7.665944686 (-9.256451%);
- final holdout touched: false;
- gate: PASS.

Tracked hashes:

- validation step56:
  `c7e663ec5e66bff5473ce1a3282dd910ea546fe7f90bc78c40083238696d0380`;
- step40-to-step56 comparison:
  `3941421b63283cdfd530c2c2be7ced5421330338192bea68c00b0db2bba14dad`.

## Active work

Record and exact-qualify the complete step56 evidence before any new training
authorization.

Do not execute step57 until the evidence commit is exact-qualified and
remote-synchronized.
