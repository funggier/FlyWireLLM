# Current Development Task

Current task:
L004 — Base-50M Research-Only Pretraining Execution

Status: ACTIVE

GitHub Issue: #6

Branch:
research/l004-base50m-pretraining

## Verified production progress

Latest verified checkpoint candidate: optimizer step 32.

- supervised tokens seen: 2,097,152 / 500,000,000;
- fraction of primary budget: 0.4194304%;
- checkpoint SHA-256:
  `136f554d7321be08ec266ac44296ae6870aadc7edb8d468812173ab240b8a463`;
- external state SHA-256:
  `57b868d833bb11c96ffce634744b59a5d02c86c5868085ac7f04f4d4476c3951`;
- fourth-tranche result SHA-256:
  `1d0cdde6be95850dc70f691ae7b59115bfe40afa384e440a323a2f4901a01c39`;
- pretraining complete: false;
- public release eligibility: not qualified.

Fourth bounded tranche step17-32 completed within authorization
`8ef449c63c6be4e2129a0604413fbfbb13f6d706a2c046d779079415981f0259`.

## Post-step32 validation gate

Byte-identical 300k validation pack:

- English: 9.584511214 -> 8.823402260 (-7.941030%);
- Thai: 9.824723932 -> 9.078124258 (-7.599192%);
- technical: 9.606904253 -> 8.857479007 (-7.800903%);
- combined: 9.672046466 -> 8.919668508 (-7.778891%);
- final holdout touched: false;
- gate: PASS.

Tracked hashes:

- validation step32:
  `34b4660037261fb7779fa8ce86ac2da5b1c323d805634bcaf10f3f0fdc5c2dc8`;
- step16-to-step32 comparison:
  `7182985f612cf6b708256abc88e12f040f5af82a58db42d1f8b9fa3bdc54ab27`.

## Active work

Record and exact-qualify the complete step32 evidence before any new training
authorization.

Do not execute step33 until the evidence commit is exact-qualified and
remote-synchronized. Any next expansion must remain bounded, predeclare its
post-tranche validation gate, and keep final holdout untouched.
