# Current Development Task

Current task:
L004 — Base-50M Research-Only Pretraining Execution

Status: ACTIVE

GitHub Issue: #6

Branch:
research/l004-base50m-pretraining

## Verified production progress

Latest verified checkpoint candidate: optimizer step 16.

- supervised tokens seen: 1,048,576 / 500,000,000;
- fraction of primary budget: 0.2097152%;
- checkpoint SHA-256:
  `07a968b8e4c8a7f6711f24cd3466c46c374e09628c7d7c65848a2166efdcd71d`;
- external state SHA-256:
  `ef8c3a15b902e2d65d6fc4597b977cd732d2969931e5926ff6b914d655651107`;
- third-tranche result SHA-256:
  `7a87e2adcd326260909f1ddff59d09c82589b13eabb7ccdc6c19ea10f963ddf3`;
- pretraining complete: false;
- public release eligibility: not qualified.

Third bounded tranche step9-16 completed within authorization
`a8be24f47fa5a98be39ebd3ba46c6a12591c2e4b5744d42a6ef5f5da8419def0`.

## Post-step16 validation gate

Byte-identical 300k validation pack:

- English: 10.102311081 -> 9.584511214 (-5.125559%);
- Thai: 10.250705522 -> 9.824723932 (-4.155632%);
- technical: 10.121698719 -> 9.606904253 (-5.086048%);
- combined: 10.158238441 -> 9.672046466 (-4.786184%);
- final holdout touched: false;
- gate: PASS.

Tracked hashes:

- validation step16:
  `d3ec4ab356b0b1d4fb89ee06f6b6d661257f59457672d9bf615b5fb4aa6cc6d5`;
- step8-to-step16 comparison:
  `bd079dcedeeb5c4d35c285ea4b8cdfa26b667f5e7a25d996bc44dc0d7d40ea18`.

## Active work

Record and exact-qualify the complete step16 evidence before any new training
authorization.

Do not execute step17 until the evidence commit is exact-qualified and
remote-synchronized. Any next expansion must remain bounded and must predeclare
its post-tranche validation gate while keeping final holdout untouched.
