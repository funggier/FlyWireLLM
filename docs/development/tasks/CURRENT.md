# Current Development Task

Current task:
L004 — Base-50M Research-Only Pretraining Execution

Status: ACTIVE

GitHub Issue: #6

Branch:
research/l004-base50m-pretraining

## Verified production progress

Latest verified checkpoint candidate: optimizer step 40.

- supervised tokens seen: 2,621,440 / 500,000,000;
- fraction of primary budget: 0.524288%;
- checkpoint SHA-256:
  `9f77df859cf1d41546d9bb919aac621e970375c175c6e9e9a93a4b200cedf95c`;
- external state SHA-256:
  `a93db123f8696d678c8c2a835ad381183018f88e30adc0612e28371333fee105`;
- fifth-tranche result SHA-256:
  `995f97250b2fd06160cd4d6f3f0c31717cd181bd5bf20858d1f9c87c41b66b2a`;
- pretraining complete: false;
- public release eligibility: not qualified.

## Rolling checkpoint pilot

- checkpoint creation: every update;
- retained fifth-tranche checkpoints: 1;
- historical fifth-tranche checkpoints pruned: 7;
- all step33-40 metrics retained;
- final checkpoint reload: PASS.

## Post-step40 validation gate

Byte-identical 300k validation pack:

- English: 8.823402260 -> 8.340751974 (-5.470115%);
- Thai: 9.078124258 -> 8.614949760 (-5.102095%);
- technical: 8.857479007 -> 8.388065623 (-5.299627%);
- combined: 8.919668508 -> 8.447922452 (-5.288829%);
- final holdout touched: false;
- gate: PASS.

Tracked hashes:

- validation step40:
  `c2dd89b43fd8e77b0dbc23f1fcc37f71f2bc5078209c5669ca98b20489e324b4`;
- step32-to-step40 comparison:
  `22b5ff56488599951ce6e4bb1decff6d2b851c6853234091e2ffff1248255a24`.

## Active work

Record and exact-qualify the complete step40 rolling evidence before any new
training authorization.

Do not execute step41 until the evidence commit is exact-qualified and
remote-synchronized. The next tranche may reuse the rolling retention policy
only after this pilot evidence is frozen.
