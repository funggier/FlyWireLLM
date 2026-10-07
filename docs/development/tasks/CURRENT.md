# Current Development Task

Current task:
L004 — Base-50M Research-Only Pretraining Execution

Status: ACTIVE

GitHub Issue: #6

Branch:
research/l004-base50m-pretraining

## Verified production progress

Latest exact-qualified checkpoint: optimizer step 32.

- supervised tokens seen: 2,097,152 / 500,000,000;
- fraction of primary budget: 0.4194304%;
- checkpoint SHA-256:
  `136f554d7321be08ec266ac44296ae6870aadc7edb8d468812173ab240b8a463`;
- step32 evidence commit:
  `8fa74f72c9ae025b64728b1d6717eab3a7452136`;
- post-step32 validation gate: PASS;
- final holdout touched: false.

## Rolling-checkpoint pilot

Qualified runner:
`3ff63c1cb9199b9e4e0e3449a4837ee10ffa3a07`.

Authorization candidate:
`configs/pretraining-tranche-l004-v5.json`.

Authorization SHA-256:
`cfff5b502da2a05e21583b8ae71e2151799aab33dc1ce1bd1eca9bb2e508c2b6`.

Bound:

- source step: 32;
- maximum additional updates: 8;
- end step: 40;
- additional supervised tokens: 524,288;
- cumulative supervised tokens at bound: 2,621,440;
- create checkpoint every update;
- retain latest fifth-tranche checkpoint only;
- prune old checkpoint only after new metric + atomic state commit;
- same 300k validation pack required after step40;
- final holdout must remain untouched;
- public release remains not qualified.

Do not execute step33 until the v5 authorization is committed,
exact-qualified, remote-synchronized, and `--validate-only` passes from the
clean exact commit.
