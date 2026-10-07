# Current Development Task

Current task:
L004 — Base-50M Research-Only Pretraining Execution

Status: ACTIVE

GitHub Issue: #6

Branch:
research/l004-base50m-pretraining

## Verified production progress

Latest exact-qualified checkpoint: optimizer step 40.

- supervised tokens seen: 2,621,440 / 500,000,000;
- fraction of primary budget: 0.524288%;
- checkpoint SHA-256:
  `9f77df859cf1d41546d9bb919aac621e970375c175c6e9e9a93a4b200cedf95c`;
- step40 evidence commit:
  `30e75e05b926224033033231a00de9ee3c6643bf`;
- post-step40 validation gate: PASS;
- final holdout touched: false.

## Sixth rolling expansion

Qualified runner:
`ff71939f7547f9d6ac8f2f510d6a604b350c525e`.

Authorization candidate:
`configs/pretraining-tranche-l004-v6.json`.

Authorization SHA-256:
`b14f53cde35373e127c5405bca7b50744c9cea6366aa03f51d060dbc7af30f6f`.

Bound:

- source step: 40;
- maximum additional updates: 16;
- end step: 56;
- additional supervised tokens: 1,048,576;
- cumulative supervised tokens at bound: 3,670,016;
- create checkpoint every update;
- retain latest sixth-tranche checkpoint only;
- prune old checkpoint only after new metric + atomic state commit;
- same 300k validation pack required after step56;
- final holdout must remain untouched;
- public release remains not qualified.

Do not execute step41 until v6 is committed, exact-qualified,
remote-synchronized, and `--validate-only` passes from the clean commit.
