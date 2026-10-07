# Current Development Task

Current task:
L004 — Base-50M Research-Only Pretraining Execution

Status: ACTIVE

GitHub Issue: #6

Branch:
research/l004-base50m-pretraining

## Verified production progress

Latest exact-qualified checkpoint: optimizer step 56.

- supervised tokens seen: 3,670,016 / 500,000,000;
- fraction of primary budget: 0.7340032%;
- checkpoint SHA-256:
  `4095c686eeff521c906a73ef34d890135cd5c4900addab54b7477f6f72a4d116`;
- step56 evidence commit:
  `5c1dde7596c0a697c803096653e62fb88b4a6580`;
- post-step56 validation gate: PASS;
- final holdout touched: false.

## Seventh rolling scaling tranche

Qualified runner:
`835d914ea91d2c22c802efab15f07f65a7e84e71`.

Authorization candidate:
`configs/pretraining-tranche-l004-v7.json`.

Authorization SHA-256:
`391250f1514f66ab5ad680cce09c7fcbe657afd32779884a96542f84d32a0620`.

Bound:

- source step: 56;
- maximum additional updates: 32;
- end step: 88;
- additional supervised tokens: 2,097,152;
- cumulative supervised tokens at bound: 5,767,168;
- create checkpoint every update;
- retain latest seventh-tranche checkpoint only;
- prune old checkpoint only after new metric + atomic state commit;
- same 300k validation pack required after step88;
- final holdout must remain untouched;
- public release remains not qualified.

Do not execute step57 until v7 is committed, exact-qualified,
remote-synchronized, and `--validate-only` passes from the clean commit.
