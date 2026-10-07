# Current Development Task

Current task:
L004 — Base-50M Research-Only Pretraining Execution

Status: ACTIVE

GitHub Issue: #6

Branch:
research/l004-base50m-pretraining

## Verified production progress

Latest verified checkpoint: optimizer step 16.

- supervised tokens seen: 1,048,576 / 500,000,000;
- fraction of primary budget: 0.2097152%;
- checkpoint SHA-256:
  `07a968b8e4c8a7f6711f24cd3466c46c374e09628c7d7c65848a2166efdcd71d`;
- third-tranche evidence commit:
  `73d78d18134f7f27b530eaea2eb4d9733435ad61`;
- post-step16 validation gate: PASS;
- final holdout touched: false.

## Fourth bounded scaling tranche

Qualified runner:
`074b429f640654db1bb8696971da8560f17c9066`.

Authorization candidate:
`configs/pretraining-tranche-l004-v4.json`.

Authorization SHA-256:
`8ef449c63c6be4e2129a0604413fbfbb13f6d706a2c046d779079415981f0259`.

Bound:

- source step: 16;
- maximum additional updates: 16;
- end step: 32;
- additional supervised tokens: 1,048,576;
- cumulative supervised tokens at bound: 2,097,152;
- checkpoint every update;
- same 300k validation pack required after step32;
- combined validation loss must not worsen vs step16;
- any category relative loss increase >0.5% fails the gate;
- final holdout must remain untouched;
- public release remains not qualified.

Do not execute step17 until the v4 authorization is committed,
exact-qualified, remote-synchronized, and `--validate-only` passes on the
clean exact commit.
