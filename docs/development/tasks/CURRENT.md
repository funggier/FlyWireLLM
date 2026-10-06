# Current Development Task

Current task:
L004 — Base-50M Research-Only Pretraining Execution

Status: ACTIVE

GitHub Issue: #6

Branch:
research/l004-base50m-pretraining

## Qualified checkpoints

Runtime/data foundation:
c216937868d0b8391fe9ed91c4ceb8269de7e93b

- 182/182 tests PASS;
- L003/L004 audits PASS;
- selection/packing/block-order external verification PASS;
- remote sync 0/0.

Bounded runner:
6b5c2946ea7b8b452e6d317de7ccd438bd23a091

- 188/188 tests PASS;
- exact runner qualification PASS;
- remote sync 0/0.

## Current authorization candidate

File:
configs/pretraining-tranche-l004-v1.json

SHA-256:
a8ed92f59480f7e922b8973e62be9a8e40f7a4afec5629d4008f7e3d5db9eb2f

Hard limits:

- checkpoint lane: research_only;
- start optimizer step: 0;
- maximum updates: 4;
- supervised tokens: 262,144;
- checkpoint every update;
- external run root:
  external://FlyWireLLM-data/L004/Runs/base50m-first-tranche-v1;
- public release eligibility: not qualified;
- automatic FlyWireModel export: blocked.

Production optimizer steps recorded remain **0**.

## Resume here

Do not run production step 1 until the authorization commit itself has:

1. clean-worktree exact qualification;
2. 188/188 tests PASS;
3. L004 audit PASS;
4. runner --validate-only PASS;
5. remote branch synchronization 0/0.

After those pass, execute only update 1, verify its checkpoint/state/metric, then
resume for the remaining three authorized updates.
