# Current Development Task

Current task:
L004 — Base-50M Research-Only Pretraining Execution

Status: ACTIVE

GitHub Issue: #6

Branch:
research/l004-base50m-pretraining

## Qualified implementation chain

Foundation:
`c216937868d0b8391fe9ed91c4ceb8269de7e93b`

First bounded runner:
`6b5c2946ea7b8b452e6d317de7ccd438bd23a091`

First bounded authorization:
`674354ea572532f3d8d5c0be856a43078b2e10ae`

First bounded-tranche evidence:
`65218c7` — 4 optimizer steps / 262,144 supervised tokens.

Validation gate:
`2151b1199346bbc26da944f731b01b84556e3ed3`

- identical 300k-token validation pack;
- English / Thai / technical all improved at step4 vs step0;
- combined loss: 10.477063405 -> 10.386913896 (-0.860446%);
- final holdout touched: false.

Second bounded continuation runner:
`3143792ebd041d9675d4711b42d0c85af591e3f8`

- exact qualification: PASS;
- remote sync: 0/0.

## Active authorization candidate

`configs/pretraining-tranche-l004-v2.json`

- source: qualified optimizer step 4 checkpoint;
- authorized updates: step 5 through step 8 only;
- additional supervised-token cap: 262,144;
- cumulative cap after tranche: 524,288;
- checkpoint every update;
- research-only lineage unchanged;
- public release eligibility: not qualified;
- authorization candidate SHA-256:
  `458cf90c0d0f38f407bc01d3a5e4af70706497610a3610347ce0cd75bcfff30d`.

Post-tranche validation is predeclared before step5: reuse the exact same
validation pack at step8, require combined loss not worse than step4 and no
category relative regression above 0.5%. Final holdout remains untouched.

## Active phase

Commit and exact-qualify the second bounded authorization, push/synchronize it,
then execute only step5-8. Do not authorize the remaining 500M run wholesale.
