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

First bounded evidence:
`65218c7` — step4 / 262,144 supervised tokens.

Validation gate:
`2151b1199346bbc26da944f731b01b84556e3ed3`

Second continuation runner:
`3143792ebd041d9675d4711b42d0c85af591e3f8`

Second bounded authorization:
`9c0df1e53c17090c1c019373397b88f77832fd84`

Authorization SHA-256:
`458cf90c0d0f38f407bc01d3a5e4af70706497610a3610347ce0cd75bcfff30d`

## Production progress

Second bounded tranche completed exactly at optimizer step 8.

- cumulative supervised tokens: 524,288;
- fraction of 500M budget: 0.1048576%;
- step8 checkpoint SHA-256:
  `2552f503438bf635f0e54a4fc1fc4bdcef3da57ad734c9ea7b0eebd379500e21`;
- step8 state SHA-256:
  `d86788a7dfd33c13ac3f1bfdacdb648ab8f6fcadc3e0253f0f964043958eca4a`;
- pretraining complete: false;
- public release eligibility: not qualified.

Training loss step5 -> step8:
10.395490 -> 10.236181.

## Post-step8 validation

The exact same 300k-token validation pack was reused.

- English: 10.371749212 -> 10.102311081 (-2.597808% vs step4);
- Thai: 10.407552466 -> 10.250705522 (-1.507049%);
- technical: 10.381440011 -> 10.121698719 (-2.501977%);
- combined: 10.386913896 -> 10.158238441 (-2.201573%);
- same validation pack: true;
- final holdout touched: false;
- predeclared post-tranche gate: PASS.

No further training tranche is authorized yet.

## Active phase

Exact-qualify and commit the step8 production + validation evidence, push and
synchronize it, then design the next bounded authorization from the verified
step8 checkpoint.

Do not authorize the remaining 500M run wholesale.
