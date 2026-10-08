# Current Development Task

Current task: L004 — Base-50M Research-Only Pretraining Execution
Status: ACTIVE
GitHub Issue: #6
Branch: research/l004-base50m-pretraining

## Verified production progress

Latest verified production checkpoint: optimizer step 425.
Exact qualification status: pending evidence commit qualification.

- supervised tokens seen: 27,852,800 / 500,000,000
- fraction of primary budget: 5.57056%
- checkpoint SHA-256:
  `b02d6d6e3d4f5f7d06a9256707a71d44ace0293b8c55d2f8b93aa2653abe47ee`
- state SHA-256:
  `36c99e89a9566243831812dd6b73276dd281b39e23157f1c45b5699c23c0a1ce`
- retained checkpoint count: 1
- historical checkpoints pruned after state commit: 127
- post-step425 validation gate: PASS
- same validation pack as step297: true
- final holdout touched: false
- pretraining complete: false
- public release eligibility: not qualified

## Fifth post-warmup accelerated tranche

Qualified runner:
`39b696a3495db40fffb2cc9ca69838058cc0c408`.

Authorization:
`configs/pretraining-tranche-l004-v16.json`.

Authorization SHA-256:
`71ce46ad2366ef60135abfc99ecb0b097782e1e322e35b906049b8c5cfe990ea`.

Production result:
`results/l004/post-warmup-5-v1.json`
SHA-256:
`eab3a1e0a280f4b9fdfcce04f5072e6928b5c9fb291d5fdb1f36ccec76f220b2`.

Step425 validation:
`results/l004/validation-step425-v1.json`
SHA-256:
`f602f2e89e3b87d478272d4aaef4da67c845c3dc842ee5846156a91fc5683ffc`.

Step297-to-step425 comparison:
`results/l004/validation-comparison-step297-step425-v1.json`
SHA-256:
`281e2eb29688209cf17ef801bf4f5d4c9de9329e7341efdb1811a1773622157f`.

Validation losses:

| Category | step297 | step425 | Relative change |
| --- | ---: | ---: | ---: |
| general English | 5.577049066 | 5.275155045 | -5.413150% |
| general Thai | 5.905298695 | 5.358594238 | -9.257863% |
| technical/scientific/code | 5.766543058 | 5.430036368 | -5.835501% |
| combined | 5.749630273 | 5.354595217 | -6.870617% |

The step425 evidence candidate is verified locally. Do not authorize any later
production tranche until the evidence commit is exact-qualified, pushed, fetched,
and confirmed synchronized. Final holdout remains sealed.
