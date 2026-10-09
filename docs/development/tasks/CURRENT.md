# Current Development Task

Current task: L004 — Base-50M Research-Only Pretraining Execution
Status: ACTIVE
Production status: PAUSED BY USER REQUEST at step1452
GitHub Issue: #6
Branch: research/l004-base50m-pretraining

## Verified production progress

Latest exact-qualified checkpoint: optimizer step 1193.

- evidence commit:
  `4d2ba99e9af6048de55c90de6f44217988079115`
- exact qualification: PASS
- pushed/fetched branch synchronization: 0 ahead / 0 behind
- supervised tokens seen: 78,184,448 / 500,000,000
- fraction of primary budget: 15.6368896%
- checkpoint SHA-256:
  `eb13f05a38317d80feb2d46b19ddc8ee90ecab67634e7f1fcff33d557b65ed13`
- state SHA-256:
  `393ace6a462dab4c3d23b27e1c8c0ddc5ca40365e1d31b55301eecf613d07174`
- retained checkpoint count: 1
- historical checkpoints pruned after state commit: 511
- pretraining complete: false
- public release eligibility: not qualified
- final holdout touched: false

## Seventh post-warmup sustained tranche

Runner commit:
`c9d79a2ef4879edbe09fb2ab8cc00f6836821274`

Authorization commit:
`fc949583a7ecbbf3e717021efc9a3183295cb9e5`

Authorization:
`configs/pretraining-tranche-l004-v18.json`

Authorization SHA-256:
`2b3b1db9067a3a6ba687f90c3875e88e7a46f3f48cefe88ccfc0ecda74cd003f`

Production result:
`results/l004/post-warmup-7-v1.json`

SHA-256:
`f8df5ac0129aa75b47eac072dbb06aad21b845db24bff2058b5a82f78ca49246`

The tranche resumed safely from step917 after an LConnect update. The persisted
checkpoint/data cursor were reverified before exactly one `--resume` runner
was started. Production then completed normally at step1193 with exit code 0.

## Step681-to-step1193 validation

Step1193 validation:
`results/l004/validation-step1193-v1.json`

SHA-256:
`5699d1bb8b629f1da7e75cc3bd11f44a2a3a56e59978dfc83952ac50e3f81b0e`

Comparison:
`results/l004/validation-comparison-step681-step1193-v1.json`

SHA-256:
`4339a97027ee49cc2ad8ef41e1d00a2b45d99202e97f4d61d15062215dfbe4b8`

| Category | step681 | step1193 | Relative change |
| --- | ---: | ---: | ---: |
| general English | 4.867667660 | 4.391436153 | -9.783567% |
| general Thai | 4.740044962 | 4.085647923 | -13.805714% |
| technical/scientific/code | 4.982780476 | 4.405080693 | -11.593924% |
| combined | 4.863497699 | 4.294054923 | -11.708503% |

Post-step1193 validation gate: PASS.

- every category improved versus step681;
- same sealed 300k validation pack: true;
- final holdout touched: false;
- exact validation rerun reproduced every loss exactly;
- exact comparison artifact was byte-identical;
- exact runtime verifier artifact was byte-identical;
- validation JSON differed only in nondeterministic `elapsed_seconds`;
- production-pinned v16/v17/v18 module and v18 runner hashes remained unchanged.

## Qualitative capability probe v2

Config:
`configs/qualitative-probe-l004-v2.json`

Config SHA-256:
`0969a229202420ee693b76e0f56649c95fa7d00d19f1fa409e0d976ab9d49206`

Tracked result:
`results/l004/qualitative-probe-step681-step1193-v2.json`

SHA-256:
`1262ad30117b268c22fc2d55cd1998fa3571cb1d02ab26c516951753356b7248`

The v2 probe reuses the exact 12 synthetic prompts and decoding settings from
v1, comparing step681 and step1193.

Observed behavior:

- Thai generation shows a substantial surface-form improvement: step1193
  produces full Thai clauses and sentence-like continuations on all four Thai
  probes where step681 still showed list/punctuation collapse on some prompts;
- semantic relevance and factual accuracy remain weak; for example the
  boiling-water prompt is still factually wrong;
- repetition remains common;
- English remains sentence-like but can be irrelevant or invent pseudo-words;
- question answering is not reliable;
- Python and JSON structure remain unreliable;
- all 24 generations reached the 32-token cap without EOS.

This probe is observational only. It reads no corpus partition and does not
change release qualification.

## Research direction

Approximately 50M parameters remains the standard FlyWireLLM research budget.
Base-50M-v1 stays the conventional Transformer control. FlyWire-inspired Sparse,
Routing, Recurrent, Circuit/local-global and gating architectures will remain
separate matched-resource experiments after the control is sufficiently
trained and characterized.

L004 remains ACTIVE because the 500M research pretraining contract is not
complete.

## Eighth post-warmup longrun authorization candidate

Candidate authorization:
`configs/pretraining-tranche-l004-v19.json`

Candidate authorization SHA-256:
`6502c901b23182f579830d71f1b8f4539484cbbf4d0b9996b4115eb1cac6b5db`

Runner implementation commit:
`46941690ee45f65a4451e1eff55d7e6872ad1b13`

Step1193 evidence commit:
`4d2ba99e9af6048de55c90de6f44217988079115`

Bounded continuation proposal:

- source optimizer step: 1193;
- production range: step1194 through step2217;
- maximum additional optimizer updates: 1,024;
- additional supervised tokens: 67,108,864;
- target cumulative supervised tokens: 145,293,312 / 500,000,000;
- target fraction of primary budget: 29.0586624%;
- step1194 learning rate: 0.000574581773;
- step2217 learning rate: 0.000504675007;
- optimizer, cosine-decay and data policy: unchanged;
- rolling checkpoint retention: latest 1 after atomic state commit;
- validation baseline: step1193;
- final holdout: sealed;
- public release: not qualified.

v19 uses a separately pinned longrun module/runner. v18 remains byte-identical.

## Production pause — step1452

Production is intentionally paused at the user's request.

Authoritative recoverable state:

- optimizer step: 1452;
- next step on resume: 1453;
- cumulative supervised tokens: 95,158,272 / 500,000,000;
- primary-budget fraction: 19.0316544%;
- v19 tranche updates complete: 259 / 1,024;
- v19 tranche updates remaining: 765;
- checkpoint SHA-256:
  `0e092418c2c586f6cbddb697dc59bdf104d5d164ae66e97238fafb44bc557210`;
- metric SHA-256:
  `75c9a2ced065c4112a4b4e5314f074e50c9420c1593662e316750973df7f9096`;
- state SHA-256:
  `f3a572c7b354e43e35dfd9b3e11d7113f70c643d12daba570070d41b81fed65e`;
- state loader `verify_files=True`: PASS;
- matching training Python processes after pause: 0;
- final holdout touched: false;
- public release: not qualified.

Pause report:
`docs/development/reports/FWLLM-20261009-l004-v19-paused-step1452.md`

An external recovery marker also exists at:
`T:\Space\Projects\ProjectsAI\FlyWireLLM-data\L004\Recovery\PAUSED-20261009-step1452.json`.

## Current work

None while production is intentionally paused. Do not start another training
runner until the user asks to resume.

## Next action

When resuming, first reverify the step1452 state/checkpoint/metric hashes and
confirm zero matching training processes. Then start exactly one v19 runner
with `--resume`; it must continue from step1453. Keep the final holdout sealed
and public release blocked.
