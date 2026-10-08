# Current Development Task

Current task: L004 — Base-50M Research-Only Pretraining Execution
Status: ACTIVE
GitHub Issue: #6
Branch: research/l004-base50m-pretraining

## Verified production progress

Latest exact-qualified checkpoint: optimizer step 681.

- step681 evidence commit:
  `a0628a604ad3caf1efe55951cba3671eb794da69`
- exact qualification: PASS
- pushed/fetched branch synchronization: 0 ahead / 0 behind
- supervised tokens seen: 44,630,016 / 500,000,000
- fraction of primary budget: 8.9260032%
- checkpoint SHA-256:
  `592497980f6737fd74e13343cff0a1fecbaa3fc1fb97c44c47521e5f38f38ea4`
- state SHA-256:
  `c1c35c918b6e93f381f0144685c1802b8e784f8fa9ac5935a566b614fc605a29`
- retained checkpoint count: 1
- historical checkpoints pruned after state commit: 255
- pretraining complete: false
- public release eligibility: not qualified

## Sixth post-warmup extended tranche

Runner commit:
`ccb04e7a111dca65eb4f2ffccfe21fd7e66280be`

Authorization commit:
`46761af6112a9b7ab0376b940c9235670e8d3355`

Authorization:
`configs/pretraining-tranche-l004-v17.json`

Authorization SHA-256:
`b7f3050fa280e0ff707a1ea3aaac42c66b9941aeb7fb5c814f8709e40261a824`

Production result:
`results/l004/post-warmup-6-v1.json`

SHA-256:
`9a52762d002ab18af93517c36f1e462317dca30aee0c0ace26ef58a4a041ede1`

Step681 validation:
`results/l004/validation-step681-v1.json`

SHA-256:
`83043c81e76913db89870e4fc2f8b1d95bc280a4d0d649380a5cbe96c8c0182b`

Step425-to-step681 comparison:
`results/l004/validation-comparison-step425-step681-v1.json`

SHA-256:
`6c172d2d2483d19672464fb511a9207ff7780d50e49b5e867bf1ed9c1a6db445`

Validation losses:

| Category | step425 | step681 | Relative change |
| --- | ---: | ---: | ---: |
| general English | 5.275155045 | 4.867667660 | -7.724652% |
| general Thai | 5.358594238 | 4.740044962 | -11.543126% |
| technical/scientific/code | 5.430036368 | 4.982780476 | -8.236702% |
| combined | 5.354595217 | 4.863497699 | -9.171515% |

Post-step681 validation gate: PASS.

- every category improved versus step425
- same sealed 300k validation pack: true
- final holdout touched: false
- exact validation rerun reproduced all category losses
- exact runtime verifier reproduced tracked evidence hash
- production-pinned v16/v17 module hashes remained unchanged

L004 remains ACTIVE because the 500M research pretraining contract is not
complete.

## Research direction

FlyWireLLM keeps approximately 50M parameters as the standard research budget.
Base-50M-v1 remains the conventional Transformer control. FlyWire-inspired
Sparse, Routing, Recurrent, Circuit/local-global, and gating experiments must
be separate variants and compared using matched or clearly reported
parameter/token/compute budgets.

The project must preserve not only final metrics but the reason for each
decision, checkpoint lineage, validation evidence, qualitative capability
progression, and the research story from blank model to conventional control
to measurable FlyWire-inspired architectural hypotheses.

See `docs/RESEARCH_ROADMAP.md` and `docs/ARCHITECTURE.md`.

## Qualitative capability probe v1

A fixed synthetic 12-prompt greedy-generation probe now compares step425 and
step681 under identical decoding settings (32 new tokens, greedy, no sampling).
It reads no train/validation/holdout records and explicitly records
`final_holdout_touched=false`.

Tracked result:
`results/l004/qualitative-probe-step425-step681-v1.json`

SHA-256:
`7c574e1480b0c9296e80701450896203599a3fef43a18641a8d476594819ef68`

Observed behavior, without assigning a benchmark score:

- Thai surface fluency is visibly stronger at step681 on some continuations,
  but semantic relevance, factual accuracy, and repetition remain weak;
- English outputs show more sentence-like syntax but still contain invented
  pseudo-words and repetition;
- question-like prompts can still collapse into punctuation/list repetition;
- Python, JSON, and technical continuations are not yet structurally reliable;
- neither checkpoint reached EOS within the 32-token generation cap on these
  prompts.

This is observational evidence only. It does not replace the quantitative
validation gate and does not qualify the model for release or assistant use.

## Seventh post-warmup sustained authorization candidate

Candidate authorization:
`configs/pretraining-tranche-l004-v18.json`

Candidate authorization SHA-256:
`2b3b1db9067a3a6ba687f90c3875e88e7a46f3f48cefe88ccfc0ecda74cd003f`

Runner commit:
`c9d79a2ef4879edbe09fb2ab8cc00f6836821274`

Evidence commit:
`ed87afecc5ad30446b1c956364b2a5d2a0729fe4`

Bounded continuation proposal:

- source optimizer step: 681;
- production range: step682 through step1193;
- maximum additional optimizer updates: 512;
- additional supervised tokens: 33,554,432;
- target cumulative supervised tokens: 78,184,448 / 500,000,000;
- target fraction of primary budget: 15.6368896%;
- step682 learning rate: 0.000593357961;
- step1193 learning rate: 0.000574629804;
- optimizer and cosine-decay policy: unchanged;
- checkpoint retention: latest 1 after atomic state commit;
- validation baseline: step681;
- final holdout: sealed;
- public release: not qualified.

The v17 production module remains byte-identical and is not reused for this
new bound. v18 uses a separately pinned sustained module/runner.

## Current work

Commit and exact-qualify v18 authorization. Production step682 is prohibited
until the authorization commit is clean, full regression/audit passes,
`--validate-only` passes from that exact commit, and GitHub synchronization is
0 ahead / 0 behind.

## Next action

1. commit v18 authorization and ledger;
2. exact-qualify full regression, L004 audit, source checkpoint, and
   `--validate-only`;
3. push/fetch and confirm exact synchronization;
4. only then start the single bounded step682-through-step1193 runner;
5. keep final holdout sealed and public release blocked.
