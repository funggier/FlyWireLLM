# FlyWireLLM Status

Updated: 2026-10-06

## Repository

- GitHub: `funggier/FlyWireLLM`
- visibility: private
- license: MIT
- portable repository state: no machine-local workspace path is required

## Release milestones

### v0.1.0

- blank/random-initialized LLM engineering baseline
- tag commit: `2ec2f502c8a4c5caa6c1f5262333e05dcacaf0a5`
- smoke parameters: 109,120
- pretrained weights: none
- release qualification: GREEN

### v0.2.0

- milestone: Research Pretraining Readiness
- tag/release commit: `fc9e0fc2f64be4574b70021226ffd97f61fd9320`
- release URL: https://github.com/funggier/FlyWireLLM/releases/tag/v0.2.0
- Base-50M architecture/tokenizer contract: qualified
- L003 research-only 500M mixture: qualified
- research-only lineage frozen before optimizer step 1
- Base-50M pretrained weights: none
- public checkpoint release eligibility: not qualified
- release ZIP SHA-256: `de495e36ff3c9d95be0869caa624049771a2b92e608863487baf22c77bfd537f`
- document ZIP SHA-256: `b0545edb1fb5fbac751f8b35350504d1b9f3662a51698a2aa484fe0343de7da1`
- release qualification: GREEN

## Base-50M-v1

Architecture frozen by L002:

- parameters: 50,213,376
- vocabulary: 32,000
- d_model: 512
- layers: 12
- query/KV heads: 8/2
- d_ff: 1,408
- initial context: 1,024
- RMSNorm + RoPE + GQA + SwiGLU + tied embeddings
- random initialization

Tokenizer:

- SentencePiece 0.2.2 Unigram
- identity normalization + byte fallback
- model SHA-256:
  `998bc75f058e554d4f50b6cbc77c52898e3446e516c3b25a126b99b113513818`
- benchmark cases: 10,150
- round-trip failures: 0
- unknown tokens: 0

## L003 production-corpus readiness

Decision:

**L003 GREEN / GLOBAL EXACT+NEAR DEDUP QUALIFIED / 500M RESEARCH-ONLY MIXTURE QUALIFIED / PRETRAINING AUTHORIZED FOR RESEARCH-ONLY LINEAGE / PUBLIC RELEASE NOT QUALIFIED**

Final global manifest:

- records total: 3,277,753
- accepted: 3,277,015
- exact duplicates: 61
- near duplicates: 677
- source artifacts: 6 unique global inputs
- decision manifest contains raw text: false

Qualified train capacity:

- research-only eligible total: 969,701,220 tokens
- general English: 522,373,720
- general Thai: 316,825,768
- technical/scientific/code: 130,501,732
- research-only 500M gap: 0
- release-safe total: 20,053,464
- release-safe 500M gap: 479,946,536

Frozen lineage:

- checkpoint lane: `research_only`
- optimizer steps completed at freeze: 0
- pretraining authorization: true
- public release eligibility: not qualified
- automatic FlyWireModel export: blocked

Frozen hashes:

- source registry:
  `5304c4cc6b302a06650d090d6731f5b0e610fd99ec257c3fd9cc671cf62f79aa`
- global report:
  `30a792d9fd26d1d4a6a1760aae7a43ae293778926be9044ebda632ec7c5054d5`
- global decisions:
  `018aa08f347ebfd747f3aa11cada563bbd27918b469c7c5f11abe0289df7770a`
- tokenizer:
  `998bc75f058e554d4f50b6cbc77c52898e3446e516c3b25a126b99b113513818`

External qualification re-hashed the tokenizer, all six accepted corpora, the
global decision manifest and the external summary: PASS.

## Current claims boundary

- Base-50M has **not** been pretrained.
- Thai/English understanding is not yet claimed.
- Conversational ability is not yet claimed.
- Research-only checkpoints may not be promoted to release-safe.
- Public redistribution of a future research-only checkpoint is not qualified.
- No trained FlyWire knowledge exists in the LLM weights.
- Automatic FlyWireModel curated export remains blocked.

## Next stage

After L003 exact-commit/merge closure, a later task may execute the first
Base-50M research-only pretraining run using the frozen L003 corpus and lineage.
That training stage must preserve the L003 hashes, train-only fit boundary and
checkpoint-lane immutability.

See `docs/RESULTS_L002.md`, `docs/RESULTS_L003.md`,
`docs/TRAINING_DATA.md` and `configs/pretraining-freeze-l003.json`.
