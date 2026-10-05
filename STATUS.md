# FlyWireLLM Status

Updated: 2026-10-06

## Repository

- GitHub: funggier/FlyWireLLM
- visibility: private
- license: MIT
- local workspace: T:\Space\Projects\ProjectsAI\FlyWireLLM

## Released baseline

- v0.1.0: blank/random-initialized LLM engineering baseline
- tag commit: 2ec2f502c8a4c5caa6c1f5262333e05dcacaf0a5
- smoke parameters: 109,120
- pretrained weights: none
- release qualification: GREEN

## L002 Base-50M-v1

Decision:

**TARGET CONTRACT GREEN / TOKENIZER ENGINEERING GREEN / DATA INVENTORY GREEN / 500M CORPUS NOT YET ACQUIRED OR RIGHTS-QUALIFIED / PRETRAINING NOT STARTED**

Frozen architecture:

- parameters: 50,213,376
- vocabulary target: 32,000
- d_model: 512
- layers: 12
- query/KV heads: 8/2
- d_ff: 1,408
- context: 1,024 initially; 2,048 later qualification target
- RMSNorm + RoPE + GQA + SwiGLU + tied embeddings
- random initialization

Tokenizer:

- SentencePiece Unigram 32K candidate trained successfully
- identity normalization
- byte fallback
- benchmark cases: 10,150
- round-trip failures: 0
- unknown tokens: 0
- compression vs v0.1.0 byte tokenizer: overall 6.252x
- Thai: 10.710x
- English: 4.031x
- mixed: 2.570x
- tokenizer engineering: GREEN
- tokenizer redistribution/production rights: NOT YET QUALIFIED

Data inventory:

- primary budget: 500,000,000 tokens
- current acquired/deduplicated train tokens: 58,334,085
- current coverage: 11.666817%
- gap to primary target: 441,665,915 tokens
- split: 99.0% train / 0.5% validation / 0.5% holdout by source group
- final holdout text excluded from tokenizer fitting
- automatic FlyWireModel training export: BLOCKED pending source/evaluation separation

Training schedule:

- 250M: 3,815 steps / 77 warmup
- 500M: 7,630 steps / 153 warmup
- 1B: 15,259 steps / 306 warmup
- global target: 65,536 tokens/update

Qualification:

- full repository pytest: 47 PASS
- L002 audit: PASS
- git diff --check: PASS
- v0.1.0 baseline tests remain included in regression

## Claims boundary

Current project state:

- blank v0.1.0 model is released and training-ready;
- Base-50M-v1 architecture and training target are defined;
- a 32K tokenizer candidate is engineering-qualified;
- Base-50M has NOT been pretrained;
- Thai/English language understanding is NOT claimed;
- conversational ability is NOT claimed;
- conditional-rights data are NOT treated as approved production training data;
- no trained FlyWire knowledge exists in LLM weights.

## Next stage

L003 should acquire and qualify the remaining approximately 441.7M train tokens, add scalable normalization/dedup/PII/provenance processing, freeze production manifests, and perform compute/memory qualification before authorizing Base-50M pretraining.

See docs/RESULTS_L002.md, docs/TOKENIZER_L002.md and docs/DATA_PLAN_L002.md.
