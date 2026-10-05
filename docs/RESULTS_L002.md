# L002 Results — Base-50M Target, Tokenizer and Data Contract

## Decision

**TARGET CONTRACT GREEN / TOKENIZER ENGINEERING GREEN / DATA INVENTORY GREEN / 500M CORPUS NOT YET ACQUIRED OR RIGHTS-QUALIFIED / PRETRAINING NOT STARTED**

## Base-50M-v1

The first real pretraining target is frozen at exactly 50,213,376 parameters under the current FlyWireLLM implementation.

- vocabulary: 32,000;
- d_model: 512;
- layers: 12;
- query heads: 8;
- KV heads: 2;
- head dimension: 64;
- SwiGLU d_ff: 1,408;
- RMSNorm;
- RoPE theta 10,000;
- tied embeddings;
- dropout 0;
- initial context 1,024;
- future context qualification target 2,048;
- random initialization std 0.02.

## Tokenizer pilot

The 8K Tatoeba pilot was lossless on 115 benchmark cases, with zero unknown tokens. It reduced byte-token count by 11.735x for Thai and 3.418x for English.

## 32K candidate

A true 32,000-token SentencePiece Unigram candidate was trained from 104,940,187 UTF-8 text bytes balanced at roughly 50 MiB per language.

Candidate provenance:

- SentencePiece 0.2.2;
- Unigram;
- identity normalization;
- byte fallback;
- model SHA-256: 998bc75f058e554d4f50b6cbc77c52898e3446e516c3b25a126b99b113513818;
- vocab SHA-256: 9ade52fa3e57f6bb096cfdcc64b37d89f794937b68d155997b48f529c841e762.

Benchmark: 10,150 validation/technical cases.

- overall compression vs byte: 6.252x;
- Thai: 10.710x;
- English: 4.031x;
- mixed: 2.570x;
- round-trip failures: 0;
- unknown tokens: 0.

The tokenizer model is kept outside the MIT source tree at this stage because the Thai engineering corpus includes conditionally licensed Wikipedia text. Hashes and reproducible training code are tracked, while rights qualification remains a separate gate.

## Corpus inventory

Current acquired/deduplicated sources provide exactly 58,334,085 train tokens under the 32K candidate:

- Thai: 38,312,940;
- English: 20,021,145.

This is 11.666817% of the primary 500M target, leaving an exact gap of 441,665,915 train tokens.

Therefore raw data availability is not the blocker, but the current approved/acquired corpus is not sufficient for the declared Base-50M run.

## Source rights states

Approved for the current engineering inventory:

- Tatoeba;
- Mozilla Common Voice, subject to release-specific verification.

Conditional:

- Thai Wikipedia;
- English Wikipedia;
- FineWeb2 Thai;
- FineWeb English;
- selected Project Gutenberg works.

Blocked:

- automatic FlyWireModel knowledge export.

FlyWireModel training export remains blocked until frozen scientific evaluation evidence and source-level rights can be separated explicitly.

## Training schedule

- 250M: 3,815 optimizer steps, 77 warmup;
- 500M: 7,630 optimizer steps, 153 warmup;
- 1B: 15,259 optimizer steps, 306 warmup;
- global target: 65,536 tokens/update.

## Claims boundary

L002 does not claim:

- that Base-50M has been pretrained;
- Thai or English language understanding;
- conversational ability;
- that conditional web/Wikimedia sources are approved for production training;
- that the 32K engineering tokenizer can be redistributed under MIT;
- that FlyWire knowledge is present in any weights.

## Next stage

L003 should acquire and qualify the missing approximately 441.7M train tokens, implement scalable normalization/dedup/PII/provenance handling, freeze production train/validation/holdout manifests, and only then authorize the first Base-50M pretraining run.
