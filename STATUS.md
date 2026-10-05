# FlyWireLLM Status

Updated: 2026-10-05

## Repository

- GitHub: `funggier/FlyWireLLM`
- visibility: private
- license: MIT
- local workspace: `T:\Space\Projects\ProjectsAI\FlyWireLLM`

## L001

Goal: blank/random-initialized training-ready LLM baseline.

Current development state:

- decoder-only causal Transformer: implemented
- RMSNorm: implemented
- RoPE: implemented
- grouped-query attention: implemented
- SwiGLU: implemented
- tied embeddings: implemented
- UTF-8 byte tokenizer: implemented
- Thai round-trip: PASS
- English round-trip: PASS
- causal-mask regression: PASS
- forward/loss/backward: PASS
- optimizer smoke step: PASS
- step-0 blank checkpoint: PASS
- safe checkpoint round-trip: PASS
- no-pretrained provenance guard: PASS
- corpus source/license/partition contract: PASS
- wheel build: PASS
- wheel includes MIT LICENSE: PASS
- full L001 regression: 26 tests PASS

## Claims boundary

Current model:

- is training-ready;
- starts from random weights;
- can losslessly tokenize Thai and English;
- cannot yet converse;
- does not yet understand Thai or English;
- does not contain trained FlyWire knowledge;
- has no language-quality qualification.

## Relationship to FlyWireModel

`FlyWireModel` remains a separate project and scientific evidence source.

Future cross-project use must be explicit and provenance-preserving. L001 does
not import FlyWireModel weights.

## Next candidates

- L002: learned tokenizer benchmark for Thai/English versus UTF-8 byte baseline;
- L003: scalable pretraining data manifest/deduplication pipeline;
- L004: larger model profiles and memory/throughput qualification;
- later: optional FlyWire-informed architecture experiments against the frozen
  plain Transformer baseline.
