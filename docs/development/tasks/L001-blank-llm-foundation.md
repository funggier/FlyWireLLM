# L001 — Blank LLM Foundation and Training-Ready Baseline

Status: `DONE`

GitHub Issue: `#1`

## Objective

Create an independently testable decoder-only Transformer from random
initialization, ready for later pretraining without claiming pretrained
knowledge or conversational ability.

## Delivered

- decoder-only causal Transformer;
- RMSNorm, RoPE, grouped-query attention and SwiGLU;
- tied token embeddings;
- deterministic random initialization;
- loss/backward/optimizer smoke path;
- checkpoint provenance with `pretrained=false`;
- UTF-8 byte tokenizer with Thai/English lossless round trip;
- CPU smoke profile and regression tests.

## Final state

L001 established the code baseline that became the v0.1.0 release.

## Claims boundary

The model is blank/random-initialized. L001 did not create language
understanding, conversation ability, or FlyWire knowledge in model weights.
