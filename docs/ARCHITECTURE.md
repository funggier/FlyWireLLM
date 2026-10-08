# Architecture

## Repository role

FlyWireLLM is the language-model repository. It is intentionally separate from
FlyWireModel, which remains the neuroscience/connectome research repository.

The baseline must stay independently understandable and benchmarkable before
FlyWire-inspired architecture experiments are introduced.

## L001 blank causal language model

The baseline is a decoder-only Transformer:

~~~text
UTF-8 bytes
    |
token embedding
    |
[ RMSNorm
  -> grouped-query causal self-attention + RoPE
  -> residual
  -> RMSNorm
  -> SwiGLU
  -> residual ] x N
    |
RMSNorm
    |
tied LM head
    |
next-token logits
~~~

### Normalization

RMSNorm is used before attention and feed-forward sublayers.

### Position representation

RoPE is applied to query and key vectors. There is no learned absolute position
embedding in the L001 baseline.

### Attention

Attention is strictly causal. Query heads may outnumber key/value heads, giving
grouped-query attention while preserving a simple baseline.

The smoke profile uses 4 query heads and 2 key/value heads.

### Feed-forward network

The feed-forward block is SwiGLU:

~~~text
SiLU(gate(x)) * up(x) -> down projection
~~~

### Weight tying

The token embedding and language-model output projection share one parameter
matrix by default.

## Random initialization boundary

The model is created only through fresh random initialization.

L001 does not import:

- pretrained Transformer weights;
- pretrained embeddings;
- pretrained tokenizer parameters;
- FlyWireModel weights.

Checkpoint metadata freezes this provenance:

- `pretrained=false`;
- `external_pretrained_weights=false`;
- `initialization_origin=random`;
- `random_initialization_origin=true`;
- `knowledge_in_weights_claimed=false`;
- initialization seed;
- training step.

A step-0 checkpoint is therefore an explicit blank model artifact.

A checkpoint after smoke/pretraining steps retains random-origin provenance but
is no longer marked `untrained`.

## Tokenization

L001 uses `UTF8ByteTokenizer`.

Four special tokens occupy IDs 0-3. Raw byte values occupy IDs 4-259.

Advantages for the blank baseline:

- no tokenizer training dependency;
- lossless Thai;
- lossless English;
- lossless mixed Thai/English;
- arbitrary Unicode support;
- stable vocabulary from the first checkpoint.

The cost is token inefficiency, especially for Thai. A learned tokenizer should
be evaluated later behind a stable tokenizer interface rather than changing the
model contract implicitly.

## Profiles

### smoke

- vocabulary: 260
- d_model: 64
- layers: 2
- query heads: 4
- KV heads: 2
- d_ff: 176
- context: 128
- parameters: 109,120

This is for CPU correctness testing only.

### small

The tracked `configs/small.json` profile provides a larger untrained research
configuration for future training experiments. Its presence is not a quality
claim.

## FlyWire-informed future experiments

FlyWireModel research may later motivate optional modules, connectivity priors,
routing schemes, recurrence, sparse structure, training curricula, or data
representations.

Every such experiment should:

1. preserve the plain L001 Transformer as a frozen baseline;
2. state exactly which FlyWire evidence motivates the change;
3. keep the change optional;
4. compare against matched-parameter or matched-compute controls;
5. avoid claiming biological knowledge exists in random weights;
6. use held-out language/model evaluations.

This keeps biological inspiration measurable rather than decorative.

## L002 Base-50M-v1 target

L002 freezes the first non-smoke architecture target while preserving the
v0.1.0 blank model as the regression baseline.

Base-50M-v1:

- vocabulary: 32,000;
- d_model: 512;
- layers: 12;
- query heads: 8;
- KV heads: 2;
- head dimension: 64;
- d_ff: 1,408;
- context: 1,024 initially;
- RMSNorm + RoPE + grouped-query attention + SwiGLU;
- tied embeddings;
- dropout: 0;
- initialization std: 0.02.

The exact parameter count implied by the current implementation is 50,213,376.
The formula is implemented in BlankLLMConfig.exact_parameter_count and is
regression-tested against the instantiated smoke model.

L002 also adds a SentencePiece Unigram tokenizer candidate with identity
normalization and byte fallback. The learned tokenizer is not coupled to the
core model implementation; the released UTF-8 byte tokenizer remains the
bootstrap/fallback baseline.

Base-50M-v1 is now being pretrained from random initialization in the
research-only L004 lineage. The latest exact-qualified milestone is optimizer
step 681 with 44,630,016 supervised training tokens. This is 8.9260032% of the
500M primary budget, so the control is still an in-progress pretrained base
model rather than a completed or release-qualified model.

## 50M research-budget policy

Approximately 50M parameters is the standard architecture-research budget for
FlyWireLLM. Base-50M-v1, at exactly 50,213,376 parameters, is the conventional
Transformer control.

FlyWire-inspired variants should remain near the same parameter budget and use
matched or explicitly reported training-token and compute budgets. A larger
parameter count is not itself an architectural improvement.

Scaling above 50M should be considered only after a variant demonstrates a
repeatable advantage over the control and a larger experiment is needed to test
whether that advantage survives scale.

## FlyWire-inspired architecture track

The baseline must not be retrofitted mid-training. FlyWire-informed mechanisms
are separate experimental variants, initially including:

- sparse connectivity;
- learned or conditional routing;
- recurrence and feedback;
- specialized circuits;
- local-before-global processing;
- learnable gating or inhibitory-style computation.

Each experiment must identify the motivating FlyWire/connectome observation,
state the mathematical mechanism actually implemented, preserve a matched
control, and report negative results as well as wins.

Validation loss is not the only success criterion. Matched quality at lower
compute, better parameter efficiency, improved stability, or stronger
qualitative capability at comparable resources can also be meaningful results.

See `docs/RESEARCH_ROADMAP.md` for the project story, experimental sequence,
and evidence policy.
