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
