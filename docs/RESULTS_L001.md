# L001 Results — Blank LLM Foundation

## Decision

**ENGINEERING GREEN / RANDOM-INITIALIZED CAUSAL-LM BASELINE TRAINING-READY /
THAI+ENGLISH TOKENIZATION LOSSLESS / PRETRAINED KNOWLEDGE ABSENT BY
CONSTRUCTION / LANGUAGE QUALITY NOT YET QUALIFIED**

## Implemented

- separate private repository `funggier/FlyWireLLM`;
- MIT License;
- decoder-only causal Transformer;
- RMSNorm;
- RoPE;
- grouped-query attention;
- SwiGLU;
- tied token embeddings;
- UTF-8 byte tokenizer;
- causal next-token loss;
- backward pass + AdamW smoke update;
- safe checkpoint load with `weights_only=True`;
- random-origin/pretrained provenance metadata;
- JSONL corpus provenance/partition contract;
- smoke and small configuration profiles;
- step-0 blank-checkpoint generator;
- one-step training smoke script.

## Smoke profile

The L001 smoke profile contains 109,120 trainable parameters.

It can be executed on CPU and is intended only to test the complete training
path.

## Language bootstrap

The tokenizer round-trips:

- Thai;
- English;
- mixed Thai/English;
- numeric identifiers;
- arbitrary Unicode.

This is tokenization readiness, not language understanding.

## Smoke optimization

With seed 1234 on the tracked synthetic training fixture, the engineering smoke
run observed:

- initial loss: 5.590452;
- optimizer-step loss: 5.590452;
- post-step loss: 5.284992.

These values demonstrate that the objective, gradients and optimizer path are
connected. They are not benchmark or language-quality results.

## Checkpoint provenance

The blank checkpoint explicitly records:

- `pretrained=false`;
- `external_pretrained_weights=false`;
- random initialization origin;
- initialization seed;
- zero training steps;
- `knowledge_in_weights_claimed=false`.

Restricted `weights_only=True` loading is retained. During development, a
PyTorch `TorchVersion` metadata object was converted to a plain string rather
than weakening safe loading.

## Qualification

- full pytest: 26 passed;
- blank-model inspection: PASS;
- blank step-0 checkpoint safe reload: PASS;
- one-step smoke checkpoint safe reload: PASS;
- git diff check: PASS;
- wheel build: PASS;
- built wheel contains MIT dist-info/licenses/LICENSE: PASS.

## Scientific boundary

FlyWireModel knowledge is not claimed to exist in L001 weights.

The two repositories have separate responsibilities:

- FlyWireModel: evidence and connectome/neuroscience research;
- FlyWireLLM: language-model implementation and training.

Future FlyWire-informed changes must be optional experiments against this plain
baseline.
