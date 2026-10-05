# FlyWireLLM

FlyWireLLM is a separate language-model repository derived from the research
direction of `funggier/FlyWireModel`.

Its first milestone is deliberately modest and testable: a **blank,
random-initialized, training-ready decoder-only language model**. It does not
ship pretrained language knowledge and it does not claim to contain FlyWire
knowledge in its random weights.

## Repository boundary

- `FlyWireModel`: connectome, neuroscience, evidence qualification and
  biologically grounded research.
- `FlyWireLLM`: tokenizer, language-model architecture, training pipeline,
  checkpoints and future pretraining/fine-tuning.

FlyWireModel research may later inspire optional architecture or data
experiments here, but those experiments must be benchmarked against the plain
Transformer baseline.

## L001 baseline

The initial baseline uses:

- decoder-only causal Transformer;
- RMSNorm;
- RoPE;
- grouped-query attention;
- SwiGLU;
- tied input/output token embeddings;
- deterministic random initialization;
- UTF-8 byte tokenizer with four reserved special tokens;
- next-token cross-entropy objective;
- checkpoint metadata that records random-init provenance and forbids imported
  pretrained weights.

The UTF-8 byte tokenizer is intentionally simple. It is training-free and can
losslessly represent Thai, English, mixed Thai/English, code and arbitrary
Unicode from day one. A learned tokenizer can replace it later behind the same
interface.

## Important claims boundary

A blank checkpoint can encode Thai and English text, but it **cannot converse**
or understand either language before training.

Likewise, random weights do **not** contain FlyWire knowledge. Future knowledge
must enter through declared training data or separately evaluated architectural
experiments.

## Quick smoke test

Use a Python environment with the project dependencies installed:

~~~powershell
python -m pip install -e ".[dev]"
python -m pytest
python scripts/smoke_train.py --output checkpoints/smoke.pt
python scripts/inspect_blank_model.py
~~~

The smoke training script performs a deterministic optimizer step on a tiny
Thai/English fixture. Its loss is only an engineering sanity check, not a model
quality metric.

## License

MIT. See [LICENSE](LICENSE).
