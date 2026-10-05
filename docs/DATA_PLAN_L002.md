# L002 Pretraining Data Plan

## Primary target

Base-50M-v1 uses 500,000,000 tokens as the primary pretraining budget. The minimum research budget is 250M tokens and the stretch budget is 1B tokens.

Target mixture:

- 40% general Thai;
- 40% general English;
- 10% technical/scientific/code;
- up to 10% FlyWire/neuroscience domain.

The FlyWire-domain share is a maximum, not a quota. Scarce domain material must not be duplicated merely to hit the percentage.

## Frozen split contract

- train: 99.0%;
- validation: 0.5%;
- holdout: 0.5%.

Partitioning occurs by source group after conservative duplicate fingerprinting. The L002 split seed is flywirellm-l002-split-v1.

Dedup fingerprints use line-ending normalization, Unicode NFC, comparison-only whitespace collapse and SHA-256. The dedup-normalized string is not silently substituted for training text.

## Source audit states

Approved sources currently include Tatoeba and Mozilla Common Voice subject to release-specific verification. Conditional sources include Thai/English Wikipedia, FineWeb/FineWeb2 and selected Project Gutenberg works. Automatic FlyWireModel export remains blocked until a dedicated provenance policy excludes frozen evaluation evidence and resolves source-level rights.

## Acquired source inventory

The currently acquired and deduplicated Thai-Wikipedia-sample plus English-Tatoeba inventory contains:

- 2,099,784 records;
- 502,254,561 UTF-8 text bytes;
- 58,940,143 total tokens under the 32K candidate;
- 58,334,085 train tokens;
- 274,895 validation tokens;
- 331,163 holdout tokens.

By training language:

- Thai: 38,312,940 tokens;
- English: 20,021,145 tokens.

The acquired train inventory is 11.666817% of the 500M primary target. The exact remaining gap is 441,665,915 tokens.

This proves that the project has enough data to build and evaluate a tokenizer, but not enough approved/acquired data yet for the declared 500M pretraining run.

## Scale-source conclusion

Raw scale is available in principle from sources such as FineWeb/FineWeb2 and Wikimedia, but web-derived rights, PII/opt-out policy, attribution, provenance and quality filtering remain separate qualification work. L002 does not silently promote conditional sources to approved training data.

## Optimizer schedule

With 65,536 global tokens per optimizer update:

| Target | Optimizer steps | Warmup | Scheduled tokens |
| ---: | ---: | ---: | ---: |
| 250M | 3,815 | 77 | 250,019,840 |
| 500M | 7,630 | 153 | 500,039,680 |
| 1B | 15,259 | 306 | 1,000,013,824 |

Base recipe: AdamW beta1=0.9, beta2=0.95, eps=1e-8, weight decay=0.1, gradient clip=1.0, peak LR=6e-4, 2% warmup, cosine decay to 6e-5, primary seed=1234, bf16 preferred with fp32 correctness fallback.

## Compute boundary

Local CPU runs qualify correctness, tokenization and data plumbing. Full Base-50M pretraining compute is a later stage and must not weaken data quality or held-out evaluation integrity.
