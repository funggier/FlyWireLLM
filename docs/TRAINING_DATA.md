# Training Data Contract

## Purpose

L001 includes a minimal JSONL corpus contract so later pretraining can start
without inventing provenance conventions after the fact.

Every training text record contains:

- `record_id`
- `text`
- `language`
- `source_id`
- `license`
- `partition`

Allowed partitions:

- `train`
- `validation`
- `holdout`

Allowed language labels in the bootstrap contract:

- `th`
- `en`
- `mixed`
- `other`

The labels describe corpus metadata; the UTF-8 tokenizer itself is not limited
to these languages.

## Provenance

The contract does not allow an omitted source or license.

This is deliberate. Future FlyWireModel-derived knowledge, public corpora,
generated data and user-authored data must remain distinguishable.

A future production corpus manifest should additionally pin source hashes,
collection dates, transformations and deduplication provenance.

## Partition boundary

The tracked example contains train, validation and holdout records only as
synthetic engineering fixtures.

Real training must keep holdout data out of optimizer updates and tokenizer
fitting.

## Thai and English

Thai and English are first-class data targets for later pretraining, but the
blank L001 model has not learned either language.

L001 proves only that:

- both languages can be encoded losslessly;
- both languages can pass through the causal LM objective;
- future corpora can label their language and source consistently.

Language capability must be evaluated after training.

## L002 scale contract

L002 extends the bootstrap JSONL contract with a source inventory and
deterministic source-group split.

Primary pretraining target: 500M tokens.

Target mixture:

- 40% general Thai;
- 40% general English;
- 10% technical/scientific/code;
- up to 10% FlyWire/neuroscience domain.

The frozen split is 99.0% train / 0.5% validation / 0.5% holdout using
source-group identities after conservative deduplication.

Source audit states are explicit: approved, conditional, or blocked.
Conditional data are not silently promoted to production training data.

The current acquired/deduplicated engineering inventory contains 58,334,085
train tokens under the 32K candidate, leaving 441,665,915 tokens to the 500M
primary target.

See docs/DATA_PLAN_L002.md and configs/corpus-sources-l002.json.
