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
