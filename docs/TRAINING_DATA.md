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

At L002 close, the acquired/deduplicated engineering inventory contained
58,334,085 train tokens under the 32K candidate, leaving 441,665,915 tokens to
the 500M primary target.

See docs/DATA_PLAN_L002.md and configs/corpus-sources-l002.json.

## L003 qualified research-only corpus

L003 adds fail-closed source rights lanes, privacy/quality screening, immutable
artifact hashes, conservative exact deduplication, SimHash64 near deduplication,
and a frozen global manifest.

The final post-dedup train capacity is:

- release-safe total: 20,053,464 tokens;
- research-only eligible total: 969,701,220 tokens;
- research-only general Thai: 316,825,768 tokens;
- research-only general English: 522,373,720 tokens;
- research-only technical/scientific/code: 130,501,732 tokens.

The 500M research-only mixture gate is qualified with zero Thai, English,
technical and total gap. The release-safe lane is not qualified for 500M.

The frozen checkpoint lane is `research_only`, declared before optimizer step
1. Training and tokenizer fitting must remain train-only; final holdout remains
excluded. A research-only checkpoint may not later be promoted to
release-safe.

This is a data/lineage authorization only. Base-50M pretraining has not started,
public checkpoint release is not qualified, and automatic FlyWireModel export
remains blocked.

See docs/RESULTS_L003.md, configs/global-manifest-inputs-l003-v2.json and
configs/pretraining-freeze-l003.json.
