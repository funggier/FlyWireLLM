# L003 Results — Production Corpus Readiness and 500M Authorization

## Decision

**L003 GREEN / TECHNICAL HEURISTIC V5 FROZEN / GLOBAL EXACT+NEAR DEDUP QUALIFIED / 500M RESEARCH-ONLY MIXTURE QUALIFIED / PRETRAINING AUTHORIZED FOR RESEARCH-ONLY LINEAGE / PUBLIC RELEASE NOT QUALIFIED**

L003 closes the production-corpus readiness gate for the first Base-50M
research run. It does **not** mean that Base-50M has been pretrained.

## Selected checkpoint lane

The selected lineage is `research_only`.

This lane may consume both `release_safe` and `research_only` sources. Once
optimizer step 1 is executed, the checkpoint lane is immutable and the
resulting lineage may not be promoted to `release_safe`.

The final freeze was created before optimizer step 1:

- optimizer steps completed: 0;
- pretraining authorized: true;
- public release eligibility: not qualified;
- automatic FlyWireModel export: blocked.

## Technical routing

The production routing classifier is frozen as
`technical-heuristic-v5`.

Deterministic 1% calibration evidence:

- FineWeb2 Thai technical tokens: 53,094;
- FineWeb English 014 technical tokens: 248,949.

Earlier v3/v4 calibration evidence remains tracked as negative history. Those
iterations were not silently rewritten or promoted.

## Five-corpus qualification attempt

The first global run used five accepted corpora and completed cross-source
exact/near deduplication successfully, but it correctly failed the 500M
research-only mixture gate.

Five-corpus result:

- records total: 2,609,182;
- records accepted: 2,608,512;
- exact duplicates excluded: 3;
- near duplicates excluded: 667;
- research total gap: 26,974,429 tokens;
- English gap: 81,481,741 tokens;
- technical gap: 62,326,681 tokens.

No mixture threshold was weakened to make this attempt pass.

## Expanded global manifest

The final registry uses six unique source artifacts. The earlier 60% FineWeb
English 014 sample was replaced by the full 014 derived corpus, not counted
twice, and FineWeb English 013 was added.

Global exact/near dedup result:

- records total: 3,277,753;
- records accepted: 3,277,015;
- records excluded: 738;
- exact duplicates: 61;
- near duplicates: 677;
- near-duplicate contract: SimHash64 + 4x16-bit LSH + Hamming distance <= 3;
- canonical partition priority: holdout > validation > train;
- canonical lane priority at equal partition priority:
  release-safe > research-only.

The tracked global report is
`results/l003/global-manifest-v2.json`.

## Qualified train capacity

Post-dedup capacity under the Base-50M 32K tokenizer:

### Release-safe lane

- general English: 20,010,764 tokens;
- general Thai: 41,824 tokens;
- technical/scientific/code: 876 tokens;
- FlyWire domain: 0 tokens;
- total: 20,053,464 tokens;
- gap to 500M: 479,946,536 tokens;
- ready: false.

### Research-only lane

- general English: 522,373,720 tokens;
- general Thai: 316,825,768 tokens;
- technical/scientific/code: 130,501,732 tokens;
- FlyWire domain: 0 tokens;
- total available: 969,701,220 tokens;
- Thai minimum gap: 0;
- English minimum gap: 0;
- technical minimum gap: 0;
- total 500M gap: 0;
- ready: true.

The available corpus intentionally exceeds 500M. A later pretraining stage must
sample the declared 500M mixture; it must not interpret 969,701,220 as a
requirement to train on every available token.

## Frozen hashes

Final source registry:

- path: `configs/global-manifest-inputs-l003-v2.json`;
- SHA-256:
  `5304c4cc6b302a06650d090d6731f5b0e610fd99ec257c3fd9cc671cf62f79aa`.

Global manifest report:

- path: `results/l003/global-manifest-v2.json`;
- SHA-256:
  `30a792d9fd26d1d4a6a1760aae7a43ae293778926be9044ebda632ec7c5054d5`.

Metadata-only global decision manifest:

- storage:
  `external://FlyWireLLM-data/L003/Global/global-manifest-v2/decisions.jsonl`;
- bytes: 1,678,379,943;
- SHA-256:
  `018aa08f347ebfd747f3aa11cada563bbd27918b469c7c5f11abe0289df7770a`;
- raw text stored in decision manifest: false.

Tokenizer:

- candidate: `base50m-unigram-32000-v1`;
- model SHA-256:
  `998bc75f058e554d4f50b6cbc77c52898e3446e516c3b25a126b99b113513818`.

## External qualification

The final qualification re-hashed the tokenizer, all six accepted external
corpora, the 1.68 GB decision manifest, and the external copy of the final
summary.

Result: `global_external_qualification=PASS`.

No raw multi-gigabyte corpus is committed to Git.

## Evaluation and lineage freeze

`configs/pretraining-freeze-l003.json` freezes:

- the source registry and its exact hash;
- the global manifest report and its exact hash;
- the global decisions hash;
- validation as `validation`;
- final holdout as `holdout`;
- model-fit partitions as train-only;
- holdout exclusion from tokenizer fitting;
- holdout exclusion from model fitting;
- the tokenizer candidate and exact model hash;
- research-only checkpoint lineage before optimizer step 1.

## Rights boundary

L003 does not make the web/Wikimedia research corpus release-safe.

In particular:

- the research-only lineage may not be promoted to release-safe;
- public release eligibility is still `not_qualified`;
- release-safe capacity remains far below 500M;
- automatic FlyWireModel curated export remains blocked;
- Common Voice archive acquisition still requires authenticated Mozilla Data
  Collective access and is not bypassed.

## Claims boundary

L003 authorizes the **data and lineage gate** for a future Base-50M
research-only pretraining run.

L003 does not claim:

- that Base-50M has been pretrained;
- Thai or English understanding;
- conversational capability;
- model-quality benchmark results;
- public redistribution rights for a checkpoint trained on the research-only
  corpus;
- that FlyWire knowledge is present in model weights.

Those claims require later training and evaluation stages.
