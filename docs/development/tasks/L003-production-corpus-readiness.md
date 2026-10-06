# L003 — Production Corpus Acquisition, Rights Lanes, and 500M Readiness

Status: `ACTIVE`

GitHub Issue: `#4`

Updated: 2026-10-06

## Objective

Build and qualify a source-grouped Thai/English/technical corpus for
Base-50M-v1, with an explicit 500,000,000-token target and fail-closed
rights/privacy/provenance controls.

## Why

L002 proved the model/tokenizer contract but did not provide enough qualified
training data. L003 must close the corpus gap without contaminating
release-safe checkpoints with research-only sources or leaking final evaluation
material into training.

## Authoritative base

- repository: `funggier/FlyWireLLM`;
- active branch: `research/l003-corpus-rights-readiness`;
- latest committed checkpoint at this update:
  `e143c3e23fed5c328422f2e92551d12a123e1ccb`;
- `main` remains at the completed L002 commit
  `528e96736c52d5ba90412bb1ee0b8c0239349e57`;
- L003 is intentionally **not merged** while acceptance is incomplete.

Live Git/GitHub state always overrides this snapshot.

## Guardrails

- no worktree reset to discard active work;
- release-safe checkpoints may consume only release-safe sources;
- research-only checkpoints may consume release-safe + research-only sources;
- a research-only checkpoint lineage cannot be promoted to release-safe;
- blocked sources enter no training lane;
- automatic FlyWireModel export remains blocked;
- final holdout/evaluation material may not be used to fit tokenizer/model
  components;
- raw/large corpora remain outside Git; tracked reports contain hashes,
  provenance, counts and metrics;
- screening `review` records are quarantined, not silently trained;
- pretraining remains unauthorized until the global corpus/mixture gates pass.

## Phase plan

| Phase | Status | Goal | Exit condition |
| --- | --- | --- | --- |
| A. Rights lanes | DONE | Separate release-safe / research-only / blocked lineage | Fail-closed policy + tests |
| B. Artifact/provenance ledger | DONE | Pin source files, hashes, acquisition metadata | Artifact validator + immutable hashes |
| C. Quality/privacy gate | DONE | Screen quality/PII/secrets without storing flagged text in reports | Accept/quarantine/reject audit |
| D. Scale acquisition | ACTIVE | Acquire enough Thai/English capacity | Pinned derived corpora and token counts |
| E. Technical routing | ACTIVE | Route technical/scientific/code data conservatively | Thai+English calibrated classifier |
| F. Baseline materialization | ACTIVE | Convert Tatoeba TH/EN + Thai Wiki sample to common accepted JSONL schema | Local materialization is complete; summaries/input registry still require checkpoint commit |
| G. Global cross-source dedup | ACTIVE | Exact + near dedup across all accepted sources with partition/lane priority | Global manifest and reproducible budget |
| H. Mixture readiness | PLANNED | Satisfy 200M Thai + 200M English + 100M technical/FlyWire policy | 500M global post-dedup mixture gate |
| I. Freeze/authorization | PLANNED | Freeze production manifests and lineage | Exact hashes + all authorization gates |
| J. Qualification/merge | PLANNED | Exact-commit tests/audit/race-check and merge | Issue #4 closed GREEN |

## Completed checkpoints

| Checkpoint | Commit | Result |
| --- | --- | --- |
| Rights-lane contract | `a915a7315e0c63d663652b824057dcd7d4903a3e` | 53-test exact-commit checkpoint; branch pushed |
| Screened acquisition pipeline | `4b0cef7` | artifact, quality, ingestion, calibration and materializers added |
| Source newline normalization | `d29e899` | source file normalization |
| Thai derived capacity | `9e6f00c` | FineWeb2 Thai derived capacity registered |
| English capacity + technical calibration | `e143c3e` | FineWeb English capacity registered; technical calibration baseline |

## Current capacity before global cross-source dedup

This is planning capacity only. It must not be treated as final pretraining
capacity until phase G completes.

Tracked web-derived research-only corpora include:

- FineWeb2 Thai accepted train tokens: 293,826,034;
- FineWeb English accepted train tokens: 120,975,441.

Screened legacy baseline before global dedup:

- release-safe Tatoeba EN train: 20,019,410 tokens;
- release-safe Tatoeba TH train: 41,836 tokens;
- research-only Thai-Wikipedia sample train: 38,172,496 tokens.

## Current work

The latest committed technical implementation checkpoint remains
`e143c3e23fed5c328422f2e92551d12a123e1ccb`. Task-ledger commits
`e6349f424d4f3c0b525b48d212c392805d6ab7d6` and
`1521bc60ada867112b6517f0ea22921cc8f603d9` are already pushed.

Active implementation work after those commits must be preserved:

1. Thai technical routing was repaired after discovering literal question-mark
   placeholders in the v2 source lexicon.
2. v3 (~9.71% Thai technical tokens) was rejected as too permissive after a
   manual spot check found SEO/gambling false positives.
3. v4 (~2.27%) reduced those false positives but still allowed a single strong
   Thai term to classify some unrelated pages.
4. v5 is the current candidate. One strong Thai term is insufficient by itself;
   the deterministic 1% calibration estimates ~1.79% Thai technical tokens and
   ~19.79% English technical tokens. It is not frozen until its final
   regression/report checkpoint is committed.
5. Baseline materialization completed locally:
   - Tatoeba TH: 6,851 accepted; train 41,836 tokens; accepted SHA-256
     `2a64b7a9e32f9a059c89a7e8dfe60f20e6c36d39004bdc1537eac74036c1737c`;
   - Tatoeba EN: 2,038,059 accepted; train 20,019,410 tokens; quarantine 3 /
     reject 75; accepted SHA-256
     `3995909ec3c15bb0700cfce9ea14ec2906b47e63e8d175098d94e96c87bd019d`;
   - Thai-Wikipedia sample: 61,538 accepted; train 38,172,496 tokens;
     quarantine 23 / reject 86; accepted SHA-256
     `05f4cf3577eae401b805fc2eaf2063b3d5fe2b840bcbcacd32064cfc898c5b0f`.
6. The first materialization launch failed before corpus processing because the
   log-output directory did not yet exist. The directories were created and
   all three jobs were rerun successfully; this was an orchestration failure,
   not a corpus/data failure.
7. A central five-corpus global-manifest input registry is now being validated.
   Focused baseline/global-input/global-dedup/technical tests currently pass
   27/27 locally, but this is WIP evidence until an exact commit is qualified.
8. `global_manifest.py` is being completed for cross-source exact and SimHash
   near deduplication with holdout > validation > train priority and
   release-safe > research-only priority. SQLite writes are being batched for
   multi-million-record scale.
9. Category/language/lane token budgets will be recomputed only after global
   dedup.

## Next action

1. Freeze/sanitize the v5 calibration reports and regression expectations.
2. Pin the three baseline summaries and central five-corpus input registry.
3. Complete global-manifest tests and exact source-file verification.
4. Create the external metadata-only global manifest; do not commit raw corpus
   text.
5. Run global exact/near dedup across all five accepted corpora.
6. Produce the post-dedup budget by partition, rights lane, language and primary
   category.
7. Evaluate the 500M mixture contract.
8. Keep L003 open unless all required mixture/rights/evaluation gates pass.

## Acceptance criteria

- [x] rights-lane contract and contamination tests;
- [x] immutable source artifact ledger;
- [x] quality/privacy screening with quarantine/reject behavior;
- [x] deterministic 99/0.5/0.5 source-group split;
- [x] large Thai and English research-only capacity acquired/materialized;
- [ ] technical classifier frozen with trustworthy Thai/English calibration;
- [ ] all baseline and derived sources use one verified global-manifest schema;
- [ ] global exact/near cross-source dedup complete;
- [ ] final post-dedup token accounting complete;
- [ ] declared 500M mixture contract satisfied for the selected training lane;
- [ ] production source/evaluation manifests frozen;
- [ ] checkpoint lane and corpus lineage frozen before optimizer step 1;
- [ ] full repository tests PASS;
- [ ] L003 audit PASS;
- [ ] `git diff --check` PASS;
- [ ] exact-commit qualification PASS;
- [ ] remote race/sync verification PASS;
- [ ] GitHub Issue #4 updated and closed only when genuinely complete.

## Blockers / dependencies

- Common Voice metadata is usable for planning, but archive download requires
  authenticated Mozilla Data Collective access/terms. Do not bypass access
  controls.
- Release-safe capacity remains far below 500M; research-only web corpora cannot
  be reclassified merely to close that gap.
- Global post-dedup capacity is not yet known.

## Claims boundary

L003 does not itself prove Thai/English language understanding, conversational
quality, or successful Base-50M pretraining. Corpus capacity before global
cross-source dedup is planning evidence only.