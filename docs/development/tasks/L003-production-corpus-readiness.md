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
| F. Baseline materialization | ACTIVE | Convert Tatoeba TH/EN + Thai Wiki sample to common accepted JSONL schema | Hash-pinned derived summaries |
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

The worktree after committed checkpoint `e143c3e` contains active work that
must be preserved:

1. Repair the Thai technical classifier. The previous Thai lexicon had literal
   question-mark placeholders in source, so its Thai calibration was invalid.
2. Calibrate the repaired classifier conservatively. A v3 attempt overclassified
   SEO/gambling/general pages; v4 improved it; current v5 makes one strong Thai
   technical term insufficient by itself.
3. Materialize Tatoeba TH, Tatoeba EN and the Thai-Wikipedia sample into the
   same accepted JSONL/provenance schema used by FineWeb/FineWeb2.
4. Complete `global_manifest.py` for cross-source exact and SimHash near
   deduplication with holdout > validation > train priority and release-safe >
   research-only priority.
5. Recompute category/language/lane token budgets only after global dedup.

Current technical calibration observations:

- v2 Thai estimate: ~0.54% technical tokens, affected by corrupted Thai terms;
- v3 Thai estimate: ~9.71%, rejected as too permissive after manual spot check;
- v4 Thai estimate: ~2.27%, improved but still allowed some single-term false
  positives;
- v5 Thai estimate: ~1.79% in the frozen 1% calibration sample;
- English v5 remains ~19.79% in the same deterministic calibration framework.

v3/v4 are negative/intermediate evidence and should not be silently relabeled as
qualified. The final current classifier still requires regression + spot-check
evidence before it is frozen.

## Next action

1. Wait for/verify the three baseline materializations.
2. Pin their accepted-file hashes and summaries.
3. Add tests for the baseline materializer and global manifest.
4. Run global dedup across all five accepted corpora.
5. Produce the post-dedup budget by partition, rights lane, language and primary
   category.
6. Evaluate the 500M mixture contract.
7. Keep L003 open unless all required mixture/rights/evaluation gates pass.

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
