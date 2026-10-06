
# FlyWireLLM Full Session Handoff — L003 Global Manifest / Dedup WIP

Date: 2026-10-06

Project: funggier/FlyWireLLM

Local workspace: T:\Space\Projects\ProjectsAI\FlyWireLLM

External data root: T:\Space\Projects\ProjectsAI\FlyWireLLM-data

GitHub Issue: #4 — L003 Production Corpus Acquisition, Rights Lanes, and 500M-Token Readiness

## Critical resume rule

DO NOT reset, clean, checkout-over, stash-and-drop, or otherwise discard the current worktree.

At handoff creation the branch contains valid uncommitted implementation work for technical routing, baseline materialization and the global cross-source manifest. Preserve it first.

Live Git/GitHub/filesystem state is authoritative if it differs from this snapshot.

## Product direction

FlyWireLLM is the language-model repository separated from funggier/FlyWireModel.

The intended sequence remains:

1. release a blank/random-initialized LLM baseline;
2. define the real Base-50M architecture/tokenizer/data contract;
3. qualify a 500M-token corpus;
4. only then authorize Base-50M pretraining.

Do not claim language capability before actual training/evaluation.

## Completed foundation

### v0.1.0 blank LLM

Released and GREEN.

- tag: v0.1.0
- tag commit: 2ec2f502c8a4c5caa6c1f5262333e05dcacaf0a5
- MIT License
- blank/random-initialized smoke model
- 109,120 parameters
- no pretrained weights
- Thai/English UTF-8 byte-tokenizer round trip qualified
- release ZIP/document ZIP/checkpoint/SHA256 assets published

Do not modify/move the existing v0.1.0 tag or release.

### L002 Base-50M contract

Completed GREEN and merged to main.

- main / origin/main: 528e96736c52d5ba90412bb1ee0b8c0239349e57
- exact Base-50M-v1 parameters: 50,213,376
- vocab target: 32,000
- d_model: 512
- layers: 12
- query/KV heads: 8 / 2
- d_ff: 1,408
- initial context: 1,024
- RMSNorm + RoPE + GQA + SwiGLU + tied embeddings
- random initialization

32K tokenizer engineering candidate:

- SentencePiece 0.2.2 Unigram
- identity normalization
- byte fallback
- exact vocab 32,000
- 10,150 benchmark cases
- round-trip failures: 0
- unknown tokens: 0
- compression vs byte:
  - overall 6.252x
  - Thai 10.710x
  - English 4.031x
  - mixed 2.570x
- model SHA-256: 998bc75f058e554d4f50b6cbc77c52898e3446e516c3b25a126b99b113513818
- vocab SHA-256: 9ade52fa3e57f6bb096cfdcc64b37d89f794937b68d155997b48f529c841e762

Tokenizer binary remains outside the MIT source tree because its engineering fit used conditionally licensed Thai Wikipedia material.

## Active L003 branch

Branch: research/l003-corpus-rights-readiness

Implementation/task-state HEAD immediately before this handoff document was created:

448226598049d504ea2f89eeae8186a0ce00d0eb

At that checkpoint:

- local branch == origin/research/l003-corpus-rights-readiness
- ahead/behind = 0/0
- main remains at L002 528e96736c52d5ba90412bb1ee0b8c0239349e57
- L003 is intentionally not merged

Recent L003 commits after main:

- a915a73 Add L003 corpus rights lanes
- 4b0cef7 Add L003 screened corpus acquisition pipeline
- d29e899 Normalize L003 source file endings
- 9e6f00c Register L003 Thai derived capacity
- e143c3e Register L003 English capacity and technical calibration
- e6349f4 Add development task ledger
- 1521bc6 Clarify active task resume pointer
- 4482265 Update L003 live task state

The latest committed implementation checkpoint is e143c3e. The later commits are task/status documentation checkpoints.

## L003 rights model

Three source states are mandatory.

### release_safe

Only release-safe sources may enter a release-safe checkpoint lineage.

### research_only

May consume release-safe + research-only sources.

Once a checkpoint has consumed research-only data it cannot be promoted back to release-safe.

### blocked

May enter no training lane.

Automatic FlyWireModel export remains blocked until source rights and frozen scientific evaluation material are explicitly separated.

Checkpoint lane becomes immutable after optimizer step 1.

## L003 phase state

- Phase A rights lanes: DONE
- Phase B artifact/provenance ledger: DONE
- Phase C quality/privacy gate: DONE
- Phase D scale acquisition: ACTIVE
- Phase E technical routing: ACTIVE
- Phase F baseline materialization: ACTIVE; local materialization complete but checkpoint commit pending
- Phase G global cross-source dedup: ACTIVE
- Phase H mixture readiness: PLANNED
- Phase I freeze/authorization: PLANNED
- Phase J qualification/merge: PLANNED

Common Voice v27 metadata was evaluated for release-safe planning, but Mozilla Data Collective archive download requires authenticated access/terms. Do not bypass that access control.

## Current pre-global-dedup capacity

These are planning numbers only. Do not call them final training capacity until the central global manifest/dedup run completes.

Tracked research-only web capacity:

- FineWeb2 Thai accepted train tokens: 293,826,034
- FineWeb English accepted train tokens: 120,975,441

Screened baseline:

- Tatoeba EN release-safe train: 20,019,410
- Tatoeba TH release-safe train: 41,836
- Thai-Wikipedia sample research-only train: 38,172,496

Pre-global-dedup sum: 473,035,217 train tokens.

The 500M mixture contract is stricter than total tokens:

- general Thai minimum: 200,000,000
- general English minimum: 200,000,000
- technical/scientific/code base target: 50,000,000
- FlyWire domain target/max: 50,000,000
- if FlyWire domain is unused, that allocation moves to technical/scientific/code
- technical + FlyWire must total 100,000,000
- general-language overage cannot substitute another category
- every record has one primary category

Therefore total raw capacity near 500M is not sufficient by itself.

## Materialized baseline files

### Tatoeba Thai

- accepted: 6,851
- train tokens: 41,836
- accepted SHA-256: 2a64b7a9e32f9a059c89a7e8dfe60f20e6c36d39004bdc1537eac74036c1737c
- accepted storage: external://FlyWireLLM-data/L003/Baseline/tatoeba-th-screened-v1/accepted.jsonl

### Tatoeba English

- accepted: 2,038,059
- quarantine: 3
- reject: 75
- train tokens: 20,019,410
- accepted SHA-256: 3995909ec3c15bb0700cfce9ea14ec2906b47e63e8d175098d94e96c87bd019d
- accepted storage: external://FlyWireLLM-data/L003/Baseline/tatoeba-en-screened-v1/accepted.jsonl

### Thai-Wikipedia sample

- accepted: 61,538
- quarantine: 23
- reject: 86
- train tokens: 38,172,496
- accepted SHA-256: 05f4cf3577eae401b805fc2eaf2063b3d5fe2b840bcbcacd32064cfc898c5b0f
- accepted storage: external://FlyWireLLM-data/L003/Baseline/thwiki-sample-v1-screened-v1/accepted.jsonl

The first materialization launch failed only because the output/log directory did not exist. Directories were created and all jobs reran successfully. Treat that as orchestration history, not a corpus failure.

## Large derived corpora

### FineWeb2 Thai

Derived ID: fineweb2-thai-train-005-00002-sample1000000-v1

- source: fineweb2-thai
- lane: research_only
- accepted records: 339,058
- accepted bytes: 3,072,938,059
- accepted SHA-256: 7b22cccde21f195ee3fc6585af2abd8ba8c3a4f63aa3ed082930a07d9ee0e633
- train tokens: 293,826,034
- validation tokens: 1,449,071
- holdout tokens: 1,479,220
- storage: external://FlyWireLLM-data/L003/FineWeb2/tha_Thai/derived/005_00002-full-v1/accepted.jsonl

### FineWeb English

Derived ID: fineweb-english-10bt-014-00000-sample600000-v1

- source: fineweb-english
- lane: research_only
- accepted records: 163,676
- accepted bytes: 603,547,565
- accepted SHA-256: 13526d486bf90e06b6a3f2cdfb6ae0f653541f6c55876da0ca052f3d3d4ad803
- train tokens: 120,975,441
- validation tokens: 539,655
- holdout tokens: 685,017
- storage: external://FlyWireLLM-data/L003/FineWeb/english/derived/014_00000-sample600000-v1/accepted.jsonl

Both remain research-only.

## Technical routing history and current WIP

The technical classifier must be calibrated before final category budgets are trusted.

History:

- v2 exposed a source-encoding bug: Thai lexicon entries had literal question-mark placeholders.
- v3 repaired UTF-8 Thai terms but classified about 9.71% Thai technical tokens and manual spot checking found SEO/gambling false positives.
- v4 reduced Thai estimate to about 2.27%, but one strong Thai term could still classify unrelated content.
- v5 is the current WIP candidate.

v5 design:

- separates strong / support / generic Thai technical terms;
- counts distinct term presence rather than repeated occurrences;
- one strong Thai term alone is insufficient;
- generic terms only add support when stronger evidence exists.

Deterministic 1% calibration results currently in the dirty worktree:

Thai FineWeb2:

- records sampled: 3,383
- technical records: 19
- technical tokens: 53,094
- total sampled tokens: 2,962,779
- estimated technical token fraction: 0.0179203376, about 1.79%

English FineWeb:

- records sampled: 1,666
- technical records: 81
- technical tokens: 248,949
- total sampled tokens: 1,257,720
- estimated technical token fraction: 0.1979367427, about 19.79%

These v5 reports are WIP and must be frozen/sanitized before they become a committed qualification result.

## Dirty worktree that must be preserved

At handoff creation the worktree contains 4 modified + 17 untracked files.

Modified:

- scripts/calibrate_technical_mix.py
- src/flywire_llm/near_duplicate.py
- src/flywire_llm/technical.py
- tests/test_technical.py

Important untracked files:

- configs/global-manifest-inputs-l003.json
- results/l003/derived/tatoeba-th-screened-v1.json
- results/l003/derived/tatoeba-en-screened-v1.json
- results/l003/derived/thwiki-sample-v1-screened-v1.json
- v3/v4/v5 Thai technical calibration reports
- v3/v4/v5 English technical calibration reports
- scripts/materialize_baseline_l003.py
- src/flywire_llm/baseline_sources.py
- src/flywire_llm/global_manifest.py
- src/flywire_llm/manifest_inputs.py
- tests/test_baseline_sources.py
- tests/test_global_manifest.py
- tests/test_manifest_inputs.py

Use git status --short on resume for the exact live list.

## Central global-manifest input registry WIP

configs/global-manifest-inputs-l003.json currently pins five accepted corpora:

1. Tatoeba TH — release_safe
2. Tatoeba EN — release_safe
3. Thai-Wikipedia sample — research_only
4. FineWeb2 Thai — research_only
5. FineWeb English — research_only

Registry contract:

- tokenizer: base50m-unigram-32000-v1
- technical classifier: technical-heuristic-v5
- near-duplicate Hamming distance: 3
- central global manifest authorization: true
- pretraining authorized: false
- every summary hash, accepted byte count and accepted SHA-256 is pinned
- accepted corpus storage must be external://
- source/artifact/rights-lane mismatches fail closed

## Global dedup WIP design

src/flywire_llm/global_manifest.py currently implements:

- recomputation/validation of content SHA-256, exact dedup fingerprint and SimHash from source text;
- canonical priority: holdout, then validation, then train;
- within a partition: release_safe before research_only;
- deterministic source/artifact tie breaking;
- exact cross-source duplicate exclusion;
- SimHash near-duplicate exclusion at Hamming distance <= 3;
- SQLite-backed indexes;
- batched commits for multi-million-record scalability;
- one primary category per accepted record;
- category routing through technical-heuristic-v5.

Important intent: a train copy must never become canonical over a matching validation/holdout copy, and research-only material must not displace an otherwise equivalent release-safe canonical record at the same partition priority.

## Current WIP validation evidence

On the dirty worktree at handoff creation:

Focused suite:

tests/test_baseline_sources.py
tests/test_manifest_inputs.py
tests/test_global_manifest.py
tests/test_technical.py
tests/test_near_duplicate.py

Result: 27/27 PASS

Full repository suite: 130 collected tests, all PASS

git diff --check: PASS

This is WIP evidence only because the implementation files/reports are not all committed yet.

## Next actions — execute in this order

1. Do not reset the worktree.
2. Refresh remote state, but preserve all current local WIP.
3. Re-run git status --short --branch.
4. Re-run the focused 27-test suite before edits.
5. Freeze/sanitize v5 technical calibration:
   - keep v5 as current candidate unless new evidence disproves it;
   - preserve v3/v4 as negative calibration history;
   - ensure reports do not embed unnecessary machine-local absolute paths if tracked-report convention requires sanitization;
   - add final regression expectations.
6. Validate/pin the three baseline summary reports and the five-input central registry.
7. Complete global-manifest runner/materializer around global_manifest.py.
8. Verify each external accepted file against pinned byte size + SHA-256 before processing.
9. Run global exact + near dedup across all five corpora.
10. Emit a metadata-only global manifest/report; do not commit raw corpus text.
11. Recompute post-dedup budgets by partition, rights lane, language and primary category.
12. Evaluate configs/mixture-l003.json:
   - at least 200M general Thai
   - at least 200M general English
   - technical + FlyWire = 100M
   - total at least 500M
13. If mixture fails, acquire more data rather than weakening the contract.
14. Freeze source/evaluation manifests and checkpoint lane before optimizer step 1.
15. Only when all L003 acceptance gates pass:
   - full tests
   - L003 audit
   - git diff --check
   - exact-commit qualification
   - fetch/race check
   - fast-forward merge to main
   - post-merge qualification
   - push main
   - verify main == origin/main
   - update/close GitHub Issue #4 GREEN

## Resume commands

Use the existing Python environment currently used for this project:

T:\Space\Projects\ProjectsAI\FlyWireModel\.venv\Scripts\python.exe

There is no requirement to create a new venv merely to resume the current WIP.

Suggested first commands:

    cd T:\Space\Projects\ProjectsAI\FlyWireLLM
    git fetch --all --prune --tags
    git status --short --branch
    git rev-parse HEAD
    git rev-parse origin/main
    git log --oneline --decorate -12

    $env:PYTHONPATH='src'
    T:\Space\Projects\ProjectsAI\FlyWireModel\.venv\Scripts\python.exe -m pytest -q tests\test_baseline_sources.py tests\test_manifest_inputs.py tests\test_global_manifest.py tests\test_technical.py tests\test_near_duplicate.py

Then read:

1. this handoff;
2. docs/development/tasks/L003-production-corpus-readiness.md;
3. configs/global-manifest-inputs-l003.json;
4. src/flywire_llm/global_manifest.py;
5. src/flywire_llm/manifest_inputs.py;
6. current git diff.

## Do not do

- do not reset/clean the worktree;
- do not merge L003 merely because pre-dedup total is near 500M;
- do not authorize Base-50M pretraining yet;
- do not count pre-global-dedup capacity as final capacity;
- do not relabel research-only web data as release-safe to close a budget gap;
- do not bypass Mozilla Data Collective authentication/terms;
- do not commit large/raw corpus text to Git;
- do not silently fit on holdout/evaluation data;
- do not automatically export FlyWireModel evidence into training;
- do not modify or move v0.1.0 release/tag;
- do not claim Thai/English understanding before actual training/evaluation.

## Final state at handoff

- v0.1.0: released GREEN
- L002: closed GREEN
- L003 Issue #4: OPEN / ACTIVE
- main: frozen at completed L002
- active branch: research/l003-corpus-rights-readiness
- implementation WIP: valid and locally passing, not yet frozen
- pretraining: NOT AUTHORIZED
- next technical milestone: freeze v5 + central five-corpus global manifest, run global cross-source dedup, then evaluate the 500M mixture contract.
