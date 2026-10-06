# L003 — Production Corpus Acquisition, Rights Lanes, and 500M Readiness

Status: `ACTIVE`

GitHub Issue: `#4`

Updated: 2026-10-06

## Objective

Build and qualify a source-grouped Thai/English/technical corpus for
Base-50M-v1, with an explicit 500,000,000-token target and fail-closed
rights/privacy/provenance controls.

## Authoritative base

- repository: `funggier/FlyWireLLM`;
- active branch: `research/l003-corpus-rights-readiness`;
- L003 base: `528e96736c52d5ba90412bb1ee0b8c0239349e57`;
- latest pre-final handoff commit:
  `dcf73905e9fd7c9483d292f2758ac249ad5c2716`;
- live Git/GitHub/runtime state overrides this snapshot.

## Guardrails

- never reset/clean away active WIP;
- release-safe checkpoints may consume only release-safe sources;
- research-only checkpoints may consume release-safe + research-only sources;
- research-only lineage may never promote to release-safe;
- checkpoint lane is immutable after optimizer step 1;
- blocked sources enter no training lane;
- automatic FlyWireModel export remains blocked;
- holdout data may not fit tokenizer/model components;
- raw multi-gigabyte corpora remain outside Git;
- review records are quarantined rather than silently trained;
- no language-capability claim is made before actual training/evaluation.

## Phase state

| Phase | Status | Result |
| --- | --- | --- |
| A. Rights lanes | DONE | release-safe / research-only / blocked enforced |
| B. Artifact/provenance ledger | DONE | immutable source size/SHA pins |
| C. Quality/privacy gate | DONE | accept/quarantine/reject fail-closed screening |
| D. Scale acquisition | DONE | Thai + English scale corpora materialized |
| E. Technical routing | DONE | technical-heuristic-v5 frozen |
| F. Baseline materialization | DONE | Tatoeba TH/EN + Thai Wiki common schema |
| G. Global cross-source dedup | DONE | exact + SimHash near dedup complete |
| H. Mixture readiness | DONE | research-only 500M contract qualified |
| I. Freeze/authorization | DONE | source/evaluation/tokenizer/lineage frozen before step 1 |
| J. Qualification/merge | ACTIVE | exact-commit + remote-sync + main fast-forward pending |

## Historical checkpoints

| Checkpoint | Commit | Result |
| --- | --- | --- |
| Rights-lane contract | `a915a7315e0c63d663652b824057dcd7d4903a3e` | 53-test checkpoint |
| Screened acquisition pipeline | `4b0cef7` | artifact/quality/ingestion pipeline |
| Source newline normalization | `d29e899` | tracked source normalization |
| Thai derived capacity | `9e6f00c` | FineWeb2 Thai capacity |
| English capacity + calibration baseline | `e143c3e` | FineWeb English + calibration |
| Task-ledger continuity | `4482265` | active WIP state recorded |
| Full handoff | `dcf7390` | 130/130 tests before continuation |

## Technical-classifier qualification

The final classifier is `technical-heuristic-v5`.

Deterministic 1% calibration:

- FineWeb2 Thai technical tokens: 53,094;
- FineWeb English 014 technical tokens: 248,949.

v3/v4 remain tracked as rejected intermediate evidence. Their reports are
sanitized and calibration-only.

## Five-corpus global attempt

The first global manifest completed correctly but did not satisfy the mixture
contract:

- records total: 2,609,182;
- accepted: 2,608,512;
- exact duplicates: 3;
- near duplicates: 667;
- research total gap: 26,974,429;
- English gap: 81,481,741;
- technical gap: 62,326,681.

Thresholds were not weakened. The corpus plan was expanded instead.

## Final expanded global manifest

Final registry: `configs/global-manifest-inputs-l003-v2.json`.

The six unique artifacts are:

1. Tatoeba Thai screened baseline;
2. Tatoeba English screened baseline;
3. Thai-Wikipedia screened sample;
4. FineWeb2 Thai full derived shard;
5. FineWeb English 013 deterministic 55% derived shard;
6. FineWeb English 014 full derived shard.

The earlier 014 60% sample is replaced by the full 014 artifact and is not
double-counted.

Global result:

- records total: 3,277,753;
- accepted: 3,277,015;
- excluded: 738;
- exact duplicates: 61;
- near duplicates: 677;
- global report:
  `results/l003/global-manifest-v2.json`;
- metadata-only decision manifest bytes: 1,678,379,943;
- decision manifest SHA-256:
  `018aa08f347ebfd747f3aa11cada563bbd27918b469c7c5f11abe0289df7770a`.

## Final qualified train capacity

Release-safe:

- general English: 20,010,764;
- general Thai: 41,824;
- technical/scientific/code: 876;
- total: 20,053,464;
- gap to 500M: 479,946,536;
- ready: false.

Research-only eligible:

- general English: 522,373,720;
- general Thai: 316,825,768;
- technical/scientific/code: 130,501,732;
- total available: 969,701,220;
- Thai gap: 0;
- English gap: 0;
- technical gap: 0;
- total gap: 0;
- ready: true.

## Frozen authorization

`configs/pretraining-freeze-l003.json` freezes the selected
`research_only` checkpoint lane before optimizer step 1.

Frozen hashes:

- source registry:
  `5304c4cc6b302a06650d090d6731f5b0e610fd99ec257c3fd9cc671cf62f79aa`;
- global manifest report:
  `30a792d9fd26d1d4a6a1760aae7a43ae293778926be9044ebda632ec7c5054d5`;
- global decisions:
  `018aa08f347ebfd747f3aa11cada563bbd27918b469c7c5f11abe0289df7770a`;
- tokenizer model:
  `998bc75f058e554d4f50b6cbc77c52898e3446e516c3b25a126b99b113513818`.

Authorization state:

- optimizer steps completed: 0;
- pretraining authorized for research-only lineage: true;
- public release eligibility: not qualified;
- automatic FlyWireModel export: blocked.

## Qualification evidence before exact-commit gate

- external re-hash of tokenizer + all six accepted corpora: PASS;
- decision-manifest full hash/size verification: PASS;
- external summary byte/hash verification: PASS;
- `global_external_qualification=PASS`;
- canonical L003 audit: PASS;
- final L003 freeze audit: PASS;
- latest full repository suite before closure: 153/153 PASS;
- `git diff --check`: PASS.

These remain WIP evidence until the exact commit is created and qualified.

## Acceptance criteria

- [x] rights-lane contract and contamination tests;
- [x] immutable source artifact ledger;
- [x] quality/privacy screening with quarantine/reject behavior;
- [x] deterministic source-group partition contract;
- [x] large Thai and English research-only capacity acquired;
- [x] technical classifier frozen with Thai/English calibration;
- [x] baseline and scale sources use verified global-manifest schema;
- [x] global exact/near cross-source dedup complete;
- [x] final post-dedup token accounting complete;
- [x] selected research-only 500M mixture contract satisfied;
- [x] source/evaluation/tokenizer manifests frozen;
- [x] checkpoint lane frozen before optimizer step 1;
- [x] external artifact/hash qualification PASS;
- [x] full repository tests PASS before exact-commit gate;
- [x] canonical L003 audit PASS before exact-commit gate;
- [x] `git diff --check` PASS before exact-commit gate;
- [ ] exact-commit qualification PASS;
- [ ] remote race/sync verification PASS;
- [ ] final ledger closure commit qualified;
- [ ] main fast-forward + post-merge qualification PASS;
- [ ] GitHub Issue #4 updated and closed.

## Current work

All technical/data/freeze gates are GREEN. Phase J is the only active phase.

## Next action

1. inspect final diff/status for accidental raw/local-path material;
2. create the implementation checkpoint commit;
3. run exact-commit full pytest, both L003 audits and diff checks;
4. push branch and verify remote synchronization/race state;
5. update this ledger to DONE referencing the qualified implementation commit;
6. qualify/push the closure commit;
7. fast-forward main and repeat post-merge qualification;
8. record final evidence on GitHub Issue #4 and close it.

## Remaining boundary

Release-safe 500M capacity is still not qualified. This does not block the
selected research-only pretraining lineage, but it does block any claim that a
future research-only checkpoint can be publicly released as release-safe.

L003 does not claim Thai/English understanding, conversational quality, or
successful Base-50M pretraining. Those require later stages.
