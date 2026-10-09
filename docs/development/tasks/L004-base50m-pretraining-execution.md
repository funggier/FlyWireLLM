# L004 — Base-50M Research-Only Pretraining Execution

Status: ACTIVE

GitHub Issue: #6

Updated: 2026-10-09

## Objective

Implement, qualify, and then execute the first real Base-50M-v1
research-only pretraining run from the frozen L003 lineage.

L004 has qualified the runtime/data path and has now exact-qualified optimizer
step 1193. Cumulative supervised training is 78,184,448 of 500,000,000 tokens
(15.6368896% of the primary budget). Full Base-50M pretraining is not complete.

## Authoritative base

- repository: funggier/FlyWireLLM;
- branch: research/l004-base50m-pretraining;
- branch base commit: aa8a00e7ee5c3e8827072a47d5466f1fbbaf7997;
- latest release: v0.2.0;
- v0.2.0 tag commit: fc9e0fc2f64be4574b70021226ffd97f61fd9320.

## Frozen L003 lineage

- checkpoint lane: research_only;
- optimizer steps completed at L003 freeze: 0;
- L003 pretraining authorization: true;
- public release eligibility: not qualified;
- automatic FlyWireModel export: blocked.

Frozen hashes:

- source registry:
  5304c4cc6b302a06650d090d6731f5b0e610fd99ec257c3fd9cc671cf62f79aa;
- global report:
  30a792d9fd26d1d4a6a1760aae7a43ae293778926be9044ebda632ec7c5054d5;
- global decisions:
  018aa08f347ebfd747f3aa11cada563bbd27918b469c7c5f11abe0289df7770a;
- tokenizer:
  998bc75f058e554d4f50b6cbc77c52898e3446e516c3b25a126b99b113513818.

## Runtime facts

Qualification machine:

- Windows 10 build 19045;
- Intel Core Ultra 5 245K;
- 14 logical CPU threads exposed to PyTorch;
- RAM: ~33.8 GB;
- PyTorch: 2.14.1+cpu;
- CUDA: unavailable;
- MKLDNN: available.

CPU bfloat16 executes but is rejected for this runtime because measured
performance is dramatically worse than fp32.

## Why the v0.1.0 checkpoint path is not reused

The legacy checkpoint path remains the blank/random-initialized v0.1.0
contract. L004 uses a dedicated pretraining format instead because the real
Base-50M run must preserve:

- frozen 32K SentencePiece tokenizer identity;
- L003 lineage hashes;
- L004 packed-data runtime hash;
- model + optimizer state;
- optimizer progress;
- exact supervised-token progress;
- packed-data cursor position;
- Torch/data RNG state;
- atomic checkpoint replacement.

The blank-checkpoint provenance contract is not weakened.

## Phases

| Phase | Status | Result |
| --- | --- | --- |
| A. Execution contract | DONE | L003 lineage + Base-50M/training/data pins verified |
| B. Base-50M tokenizer runtime | DONE | exact model hash, vocab=32K, special-token contract |
| C. Training schedule/runtime | DONE | AdamW + warmup/cosine + clipping + token-weighted accumulation |
| D. Pretraining checkpoint format | DONE | model/optimizer/progress/RNG/data cursor/lineage + atomic save |
| E. Deterministic resume | DONE | interrupted vs continuous model/data-cursor equivalence |
| F. Mixture selection/materialization | DONE | exact 500M selected, packed, shuffled, verified |
| G. Base-50M micro qualification | DONE | real forward/backward/update with frozen tokenizer + packed data |
| H. Throughput/memory qualification | DONE | fp32 local runtime selected; bf16 rejected |
| I. First bounded training tranche | DONE | 4 updates / 262,144 supervised tokens; checkpoints verified |
| J. Validation gate | ACTIVE | deterministic step-0 vs step-4 validation before any larger tranche |

## Phase F — exact 500M selection

Selection contract:

- lane: research_only;
- partition: train only;
- deterministic key:
  sha256(seed_decimal + '|' + canonical_id);
- seed: 1234;
- raw text in selection manifest: false;
- selected records: 1,358,839;
- partial records: exactly 3, one final prefix-truncated record per active
  category.

Selected content tokens:

- general Thai: 200,000,000;
- general English: 200,000,000;
- technical/scientific/code: 100,000,000;
- total: 500,000,000.

Selection manifest:

- bytes: 580,186,273;
- SHA-256:
  7729f79a0447238289de55a9ada52390a7e0a7291ab49f714b53b14e7da90268.

## Packed-data contract

Packed uint16 token stream:

- physical tokens: 501,358,839;
- content/supervised tokens: 500,000,000;
- EOS separators: 1,358,839;
- bytes: 1,002,717,678;
- SHA-256:
  07a0faf43c241f3e232a6200bc1732381d18e9ef1b2cd77c185d2fb4c4296a15.

Block index:

- blocks: 489,609;
- input positions: 501,358,838;
- supervised positions: 500,000,000;
- EOS input positions masked from loss: 1,358,838;
- block-index SHA-256:
  dd1ebd08a69eff0f44b0714b16eb5c257feb6d3e9879489d7d5e05f4c8dcf0d9.

Deterministic block order:

- blocks: 489,609;
- dtype: uint32 little-endian;
- complete permutation: true;
- key: sha256('block|' + seed_decimal + '|' + block_index_decimal);
- seed: 1234;
- SHA-256:
  146f12fdccd0d2694f0e81e3f85f2a786cb4d72fc47b4302990d6c8696eecc6d.

Full re-hash, token scan, block structure and deterministic-order verification:
PASS.

## Loss/accounting correction

Gradient accumulation is weighted by the number of non-ignored supervised
targets in each microbatch.

This is required because EOS input positions are masked with target=-100, so
microbatches do not contain exactly equal supervised-token counts. Equal
microbatch weighting would bias the objective.

A regression compares accumulated updates against one combined-batch reference
with unequal valid-target counts.

## Data cursor/resume contract

The packed-data cursor stores:

- block-order position;
- within-block input offset;
- supervised tokens consumed.

The final microbatch of an update may stop in the middle of a block to hit
65,536 supervised tokens exactly. The next update resumes from that exact
within-block offset.

Checkpoint data-state supervised progress must equal checkpoint training
progress. A deterministic regression verifies interrupt/save/load/resume from
a mid-block cursor produces the same next data and exact same model weights as
continuous execution.

## Runtime qualification

The selected production-local profile remains:

- device: CPU;
- precision: fp32;
- threads: 12;
- physical sequence length: 1024;
- micro batch size: 1;
- global supervised tokens/update: 65,536;
- planned optimizer steps: 7,630;
- full 65,536-token updates: 7,629;
- final supervised-token update: 25,856.

A faster seq512/batch4 synthetic profile reached ~1,824 tokens/s, but it is not
selected because using 512-token contexts for the full run would weaken the
frozen Base-50M 1024-context training semantics.

Actual frozen packed-data qualification:

- supervised tokens: 65,536;
- physical input positions: 65,800;
- EOS-masked input positions: 264;
- microbatches: 65;
- update time: 46.2343 s;
- throughput: 1,417.48 supervised tokens/s;
- peak working set: ~2.52 GB;
- peak pagefile/private allocation envelope: ~3.91 GB;
- cursor after update: block-order position 64, within-block offset 264;
- production optimizer steps recorded: 0.

Pure-compute projection at the measured packed-data rate:

- ~352,740 seconds;
- ~98.0 hours;
- ~4.08 days.

This is a lower-bound projection before checkpoint, evaluation, logging and
other operational overhead.

## Runtime/config hashes at this checkpoint candidate

- execution contract:
  f9e533804cb7f10a8eb6e0aa11c3b1c1b3721401961ef5725127f885e24c63bb;
- data runtime:
  2d99c61c84fbfffb0ba5f23434bf6e8b92a92b3f588d7716238c0ebc76ee424b;
- local runtime:
  97f689652387deb034c6c21956bf44c6b7fe62e98baafdee442ee78af1ddf626;
- runtime qualification summary:
  69a3f4402dc36865594b7ac93240b57fdbba3290be9f42e1692bd74ae4a9d9af.

## Guardrails

- never reset/clean active WIP;
- never mutate or silently replace L003 frozen hashes;
- research-only lineage can never be promoted to release-safe;
- model/tokenizer fit uses train only;
- validation and holdout are not training samples;
- no raw corpus enters Git;
- no large training checkpoint enters Git;
- no public checkpoint release is implied;
- step-0 state cannot claim completed pretraining;
- checkpoints retain random initialization origin plus optimization progress;
- resume must preserve optimizer/RNG/data cursor exactly;
- checkpoints are written atomically;
- no unbounded 500M-token process before bounded tranche qualification;
- exact-commit tests/runtime evidence are required before every authorization expansion.

## Acceptance criteria for runtime/data foundation checkpoint

- [x] L004 execution contract verifies L003 freeze and tokenizer hash;
- [x] Base-50M SentencePiece runtime rejects hash/vocab mismatch;
- [x] optimizer/schedule matches training-base-50m-v1.json;
- [x] token-weighted gradient accumulation implemented and tested;
- [x] dedicated pretraining checkpoint format implemented;
- [x] checkpoint round trip restores optimizer/progress/RNG/lineage;
- [x] packed-data cursor is checkpointed and restored;
- [x] deterministic mid-block resume equivalence passes;
- [x] exact 500M train-only selection materialized;
- [x] selection manifest full verification passes;
- [x] packed uint16 stream full hash/token scan passes;
- [x] block index structural verification passes;
- [x] block-order permutation/deterministic-order verification passes;
- [x] Base-50M packed-data 8,192-token update passes;
- [x] Base-50M packed-data 65,536-token update passes;
- [x] local throughput/memory qualification completed;
- [x] full repository regression on final foundation WIP passes: 182/182;
- [x] exact foundation commit qualification passes: `c216937868d0b8391fe9ed91c4ceb8269de7e93b`;
- [x] remote branch synchronization passes: 0/0 at `c216937868d0b8391fe9ed91c4ceb8269de7e93b`.

## Qualified foundation and runner

Runtime/data foundation commit:
`c216937868d0b8391fe9ed91c4ceb8269de7e93b`.

- exact qualification: PASS;
- full pytest: 182/182 PASS;
- L003/L004 audits: PASS;
- selection/packing/block-order external verification: PASS;
- remote branch sync after push: 0/0.

Bounded-runner implementation commit:
`6b5c2946ea7b8b452e6d317de7ccd438bd23a091`.

- exact qualification: PASS;
- full pytest: 188/188 PASS;
- runner unit/authorization/state tests: PASS;
- remote branch sync after push: 0/0.

## First bounded-tranche authorization

Authorization commit:
`674354ea572532f3d8d5c0be856a43078b2e10ae`.

Authorization SHA-256:
`a8ed92f59480f7e922b8973e62be9a8e40f7a4afec5629d4008f7e3d5db9eb2f`.

The authorization pins foundation `c216937868d0b8391fe9ed91c4ceb8269de7e93b`
and runner `6b5c2946ea7b8b452e6d317de7ccd438bd23a091`, limits execution to 4
optimizer updates / 262,144 supervised tokens, requires a checkpoint every
update, preserves the research-only lane, and keeps public release/export
blocked.

Exact authorization qualification: 188/188 tests PASS, L004 audit PASS,
`--validate-only` PASS, packed-data hashes PASS, remote sync 0/0.

## First bounded-tranche result

Production progress:

- optimizer steps: 4;
- supervised tokens: 262,144;
- fraction of 500M budget: 0.0524288%;
- pretraining started: true;
- pretraining complete: false;
- public release eligibility: not qualified.

Per-step training evidence:

| Step | LR | Mean loss | Checkpoint SHA-256 |
| ---: | ---: | ---: | --- |
| 1 | 0.0000039216 | 10.474073 | `0e9ab3532c04f9b407dafc68d9c6198b4b387fdeeb5ac1f08c25fa5612fea371` |
| 2 | 0.0000078431 | 10.464879 | `c3ae2be3456436fd885f695f328954d33e4ad231e1221fa7aab261c571947b83` |
| 3 | 0.0000117647 | 10.443707 | `bf835df7eb3f9bd994b4896e8d621f6f1b71ac6660f27c4d8a47a8dbb44d579d` |
| 4 | 0.0000156863 | 10.420443 | `d4ab8703cf723600e1fa6606d28674b06638284dd1f50ea2b771d6a96f947266` |

Each checkpoint is 602,706,571 bytes and remains outside Git.

Final state SHA-256:
`8be4f6cbe5c6b5ff84095b44aea25d418a44f449c739a5d642b8fcfc377c1c78`.

Tracked result:
`results/l004/first-bounded-tranche-v1.json`
SHA-256:
`fd906c4a7170dc3fcce6a673dc920d352559a9a9760ab31219163c18995d3908`.

All four checkpoint/metric hashes were re-verified and step-4 loaded back
successfully. An attempted step 5 was rejected and the state hash remained
unchanged. The falling training loss is operational evidence only, not a
language-quality claim.

## Phase J validation gate

Validation-only materialization uses the frozen global decision manifest with
partition `validation` only. Final holdout was not read.

Deterministic validation pack:

- selection seed: 4321;
- total supervised validation tokens: 300,000;
- general English: 100,000 tokens / 597 records;
- general Thai: 100,000 tokens / 130 records;
- technical/scientific/code: 100,000 tokens / 26 records;
- selection records: 753;
- selection SHA-256:
  `90b7727239d487d1eef5bcf000865dd9df33122bc28b754ead394cc9cd99ba25`;
- raw text stored in tracked metadata: false;
- final holdout touched: false.

Validation pack hashes:

- English:
  `b1d143aa16a5ffc347a7a956b7c66124f4c772128e8ff962a0ba4c10ba5fd6f7`;
- Thai:
  `42144c77105c3c05f09e153fb2516b7ea41e9e258bf961e7e084be2d3ec5b6f6`;
- technical:
  `c62792991671ad7dbe2121fccfa533475d6db6211c47d3111cfcc4a38af00747`.

Identical validation data was evaluated on deterministic step 0 and the
qualified step-4 checkpoint:

| Category | Step 0 loss | Step 4 loss | Relative change |
| --- | ---: | ---: | ---: |
| general English | 10.483630929 | 10.371749212 | -1.067204% |
| general Thai | 10.460858983 | 10.407552466 | -0.509581% |
| technical/scientific/code | 10.486700305 | 10.381440011 | -1.003750% |
| combined | 10.477063405 | 10.386913896 | -0.860446% |

Predeclared validation gate:

- combined loss must not increase;
- no category may regress by more than 0.5% relative;
- all losses must remain finite.

Result: **PASS**. All three categories improved and combined loss improved by
0.860446%.

Tracked evidence hashes:

- validation config:
  `1909c06ae22d39848b1f9e81dcd0618203fb4c5308fcc5004a0f639770f19648`;
- validation pack summary:
  `527d2a6d2b1d08e52d1b4c5cfe6d9bbda265fc39c962bad57fe63ccf8b948458`;
- step-0 result:
  `3bedb956dedeb837ef6ab2cde87dba2ffb71989169dc152222f42de9d259378a`;
- step-4 result:
  `8d4f6bb17fdcad70f9bfa771f4a77cef058c0d350ab2e696638118806e1e20f9`;
- comparison:
  `937012b1586cbe583d9fe20463a156bc98dd0e14f4dabf3f6027d2035bf4b50f`.

## Phase K second bounded continuation

Validation gate commit:
`2151b1199346bbc26da944f731b01b84556e3ed3`.

- exact qualification: PASS;
- full pytest: 188/188 PASS;
- L004 audit: PASS;
- first tranche verification: PASS;
- validation verification: PASS;
- remote branch sync after push: 0/0.

Second-tranche runner commit:
`3143792ebd041d9675d4711b42d0c85af591e3f8`.

- continuation guardrail regression: 5/5 PASS;
- full repository regression: 193 tests PASS;
- validation gate remains PASS;
- exact runner qualification: PASS;
- remote branch sync after push: 0/0.

Second bounded authorization:
`configs/pretraining-tranche-l004-v2.json`.

Authorization commit:
`9c0df1e53c17090c1c019373397b88f77832fd84`.

- authorization id: `base50m-second-tranche-v1`;
- source optimizer step: 4;
- source checkpoint SHA-256:
  `d4ab8703cf723600e1fa6606d28674b06638284dd1f50ea2b771d6a96f947266`;
- source supervised tokens: 262,144;
- maximum additional optimizer updates: 4;
- authorized end step: 8;
- additional supervised-token cap: 262,144;
- cumulative supervised-token cap: 524,288;
- checkpoint cadence: every optimizer update;
- run root:
  `external://FlyWireLLM-data/L004/Runs/base50m-second-tranche-v1`;
- authorization SHA-256:
  `458cf90c0d0f38f407bc01d3a5e4af70706497610a3610347ce0cd75bcfff30d`;
- exact authorization qualification: PASS;
- remote sync before execution: 0/0.

The post-tranche validation gate was predeclared before step 5: evaluate the
same validation pack at step 8, require finite losses, combined loss no worse
than step 4, and no category relative regression greater than 0.5%. Final
holdout must remain untouched.

Second bounded production result:

| Step | LR | Mean loss | Checkpoint SHA-256 |
| ---: | ---: | ---: | --- |
| 5 | 0.0000196078 | 10.395490 | `ffb67c03f13d5a63dee1e6d6626265d72eaf6c801dafb209da93c0820aa83f81` |
| 6 | 0.0000235294 | 10.354821 | `88fe5f28903299d01b9ffafe73d9fbea0f7875f3e9f3a43826b9678dec49d2a9` |
| 7 | 0.0000274510 | 10.300193 | `2176f76c4ef1f4641631738d8aa9079c14dd14935d7282876b0c33c1ae75cc2e` |
| 8 | 0.0000313725 | 10.236181 | `2552f503438bf635f0e54a4fc1fc4bdcef3da57ad734c9ea7b0eebd379500e21` |

Cumulative production progress after step 8:

- optimizer steps: 8;
- supervised tokens seen: 524,288;
- fraction of 500M budget: 0.1048576%;
- final data cursor: order position 513 / block offset 400;
- state SHA-256:
  `d86788a7dfd33c13ac3f1bfdacdb648ab8f6fcadc3e0253f0f964043958eca4a`;
- tracked result SHA-256:
  `06ce67caa95a153798398cdd4784f3bb8c6c6e0194eb8ba573ebd86e79cb5a29`;
- pretraining complete: false;
- public release eligibility: not qualified.

All step5-8 checkpoint/metric hashes were re-verified and the final step8
checkpoint/cursor loaded successfully.

Post-step8 validation on the byte-identical 300k-token validation pack:

| Category | Step 4 loss | Step 8 loss | Relative change |
| --- | ---: | ---: | ---: |
| general English | 10.371749212 | 10.102311081 | -2.597808% |
| general Thai | 10.407552466 | 10.250705522 | -1.507049% |
| technical/scientific/code | 10.381440011 | 10.121698719 | -2.501977% |
| combined | 10.386913896 | 10.158238441 | -2.201573% |

Post-tranche gate: **PASS**. All categories improved, the combined loss
improved, the validation packs are byte-identical to the step4 gate, and final
holdout remains untouched.

Tracked evidence hashes:

- step8 validation:
  `f596b4df0e5cad3acf669cefcf54007d3916e427ab1cf854e873587585c39c1f`;
- step4-to-step8 comparison:
  `a1564c28fbcb0a0c90c940b69fc91a38123c418caae551759633e7165dee475f`.

## Phase L third bounded expansion authorization

Third-tranche runner commit:
`4e0ef81406da2adc2db44a0fc08b13a9ebde57b4`.

- exact runner qualification: PASS;
- full repository regression: PASS;
- L004 audit: PASS;
- first/second tranche re-verification: PASS;
- validation gate re-verification: PASS;
- remote branch sync after push: 0/0.

Authorization candidate:
`configs/pretraining-tranche-l004-v3.json`.

- authorization id: `base50m-third-tranche-v1`;
- authorization SHA-256:
  `a8be24f47fa5a98be39ebd3ba46c6a12591c2e4b5744d42a6ef5f5da8419def0`;
- source optimizer step: 8;
- source checkpoint SHA-256:
  `2552f503438bf635f0e54a4fc1fc4bdcef3da57ad734c9ea7b0eebd379500e21`;
- maximum additional optimizer updates: 8;
- authorized end step: 16;
- additional supervised-token cap: 524,288;
- cumulative supervised-token cap: 1,048,576;
- checkpoint cadence: every optimizer update;
- run root:
  `external://FlyWireLLM-data/L004/Runs/base50m-third-tranche-v1`;
- public release eligibility: not qualified;
- automatic FlyWireModel export: blocked.

The post-tranche gate is predeclared before step 9: reuse the identical 300k
validation pack at step 16, require finite losses, combined loss no worse than
step 8, no category regression greater than 0.5%, and keep final holdout
untouched.

## Third bounded-tranche result

Authorization commit:
`45523f8073d66bb60d888b7f818a93959bd0ca01`.

The authorization was exact-qualified, validate-only passed, and the branch
was remote-synchronized 0/0 before optimizer step 9.

Production result:

| Step | LR | Mean loss | Checkpoint SHA-256 |
| ---: | ---: | ---: | --- |
| 9 | 0.0000352941 | 10.176307 | `80ee1e72cc46111bcf33fd8b865d79ebe0b81bec38b7d3b2a6005be87bafddb8` |
| 10 | 0.0000392157 | 10.082878 | `9a8e0946f00f4acbb2aab70b7c13ec7bcc2e3d62938c8e8c120f87956098410e` |
| 11 | 0.0000431373 | 10.032805 | `9af4aab0d1892d9077875b664501d46b140b96bd3b90effd4047f6b4161af2a9` |
| 12 | 0.0000470588 | 9.943532 | `1a38c210ac2ef3e4b5b5c0d81e134407c1ef7d740fabfc3a9df4db6f9c9f4148` |
| 13 | 0.0000509804 | 9.895436 | `5c54e21860f0a17eca4aefc4cd10c380fa5a1778e009b168df451e87e430ded6` |
| 14 | 0.0000549020 | 9.866211 | `79426feaf51d303acabd29d39422a9327268b8e20a575aa6fc8ecaa1f4e49b5b` |
| 15 | 0.0000588235 | 9.765690 | `d2c9eb368ff7c8a3b453de102c8addbca65d32d91d23f1b44498045f1d9ee65d` |
| 16 | 0.0000627451 | 9.705748 | `07a968b8e4c8a7f6711f24cd3466c46c374e09628c7d7c65848a2166efdcd71d` |

Cumulative progress after step 16:

- optimizer steps: 16 / 7,630;
- supervised tokens: 1,048,576 / 500,000,000;
- fraction of primary budget: 0.2097152%;
- training loss step9 to step16: 10.176307 -> 9.705748;
- state SHA-256:
  `ef8c3a15b902e2d65d6fc4597b977cd732d2969931e5926ff6b914d655651107`;
- tracked third-tranche result SHA-256:
  `7a87e2adcd326260909f1ddff59d09c82589b13eabb7ccdc6c19ea10f963ddf3`;
- pretraining complete: false;
- public release eligibility: not qualified.

All step9-16 checkpoint/metric hashes were re-verified and the final step16
checkpoint/cursor loaded successfully.

Post-step16 validation on the byte-identical 300k-token validation pack:

| Category | Step 8 loss | Step 16 loss | Relative change |
| --- | ---: | ---: | ---: |
| general English | 10.102311081 | 9.584511214 | -5.125559% |
| general Thai | 10.250705522 | 9.824723932 | -4.155632% |
| technical/scientific/code | 10.121698719 | 9.606904253 | -5.086048% |
| combined | 10.158238441 | 9.672046466 | -4.786184% |

Post-tranche gate: **PASS**. Every category improved, combined validation loss
improved, the validation packs are unchanged, and final holdout remains
untouched.

Tracked evidence hashes:

- step16 validation:
  `d3ec4ab356b0b1d4fb89ee06f6b6d661257f59457672d9bf615b5fb4aa6cc6d5`;
- step8-to-step16 comparison:
  `bd079dcedeeb5c4d35c285ea4b8cdfa26b667f5e7a25d996bc44dc0d7d40ea18`.

## Phase M fourth bounded scaling authorization

Step16 evidence commit:
`73d78d18134f7f27b530eaea2eb4d9733435ad61`.

Fourth-tranche runner commit:
`074b429f640654db1bb8696971da8560f17c9066`.

- step16 evidence exact qualification: PASS;
- runner exact qualification: PASS;
- full repository regression: PASS;
- L004 audit: PASS;
- third-tranche re-verification: PASS;
- step16 validation gate re-verification: PASS;
- remote branch sync after both commits: 0/0.

Authorization candidate:
`configs/pretraining-tranche-l004-v4.json`.

- authorization id: `base50m-fourth-tranche-v1`;
- authorization SHA-256:
  `8ef449c63c6be4e2129a0604413fbfbb13f6d706a2c046d779079415981f0259`;
- source optimizer step: 16;
- source checkpoint SHA-256:
  `07a968b8e4c8a7f6711f24cd3466c46c374e09628c7d7c65848a2166efdcd71d`;
- maximum additional optimizer updates: 16;
- authorized end step: 32;
- additional supervised-token cap: 1,048,576;
- cumulative supervised-token cap: 2,097,152;
- checkpoint cadence: every optimizer update;
- run root:
  `external://FlyWireLLM-data/L004/Runs/base50m-fourth-tranche-v1`;
- public release eligibility: not qualified;
- automatic FlyWireModel export: blocked.

The post-tranche gate is predeclared before step17: reuse the identical 300k
validation pack at step32, require finite losses, combined loss no worse than
step16, no category relative regression greater than 0.5%, and keep final
holdout untouched.

## Fourth bounded-tranche result

Authorization commit:
`9dc004dc0df0d233d2504aeaf238d396cc2eed4b`.

The authorization was exact-qualified, `--validate-only` passed, and the
branch was remote-synchronized before execution. The completed external run
was then reloaded from the same authorization rather than overwritten.

Production result:

- optimizer steps: 17 through 32;
- cumulative optimizer steps: 32 / 7,630;
- additional supervised tokens: 1,048,576;
- cumulative supervised tokens: 2,097,152 / 500,000,000;
- fraction of primary budget: 0.4194304%;
- training loss step17 to step32: 9.697028 -> 9.008830;
- final checkpoint SHA-256:
  `136f554d7321be08ec266ac44296ae6870aadc7edb8d468812173ab240b8a463`;
- final state SHA-256:
  `57b868d833bb11c96ffce634744b59a5d02c86c5868085ac7f04f4d4476c3951`;
- tracked fourth-tranche result SHA-256:
  `1d0cdde6be95850dc70f691ae7b59115bfe40afa384e440a323a2f4901a01c39`;
- final data cursor: order position 2053 / block offset 970;
- pretraining complete: false;
- public release eligibility: not qualified.

All step17-32 checkpoint and metric hashes were re-verified, and the final
step32 checkpoint plus data cursor loaded successfully.

Post-step32 validation on the byte-identical 300k-token validation pack:

| Category | Step 16 loss | Step 32 loss | Relative change |
| --- | ---: | ---: | ---: |
| general English | 9.584511214 | 8.823402260 | -7.941030% |
| general Thai | 9.824723932 | 9.078124258 | -7.599192% |
| technical/scientific/code | 9.606904253 | 8.857479007 | -7.800903% |
| combined | 9.672046466 | 8.919668508 | -7.778891% |

Post-tranche gate: **PASS**. Every category improved, combined validation loss
improved, the validation packs are unchanged, and final holdout remains
untouched.

Tracked evidence hashes:

- step32 validation:
  `34b4660037261fb7779fa8ce86ac2da5b1c323d805634bcaf10f3f0fdc5c2dc8`;
- step16-to-step32 comparison:
  `7182985f612cf6b708256abc88e12f040f5af82a58db42d1f8b9fa3bdc54ab27`.

## Phase N rolling-checkpoint pilot authorization

Step32 evidence commit:
`8fa74f72c9ae025b64728b1d6717eab3a7452136`.

Rolling runner commit:
`3ff63c1cb9199b9e4e0e3449a4837ee10ffa3a07`.

Authorization:
`configs/pretraining-tranche-l004-v5.json`.

- authorization id: `base50m-fifth-tranche-v1`;
- authorization SHA-256:
  `cfff5b502da2a05e21583b8ae71e2151799aab33dc1ce1bd1eca9bb2e508c2b6`;
- source optimizer step: 32;
- source checkpoint SHA-256:
  `136f554d7321be08ec266ac44296ae6870aadc7edb8d468812173ab240b8a463`;
- maximum additional optimizer updates: 8;
- authorized end step: 40;
- additional supervised-token cap: 524,288;
- cumulative supervised-token cap: 2,621,440;
- checkpoint creation cadence: every optimizer update;
- rolling checkpoint retention: latest 1 checkpoint in this tranche;
- prior checkpoint may be pruned only after the new metric and atomic state
  commit succeed;
- run root:
  `external://FlyWireLLM-data/L004/Runs/base50m-fifth-tranche-v1`;
- post-tranche validation baseline: step32;
- candidate validation step: 40;
- same 300k validation pack required;
- final holdout must remain untouched;
- public release eligibility: not qualified.

This pilot exists to prove bounded training remains restart-safe while avoiding
unbounded checkpoint storage growth.

## Fifth rolling-checkpoint pilot result

Authorization commit:
`8f5882861a52a57cbcbc10aafb1768373b22aed3`.

The authorization was exact-qualified, `--validate-only` passed, and the
branch was remote-synchronized before optimizer step33.

Production result:

- optimizer steps: 33 through 40;
- cumulative optimizer steps: 40 / 7,630;
- additional supervised tokens: 524,288;
- cumulative supervised tokens: 2,621,440 / 500,000,000;
- fraction of primary budget: 0.524288%;
- training loss step33 to step40: 8.937551 -> 8.590367;
- final checkpoint SHA-256:
  `9f77df859cf1d41546d9bb919aac621e970375c175c6e9e9a93a4b200cedf95c`;
- final state SHA-256:
  `a93db123f8696d678c8c2a835ad381183018f88e30adc0612e28371333fee105`;
- tracked fifth-tranche result SHA-256:
  `995f97250b2fd06160cd4d6f3f0c31717cd181bd5bf20858d1f9c87c41b66b2a`;
- final data cursor: order position 2567 / block offset 870;
- pretraining complete: false;
- public release eligibility: not qualified.

Rolling retention result:

- checkpoint creation cadence: every optimizer update;
- retained full checkpoints in fifth-tranche run root: 1;
- historical checkpoints pruned after atomic state commit: 7;
- retained checkpoint: step40;
- all step33-40 metrics retained and hash-verified;
- final step40 checkpoint loaded successfully with matching optimizer/data
  progress.

Post-step40 validation on the byte-identical 300k-token validation pack:

| Category | Step 32 loss | Step 40 loss | Relative change |
| --- | ---: | ---: | ---: |
| general English | 8.823402260 | 8.340751974 | -5.470115% |
| general Thai | 9.078124258 | 8.614949760 | -5.102095% |
| technical/scientific/code | 8.857479007 | 8.388065623 | -5.299627% |
| combined | 8.919668508 | 8.447922452 | -5.288829% |

Post-tranche gate: **PASS**. Every category improved, combined validation loss
improved, the validation packs are unchanged, and final holdout remains
untouched.

Tracked evidence hashes:

- step40 validation:
  `c2dd89b43fd8e77b0dbc23f1fcc37f71f2bc5078209c5669ca98b20489e324b4`;
- step32-to-step40 comparison:
  `22b5ff56488599951ce6e4bb1decff6d2b851c6853234091e2ffff1248255a24`.

## Phase O sixth bounded rolling expansion authorization

Step40 evidence commit:
`30e75e05b926224033033231a00de9ee3c6643bf`.

Rolling expansion runner commit:
`ff71939f7547f9d6ac8f2f510d6a604b350c525e`.

Authorization:
`configs/pretraining-tranche-l004-v6.json`.

- authorization id: `base50m-sixth-tranche-v1`;
- authorization SHA-256:
  `b14f53cde35373e127c5405bca7b50744c9cea6366aa03f51d060dbc7af30f6f`;
- source optimizer step: 40;
- source checkpoint SHA-256:
  `9f77df859cf1d41546d9bb919aac621e970375c175c6e9e9a93a4b200cedf95c`;
- maximum additional optimizer updates: 16;
- authorized end step: 56;
- additional supervised-token cap: 1,048,576;
- cumulative supervised-token cap: 3,670,016;
- checkpoint creation cadence: every optimizer update;
- rolling checkpoint retention: latest 1 checkpoint in this tranche;
- prune prior checkpoint only after new metric + atomic state commit;
- post-tranche validation baseline: step40;
- candidate validation step: 56;
- same 300k validation pack required;
- final holdout must remain untouched;
- public release eligibility: not qualified.

## Sixth bounded rolling expansion result

Authorization commit:
`3bf4dc264f83ab64f50d8652a179e794e3d691c9`.

Production result:

- optimizer steps: 41 through 56;
- cumulative optimizer steps: 56 / 7,630;
- additional supervised tokens: 1,048,576;
- cumulative supervised tokens: 3,670,016 / 500,000,000;
- fraction of primary budget: 0.7340032%;
- training loss step41 to step56: 8.486466 -> 7.698671;
- final checkpoint SHA-256:
  `4095c686eeff521c906a73ef34d890135cd5c4900addab54b7477f6f72a4d116`;
- final state SHA-256:
  `52ebd96c161855febc8bf1cf45c8b0b81780b1be264a8bfa37634f045cecf86e`;
- tracked sixth-tranche result SHA-256:
  `ddab037da4623f706ae5d162cb2dd088c045b3972467c4497be2f68c87cb253f`;
- final data cursor: order position 3594 / block offset 657;
- pretraining complete: false;
- public release eligibility: not qualified.

Rolling retention:

- full checkpoints retained in sixth-tranche run root: 1;
- historical checkpoints pruned after atomic state commit: 15;
- metrics retained and hash-verified: 16 / 16;
- final checkpoint reload: PASS.

Post-step56 validation on the byte-identical 300k-token validation pack:

| Category | Step 40 loss | Step 56 loss | Relative change |
| --- | ---: | ---: | ---: |
| general English | 8.340751974 | 7.532882587 | -9.685810% |
| general Thai | 8.614949760 | 7.848998205 | -8.890958% |
| technical/scientific/code | 8.388065623 | 7.615953266 | -9.204892% |
| combined | 8.447922452 | 7.665944686 | -9.256451% |

Post-tranche gate: **PASS**. Every category improved, combined validation loss
improved, the validation packs are unchanged, and final holdout remains
untouched.

Tracked evidence hashes:

- step56 validation:
  `c7e663ec5e66bff5473ce1a3282dd910ea546fe7f90bc78c40083238696d0380`;
- step40-to-step56 comparison:
  `3941421b63283cdfd530c2c2be7ced5421330338192bea68c00b0db2bba14dad`.

## Phase P seventh bounded rolling scaling authorization

Step56 evidence commit:
`5c1dde7596c0a697c803096653e62fb88b4a6580`.

Rolling scaling runner commit:
`835d914ea91d2c22c802efab15f07f65a7e84e71`.

Authorization:
`configs/pretraining-tranche-l004-v7.json`.

- authorization id: `base50m-seventh-tranche-v1`;
- authorization SHA-256:
  `391250f1514f66ab5ad680cce09c7fcbe657afd32779884a96542f84d32a0620`;
- source optimizer step: 56;
- source checkpoint SHA-256:
  `4095c686eeff521c906a73ef34d890135cd5c4900addab54b7477f6f72a4d116`;
- maximum additional optimizer updates: 32;
- authorized end step: 88;
- additional supervised-token cap: 2,097,152;
- cumulative supervised-token cap: 5,767,168;
- checkpoint creation cadence: every optimizer update;
- rolling checkpoint retention: latest 1 checkpoint in this tranche;
- prune prior checkpoint only after new metric + atomic state commit;
- post-tranche validation baseline: step56;
- candidate validation step: 88;
- same 300k validation pack required;
- final holdout must remain untouched;
- public release eligibility: not qualified.

## Seventh bounded rolling scaling result

Authorization commit:
`668c8effbae0cd31d98c472159f0ec8df708b4f5`.

Production result:

- optimizer steps: 57 through 88;
- cumulative optimizer steps: 88 / 7,630;
- additional supervised tokens: 2,097,152;
- cumulative supervised tokens: 5,767,168 / 500,000,000;
- fraction of primary budget: 1.1534336%;
- training loss step57 to step88: 7.753659 -> 7.225092;
- final checkpoint SHA-256:
  `e5c6427334044b300c98c964499a14cb9126db87a2996b9ca5505f417e6bd27a`;
- final state SHA-256:
  `263f0315f019fe4ec5cbb3c8e7f923427aa0c0e0d1ba2088df4c83840b274d9a`;
- tracked seventh-tranche result SHA-256:
  `d6961d541c556b4af57dba481c49586639040af514452a0303e3e26bb8c37fc9`;
- final data cursor: order position 5649 / block offset 232;
- pretraining complete: false;
- public release eligibility: not qualified.

Rolling retention:

- full checkpoints retained in seventh-tranche run root: 1;
- historical checkpoints pruned after atomic state commit: 31;
- metrics retained and hash-verified: 32 / 32;
- final checkpoint reload: PASS.

Post-step88 validation on the byte-identical 300k-token validation pack:

| Category | Step 56 loss | Step 88 loss | Relative change |
| --- | ---: | ---: | ---: |
| general English | 7.532882587 | 6.991381387 | -7.188499% |
| general Thai | 7.848998205 | 7.424897029 | -5.403252% |
| technical/scientific/code | 7.615953266 | 7.127175571 | -6.417814% |
| combined | 7.665944686 | 7.181151329 | -6.323987% |

Post-tranche gate: **PASS**. Every category improved, combined validation loss
improved, the validation packs are unchanged, and final holdout remains
untouched.

Tracked evidence hashes:

- step88 validation:
  `cab6d93ffef975042ee258e5400dd4974aa881a90085d6579d146defcb34fc18`;
- step56-to-step88 comparison:
  `96831b5949c5f15a97108b78de1cef86b78aa8ba091af998dd7a9ca7832c1313`.

## Phase R warmup-aware eighth tranche authorization

Step88 evidence commit:
`61f969000a330d686ebbabb0d6387c6e4a283fa9`.

Warmup-guard runner commit:
`79e3bd76a5088e6d9b60e3ffa647674edb78dbe6`.

Authorization:
`configs/pretraining-tranche-l004-v8.json`.

- authorization id: `base50m-eighth-tranche-v1`;
- authorization SHA-256:
  `ffc8bfccec3bd0ec655d2899b3ac20417fd496fd39d2de906fc4dc18c5333367`;
- source optimizer step: 88;
- source checkpoint SHA-256:
  `e5c6427334044b300c98c964499a14cb9126db87a2996b9ca5505f417e6bd27a`;
- maximum additional optimizer updates: 16;
- authorized end step: 104;
- additional supervised-token cap: 1,048,576;
- cumulative supervised-token cap: 6,815,744;
- checkpoint creation cadence: every optimizer update;
- rolling checkpoint retention: latest 1 checkpoint in this tranche;
- prune prior checkpoint only after new metric + atomic state commit;
- post-tranche validation baseline: step88;
- candidate validation step: 104;
- same 300k validation pack required;
- final holdout must remain untouched;
- public release eligibility: not qualified.

Rationale: the optimizer is still in the declared 153-step warmup. The
validation cadence is intentionally tightened to 16 updates while learning
rate continues to rise.

## Eighth warmup-aware tranche result

Authorization commit:
`0899b22ec1c403bd9b21edc66e8d3d14a0b0c536`.

Production result:

- optimizer steps: 89 through 104;
- cumulative optimizer steps: 104 / 7,630;
- additional supervised tokens: 1,048,576;
- cumulative supervised tokens: 6,815,744 / 500,000,000;
- fraction of primary budget: 1.3631488%;
- training loss step89 to step104: 7.243646 -> 6.987964;
- final checkpoint SHA-256:
  `0f61a87d497543af516f3818d2d47ce21412fec009ac7a08492ee4396952726b`;
- final state SHA-256:
  `e2a98f2466436c163cc519082ee7e718c1fe55ec1621296337392a5d720cdd89`;
- tracked eighth-tranche result SHA-256:
  `caecd23b8b79aba1f6e2e76e1d81152387ff3186be0fe5ba37d0a268dd2ca6e6`;
- final data cursor: order position 6675 / block offset 745;
- pretraining complete: false;
- public release eligibility: not qualified.

Rolling retention:

- full checkpoints retained in eighth-tranche run root: 1;
- historical checkpoints pruned after atomic state commit: 15;
- metrics retained and hash-verified: 16 / 16;
- final checkpoint reload: PASS.

Post-step104 validation on the byte-identical 300k-token validation pack:

| Category | Step 88 loss | Step 104 loss | Relative change |
| --- | ---: | ---: | ---: |
| general English | 6.991381387 | 6.776078678 | -3.079545% |
| general Thai | 7.424897029 | 7.331277076 | -1.260892% |
| technical/scientific/code | 7.127175571 | 6.907709166 | -3.079290% |
| combined | 7.181151329 | 7.005021640 | -2.452666% |

Post-tranche gate: **PASS**. Every category improved, combined validation loss
improved, the validation packs are unchanged, and final holdout remains
untouched.

Tracked evidence hashes:

- step104 validation:
  `4652344185a436f3185626859773b78efe49ddf517c373538c87694df935aa60`;
- step88-to-step104 comparison:
  `7f03b35b296f2ee23863ad52ccf3f5b480c1be5fed725886a3dd61421e22a109`.

## Phase S ninth warmup-aware tranche authorization

Step104 evidence commit:
`7997a1747cd822b2e36e44a8c6069bab22efe8e7`.

Warmup follow-up runner commit:
`7af23fc1f62a69cf08b03cb1bf16ed4720915ba1`.

Authorization:
`configs/pretraining-tranche-l004-v9.json`.

- authorization id: `base50m-ninth-tranche-v1`;
- authorization SHA-256:
  `90bd19426caa899732f71fcd0ee6511f89e038f6d0cb678ee3b8b54f4ae1980b`;
- source optimizer step: 104;
- source checkpoint SHA-256:
  `0f61a87d497543af516f3818d2d47ce21412fec009ac7a08492ee4396952726b`;
- maximum additional optimizer updates: 16;
- authorized end step: 120;
- additional supervised-token cap: 1,048,576;
- cumulative supervised-token cap: 7,864,320;
- checkpoint creation cadence: every optimizer update;
- rolling checkpoint retention: latest 1 checkpoint in this tranche;
- prune prior checkpoint only after new metric + atomic state commit;
- post-tranche validation baseline: step104;
- candidate validation step: 120;
- same 300k validation pack required;
- final holdout must remain untouched;
- public release eligibility: not qualified.

Rationale: optimizer warmup continues through step153, so the 16-update
validation cadence remains in force.

## Ninth warmup-aware tranche result

Authorization commit:
`3d3a7c9161d80f00f992908a2f0e219e47ea2b56`.

Production result:

- optimizer steps: 105 through 120;
- cumulative optimizer steps: 120 / 7,630;
- additional supervised tokens: 1,048,576;
- cumulative supervised tokens: 7,864,320 / 500,000,000;
- fraction of primary budget: 1.572864%;
- training loss step105 to step120: 7.021231 -> 6.854975;
- final checkpoint SHA-256:
  `f8b6f447124c9f08250da126fcecca827ffbbb8b23d93338934a01031ea5086d`;
- final state SHA-256:
  `25cf31506c855ef85aed5f4e0d3de9563a2c7206f9e368767ecea71fc4e4bb7b`;
- tracked ninth-tranche result SHA-256:
  `fd25888ce3b055de25737a2c0a6a6479f75ad3f01191b6bbfd8a359a02c3bbf1`;
- final data cursor: order position 7702 / block offset 337;
- pretraining complete: false;
- public release eligibility: not qualified.

Rolling retention:

- full checkpoints retained in ninth-tranche run root: 1;
- historical checkpoints pruned after atomic state commit: 15;
- metrics retained and hash-verified: 16 / 16;
- final checkpoint reload: PASS.

Post-step120 validation on the byte-identical 300k-token validation pack:

| Category | Step 104 loss | Step 120 loss | Relative change |
| --- | ---: | ---: | ---: |
| general English | 6.776078678 | 6.567051706 | -3.084778% |
| general Thai | 7.331277076 | 7.236243482 | -1.296276% |
| technical/scientific/code | 6.907709166 | 6.737619036 | -2.462323% |
| combined | 7.005021640 | 6.846971408 | -2.256242% |

Post-tranche gate: **PASS**. Every category improved, combined validation loss
improved, the validation packs are unchanged, and final holdout remains
untouched.

Tracked evidence hashes:

- step120 validation:
  `854aa31870e7af6c7454a3a7608130c67623f071d60980d3208bc29474896c43`;
- step104-to-step120 comparison:
  `6d7a984b80fe5aeec65c9569d26f16192606f65dce55f9274e0ba54fddb761f3`.

## Phase T tenth warmup-aware tranche authorization

Step120 evidence commit:
`9f951c0c33b00006e7ffb376626acfcf4c60b30d`.

Warmup penultimate runner commit:
`44586b6f643e152776129297be474e487111d694`.

Authorization:
`configs/pretraining-tranche-l004-v10.json`.

- authorization id: `base50m-tenth-tranche-v1`;
- authorization SHA-256:
  `1fc3a96cf3d64e86d3bfd533f8ab5e7509b9dd69d0bb7773d1cdd9b95a75182a`;
- source optimizer step: 120;
- source checkpoint SHA-256:
  `f8b6f447124c9f08250da126fcecca827ffbbb8b23d93338934a01031ea5086d`;
- maximum additional optimizer updates: 16;
- authorized end step: 136;
- additional supervised-token cap: 1,048,576;
- cumulative supervised-token cap: 8,912,896;
- checkpoint creation cadence: every optimizer update;
- rolling checkpoint retention: latest 1 checkpoint in this tranche;
- prune prior checkpoint only after new metric + atomic state commit;
- post-tranche validation baseline: step120;
- candidate validation step: 136;
- same 300k validation pack required;
- final holdout must remain untouched;
- public release eligibility: not qualified.

Rationale: warmup remains active through step153. Step136 leaves 17 updates to
the exact warmup boundary, enabling one final warmup-boundary tranche.

## Tenth warmup-aware tranche result

Authorization commit:
`d5840a7aa9c6764605a682dc5c62b5ab571de642`.

Production result:

- optimizer steps: 121 through 136;
- cumulative optimizer steps: 136 / 7,630;
- additional supervised tokens: 1,048,576;
- cumulative supervised tokens: 8,912,896 / 500,000,000;
- fraction of primary budget: 1.7825792%;
- training loss step121 to step136: 6.953884 -> 6.756546;
- final checkpoint SHA-256:
  `8ee3cb135acd1058321c3838919dcd6e61b0fa98d37edd8cd2b32fb17d08f3a1`;
- final state SHA-256:
  `3c2d16cd47bbff977e52ec7b5a022dcbc5125eb0646227d14e4121c3fa85a5cb`;
- tracked tenth-tranche result SHA-256:
  `fc762526743473932b628ea50456eb668dc6ed9592b7baaff67edb331c4d2b22`;
- final data cursor: order position 8728 / block offset 759;
- pretraining complete: false;
- public release eligibility: not qualified.

Rolling retention:

- full checkpoints retained in tenth-tranche run root: 1;
- historical checkpoints pruned after atomic state commit: 15;
- metrics retained and hash-verified: 16 / 16;
- final checkpoint reload: PASS.

Post-step136 validation on the byte-identical 300k-token validation pack:

| Category | Step 120 loss | Step 136 loss | Relative change |
| --- | ---: | ---: | ---: |
| general English | 6.567051706 | 6.374071930 | -2.938606% |
| general Thai | 7.236243482 | 7.094012272 | -1.965539% |
| technical/scientific/code | 6.737619036 | 6.578042986 | -2.368434% |
| combined | 6.846971408 | 6.682042396 | -2.408788% |

Post-tranche gate: **PASS**. Every category improved, combined validation loss
improved, the validation packs are unchanged, and final holdout remains
untouched.

Tracked evidence hashes:

- step136 validation:
  `a0710642d686aa6cbed38d513eabfc540312f8e8c7fc242a98d65e9e5d90604b`;
- step120-to-step136 comparison:
  `2790a9f846a3c11349c4b9e0bc4a7d8981654c9e36ca9ed6ec6d50404ba0d64b`.

## Phase U exact warmup-boundary authorization

Step136 evidence commit:
`ead9cbe456a1ac5ce0e92f0d6a8281d93aafe9d0`.

Warmup-boundary runner commit:
`ee8c36a082cab5e07d0346ff530902e4b6a4136e`.

Authorization:
`configs/pretraining-tranche-l004-v11.json`.

- authorization id: `base50m-warmup-boundary-v1`;
- authorization SHA-256:
  `b8482919369a8cc402f8addaa7c1a122d59a1fd28ce52e9dfe9b78fb78eb1f3b`;
- source optimizer step: 136;
- source checkpoint SHA-256:
  `8ee3cb135acd1058321c3838919dcd6e61b0fa98d37edd8cd2b32fb17d08f3a1`;
- maximum additional optimizer updates: 17;
- authorized end step: 153;
- additional supervised-token cap: 1,114,112;
- cumulative supervised-token cap: 10,027,008;
- checkpoint creation cadence: every optimizer update;
- rolling checkpoint retention: latest 1 checkpoint in this tranche;
- prune prior checkpoint only after new metric + atomic state commit;
- post-tranche validation baseline: step136;
- candidate validation step: 153;
- same 300k validation pack required;
- final holdout must remain untouched;
- public release eligibility: not qualified.

This tranche ends exactly at the declared optimizer warmup boundary. It must
not execute optimizer step154.

## Warmup-boundary production result

Authorization commit:
`655297bbe7154b26e1fca0cdf154bbfcbda962d5`.

Production result:

- optimizer steps: 137 through 153;
- cumulative optimizer steps: 153 / 7,630;
- additional supervised tokens: 1,114,112;
- cumulative supervised tokens: 10,027,008 / 500,000,000;
- fraction of primary budget: 2.0054016%;
- training loss step137 to step153: 6.664082 -> 6.429739;
- peak learning rate reached at step153: 0.0006000000;
- final checkpoint SHA-256:
  `f00d36e288094aab5bc0bec52a62dd134e92505ee6a87f3797e8e70d05cc3700`;
- final state SHA-256:
  `760052bfc230ddcf6b7ba3bbc70255d5090060f5e8982241287d273eda9c3f02`;
- tracked warmup-boundary result SHA-256:
  `49c7586d901964812a71b67caebbe324e735778fb748dc574146b02cefac1950`;
- final data cursor: order position 9819 / block offset 6;
- optimizer warmup complete: true;
- pretraining complete: false;
- public release eligibility: not qualified.

Rolling retention:

- full checkpoints retained in warmup-boundary run root: 1;
- historical checkpoints pruned after atomic state commit: 16;
- metrics retained and hash-verified: 17 / 17;
- final checkpoint reload: PASS.

Post-step153 validation on the byte-identical 300k-token validation pack:

| Category | Step 136 loss | Step 153 loss | Relative change |
| --- | ---: | ---: | ---: |
| general English | 6.374071930 | 6.210462363 | -2.566798% |
| general Thai | 7.094012272 | 6.872962925 | -3.115999% |
| technical/scientific/code | 6.578042986 | 6.412851243 | -2.511260% |
| combined | 6.682042396 | 6.498758844 | -2.742927% |

Post-tranche gate: **PASS**. Every category improved, combined validation loss
improved, the validation packs are unchanged, and final holdout remains
untouched.

Tracked evidence hashes:

- step153 validation:
  `b8d53069dac09fdcf899d92cff71636b51ec14b18ccb1ecd7b9339df59e6c13b`;
- step136-to-step153 comparison:
  `1e484d263091c6d185e82e95315b4ee53788f5328b12b6540a371d6a4f36da42`.

## Phase V first post-warmup cosine-decay authorization

Step153 evidence commit:
`988e10fd0c419dee91abb00224f0d96e0e02e1d1`.

Post-warmup runner commit:
`ed93936c426a3b6212659ddf7e30a835ea8b86bd`.

Authorization:
`configs/pretraining-tranche-l004-v12.json`.

- authorization id: `base50m-post-warmup-1-v1`;
- authorization SHA-256:
  `d8e1ffe91ab4af0aed58fc5635674e9b87923b2cc96347344283706743d94846`;
- source optimizer step: 153;
- source checkpoint SHA-256:
  `f00d36e288094aab5bc0bec52a62dd134e92505ee6a87f3797e8e70d05cc3700`;
- maximum additional optimizer updates: 16;
- authorized end step: 169;
- additional supervised-token cap: 1,048,576;
- cumulative supervised-token cap: 11,075,584;
- schedule: existing qualified cosine decay, no new LR policy;
- learning rate step154: 0.0006000000;
- learning rate step169: 0.0005999939;
- checkpoint creation cadence: every optimizer update;
- rolling checkpoint retention: latest 1 checkpoint in this tranche;
- prune prior checkpoint only after new metric + atomic state commit;
- post-tranche validation baseline: step153;
- candidate validation step: 169;
- same 300k validation pack required;
- final holdout must remain untouched;
- public release eligibility: not qualified.

## First post-warmup cosine-decay tranche result

Authorization commit:
`f11a412d04e7c50311230297f1f29b6116e8c3c7`.

Production result:

- optimizer steps: 154 through 169;
- cumulative optimizer steps: 169 / 7,630;
- additional supervised tokens: 1,048,576;
- cumulative supervised tokens: 11,075,584 / 500,000,000;
- fraction of primary budget: 2.2151168%;
- training loss step154 to step169: 6.552233 -> 6.391935;
- learning rate step154 to step169: 0.0006000000 -> 0.0005999939;
- final checkpoint SHA-256:
  `ccce01d29686a76edfe52fbaa531cf198a82bfd8aa17ded34625b8c6f14b979e`;
- final state SHA-256:
  `175ebccef8707fda0af0e65cf0646ac12f72b6464ab54c20ac6fa81be25124e6`;
- tracked post-warmup result SHA-256:
  `7069138a1910643eb577008076fdbafab2c6cfe52b74861539074d788df6a563`;
- final data cursor: order position 10846 / block offset 3;
- optimizer warmup complete: true;
- pretraining complete: false;
- public release eligibility: not qualified.

Rolling retention:

- full checkpoints retained in post-warmup run root: 1;
- historical checkpoints pruned after atomic state commit: 15;
- metrics retained and hash-verified: 16 / 16;
- final checkpoint reload: PASS.

Post-step169 validation on the byte-identical 300k-token validation pack:

| Category | Step 153 loss | Step 169 loss | Relative change |
| --- | ---: | ---: | ---: |
| general English | 6.210462363 | 6.117310139 | -1.499924% |
| general Thai | 6.872962925 | 6.745732500 | -1.851173% |
| technical/scientific/code | 6.412851243 | 6.337900238 | -1.168763% |
| combined | 6.498758844 | 6.400314292 | -1.514821% |

Post-tranche gate: **PASS**. Every category improved, combined validation loss
improved, the validation packs are unchanged, and final holdout remains
untouched.

Tracked evidence hashes:

- step169 validation:
  `eb6489be84dbf2ff06d7ff46905a2a6c9f84bb303c1053552c174e443881d962`;
- step153-to-step169 comparison:
  `0f01fef42571bb756cad34ed00b9d3eb36ca7240788dd70443c51baa6b1b631f`.

## Phase W second post-warmup cosine-decay scaling authorization

Step169 evidence commit:
`068a2a1eaec7affa84d719d4c1d847fe90ef5a88`.

Post-warmup scaling runner commit:
`d30273244407936a93f4bf7bb341828d2a357cbb`.

Authorization:
`configs/pretraining-tranche-l004-v13.json`.

- authorization id: `base50m-post-warmup-2-v1`;
- authorization SHA-256:
  `b72536d3968a89043b3d2f453ccd5b2258da9e2f1e0ec956733fc1f536857f02`;
- source optimizer step: 169;
- source checkpoint SHA-256:
  `ccce01d29686a76edfe52fbaa531cf198a82bfd8aa17ded34625b8c6f14b979e`;
- maximum additional optimizer updates: 32;
- authorized end step: 201;
- additional supervised-token cap: 2,097,152;
- cumulative supervised-token cap: 13,172,736;
- schedule: existing qualified cosine decay, no new LR policy;
- learning rate step170: 0.0005999931;
- learning rate step201: 0.0005999451;
- checkpoint creation cadence: every optimizer update;
- rolling checkpoint retention: latest 1 checkpoint in this tranche;
- prune prior checkpoint only after new metric + atomic state commit;
- post-tranche validation baseline: step169;
- candidate validation step: 201;
- same 300k validation pack required;
- final holdout must remain untouched;
- public release eligibility: not qualified.

This doubles the prior post-warmup tranche length from 16 to 32 updates while
remaining bounded and keeping the same validation and retention contracts.

## Second post-warmup cosine-decay scaling result

Authorization commit:
`4fe9324797f438f299bb4230cc11bca4dea72cee`.

Production result:

- optimizer steps: 170 through 201;
- cumulative optimizer steps: 201 / 7,630;
- additional supervised tokens: 2,097,152;
- cumulative supervised tokens: 13,172,736 / 500,000,000;
- fraction of primary budget: 2.6345472%;
- training loss step170 to step201: 6.444974 -> 6.215267;
- learning rate step170 to step201: 0.0005999931 -> 0.0005999451;
- final checkpoint SHA-256:
  `11fe2a45b65d43bec6a6710d64c2110a1be4cc11042e1fcbf1b6197a9feaa2a0`;
- final state SHA-256:
  `267f3e3bdcbb01743cd92d800518fda502d9b9642ae00186638e9b5ecd9183ac`;
- tracked post-warmup-2 result SHA-256:
  `75ac56438958999c683518dbc04f06c98672d649fc3a9074328d2aac6dd39dfd`;
- final data cursor: order position 12898 / block offset 493;
- optimizer warmup complete: true;
- pretraining complete: false;
- public release eligibility: not qualified.

Rolling retention:

- full checkpoints retained in post-warmup-2 run root: 1;
- historical checkpoints pruned after atomic state commit: 31;
- metrics retained and hash-verified: 32 / 32;
- final checkpoint reload: PASS.

Post-step201 validation on the byte-identical 300k-token validation pack:

| Category | Step 169 loss | Step 201 loss | Relative change |
| --- | ---: | ---: | ---: |
| general English | 6.117310139 | 5.931827876 | -3.032089% |
| general Thai | 6.745732500 | 6.452203094 | -4.351335% |
| technical/scientific/code | 6.337900238 | 6.140651583 | -3.112208% |
| combined | 6.400314292 | 6.174894184 | -3.522016% |

Post-tranche gate: **PASS**. Every category improved, combined validation loss
improved, the validation packs are unchanged, and final holdout remains
untouched.

Tracked evidence hashes:

- step201 validation:
  `1b84b6ff741701ada4a1980030a668ed217bf1fff368aa6c86adff0fb0ad3b2d`;
- step169-to-step201 comparison:
  `a791e099ab10ba971d6313bfa1b79578746c0f5223076bbcda56e8a2ede97385`.

## Phase X third post-warmup cosine-decay authorization

Step201 evidence commit:
`789edfeff672a0bfe56fdf909742746b46ab3bf6`.

Post-warmup steady runner commit:
`585aad92e6140725b8ccf7e75e0f107c974a378a`.

Authorization:
`configs/pretraining-tranche-l004-v14.json`.

- authorization id: `base50m-post-warmup-3-v1`;
- authorization SHA-256:
  `0e776c62ab1fde305e9524a422a512ffe895c50f438c4901e9b83a857fb04680`;
- source optimizer step: 201;
- source checkpoint SHA-256:
  `11fe2a45b65d43bec6a6710d64c2110a1be4cc11042e1fcbf1b6197a9feaa2a0`;
- maximum additional optimizer updates: 32;
- authorized end step: 233;
- additional supervised-token cap: 2,097,152;
- cumulative supervised-token cap: 15,269,888;
- schedule: existing qualified cosine decay, no new LR policy;
- learning rate step202: 0.0005999428;
- learning rate step233: 0.0005998475;
- checkpoint creation cadence: every optimizer update;
- rolling checkpoint retention: latest 1 checkpoint in this tranche;
- prune prior checkpoint only after new metric + atomic state commit;
- post-tranche validation baseline: step201;
- candidate validation step: 233;
- same 300k validation pack required;
- final holdout must remain untouched;
- public release eligibility: not qualified.

## Third post-warmup cosine-decay result

Authorization commit:
`7be038d29946d34ddbd845ddd797b6b070778c8b`.

Production result:

- optimizer steps: 202 through 233;
- cumulative optimizer steps: 233 / 7,630;
- additional supervised tokens: 2,097,152;
- cumulative supervised tokens: 15,269,888 / 500,000,000;
- fraction of primary budget: 3.0539776%;
- training loss step202 to step233: 6.117244 -> 5.872188;
- learning rate step202 to step233: 0.0005999428 -> 0.0005998475;
- final checkpoint SHA-256:
  `1513602d920fe57b7bb12ee7ee69cdf77ea23f39f0253cdabb718e55413dd964`;
- final state SHA-256:
  `93f5256d1fa82d29dcfb95c997518afd46901101475af4c553816e42c2edf7ed`;
- tracked post-warmup-3 result SHA-256:
  `e26e181ec16be56a23155544fa5f907e64bc5fd85ea47a39abacc45ef691b12e`;
- final data cursor: order position 14951 / block offset 843;
- optimizer warmup complete: true;
- pretraining complete: false;
- public release eligibility: not qualified.

Rolling retention:

- full checkpoints retained in post-warmup-3 run root: 1;
- historical checkpoints pruned after atomic state commit: 31;
- metrics retained and hash-verified: 32 / 32;
- final checkpoint reload: PASS.

Post-step233 validation on the byte-identical 300k-token validation pack:

| Category | Step 201 loss | Step 233 loss | Relative change |
| --- | ---: | ---: | ---: |
| general English | 5.931827876 | 5.791585690 | -2.364232% |
| general Thai | 6.452203094 | 6.260309313 | -2.974082% |
| technical/scientific/code | 6.140651583 | 6.005837822 | -2.195431% |
| combined | 6.174894184 | 6.019244275 | -2.520689% |

Post-tranche gate: **PASS**. Every category improved, combined validation loss
improved, the validation packs are unchanged, and final holdout remains
untouched.

Tracked evidence hashes:

- step233 validation:
  `89ab8cfc6bc0ed2f3923f53407ba9d432d5a76ac0086497a17545d79d6913561`;
- step201-to-step233 comparison:
  `cd9c4f78e74200de9af3731b9cbeb4f3fecb07b5b9c2118c64348837bba687de`.

## Phase Y fourth post-warmup cosine-decay expansion authorization

Step233 evidence commit:
`2c4650b574cd9a86989313b0d43ecdee3f0f52c3`.

Expanded post-warmup runner commit:
`db93205e6a105b2697449f6517ffa2f8617ea7ae`.

Authorization:
`configs/pretraining-tranche-l004-v15.json`.

- authorization id: `base50m-post-warmup-4-v1`;
- authorization SHA-256:
  `84b8fdd089c1211e78d6359589886d4ed83b38b328d5222d5f418cae4efbd7a9`;
- source optimizer step: 233;
- source checkpoint SHA-256:
  `1513602d920fe57b7bb12ee7ee69cdf77ea23f39f0253cdabb718e55413dd964`;
- maximum additional optimizer updates: 64;
- authorized end step: 297;
- additional supervised-token cap: 4,194,304;
- cumulative supervised-token cap: 19,464,192;
- schedule: existing qualified cosine decay, no new LR policy;
- learning rate step234: 0.000599843647;
- learning rate step297: 0.000599505950;
- checkpoint creation cadence: every optimizer update;
- rolling checkpoint retention: latest 1 checkpoint in this tranche;
- prune prior checkpoint only after new metric + atomic state commit;
- post-tranche validation baseline: step233;
- candidate validation step: 297;
- same 300k validation pack required;
- final holdout must remain untouched;
- public release eligibility: not qualified.

This doubles the bounded post-warmup tranche from 32 to 64 updates after two
consecutive 32-update PASS gates, while preserving the exact same optimizer,
retention, and validation contracts.

## Fourth post-warmup cosine-decay result

Authorization commit:
`527792cdf3eb59ec5ad4624a7b6dc10f32afa6e9`.

Production result:

- optimizer steps: 234 through 297;
- cumulative optimizer steps: 297 / 7,630;
- additional supervised tokens: 4,194,304;
- cumulative supervised tokens: 19,464,192 / 500,000,000;
- fraction of primary budget: 3.8928384%;
- training loss step234 to step297: 6.049393 -> 5.593243;
- learning rate step234 to step297: 0.000599843647 -> 0.000599505950;
- final checkpoint SHA-256:
  `f0917d81130d3fb8ff6c59d28cfd0b662633b68ae96b36c5d75caceb179870fc`;
- final state SHA-256:
  `bcb97673f3ce6a02f6149e03ee8c7ddecdc8ab19a647e9ec8c41c9f9fb1e28bf`;
- tracked post-warmup-4 result SHA-256:
  `83722298982d9deaa929a7fd89baf527d589f8b16a188b637198723100792450`;
- final data cursor: order position 19059 / block offset 578;
- optimizer warmup complete: true;
- pretraining complete: false;
- public release eligibility: not qualified.

Rolling retention:

- full checkpoints retained in post-warmup-4 run root: 1;
- historical checkpoints pruned after atomic state commit: 63;
- metrics retained and hash-verified: 64 / 64;
- final checkpoint reload: PASS.

Post-step297 validation on the byte-identical 300k-token validation pack:

| Category | Step 233 loss | Step 297 loss | Relative change |
| --- | ---: | ---: | ---: |
| general English | 5.791585690 | 5.577049066 | -3.704281% |
| general Thai | 6.260309313 | 5.905298695 | -5.670816% |
| technical/scientific/code | 6.005837822 | 5.766543058 | -3.984369% |
| combined | 6.019244275 | 5.749630273 | -4.479200% |

Post-tranche gate: **PASS**. Every category improved, combined validation loss
improved, the validation packs are unchanged, and final holdout remains
untouched.

Tracked evidence hashes:

- step297 validation:
  `f1e08c42cbe62e381837370a3d3dc7000e79f0c9622311dc706140fd19906e57`;
- step233-to-step297 comparison:
  `b7e048dbc355171e6fa46017dc57f1fbc8ef9475db98281732d98fa0b419e0ed`.

## Phase Z fifth post-warmup cosine-decay acceleration authorization

Step297 evidence commit:
`4b8775c324f004d406a5340324c85ac37e8c8642`.

Accelerated post-warmup runner commit:
`39b696a3495db40fffb2cc9ca69838058cc0c408`.

Authorization:
`configs/pretraining-tranche-l004-v16.json`.

- authorization id: `base50m-post-warmup-5-v1`;
- authorization SHA-256:
  `71ce46ad2366ef60135abfc99ecb0b097782e1e322e35b906049b8c5cfe990ea`;
- source optimizer step: 297;
- source checkpoint SHA-256:
  `f0917d81130d3fb8ff6c59d28cfd0b662633b68ae96b36c5d75caceb179870fc`;
- maximum additional optimizer updates: 128;
- authorized end step: 425;
- additional supervised-token cap: 8,388,608;
- cumulative supervised-token cap: 27,852,800;
- schedule: existing qualified cosine decay, no LR-policy change;
- learning rate step298: 0.000599499066;
- learning rate step425: 0.000598238658;
- checkpoint creation cadence: every optimizer update;
- rolling checkpoint retention: latest 1 checkpoint in this tranche;
- prune prior checkpoint only after new metric + atomic state commit;
- post-tranche validation baseline: step297;
- candidate validation step: 425;
- same 300k validation pack required;
- final holdout must remain untouched;
- public release eligibility: not qualified.

This doubles the validated 64-update post-warmup cadence to 128 updates after
step297 improved every validation category, while preserving optimizer,
retention, validation, and holdout contracts.

## Phase Z fifth post-warmup production evidence

The v16 production runner completed normally at the authorized hard stop:
optimizer step 425 with 27,852,800 cumulative supervised tokens.

Verified runtime evidence:

- final checkpoint: `step-000425.pt`;
- final checkpoint SHA-256:
  `b02d6d6e3d4f5f7d06a9256707a71d44ace0293b8c55d2f8b93aa2653abe47ee`;
- final state SHA-256:
  `36c99e89a9566243831812dd6b73276dd281b39e23157f1c45b5699c23c0a1ce`;
- retained checkpoint count: 1;
- historical tranche checkpoints pruned after state commit: 127;
- metrics present and verified: step298 through step425;
- final checkpoint load: PASS;
- pretraining complete: false;
- public release eligibility: not qualified.

Tracked production result:
`results/l004/post-warmup-5-v1.json`.

SHA-256:
`eab3a1e0a280f4b9fdfcce04f5072e6928b5c9fb291d5fdb1f36ccec76f220b2`.

Step425 was then evaluated on the same sealed 300k validation pack used for
step297. No final-holdout material was read or materialized.

| Category | step297 | step425 | Relative change |
| --- | ---: | ---: | ---: |
| general English | 5.577049066 | 5.275155045 | -5.413150% |
| general Thai | 5.905298695 | 5.358594238 | -9.257863% |
| technical/scientific/code | 5.766543058 | 5.430036368 | -5.835501% |
| combined | 5.749630273 | 5.354595217 | -6.870617% |

Post-tranche validation gate: **PASS**.

- every category is finite and improved versus step297;
- combined validation loss improved versus step297;
- same validation pack: true;
- final holdout touched: false.

Tracked validation evidence:

- step425 validation:
  `results/l004/validation-step425-v1.json`;
  SHA-256:
  `f602f2e89e3b87d478272d4aaef4da67c845c3dc842ee5846156a91fc5683ffc`;
- step297-to-step425 comparison:
  `results/l004/validation-comparison-step297-step425-v1.json`;
  SHA-256:
  `281e2eb29688209cf17ef801bf4f5d4c9de9329e7341efdb1811a1773622157f`.

## Step425 evidence exact qualification

Evidence commit:
`691cb9d6e6640f8e4c7c36f3927ae12cdb6e8121`.

Exact qualification from a clean worktree: **PASS**.

Qualification included:

- full repository pytest: PASS;
- L004 audit: PASS;
- step298-through-step425 runtime/state/checkpoint verification: PASS;
- step425 validation rerun to a temporary output: PASS;
- step297-to-step425 predeclared validation gate: PASS;
- same validation pack: true;
- final holdout touched: false;
- HEAD before/after qualification matched the evidence commit;
- worktree before/after qualification was clean.

The evidence commit was pushed non-force, fetched again, and confirmed equal to
`origin/research/l004-base50m-pretraining` at 0 ahead / 0 behind.

## Phase AA sixth post-warmup cosine-decay extension authorization and result

Step425 exact-qualified evidence commit:
`691cb9d6e6640f8e4c7c36f3927ae12cdb6e8121`.

Extended post-warmup runner commit:
`ccb04e7a111dca65eb4f2ffccfe21fd7e66280be`.

Runner exact qualification: **PASS**.
Runner commit was pushed/fetched and confirmed 0 ahead / 0 behind.

Authorization:
`configs/pretraining-tranche-l004-v17.json`.

Authorization SHA-256:
`b7f3050fa280e0ff707a1ea3aaac42c66b9941aeb7fb5c814f8709e40261a824`.

- authorization id: `base50m-post-warmup-6-v1`;
- source optimizer step: 425;
- source checkpoint SHA-256:
  `b02d6d6e3d4f5f7d06a9256707a71d44ace0293b8c55d2f8b93aa2653abe47ee`;
- maximum additional optimizer updates: 256;
- authorized end step: 681;
- additional supervised-token cap: 16,777,216;
- cumulative supervised-token cap: 44,630,016;
- fraction of 500M budget at bound: 8.9260032%;
- schedule: existing qualified cosine decay, no LR-policy change;
- learning rate step426: 0.000598225697;
- learning rate step681: 0.000593382946;
- checkpoint creation cadence: every optimizer update;
- rolling checkpoint retention: latest 1 checkpoint;
- prune prior checkpoint only after new metric + atomic state commit;
- post-tranche validation baseline: step425;
- candidate validation step: 681;
- same sealed 300k validation pack required;
- final holdout must remain untouched;
- public release eligibility: not qualified.

This doubles the validated 128-update cadence to 256 updates after step425
improved all three validation categories and combined loss, while preserving
the same optimizer, retention, validation, and holdout contracts.

The new runner is isolated from the v16 runner/module, so v16 pinned production
provenance remains byte-identical. The new state loader explicitly verifies
`tranche_complete=false` before step681 and `true` at step681.

## Phase AA sixth post-warmup production evidence

Authorization commit:
`46761af6112a9b7ab0376b940c9235670e8d3355`.

The bounded v17 production runner completed normally at optimizer step 681.

Verified runtime evidence:

- optimizer steps: 426 through 681;
- cumulative optimizer step: 681 / 7,630;
- additional supervised tokens: 16,777,216;
- cumulative supervised tokens: 44,630,016 / 500,000,000;
- fraction of primary budget: 8.9260032%;
- final checkpoint: `step-000681.pt`;
- final checkpoint SHA-256:
  `592497980f6737fd74e13343cff0a1fecbaa3fc1fb97c44c47521e5f38f38ea4`;
- final state SHA-256:
  `c1c35c918b6e93f381f0144685c1802b8e784f8fa9ac5935a566b614fc605a29`;
- retained checkpoint count: 1;
- historical tranche checkpoints pruned after state commit: 255;
- metrics present and verified: step426 through step681;
- final checkpoint load: PASS;
- pretraining complete: false;
- public release eligibility: not qualified.

Tracked production result:
`results/l004/post-warmup-6-v1.json`.

SHA-256:
`9a52762d002ab18af93517c36f1e462317dca30aee0c0ace26ef58a4a041ede1`.

Step681 was evaluated on the same sealed 300k validation pack used for step425.
No final-holdout material was read or materialized.

| Category | step425 | step681 | Relative change |
| --- | ---: | ---: | ---: |
| general English | 5.275155045 | 4.867667660 | -7.724652% |
| general Thai | 5.358594238 | 4.740044962 | -11.543126% |
| technical/scientific/code | 5.430036368 | 4.982780476 | -8.236702% |
| combined | 5.354595217 | 4.863497699 | -9.171515% |

Post-tranche validation gate: **PASS**.

- every category is finite and improved versus step425;
- combined validation loss improved versus step425;
- same validation pack: true;
- final holdout touched: false.

Tracked validation evidence:

- step681 validation:
  `results/l004/validation-step681-v1.json`;
  SHA-256:
  `83043c81e76913db89870e4fc2f8b1d95bc280a4d0d649380a5cbe96c8c0182b`;
- step425-to-step681 comparison:
  `results/l004/validation-comparison-step425-step681-v1.json`;
  SHA-256:
  `6c172d2d2483d19672464fb511a9207ff7780d50e49b5e867bf1ed9c1a6db445`.

## Step681 evidence exact qualification

Evidence commit:
`a0628a604ad3caf1efe55951cba3671eb794da69`.

Exact qualification from a clean worktree: **PASS**.

Qualification included:

- full repository pytest: PASS;
- L004 audit: PASS;
- step426-through-step681 runtime/state/checkpoint verification: PASS;
- deterministic verifier output matched the tracked evidence SHA-256;
- step681 validation rerun to a temporary output: PASS;
- all three rerun category losses exactly matched the tracked validation;
- step425-to-step681 predeclared validation gate: PASS;
- same validation pack: true;
- final holdout touched: false;
- v16 and v17 production-pinned module hashes remained unchanged;
- HEAD before/after qualification matched the evidence commit;
- worktree before/after qualification was clean.

The evidence commit was pushed non-force, fetched again, and confirmed equal to
`origin/research/l004-base50m-pretraining` at 0 ahead / 0 behind.

## Research-budget direction

Approximately 50M parameters is the standard FlyWireLLM research budget.
Base-50M-v1 remains the conventional Transformer control; future FlyWire-
inspired Sparse, Routing, Recurrent, Circuit/local-global and gating variants
must remain near the same parameter budget and use matched or explicitly
reported token/compute budgets.

The project must preserve the research story and decision rationale alongside
metrics. See `docs/RESEARCH_ROADMAP.md` and `docs/ARCHITECTURE.md`.

## Step425-to-step681 qualitative capability probe

A fixed synthetic generation probe was added after the quantitative step681
evidence gate. It uses 12 versioned prompts: four Thai, four English, and four
technical/structured-code prompts.

Decoding contract:

- greedy decoding only;
- 32 maximum new tokens;
- no sampling;
- stop on EOS if emitted;
- same prompts/settings for both checkpoints;
- no corpus partition is read;
- final holdout touched: false.

Tracked result:
`results/l004/qualitative-probe-step425-step681-v1.json`.

SHA-256:
`7c574e1480b0c9296e80701450896203599a3fef43a18641a8d476594819ef68`.

Observations are intentionally descriptive rather than scored:

- step681 shows stronger Thai surface-form generation on some prompts than
  step425, but relevance and factual correctness remain poor and repetition is
  still common;
- English at step681 is more sentence-like in several prompts but still creates
  pseudo-words and repetitive constructions;
- question-like prompts remain fragile and can collapse into punctuation;
- Python, JSON, and technical continuation do not yet preserve the requested
  structure reliably;
- all 24 generations reached the 32-token cap without EOS.

This is expected early-base-model behavior at only 44.63M supervised tokens.
The probe is observational evidence only and does not change release status.

## Current work

Exact-qualify the versioned qualitative probe and synchronize its evidence.
After that, design the next bounded continuation from step681 while preserving
v17 production provenance byte-for-byte.

## Next action

1. exact-rerun the qualitative probe and require deterministic output;
2. push/fetch the probe/documentation milestone and verify 0/0 sync;
3. design a separately pinned next continuation runner/authorization;
4. continue bounded pretraining only after exact qualification;
5. keep final holdout sealed and public release blocked.

## Phase AB seventh post-warmup sustained authorization candidate

The step681 quantitative validation gate passed and the versioned qualitative
probe confirmed the model is still an early base model: language form is
improving, while semantic relevance, repetition control, question answering,
and structured/code generation remain immature. Continuing conventional
Base-50M pretraining is therefore preferred over changing architecture now.

A separate sustained continuation implementation preserves v17 provenance.

Runner commit:
`c9d79a2ef4879edbe09fb2ab8cc00f6836821274`.

Candidate authorization:
`configs/pretraining-tranche-l004-v18.json`.

Candidate authorization SHA-256:
`2b3b1db9067a3a6ba687f90c3875e88e7a46f3f48cefe88ccfc0ecda74cd003f`.

Bounded contract:

- source: exact-qualified step681 checkpoint;
- source checkpoint SHA-256:
  `592497980f6737fd74e13343cff0a1fecbaa3fc1fb97c44c47521e5f38f38ea4`;
- source cumulative supervised tokens: 44,630,016;
- authorized optimizer steps: 682 through 1193;
- maximum additional updates: 512;
- additional supervised tokens: 33,554,432;
- target cumulative supervised tokens: 78,184,448;
- primary-budget fraction at the boundary: 15.6368896%;
- step682 LR: 0.000593357961;
- step1193 LR: 0.000574629804;
- optimizer/schedule/data policy: unchanged;
- rolling checkpoint retention: 1;
- prune prior checkpoint only after metric + atomic state commit;
- validation baseline/candidate: step681 -> step1193 on the same sealed pack;
- final holdout must remain untouched;
- public release remains not qualified.

Production is forbidden until this authorization is committed, exact-qualified,
pushed/fetched, and confirmed 0/0 synchronized.


## Phase AB seventh post-warmup sustained production evidence

Authorization commit:
`fc949583a7ecbbf3e717021efc9a3183295cb9e5`.

The v18 sustained production runner completed normally at the authorized hard
stop, optimizer step 1193.

During the tranche, LConnect itself was upgraded. Production was deliberately
stopped only after a committed rolling state existed at step917. The checkpoint,
metric, state hash, optimizer progress and packed-data cursor were verified,
the remaining Python process tree was explicitly stopped, and durable recovery
markers were written outside the repository. After LConnect reconnected, those
hashes were reverified before exactly one runner was launched with `--resume`.
The resumed runner continued at step918 and completed step1193 with exit code 0.

Verified runtime evidence:

- optimizer steps in this tranche: 682 through 1193;
- cumulative optimizer step: 1193 / 7,630;
- additional supervised tokens: 33,554,432;
- cumulative supervised tokens: 78,184,448 / 500,000,000;
- fraction of primary budget: 15.6368896%;
- final checkpoint: `step-001193.pt`;
- final checkpoint bytes: 602,706,635;
- final checkpoint SHA-256:
  `eb13f05a38317d80feb2d46b19ddc8ee90ecab67634e7f1fcff33d557b65ed13`;
- final state SHA-256:
  `393ace6a462dab4c3d23b27e1c8c0ddc5ca40365e1d31b55301eecf613d07174`;
- retained checkpoint count: 1;
- historical tranche checkpoints pruned after state commit: 511;
- metrics present and verified: 512, step682 through step1193;
- final checkpoint load: PASS;
- pretraining complete: false;
- public release eligibility: not qualified.

Checkpoint serialization metadata caused a legitimate byte-size transition
inside this tranche: 324 metric records report 602,706,571-byte checkpoints and
188 report 602,706,635-byte checkpoints, beginning at step1006. Runtime evidence
confirmed this was not corruption: each metric committed its own byte size and
SHA-256, and the retained final file exactly matches the final metric/state.

Tracked production result:
`results/l004/post-warmup-7-v1.json`.

SHA-256:
`f8df5ac0129aa75b47eac072dbb06aad21b845db24bff2058b5a82f78ca49246`.

### Step1193 quantitative validation

Step1193 was evaluated on the same sealed 300k validation pack used at step681.
No final-holdout material was read or materialized.

| Category | step681 | step1193 | Relative change |
| --- | ---: | ---: | ---: |
| general English | 4.867667660 | 4.391436153 | -9.783567% |
| general Thai | 4.740044962 | 4.085647923 | -13.805714% |
| technical/scientific/code | 4.982780476 | 4.405080693 | -11.593924% |
| combined | 4.863497699 | 4.294054923 | -11.708503% |

Post-tranche validation gate: **PASS**.

- every category is finite and improved versus step681;
- combined validation loss improved versus step681;
- same validation pack: true;
- final holdout touched: false.

Tracked validation evidence:

- step1193 validation:
  `results/l004/validation-step1193-v1.json`;
  SHA-256:
  `5699d1bb8b629f1da7e75cc3bd11f44a2a3a56e59978dfc83952ac50e3f81b0e`;
- step681-to-step1193 comparison:
  `results/l004/validation-comparison-step681-step1193-v1.json`;
  SHA-256:
  `4339a97027ee49cc2ad8ef41e1d00a2b45d99202e97f4d61d15062215dfbe4b8`.

### Step681-to-step1193 qualitative capability probe

The v2 qualitative probe uses the exact same 12 synthetic prompts and greedy
decoding contract as v1, but compares step681 with step1193.

Config:
`configs/qualitative-probe-l004-v2.json`.

Config SHA-256:
`0969a229202420ee693b76e0f56649c95fa7d00d19f1fa409e0d976ab9d49206`.

Tracked result:
`results/l004/qualitative-probe-step681-step1193-v2.json`.

SHA-256:
`1262ad30117b268c22fc2d55cd1998fa3571cb1d02ab26c516951753356b7248`.

Observed behavior is intentionally descriptive rather than scored:

- Thai surface-form generation improves materially at step1193, including full
  clauses and sentence-like continuations on prompts where step681 could
  collapse into punctuation or list patterns;
- semantic relevance and factual accuracy are still weak;
- repetition remains common;
- English remains sentence-like but can be irrelevant or invent pseudo-words;
- question answering is not reliable;
- Python and JSON structure remain unreliable;
- all generations reached the fixed 32-token cap without EOS.

The probe is observational only, reads no corpus partition, and does not change
release status.

## Step1193 evidence exact qualification

Evidence commit:
`4d2ba99e9af6048de55c90de6f44217988079115`.

Exact qualification from a clean worktree: **PASS**.

Qualification included:

- full repository pytest: PASS;
- L004 audit: PASS;
- step682-through-step1193 runtime/state/checkpoint verification: PASS;
- deterministic runtime-verifier temporary artifact byte-identical to tracked
  result SHA-256;
- step1193 validation rerun: all category and combined losses exactly matched;
- the validation JSON differed only in nondeterministic `elapsed_seconds`;
- step681-to-step1193 comparison temporary artifact byte-identical to the
  tracked comparison and validation gate PASS;
- qualitative v2 exact rerun temporary artifact byte-identical to tracked
  evidence;
- same validation pack: true;
- final holdout touched: false;
- v16, v17 and v18 production-pinned module hashes remained unchanged;
- v18 production runner hash remained unchanged;
- HEAD before/after qualification matched the evidence commit;
- worktree before/after qualification was clean.

The evidence commit was pushed non-force, fetched again, and confirmed equal to
`origin/research/l004-base50m-pretraining` at 0 ahead / 0 behind.

## Current work after step1193

The Transformer control is still learning materially at 15.6368896% of the
primary token budget. The next step is another separately pinned bounded
continuation from the exact-qualified step1193 state. Existing v18 production
files remain immutable.

## Next action after step1193

1. implement a separately pinned next-continuation module/runner;
2. use step1193 as the exact source checkpoint;
3. preserve optimizer, cosine schedule, data mixture and checkpoint-retention
   semantics;
4. exact-qualify a new authorization before production;
5. keep the final holdout sealed and public release blocked.


## Phase AC eighth post-warmup longrun authorization candidate

Step1193 remains the exact-qualified evidence source:
`4d2ba99e9af6048de55c90de6f44217988079115`.

The next continuation keeps the same model, optimizer, cosine schedule, packed
data and rolling-checkpoint semantics, while doubling the qualified 512-update
cadence to a bounded 1,024-update tranche.

Longrun runner implementation commit:
`46941690ee45f65a4451e1eff55d7e6872ad1b13`.

Candidate authorization:
`configs/pretraining-tranche-l004-v19.json`.

Candidate authorization SHA-256:
`6502c901b23182f579830d71f1b8f4539484cbbf4d0b9996b4115eb1cac6b5db`.

Bounded contract:

- source optimizer step: 1193;
- source cumulative supervised tokens: 78,184,448;
- source checkpoint SHA-256:
  `eb13f05a38317d80feb2d46b19ddc8ee90ecab67634e7f1fcff33d557b65ed13`;
- source checkpoint bytes: 602,706,635;
- authorized optimizer steps: 1194 through 2217;
- maximum additional updates: 1,024;
- additional supervised tokens: 67,108,864;
- target cumulative supervised tokens: 145,293,312;
- primary-budget fraction at the boundary: 29.0586624%;
- step1194 LR: 0.000574581773;
- step2217 LR: 0.000504675007;
- optimizer/schedule/data policy: unchanged;
- rolling checkpoint retention: 1;
- prior checkpoint pruning only after metric + atomic state commit;
- validation baseline/candidate: step1193 -> step2217 on the same sealed pack;
- maximum allowed per-category relative loss increase: 0.5%;
- combined loss must not increase versus step1193;
- final holdout must remain untouched;
- public release remains not qualified.

v19 uses a separate `pretraining_post_warmup_longrun.py` module and longrun
runner. The v18 module and runner remain immutable.

Production is forbidden until v19 authorization is committed, exact-qualified,
pushed/fetched, and confirmed 0/0 synchronized.
