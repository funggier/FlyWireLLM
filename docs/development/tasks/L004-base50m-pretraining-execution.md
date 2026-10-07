# L004 — Base-50M Research-Only Pretraining Execution

Status: ACTIVE

GitHub Issue: #6

Updated: 2026-10-07

## Objective

Implement, qualify, and then execute the first real Base-50M-v1
research-only pretraining run from the frozen L003 lineage.

L004 has qualified the runtime/data path and completed the first bounded
production tranche. Pretraining has started, but only 4 of 7,630 planned
optimizer updates (262,144 of 500,000,000 supervised tokens) have executed.
Full Base-50M pretraining is not complete.

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

## Current work

Record and exact-qualify the third bounded-tranche + post-step16 validation
evidence before any further training authorization.

## Next action

1. full-regression the complete step16 evidence WIP;
2. commit and exact-qualify the third-tranche evidence;
3. push and confirm remote synchronization;
4. only after that, design the next bounded expansion from verified step16;
5. continue to use the identical validation pack and keep final holdout
   untouched before every subsequent expansion.
