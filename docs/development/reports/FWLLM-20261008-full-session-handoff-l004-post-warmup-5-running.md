# FlyWireLLM Full Session Handoff - L004 Post-Warmup-5 Running

Date: 2026-10-08  
Project: `funggier/FlyWireLLM`  
Task: L004 - Base-50M Research-Only Pretraining Execution  
GitHub issue: #6  
Branch: `research/l004-base50m-pretraining`

## Read this first

This handoff was written while a production pretraining process was **still
running**. The live Git/GitHub/runtime state is authoritative over any older
handoff, task prose, or conversational context.

Hard rules for the next session:

1. **Do not reset, clean, checkout away, or discard the worktree.**
2. **Do not launch a second training runner if the current process is still
   alive.**
3. Re-read Git, remote, the managed process session, and external
   `state.json` before taking any action.
4. Preserve rolling checkpoint retention and atomic state-before-prune order.
5. Final holdout must remain untouched.
6. Public release remains **not qualified**.
7. Do not claim L004 complete until the full 500,000,000 supervised-token
   contract and later qualification requirements are actually satisfied.
8. Do not modify the training/data path merely to make a test pass.
9. Keep using exact-commit qualification before authorizing each new
   production tranche.

## Authoritative repository/runtime locations

Repository:

`T:\Space\Projects\ProjectsAI\FlyWireLLM`

External training data/runtime root:

`T:\Space\Projects\ProjectsAI\FlyWireLLM-data`

Python runtime:

`T:\Space\Projects\ProjectsAI\FlyWireModel\.venv\Scripts\python.exe`

Current branch:

`research/l004-base50m-pretraining`

At handoff discovery, Git was clean and synchronized:

- local HEAD:
  `32a2eed73fa4a6c66e0533c0b6411e9413cb6bbb`
- remote:
  `origin/research/l004-base50m-pretraining`
- ahead/behind: `0 / 0`

The handoff report itself is a documentation-only change created after that
snapshot. Re-check HEAD/remote after reading this file.

## Important correction versus older context

Do **not** resume from step153.

The repository and runtime progressed substantially after the earlier
step153 context. Exact-qualified post-warmup evidence already exists through
**optimizer step297** and a new accelerated production tranche is currently
running from step298 through step425.

Recent authoritative commit chain at handoff discovery:

- `32a2eed` - Authorize accelerated post-warmup L004 tranche
- `39b696a` - Add accelerated post-warmup L004 tranche runner
- `4b8775c` - Record L004 step297 post-warmup evidence
- `527792c` - Authorize expanded post-warmup L004 tranche
- `db93205` - Add expanded post-warmup L004 tranche runner
- `2c4650b` - Record L004 step233 post-warmup evidence
- `7be038d` - Authorize third post-warmup L004 tranche
- `585aad9` - Add steady post-warmup L004 tranche runner
- `789edfe` - Record L004 step201 post-warmup evidence
- `4fe9324` - Authorize scaled post-warmup L004 tranche
- `d302732` - Add scaled post-warmup L004 tranche runner
- `068a2a1` - Record L004 step169 post-warmup evidence
- `f11a412` - Authorize first post-warmup L004 tranche
- `ed93936` - Add bounded post-warmup L004 runner
- `988e10f` - Record L004 step153 warmup boundary evidence
- `655297b` - Authorize exact L004 warmup boundary

## Latest completed exact-qualified boundary: step297

`docs/development/tasks/CURRENT.md` records step297 as the latest completed
exact-qualified checkpoint before the active tranche.

Step297 evidence commit:

`4b8775c324f004d406a5340324c85ac37e8c8642`

Checkpoint SHA-256:

`f0917d81130d3fb8ff6c59d28cfd0b662633b68ae96b36c5d75caceb179870fc`

Progress at step297:

- supervised tokens: 19,464,192 / 500,000,000
- fraction of primary budget: 3.8928384%
- validation gate: PASS
- final holdout touched: false
- pretraining complete: false
- public release eligibility: not qualified

Step233 -> step297 validation gate on the same 300k-token pack:

| Category | Step233 | Step297 | Relative change |
| --- | ---: | ---: | ---: |
| general English | 5.791585690 | 5.577049066 | -3.704281% |
| general Thai | 6.260309313 | 5.905298695 | -5.670816% |
| technical/scientific/code | 6.005837822 | 5.766543058 | -3.984369% |
| combined | 6.019244275 | 5.749630273 | -4.479200% |

All category gates passed, the same validation pack was used, and the final
holdout remained untouched.

## Active authorization: v16 post-warmup accelerated tranche

Authorization file:

`configs/pretraining-tranche-l004-v16.json`

Authorization id:

`base50m-post-warmup-5-v1`

Authorization SHA-256:

`71ce46ad2366ef60135abfc99ecb0b097782e1e322e35b906049b8c5cfe990ea`

Authorization commit / current HEAD at discovery:

`32a2eed73fa4a6c66e0533c0b6411e9413cb6bbb`

Runner commit:

`39b696a3495db40fffb2cc9ca69838058cc0c408`

Evidence source commit:

`4b8775c324f004d406a5340324c85ac37e8c8642`

Source checkpoint:

`external://FlyWireLLM-data/L004/Runs/base50m-post-warmup-4-v1/checkpoints/step-000297.pt`

Source checkpoint SHA-256:

`f0917d81130d3fb8ff6c59d28cfd0b662633b68ae96b36c5d75caceb179870fc`

Authorized bound:

- source optimizer step: 297
- first production step: 298
- maximum additional updates: 128
- hard stop: step425
- additional supervised-token cap: 8,388,608
- cumulative supervised-token cap: 27,852,800
- checkpoint every update
- rolling retention: 1 latest full checkpoint
- prior checkpoint pruned only after new metric + atomic state commit
- candidate validation step: 425
- baseline validation step: 297
- same 300k validation pack required
- final holdout must remain untouched

The v16 runner exact qualification passed:

`39b696a3495db40fffb2cc9ca69838058cc0c408`

The v16 authorization exact qualification also passed before production was
started. The managed exact-qualification session was:

`flywirellm-l004-v16-auth-exact`

and exited successfully with
`post_warmup_accelerated_authorization_exact_qualification=PASS`.

## Production process currently running

Managed process session:

`proc-1791437385568-350`

Label:

`flywirellm-l004-post-warmup-5-step298-425`

PID at handoff snapshot:

`14076`

Command:

`scripts\run_pretraining_post_warmup_accelerated_l004.py --external-root T:\Space\Projects\ProjectsAI\FlyWireLLM-data`

The process was **running** when this handoff was written.

### Live snapshot captured during handoff

The first live snapshot captured:

- optimizer step: 373
- cumulative supervised tokens: 24,444,928
- physical input positions: 24,511,279
- microbatches seen: 24,309
- data cursor order position: 23,936
- block input offset: 815
- tranche complete: false
- metrics present: 76
- retained checkpoint count: 1
- retained checkpoint:
  `step-000373.pt`
- checkpoint SHA-256:
  `a9c0e0fdfcea48d2c1186f6f5aa4785b18d4adf3a2447f1b63bd4815562071a1`
- step373 loss: 5.591201
- learning rate at step373: 0.0005988473
- stderr: empty
- Git ahead/behind at snapshot: 0 / 0

External live state file:

`T:\Space\Projects\ProjectsAI\FlyWireLLM-data\L004\Runs\base50m-post-warmup-5-v1\state.json`

Checkpoint directory:

`T:\Space\Projects\ProjectsAI\FlyWireLLM-data\L004\Runs\base50m-post-warmup-5-v1\checkpoints`

Metrics directory:

`T:\Space\Projects\ProjectsAI\FlyWireLLM-data\L004\Runs\base50m-post-warmup-5-v1\metrics`

**Important:** this process continued running after the snapshot, so step373
is not a resume instruction. It is only evidence of a valid recovery point.
Always read the live session + `state.json` first.

## Correct resume decision tree

### Case A - managed process is still running

1. Read session:
   `proc-1791437385568-350`.
2. Read external `state.json`.
3. Confirm Git branch and remote synchronization.
4. **Do not launch another runner.**
5. Monitor the existing process until it stops at step425.
6. Confirm exit code 0 and no stderr/integrity errors.

### Case B - process already completed at step425

Do not rerun training.

Proceed directly to:

1. verify external state/checkpoint/metrics for steps298-425;
2. confirm exactly one retained full checkpoint;
3. confirm historical checkpoints were pruned only after atomic state commit;
4. generate the tracked post-warmup-5 result JSON;
5. extend the validation evaluator for step425 if not already present;
6. evaluate step425 on the unchanged 300k validation pack;
7. compare step297 vs step425 using the predeclared v16 validation gate;
8. require `final_holdout_touched=false`;
9. update task ledger/CURRENT;
10. full-regression;
11. commit step425 evidence;
12. exact-qualify that exact commit with validation output directed to a temp
    file;
13. require a clean/non-mutated worktree after qualification;
14. push and fetch;
15. confirm remote synchronization before authorizing any later tranche.

### Case C - process stopped before step425

Do not restart from step297 and do not delete the run root.

1. Inspect `state.json`.
2. Verify the checkpoint and metric hashes referenced by state.
3. Verify the latest retained checkpoint loads with the pinned lineage.
4. Treat that state/checkpoint as the recovery boundary.
5. Resume only through the existing v16 runner/authorization semantics.
6. Never execute beyond step425 under v16.
7. Preserve all metrics already written.

## Expected step425 verification contract

When production reaches the v16 hard stop:

- optimizer step must equal 425;
- cumulative supervised tokens must equal 27,852,800;
- the run must remain research-only;
- pretraining complete must still be false;
- public release eligibility must still be not qualified;
- final holdout must remain untouched;
- full retained checkpoints inside post-warmup-5 run root should be 1;
- metrics should cover every authorized production update step298-425;
- final checkpoint must load with the expected lineage and optimizer/data
  progress;
- validation must use the same byte-identical 300k validation pack as step297.

Validation gate from v16:

- combined loss must not increase versus step297;
- each category may increase by at most 0.5% relative to step297;
- all losses must be finite;
- same validation pack required;
- final holdout must remain untouched.

## Long-run L004 facts that remain authoritative

Runtime contract:

- model parameters: 50,213,376
- tokenizer vocab: 32,000
- primary supervised-token budget: 500,000,000
- global supervised tokens per optimizer update: 65,536
- planned optimizer steps: 7,630
- optimizer warmup steps: 153
- warmup is already complete
- post-warmup schedule is cosine decay
- checkpoint lane: research-only

Pinned core identities from L004 audit:

- tokenizer SHA-256:
  `998bc75f058e554d4f50b6cbc77c52898e3446e516c3b25a126b99b113513818`
- data runtime SHA-256:
  `2d99c61c84fbfffb0ba5f23434bf6e8b92a92b3f588d7716238c0ebc76ee424b`
- token stream SHA-256:
  `07a0faf43c241f3e232a6200bc1732381d18e9ef1b2cd77c185d2fb4c4296a15`
- block order SHA-256:
  `146f12fdccd0d2694f0e81e3f85f2a786cb4d72fc47b4302990d6c8696eecc6d`
- source manifest SHA-256:
  `5304c4cc6b302a06650d090d6731f5b0e610fd99ec257c3fd9cc671cf62f79aa`
- global manifest report SHA-256:
  `30a792d9fd26d1d4a6a1760aae7a43ae293778926be9044ebda632ec7c5054d5`
- decisions SHA-256:
  `018aa08f347ebfd747f3aa11cada563bbd27918b469c7c5f11abe0289df7770a`

## Files to read first in the next session

1. this handoff;
2. `docs/development/tasks/CURRENT.md`;
3. `docs/development/tasks/L004-base50m-pretraining-execution.md`;
4. `configs/pretraining-tranche-l004-v16.json`;
5. `src/flywire_llm/pretraining_post_warmup_accelerated.py`;
6. `scripts/run_pretraining_post_warmup_accelerated_l004.py`;
7. live external `base50m-post-warmup-5-v1/state.json`;
8. `results/l004/post-warmup-4-v1.json`;
9. `results/l004/validation-step297-v1.json`;
10. `results/l004/validation-comparison-step233-step297-v1.json`.

## First commands/checks for the next session

Use Git/GitHub/runtime as authoritative source before documents:

```powershell
cd T:\Space\Projects\ProjectsAI\FlyWireLLM
git status --short --branch
git rev-parse HEAD
git fetch origin --prune
git rev-list --left-right --count HEAD...origin/research/l004-base50m-pretraining
git log -12 --oneline --decorate
```

Then inspect the managed session
`proc-1791437385568-350` using LConnect. If it still exists and is running,
monitor it; do not start a duplicate.

Then inspect:

```powershell
Get-Content T:\Space\Projects\ProjectsAI\FlyWireLLM-data\L004\Runs\base50m-post-warmup-5-v1\state.json
Get-ChildItem T:\Space\Projects\ProjectsAI\FlyWireLLM-data\L004\Runs\base50m-post-warmup-5-v1\checkpoints
(Get-ChildItem T:\Space\Projects\ProjectsAI\FlyWireLLM-data\L004\Runs\base50m-post-warmup-5-v1\metrics -File).Count
```

## Recommended next-session opening prompt

> ทำ `funggier/FlyWireLLM` ต่อจาก Full Handoff ล่าสุด โดยอ่าน
> `docs/development/reports/FWLLM-20261008-full-session-handoff-l004-post-warmup-5-running.md`
> ก่อน ยึด Git/GitHub/runtime สดเป็น authoritative source ห้าม reset/clean
> worktree และห้ามเปิด training runner ซ้ำถ้า session
> `proc-1791437385568-350` ยังรันอยู่ ให้ตรวจ live state ก่อน ถ้ายังรันให้
> monitor จนถึง hard stop step425; ถ้าจบแล้วให้ทำ verify + validation
> step425 + exact-qualified evidence ต่อ โดย final holdout ต้องไม่ถูกแตะ

## Handoff status

**ACTIVE / RUNNING.**

The current production tranche was healthy at the handoff snapshot and was
still progressing toward hard stop step425. The next session must begin by
refreshing the live state rather than treating any step number in this report
as the final current step.
