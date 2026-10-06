# Development Task Ledger

This directory is the authoritative human-readable work ledger for FlyWireLLM.

GitHub Issues remain the external task tracker. Files here explain the local
engineering state in enough detail that a new session can resume work without
reconstructing intent from commit history.

## Required workflow

For every non-trivial development stage or release task:

1. Create or update one task file before implementation begins.
2. Set exactly one status: `PLANNED`, `ACTIVE`, `BLOCKED`, or `DONE`.
3. Record the authoritative base branch and exact base commit.
4. State the goal, motivation, scope, non-scope, invariants, and acceptance
   criteria before changing code/data.
5. Keep **Current work** and **Next action** current while the task is ACTIVE.
6. Record checkpoints with exact commit SHAs and evidence.
7. If blocked, record the blocker and the condition required to resume.
8. Close a task only after exact-commit qualification and remote-sync evidence.
9. Never rewrite historical evidence to make an old checkpoint look current.
10. If live Git/GitHub/runtime state disagrees with a task document, live state
    wins and the task document must be corrected.

## Status meanings

| Status | Meaning |
| --- | --- |
| `PLANNED` | Accepted work, implementation has not started. |
| `ACTIVE` | Work is currently in progress. |
| `BLOCKED` | Work cannot proceed until a named dependency/decision is resolved. |
| `DONE` | Acceptance criteria passed on an exact commit and the result is recorded. |

## File naming

- Development stages: `LNNN-short-slug.md`
- Release tasks: `RNNN-short-slug.md`
- Template: `TASK-TEMPLATE.md`
- Current-task pointer: `CURRENT.md`

A task file should correspond to one GitHub Issue whenever practical.

## Current ledger

| Task | Status | GitHub | Result |
| --- | --- | --- | --- |
| L001 Blank LLM foundation | DONE | #1 | Random-initialized training-ready baseline |
| R001 v0.1.0 release | DONE | #2 | First MIT release |
| L002 Base-50M/tokenizer/data contract | DONE | #3 | Base-50M + 32K tokenizer engineering gate |
| L003 production corpus readiness | DONE | #4 | 500M research-only corpus qualified; pretraining lineage authorized |

For the exact current action, read [CURRENT.md](CURRENT.md).
