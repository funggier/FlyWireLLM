# Current Development Task

Current task:
L004 — Base-50M Research-Only Pretraining Execution

Status: ACTIVE

GitHub Issue: #6

Branch:
research/l004-base50m-pretraining

Base commit:
aa8a00e7ee5c3e8827072a47d5466f1fbbaf7997

## Current state

Runtime/data foundation phases A-H are technically GREEN in the worktree.

Key facts:

- production optimizer steps recorded: 0;
- checkpoint lane: research_only;
- exact selected content tokens: 500,000,000;
- packed physical tokens: 501,358,839;
- token-stream SHA-256:
  07a0faf43c241f3e232a6200bc1732381d18e9ef1b2cd77c185d2fb4c4296a15;
- shuffled block-order SHA-256:
  146f12fdccd0d2694f0e81e3f85f2a786cb4d72fc47b4302990d6c8696eecc6d;
- selected production-local context: 1024 / fp32 / CPU / 12 threads;
- actual packed-data global update qualification:
  65,536 supervised tokens in 46.2343 s (~1,417.5 tok/s);
- projected pure compute for 500M: ~98.0 h before operational overhead;
- deterministic mid-block checkpoint/resume equivalence: PASS.

The faster seq512/batch4 result is evidence only and is not selected because
the first production run must preserve 1024-context training semantics.

## Resume here

Do not reset or clean the worktree.

Before production optimizer step 1:

1. regenerate runtime-qualification summary hashes after final edits;
2. run full pytest;
3. run L003 final audit and L004 audit;
4. re-run selection, packing and deterministic block-order verifiers;
5. run diff/path/raw-artifact safety checks;
6. create and qualify the exact L004 foundation commit;
7. push and verify remote branch synchronization;
8. create a separate first-bounded-tranche authorization pinned to that commit.

Only after those gates may Phase I execute a real optimizer step.
