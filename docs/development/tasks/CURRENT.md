# Current Development Task

Current task: [L003 — Production Corpus Acquisition, Rights Lanes, and 500M Readiness](L003-production-corpus-readiness.md)

Status: `ACTIVE`

GitHub Issue: `#4`

L003 technical/data/freeze gates are GREEN. The selected checkpoint lane is
`research_only`; the post-dedup 500M mixture is qualified and the freeze was
created before optimizer step 1.

## Resume here

Do **not** reset or clean the worktree.

The only remaining L003 work is Phase J qualification/merge:

1. inspect the final diff for accidental raw corpus or machine-local paths;
2. create the implementation checkpoint commit;
3. exact-commit full pytest + L003 audits + diff checks;
4. push and verify remote race/sync state;
5. close the task ledger on a second qualified commit;
6. fast-forward `main` and repeat post-merge qualification;
7. update and close GitHub Issue #4.

Key frozen evidence:

- final global report:
  `results/l003/global-manifest-v2.json`;
- research-only available train tokens: 969,701,220;
- source registry SHA-256:
  `5304c4cc6b302a06650d090d6731f5b0e610fd99ec257c3fd9cc671cf62f79aa`;
- global report SHA-256:
  `30a792d9fd26d1d4a6a1760aae7a43ae293778926be9044ebda632ec7c5054d5`;
- decision manifest SHA-256:
  `018aa08f347ebfd747f3aa11cada563bbd27918b469c7c5f11abe0289df7770a`;
- tokenizer SHA-256:
  `998bc75f058e554d4f50b6cbc77c52898e3446e516c3b25a126b99b113513818`.

Pretraining is authorized only for the frozen research-only lineage. Base-50M
pretraining itself has not started, and public release eligibility remains not
qualified.
