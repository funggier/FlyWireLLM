# L002 — Base-50M Target, Thai-English Tokenizer, and Data Contract

Status: `DONE`

GitHub Issue: `#3`

Updated: 2026-10-06

## Objective

Freeze the first real FlyWireLLM pretraining target and qualify its tokenizer,
data split, source policy, and training schedule without claiming pretraining.

## Final result

Decision:

**TARGET CONTRACT GREEN / TOKENIZER ENGINEERING GREEN / DATA INVENTORY GREEN /
500M CORPUS NOT YET ACQUIRED OR RIGHTS-QUALIFIED / PRETRAINING NOT STARTED**

Base-50M-v1:

- exact parameters: 50,213,376;
- vocabulary: 32,000;
- d_model: 512;
- layers: 12;
- query/KV heads: 8/2;
- d_ff: 1,408;
- initial context: 1,024.

Tokenizer engineering:

- SentencePiece Unigram 32K;
- identity normalization + byte fallback;
- 10,150 benchmark cases;
- round-trip failures: 0;
- unknown tokens: 0;
- compression vs byte: overall 6.252x, Thai 10.710x, English 4.031x,
  mixed 2.570x.

## Final evidence

- final commit:
  `528e96736c52d5ba90412bb1ee0b8c0239349e57`;
- full pytest: 47 PASS;
- L002 audit: PASS;
- `git diff --check`: PASS;
- `main = origin/main` at the final commit;
- GitHub Actions had 0 runs; no CI success was claimed.

## Follow-on

L003 owns production-corpus acquisition, rights lanes, quality/privacy
screening, global cross-source deduplication, mixture readiness, and final
pretraining authorization.
