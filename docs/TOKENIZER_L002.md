# L002 Tokenizer Design

## Target

FlyWireLLM Base-50M-v1 targets a learned 32,000-token SentencePiece Unigram tokenizer. The released v0.1.0 UTF-8 byte tokenizer remains the frozen baseline.

Required properties:

- exact Thai, English and mixed Thai/English round-trip;
- byte fallback so arbitrary UTF-8 avoids unknown-token loss;
- identity normalization;
- preserved repeated whitespace and technical identifiers;
- special IDs: pad=0, bos=1, eos=2, unk=3;
- tokenizer fitting uses train only;
- final holdout text is excluded from tokenizer development;
- Thai, English and mixed-language efficiency are reported separately.

## 8K Tatoeba pilot

The first pilot used the 2026-10-03 Tatoeba Thai/English sentence exports under CC BY 2.0 FR. Exact-content fingerprints were computed before partitioning, so duplicate content cannot cross model partitions.

Fit was balanced by UTF-8 bytes:

- Thai: 6,780 records / 440,701 bytes;
- English: 10,962 records / 440,710 bytes.

The 8,192-token pilot benchmark used 115 validation/technical cases and achieved:

- round-trip failures: 0;
- unknown tokens: 0;
- overall compression versus byte tokenizer: 4.729x;
- Thai compression: 11.735x;
- English compression: 3.418x;
- mixed compression: 1.933x.

Decision: Unigram + identity normalization + byte fallback remains the Base-50M candidate design.

## 32K engineering candidate

The 32K engineering corpus combines a pinned Thai Wikipedia sample with English Tatoeba. The fit target was 50 MiB of text per language.

Materialized fit:

- Thai: 7,814 documents / 52,511,355 UTF-8 bytes;
- English: 1,304,156 sentences / 52,428,832 UTF-8 bytes;
- total fit records: 1,311,970;
- total fit text bytes: 104,940,187;
- fit SHA-256: 01f51cec4eb6926e8654fe8d920d498e342351dbc4b07f4b50f0ec4c6be690ad.

Validation:

- 10,138 records;
- 929,878 UTF-8 text bytes;
- SHA-256: ebb2dfa1c8b4f47e1608d39bb5c5b05c92096a31fd1838345016e5287288cf6b.

Holdout text was not materialized.

The exact 32,000-token Unigram candidate trained successfully. Its model SHA-256 is 998bc75f058e554d4f50b6cbc77c52898e3446e516c3b25a126b99b113513818 and vocabulary SHA-256 is 9ade52fa3e57f6bb096cfdcc64b37d89f794937b68d155997b48f529c841e762.

## 32K benchmark

The benchmark contains 10,150 validation and technical cases.

| Slice | Cases | Compression vs byte | Round-trip failures | Unknown tokens |
| --- | ---: | ---: | ---: | ---: |
| Thai | 50 | 10.710x | 0 | 0 |
| English | 10,094 | 4.031x | 0 | 0 |
| Mixed | 6 | 2.570x | 0 | 0 |
| Overall | 10,150 | 6.252x | 0 | 0 |

The English validation set is much larger by record count because Tatoeba sentences are short. Thai and English results are therefore reported separately instead of treating the overall average as sufficient.

## Rights boundary

Thai Wikipedia is retained as a conditional-rights engineering source. The successful 32K candidate is therefore an engineering qualification, not permission to redistribute that tokenizer under the repository MIT license or to use the underlying text for production pretraining without a separate compliance decision.

Tokenizer engineering and corpus-rights qualification remain separate gates.

## Decision

**TOKENIZER ENGINEERING GREEN / BASE-50M 32K UNIGRAM CANDIDATE REPRODUCIBLE AND LOSSLESS ON L002 BENCHMARK / CORPUS RIGHTS AND 500M PRETRAINING READINESS REMAIN SEPARATE**
