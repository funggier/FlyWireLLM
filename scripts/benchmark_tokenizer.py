from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from flywire_llm.corpus import iter_jsonl_records
from flywire_llm.sentencepiece_tokenizer import (
    SentencePieceTokenizer,
    benchmark_tokenizer,
    load_benchmark_cases,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark SentencePiece candidate against UTF-8 byte baseline."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--validation-corpus")
    parser.add_argument("--cases")
    parser.add_argument("--output-json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.validation_corpus and not args.cases:
        raise SystemExit("provide --validation-corpus and/or --cases")

    cases: list[dict[str, str]] = []
    if args.validation_corpus:
        for record in iter_jsonl_records(args.validation_corpus):
            if record.partition != "validation":
                raise RuntimeError(
                    "validation corpus contains a non-validation record"
                )
            cases.append(
                {
                    "case_id": record.record_id,
                    "language": record.language,
                    "category": "natural_validation",
                    "text": record.text,
                }
            )
    if args.cases:
        cases.extend(load_benchmark_cases(args.cases))

    tokenizer = SentencePieceTokenizer(args.model)
    rows = benchmark_tokenizer(tokenizer, cases)

    grouped: dict[str, dict[str, float | int]] = {}
    by_language: dict[str, list] = defaultdict(list)
    for row in rows:
        by_language[row.language].append(row)

    for language, subset in sorted(by_language.items()):
        byte_tokens = sum(row.byte_tokens for row in subset)
        candidate_tokens = sum(row.candidate_tokens for row in subset)
        characters = sum(row.characters for row in subset)
        grouped[language] = {
            "cases": len(subset),
            "characters": characters,
            "byte_tokens": byte_tokens,
            "candidate_tokens": candidate_tokens,
            "candidate_tokens_per_char": candidate_tokens / characters,
            "compression_vs_byte": byte_tokens / candidate_tokens,
            "roundtrip_failures": sum(
                not row.exact_roundtrip for row in subset
            ),
            "unk_tokens": sum(row.unk_tokens for row in subset),
        }

    total_byte = sum(row.byte_tokens for row in rows)
    total_candidate = sum(row.candidate_tokens for row in rows)
    payload = {
        "model": str(Path(args.model)),
        "vocab_size": tokenizer.vocab_size,
        "cases": len(rows),
        "overall": {
            "byte_tokens": total_byte,
            "candidate_tokens": total_candidate,
            "compression_vs_byte": total_byte / total_candidate,
            "roundtrip_failures": sum(
                not row.exact_roundtrip for row in rows
            ),
            "unk_tokens": sum(row.unk_tokens for row in rows),
        },
        "by_language": grouped,
        "rows": [
            {
                "case_id": row.case_id,
                "language": row.language,
                "category": row.category,
                "characters": row.characters,
                "utf8_bytes": row.utf8_bytes,
                "byte_tokens": row.byte_tokens,
                "candidate_tokens": row.candidate_tokens,
                "compression_vs_byte": row.compression_vs_byte,
                "exact_roundtrip": row.exact_roundtrip,
                "unk_tokens": row.unk_tokens,
            }
            for row in rows
        ],
    }
    rendered = json.dumps(payload, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output_json:
        Path(args.output_json).write_text(
            rendered + "\n",
            encoding="utf-8",
            newline="\n",
        )


if __name__ == "__main__":
    main()
