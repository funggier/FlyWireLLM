from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from flywire_llm.sentencepiece_tokenizer import SentencePieceTokenizer
from flywire_llm.tokenizer_corpus import (
    deduplicate,
    iter_tatoeba_english,
    iter_wikiextractor_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count exact 32K tokens in acquired deduplicated L002 sources."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--thai-wiki-dir", required=True)
    parser.add_argument("--english-tatoeba", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _empty() -> dict[str, int]:
    return {
        "records": 0,
        "utf8_bytes": 0,
        "tokens": 0,
    }


def main() -> None:
    args = parse_args()
    tokenizer = SentencePieceTokenizer(args.model)
    thai, thai_duplicates = deduplicate(
        iter_wikiextractor_json(args.thai_wiki_dir)
    )
    english, english_duplicates = deduplicate(
        iter_tatoeba_english(args.english_tatoeba)
    )

    by_language = {"th": _empty(), "en": _empty()}
    by_partition = {
        "train": _empty(),
        "validation": _empty(),
        "holdout": _empty(),
    }
    cross: dict[str, dict[str, dict[str, int]]] = {
        partition: {"th": _empty(), "en": _empty()}
        for partition in by_partition
    }

    for row in thai + english:
        tokens = len(tokenizer.encode(row.text))
        for bucket in (
            by_language[row.language],
            by_partition[row.partition],
            cross[row.partition][row.language],
        ):
            bucket["records"] += 1
            bucket["utf8_bytes"] += row.utf8_bytes
            bucket["tokens"] += tokens

    total = _empty()
    for language in ("th", "en"):
        for key in total:
            total[key] += by_language[language][key]

    payload = {
        "schema_version": 1,
        "stage": "L002",
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "deduplication": {
            "thai_duplicates_removed": thai_duplicates,
            "english_duplicates_removed": english_duplicates,
        },
        "total": total,
        "by_language": by_language,
        "by_partition": by_partition,
        "partition_language": cross,
        "primary_target_tokens": 500_000_000,
        "train_fraction_of_primary_target": (
            by_partition["train"]["tokens"] / 500_000_000
        ),
        "train_gap_to_primary_target": max(
            0,
            500_000_000 - by_partition["train"]["tokens"],
        ),
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
