from __future__ import annotations

import argparse
import json
from pathlib import Path

from flywire_llm.tokenizer_corpus import (
    deduplicate,
    iter_tatoeba_english,
    iter_wikiextractor_json,
    partition_rows,
    rows_stats,
    select_bytes,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build deterministic L002 Thai/English tokenizer corpus."
    )
    parser.add_argument("--thai-wiki-dir", required=True)
    parser.add_argument("--english-tatoeba", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--fit-bytes-per-language",
        type=int,
        default=50 * 1024 * 1024,
    )
    parser.add_argument(
        "--validation-bytes-per-language",
        type=int,
        default=512 * 1024,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    thai, thai_duplicates = deduplicate(
        iter_wikiextractor_json(args.thai_wiki_dir)
    )
    english, english_duplicates = deduplicate(
        iter_tatoeba_english(args.english_tatoeba)
    )

    thai_train = partition_rows(thai, "train")
    english_train = partition_rows(english, "train")
    thai_validation = partition_rows(thai, "validation")
    english_validation = partition_rows(english, "validation")
    thai_holdout = partition_rows(thai, "holdout")
    english_holdout = partition_rows(english, "holdout")

    fit_thai = select_bytes(
        thai_train,
        args.fit_bytes_per_language,
    )
    fit_english = select_bytes(
        english_train,
        args.fit_bytes_per_language,
    )
    validation_thai = select_bytes(
        thai_validation,
        args.validation_bytes_per_language,
    )
    validation_english = select_bytes(
        english_validation,
        args.validation_bytes_per_language,
    )

    fit_path = output / "tokenizer_fit_32k.jsonl"
    validation_path = output / "tokenizer_validation_32k.jsonl"
    fit_stats = write_jsonl(fit_path, fit_thai + fit_english)
    validation_stats = write_jsonl(
        validation_path,
        validation_thai + validation_english,
    )

    summary = {
        "schema_version": 1,
        "stage": "L002",
        "builder": "thai-wikipedia+tatoeba-english-v1",
        "split_seed": "flywirellm-l002-split-v1",
        "split_group": "dedup_fingerprint",
        "fit_target_bytes_per_language": args.fit_bytes_per_language,
        "validation_target_bytes_per_language": (
            args.validation_bytes_per_language
        ),
        "sources": {
            "th": {
                "source_id": "wikimedia-thwiki-2026-10-01",
                "deduplicated_records": len(thai),
                "duplicates_removed": thai_duplicates,
                "partitions": {
                    "train": rows_stats(thai_train),
                    "validation": rows_stats(thai_validation),
                    "holdout": rows_stats(thai_holdout),
                },
            },
            "en": {
                "source_id": "tatoeba-sentences-2026-10-03",
                "deduplicated_records": len(english),
                "duplicates_removed": english_duplicates,
                "partitions": {
                    "train": rows_stats(english_train),
                    "validation": rows_stats(english_validation),
                    "holdout": rows_stats(english_holdout),
                },
            },
        },
        "materialized": {
            "fit": {
                "th": rows_stats(fit_thai),
                "en": rows_stats(fit_english),
                "jsonl_records": fit_stats[0],
                "text_utf8_bytes": fit_stats[1],
                "sha256": fit_stats[2],
            },
            "validation": {
                "th": rows_stats(validation_thai),
                "en": rows_stats(validation_english),
                "jsonl_records": validation_stats[0],
                "text_utf8_bytes": validation_stats[1],
                "sha256": validation_stats[2],
            },
            "holdout_text": False,
        },
    }
    summary_path = output / "corpus_32k_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        f"fit_records={fit_stats[0]} "
        f"fit_text_bytes={fit_stats[1]} "
        f"fit_sha256={fit_stats[2]}"
    )
    print(
        f"validation_records={validation_stats[0]} "
        f"validation_text_bytes={validation_stats[1]} "
        f"validation_sha256={validation_stats[2]}"
    )
    print(
        "holdout_records="
        f"th:{len(thai_holdout)},en:{len(english_holdout)}"
    )
    print("holdout_text_materialized=false")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
