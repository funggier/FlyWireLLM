from __future__ import annotations

import argparse
import bz2
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from flywire_llm.data_contract import (
    assign_partition,
    dedup_fingerprint,
)


SOURCE_ID = "tatoeba-sentences-2026-10-03"
LICENSE = "CC BY 2.0 FR"
LICENSE_URL = "https://creativecommons.org/licenses/by/2.0/fr/"


@dataclass(frozen=True)
class Row:
    sentence_id: str
    language: str
    text: str
    fingerprint: str
    partition: str

    @property
    def record_id(self) -> str:
        prefix = "TH" if self.language == "th" else "EN"
        return f"TATOEBA-{prefix}-{self.sentence_id}"

    @property
    def utf8_bytes(self) -> int:
        return len(self.text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_tatoeba(path: Path, language: str) -> list[Row]:
    expected_code = "tha" if language == "th" else "eng"
    rows: list[Row] = []
    with bz2.open(path, "rt", encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, start=1):
            line = raw.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 2)
            if len(parts) != 3:
                raise RuntimeError(
                    f"{path}:{line_number}: expected 3 TSV fields"
                )
            sentence_id, language_code, text = parts
            if language_code != expected_code:
                raise RuntimeError(
                    f"{path}:{line_number}: expected {expected_code}, "
                    f"got {language_code}"
                )
            fingerprint = dedup_fingerprint(text)
            rows.append(
                Row(
                    sentence_id=sentence_id,
                    language=language,
                    text=text,
                    fingerprint=fingerprint,
                    partition=assign_partition(
                        SOURCE_ID,
                        fingerprint,
                    ),
                )
            )
    return rows


def deduplicate(rows: list[Row]) -> tuple[list[Row], int]:
    chosen: dict[str, Row] = {}
    duplicates = 0
    for row in sorted(rows, key=lambda item: (item.fingerprint, item.record_id)):
        if row.fingerprint in chosen:
            duplicates += 1
            continue
        chosen[row.fingerprint] = row
    return list(chosen.values()), duplicates


def deterministic_rank(row: Row) -> bytes:
    return hashlib.sha256(row.record_id.encode("utf-8")).digest()


def balance_by_utf8_bytes(
    thai: list[Row],
    english: list[Row],
) -> tuple[list[Row], dict[str, int]]:
    thai_total = sum(row.utf8_bytes for row in thai)
    english_total = sum(row.utf8_bytes for row in english)
    target = min(thai_total, english_total)

    def take(rows: list[Row], target_bytes: int) -> list[Row]:
        selected: list[Row] = []
        total = 0
        for row in sorted(rows, key=deterministic_rank):
            if total >= target_bytes:
                break
            selected.append(row)
            total += row.utf8_bytes
        return selected

    thai_selected = (
        thai if thai_total <= target else take(thai, target)
    )
    english_selected = (
        english if english_total <= target else take(english, target)
    )
    selected = sorted(
        thai_selected + english_selected,
        key=lambda row: row.record_id,
    )
    return selected, {
        "target_utf8_bytes_per_language": target,
        "thai_utf8_bytes": sum(row.utf8_bytes for row in thai_selected),
        "english_utf8_bytes": sum(
            row.utf8_bytes for row in english_selected
        ),
        "thai_records": len(thai_selected),
        "english_records": len(english_selected),
    }


def write_jsonl(path: Path, rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            payload = {
                "record_id": row.record_id,
                "text": row.text,
                "language": row.language,
                "source_id": SOURCE_ID,
                "license": LICENSE,
                "partition": row.partition,
                "source_group_id": row.fingerprint,
                "source_sentence_id": row.sentence_id,
                "dedup_fingerprint": row.fingerprint,
                "source_url": "https://tatoeba.org/en/downloads",
            }
            stream.write(json.dumps(payload, ensure_ascii=False))
            stream.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--thai", required=True)
    parser.add_argument("--english", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    thai_path = Path(args.thai)
    english_path = Path(args.english)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    raw_thai = load_tatoeba(thai_path, "th")
    raw_english = load_tatoeba(english_path, "en")
    thai, thai_duplicates = deduplicate(raw_thai)
    english, english_duplicates = deduplicate(raw_english)

    by_partition_language: dict[str, dict[str, list[Row]]] = {
        partition: {"th": [], "en": []}
        for partition in ("train", "validation", "holdout")
    }
    for row in thai + english:
        by_partition_language[row.partition][row.language].append(row)

    fit, fit_balance = balance_by_utf8_bytes(
        by_partition_language["train"]["th"],
        by_partition_language["train"]["en"],
    )
    validation, validation_balance = balance_by_utf8_bytes(
        by_partition_language["validation"]["th"],
        by_partition_language["validation"]["en"],
    )

    # The final holdout text is deliberately not materialized by this pilot.
    # Only aggregate counts are recorded so tokenizer development cannot
    # accidentally consume it.
    write_jsonl(output / "tokenizer_fit.jsonl", fit)
    write_jsonl(output / "tokenizer_validation.jsonl", validation)

    summary = {
        "schema_version": 1,
        "source_id": SOURCE_ID,
        "snapshot_date": "2026-10-03",
        "license": LICENSE,
        "license_url": LICENSE_URL,
        "raw_artifacts": {
            "thai": {
                "path": str(thai_path),
                "sha256": sha256_file(thai_path),
                "raw_records": len(raw_thai),
                "deduplicated_records": len(thai),
                "duplicates_removed": thai_duplicates,
            },
            "english": {
                "path": str(english_path),
                "sha256": sha256_file(english_path),
                "raw_records": len(raw_english),
                "deduplicated_records": len(english),
                "duplicates_removed": english_duplicates,
            },
        },
        "partition_counts": {
            partition: {
                language: len(rows)
                for language, rows in language_rows.items()
            }
            for partition, language_rows in by_partition_language.items()
        },
        "fit_materialization": fit_balance,
        "validation_materialization": validation_balance,
        "holdout_materialized": False,
        "split_seed": "flywirellm-l002-split-v1",
        "split_group": "dedup_fingerprint",
        "fit_sha256": sha256_file(output / "tokenizer_fit.jsonl"),
        "validation_sha256": sha256_file(
            output / "tokenizer_validation.jsonl"
        ),
    }
    (output / "pilot_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print(
        "raw_records="
        f"th:{len(raw_thai)},en:{len(raw_english)}"
    )
    print(
        "deduplicated_records="
        f"th:{len(thai)},en:{len(english)}"
    )
    print(
        "fit_records="
        f"th:{fit_balance['thai_records']},"
        f"en:{fit_balance['english_records']}"
    )
    print(
        "validation_records="
        f"th:{validation_balance['thai_records']},"
        f"en:{validation_balance['english_records']}"
    )
    print(
        "holdout_records="
        f"th:{len(by_partition_language['holdout']['th'])},"
        f"en:{len(by_partition_language['holdout']['en'])}"
    )
    print(f"output={output}")


if __name__ == "__main__":
    main()
