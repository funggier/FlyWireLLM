from __future__ import annotations

import argparse
import bz2
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from flywire_llm.data_contract import assign_partition, dedup_fingerprint
from flywire_llm.quality import screen_text
from flywire_llm.sentencepiece_tokenizer import SentencePieceTokenizer
from flywire_llm.tokenizer_corpus import iter_wikiextractor_json


TATOEBA_SPLIT_SOURCE = "tatoeba-sentences-2026-10-03"


@dataclass(frozen=True)
class AuditRow:
    source_id: str
    artifact_id: str
    rights_lane: str
    language: str
    partition: str
    record_id: str
    source_group_id: str
    text: str


def iter_tatoeba(
    path: str | Path,
    *,
    language: str,
    code: str,
    artifact_id: str,
) -> Iterator[AuditRow]:
    seen: set[str] = set()
    with bz2.open(path, "rt", encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, start=1):
            line = raw.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 2)
            if len(parts) != 3:
                raise RuntimeError(
                    f"{path}:{line_number}: expected three TSV fields"
                )
            sentence_id, observed_code, text = parts
            if observed_code != code:
                raise RuntimeError(
                    f"{path}:{line_number}: expected {code}, got "
                    f"{observed_code}"
                )
            fingerprint = dedup_fingerprint(text)
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            yield AuditRow(
                source_id="tatoeba-sentences",
                artifact_id=artifact_id,
                rights_lane="release_safe",
                language=language,
                partition=assign_partition(
                    TATOEBA_SPLIT_SOURCE,
                    fingerprint,
                ),
                record_id=f"TATOEBA-{code.upper()}-{sentence_id}",
                source_group_id=fingerprint,
                text=text,
            )


def iter_thwiki(path: str | Path) -> Iterator[AuditRow]:
    seen: set[str] = set()
    for row in iter_wikiextractor_json(path):
        if row.source_group_id in seen:
            continue
        seen.add(row.source_group_id)
        yield AuditRow(
            source_id="wikimedia-thwiki",
            artifact_id="thwiki-pages-articles-2026-10-01",
            rights_lane="research_only",
            language="th",
            partition=row.partition,
            record_id=row.record_id,
            source_group_id=row.source_group_id,
            text=row.text,
        )


def _new_stats() -> dict:
    return {
        "records": 0,
        "characters": 0,
        "utf8_bytes": 0,
        "tokens": 0,
        "status": {
            "accept": {"records": 0, "tokens": 0},
            "quarantine": {"records": 0, "tokens": 0},
            "reject": {"records": 0, "tokens": 0},
        },
        "findings": Counter(),
    }


def audit_rows(
    rows: Iterable[AuditRow],
    *,
    tokenizer: SentencePieceTokenizer,
) -> tuple[dict, dict]:
    overall = _new_stats()
    dimensions: dict[str, dict] = {}

    for row in rows:
        screening = screen_text(row.text)
        if screening.disposition.value == "accept":
            status = "accept"
        elif screening.disposition.value == "review":
            status = "quarantine"
        else:
            status = "reject"
        token_count = len(tokenizer.encode(row.text))

        keys = [
            "overall",
            f"lane:{row.rights_lane}",
            f"source:{row.source_id}",
            f"language:{row.language}",
            f"partition:{row.partition}",
            (
                f"lane:{row.rights_lane}|partition:{row.partition}"
                f"|language:{row.language}"
            ),
        ]
        for key in keys:
            stats = overall if key == "overall" else dimensions.setdefault(
                key, _new_stats()
            )
            stats["records"] += 1
            stats["characters"] += screening.characters
            stats["utf8_bytes"] += screening.utf8_bytes
            stats["tokens"] += token_count
            stats["status"][status]["records"] += 1
            stats["status"][status]["tokens"] += token_count
            for finding in screening.findings:
                stats["findings"][finding.code] += 1

    def finalize(stats: dict) -> dict:
        findings = dict(sorted(stats["findings"].items()))
        result = dict(stats)
        result["findings"] = findings
        total_tokens = result["tokens"]
        total_records = result["records"]
        result["accepted_token_fraction"] = (
            result["status"]["accept"]["tokens"] / total_tokens
            if total_tokens
            else 0.0
        )
        result["accepted_record_fraction"] = (
            result["status"]["accept"]["records"] / total_records
            if total_records
            else 0.0
        )
        return result

    return finalize(overall), {
        key: finalize(value)
        for key, value in sorted(dimensions.items())
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tatoeba-thai", required=True)
    parser.add_argument("--tatoeba-english", required=True)
    parser.add_argument("--thwiki-dir", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = SentencePieceTokenizer(args.model)

    def all_rows() -> Iterator[AuditRow]:
        yield from iter_tatoeba(
            args.tatoeba_thai,
            language="th",
            code="tha",
            artifact_id="tatoeba-tha-sentences-2026-10-03",
        )
        yield from iter_tatoeba(
            args.tatoeba_english,
            language="en",
            code="eng",
            artifact_id="tatoeba-eng-sentences-2026-10-03",
        )
        yield from iter_thwiki(args.thwiki_dir)

    overall, dimensions = audit_rows(all_rows(), tokenizer=tokenizer)

    release_train = dimensions[
        "lane:release_safe|partition:train|language:en"
    ]["status"]["accept"]["tokens"] + dimensions[
        "lane:release_safe|partition:train|language:th"
    ]["status"]["accept"]["tokens"]
    research_train = dimensions[
        "lane:research_only|partition:train|language:th"
    ]["status"]["accept"]["tokens"]
    payload = {
        "schema_version": 1,
        "stage": "L003",
        "tokenizer": "base50m-unigram-32000-v1",
        "screening_policy": {
            "accept": "eligible for the next manifest gate",
            "review": "quarantine and exclude from automatic training",
            "reject": "exclude from training",
            "raw_flagged_text_persisted_in_report": False,
        },
        "overall": overall,
        "dimensions": dimensions,
        "screened_train_budget": {
            "release_safe_tokens": release_train,
            "release_safe_gap_to_500m": max(
                0, 500_000_000 - release_train
            ),
            "research_only_tokens": research_train,
            "combined_research_eligible_tokens": (
                release_train + research_train
            ),
            "combined_research_gap_to_500m": max(
                0,
                500_000_000 - release_train - research_train,
            ),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "overall": payload["overall"],
                "screened_train_budget": payload[
                    "screened_train_budget"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
