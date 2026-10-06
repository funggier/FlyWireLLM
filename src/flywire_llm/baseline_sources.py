from __future__ import annotations

import bz2
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .data_contract import assign_partition, dedup_fingerprint
from .tokenizer_corpus import iter_wikiextractor_json


TATOEBA_SPLIT_SOURCE = "tatoeba-sentences-2026-10-03"


@dataclass(frozen=True)
class BaselineSourceRow:
    record_id: str
    text: str
    language: str
    category: str
    source_id: str
    artifact_id: str
    rights_lane: str
    license: str
    source_group_id: str
    partition: str


def iter_tatoeba(
    path: str | Path,
    *,
    language: str,
    code: str,
    artifact_id: str,
) -> Iterator[BaselineSourceRow]:
    if language not in {"th", "en"}:
        raise ValueError("language must be th or en")
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
            yield BaselineSourceRow(
                record_id=f"TATOEBA-{code.upper()}-{sentence_id}",
                text=text,
                language=language,
                category=(
                    "general_thai"
                    if language == "th"
                    else "general_english"
                ),
                source_id="tatoeba-sentences",
                artifact_id=artifact_id,
                rights_lane="release_safe",
                license="CC BY 2.0 FR",
                source_group_id=fingerprint,
                partition=assign_partition(
                    TATOEBA_SPLIT_SOURCE,
                    fingerprint,
                ),
            )


def iter_thwiki_sample(
    root: str | Path,
) -> Iterator[BaselineSourceRow]:
    seen: set[str] = set()
    for row in iter_wikiextractor_json(root):
        if row.source_group_id in seen:
            continue
        seen.add(row.source_group_id)
        yield BaselineSourceRow(
            record_id=row.record_id,
            text=row.text,
            language="th",
            category="general_thai",
            source_id="wikimedia-thwiki",
            artifact_id="thwiki-pages-articles-2026-10-01",
            rights_lane="research_only",
            license="CC BY-SA 4.0 / GFDL source text",
            source_group_id=row.source_group_id,
            partition=row.partition,
        )
