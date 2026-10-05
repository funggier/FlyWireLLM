from __future__ import annotations

import bz2
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from .data_contract import assign_partition, dedup_fingerprint


THWIKI_SOURCE_ID = "wikimedia-thwiki-2026-10-01"
TATOEBA_SOURCE_ID = "tatoeba-sentences-2026-10-03"


@dataclass(frozen=True)
class SourceRow:
    record_id: str
    source_id: str
    source_group_id: str
    text: str
    language: str
    license: str
    partition: str
    source_url: str

    @property
    def utf8_bytes(self) -> int:
        return len(self.text.encode("utf-8"))


def deterministic_rank(row: SourceRow) -> bytes:
    return hashlib.sha256(
        f"{row.source_id}\0{row.record_id}".encode("utf-8")
    ).digest()


def clean_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def iter_wikiextractor_json(root: str | Path) -> Iterator[SourceRow]:
    root = Path(root)
    for path in sorted(root.rglob("wiki_*")):
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as stream:
            for line_number, raw in enumerate(stream, start=1):
                if not raw.strip():
                    continue
                payload = json.loads(raw)
                if not isinstance(payload, dict):
                    raise RuntimeError(
                        f"{path}:{line_number}: expected JSON object"
                    )
                page_id = str(payload.get("id", "")).strip()
                revision_id = str(payload.get("revid", "")).strip()
                text = clean_text(str(payload.get("text", "")))
                if not page_id or not text:
                    continue
                fingerprint = dedup_fingerprint(text)
                yield SourceRow(
                    record_id=(
                        f"THWIKI-{page_id}-{revision_id}"
                        if revision_id
                        else f"THWIKI-{page_id}"
                    ),
                    source_id=THWIKI_SOURCE_ID,
                    source_group_id=fingerprint,
                    text=text,
                    language="th",
                    license="CC BY-SA 4.0 / GFDL source text",
                    partition=assign_partition(
                        THWIKI_SOURCE_ID,
                        fingerprint,
                    ),
                    source_url=str(payload.get("url", "")).strip()
                    or "https://th.wikipedia.org/",
                )


def iter_tatoeba_english(path: str | Path) -> Iterator[SourceRow]:
    path = Path(path)
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
            if language_code != "eng":
                raise RuntimeError(
                    f"{path}:{line_number}: expected eng"
                )
            text = clean_text(text)
            if not text:
                continue
            fingerprint = dedup_fingerprint(text)
            yield SourceRow(
                record_id=f"TATOEBA-EN-{sentence_id}",
                source_id=TATOEBA_SOURCE_ID,
                source_group_id=fingerprint,
                text=text,
                language="en",
                license="CC BY 2.0 FR",
                partition=assign_partition(
                    TATOEBA_SOURCE_ID,
                    fingerprint,
                ),
                source_url="https://tatoeba.org/en/downloads",
            )


def deduplicate(rows: Iterable[SourceRow]) -> tuple[list[SourceRow], int]:
    chosen: dict[tuple[str, str], SourceRow] = {}
    duplicates = 0
    for row in rows:
        key = (row.source_id, row.source_group_id)
        current = chosen.get(key)
        if current is None or row.record_id < current.record_id:
            if current is not None:
                duplicates += 1
            chosen[key] = row
        else:
            duplicates += 1
    return list(chosen.values()), duplicates


def select_bytes(
    rows: Iterable[SourceRow],
    target_bytes: int,
) -> list[SourceRow]:
    if target_bytes <= 0:
        raise ValueError("target_bytes must be positive")
    selected: list[SourceRow] = []
    total = 0
    for row in sorted(rows, key=deterministic_rank):
        if total >= target_bytes:
            break
        selected.append(row)
        total += row.utf8_bytes
    return selected


def partition_rows(
    rows: Iterable[SourceRow],
    partition: str,
) -> list[SourceRow]:
    return [row for row in rows if row.partition == partition]


def write_jsonl(
    path: str | Path,
    rows: Iterable[SourceRow],
) -> tuple[int, int, str]:
    path = Path(path)
    count = 0
    total_bytes = 0
    digest = hashlib.sha256()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        for row in sorted(rows, key=lambda item: item.record_id):
            payload = {
                "record_id": row.record_id,
                "text": row.text,
                "language": row.language,
                "source_id": row.source_id,
                "license": row.license,
                "partition": row.partition,
                "source_group_id": row.source_group_id,
                "source_url": row.source_url,
            }
            encoded = (
                json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            ).encode("utf-8")
            stream.write(encoded)
            digest.update(encoded)
            count += 1
            total_bytes += row.utf8_bytes
    return count, total_bytes, digest.hexdigest()


def rows_stats(rows: Iterable[SourceRow]) -> dict[str, int]:
    rows = list(rows)
    return {
        "records": len(rows),
        "utf8_bytes": sum(row.utf8_bytes for row in rows),
    }
