from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


class CorpusContractError(ValueError):
    pass


_ALLOWED_PARTITIONS = {"train", "validation", "holdout"}
_ALLOWED_LANGUAGES = {"th", "en", "mixed", "other"}


@dataclass(frozen=True)
class TextRecord:
    record_id: str
    text: str
    language: str
    source_id: str
    license: str
    partition: str


def iter_jsonl_records(path: str | Path) -> Iterator[TextRecord]:
    path = Path(path)
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CorpusContractError(
                    f"{path}:{line_number}: invalid JSON"
                ) from exc
            if not isinstance(payload, dict):
                raise CorpusContractError(
                    f"{path}:{line_number}: record must be a JSON object"
                )

            required = {
                "record_id",
                "text",
                "language",
                "source_id",
                "license",
                "partition",
            }
            missing = sorted(required - set(payload))
            if missing:
                raise CorpusContractError(
                    f"{path}:{line_number}: missing fields: {', '.join(missing)}"
                )

            values: dict[str, str] = {}
            for field in required:
                value = payload[field]
                if not isinstance(value, str) or not value.strip():
                    raise CorpusContractError(
                        f"{path}:{line_number}: {field} must be a non-empty string"
                    )
                values[field] = value.strip()

            record_id = values["record_id"]
            if record_id in seen:
                raise CorpusContractError(
                    f"{path}:{line_number}: duplicate record_id {record_id!r}"
                )
            seen.add(record_id)

            if values["partition"] not in _ALLOWED_PARTITIONS:
                raise CorpusContractError(
                    f"{path}:{line_number}: unsupported partition "
                    f"{values['partition']!r}"
                )
            if values["language"] not in _ALLOWED_LANGUAGES:
                raise CorpusContractError(
                    f"{path}:{line_number}: unsupported language "
                    f"{values['language']!r}"
                )

            yield TextRecord(
                record_id=record_id,
                text=payload["text"],
                language=values["language"],
                source_id=values["source_id"],
                license=values["license"],
                partition=values["partition"],
            )


def load_partition(path: str | Path, partition: str) -> list[TextRecord]:
    if partition not in _ALLOWED_PARTITIONS:
        raise CorpusContractError(f"unsupported partition {partition!r}")
    return [
        record
        for record in iter_jsonl_records(path)
        if record.partition == partition
    ]
