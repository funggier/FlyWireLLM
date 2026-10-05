from __future__ import annotations

import json
from pathlib import Path

import pytest

from flywire_llm.corpus import (
    CorpusContractError,
    iter_jsonl_records,
    load_partition,
)


def test_example_corpus_has_thai_english_and_frozen_partitions():
    path = Path("examples/corpus/sample.jsonl")
    records = list(iter_jsonl_records(path))
    assert {record.language for record in records} == {"th", "en", "mixed"}
    assert {record.partition for record in records} == {
        "train",
        "validation",
        "holdout",
    }
    assert len(load_partition(path, "train")) == 2
    assert len(load_partition(path, "validation")) == 1
    assert len(load_partition(path, "holdout")) == 1


def test_duplicate_record_id_is_rejected(tmp_path: Path):
    path = tmp_path / "duplicate.jsonl"
    row = {
        "record_id": "R1",
        "text": "hello",
        "language": "en",
        "source_id": "synthetic",
        "license": "CC0-1.0",
        "partition": "train",
    }
    path.write_text(
        json.dumps(row) + "\n" + json.dumps(row) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(CorpusContractError, match="duplicate"):
        list(iter_jsonl_records(path))


def test_missing_license_is_rejected(tmp_path: Path):
    path = tmp_path / "bad.jsonl"
    row = {
        "record_id": "R1",
        "text": "สวัสดี",
        "language": "th",
        "source_id": "synthetic",
        "partition": "train",
    }
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(CorpusContractError, match="missing fields"):
        list(iter_jsonl_records(path))
