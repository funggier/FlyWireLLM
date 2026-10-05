from __future__ import annotations

import json
from pathlib import Path

import pytest

from flywire_llm.data_contract import (
    assign_partition,
    dedup_fingerprint,
)

from flywire_llm import tokenizer_corpus as builder


def test_wikiextractor_parser_preserves_page_text_and_split(tmp_path):
    root = tmp_path / "wiki"
    chunk = root / "AA" / "wiki_00"
    chunk.parent.mkdir(parents=True)
    rows = [
        {
            "id": "10",
            "revid": "101",
            "url": "https://th.wikipedia.org/wiki/A",
            "title": "A",
            "text": "ภาษาไทยตัวอย่างหนึ่ง",
        },
        {
            "id": "11",
            "revid": "102",
            "url": "https://th.wikipedia.org/wiki/B",
            "title": "B",
            "text": "ภาษาไทยตัวอย่างสอง",
        },
    ]
    chunk.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )
    parsed = list(builder.iter_wikiextractor_json(root))
    assert [row.record_id for row in parsed] == [
        "THWIKI-10-101",
        "THWIKI-11-102",
    ]
    for row in parsed:
        assert row.language == "th"
        assert row.partition == assign_partition(
            builder.THWIKI_SOURCE_ID,
            dedup_fingerprint(row.text),
        )


def test_wikipedia_exact_duplicate_is_deduplicated_and_grouped(tmp_path):
    root = tmp_path / "wiki"
    chunk = root / "AA" / "wiki_00"
    chunk.parent.mkdir(parents=True)
    text = "เนื้อหาเดียวกัน"
    rows = [
        {
            "id": "20",
            "revid": "201",
            "url": "https://example/A",
            "title": "A",
            "text": text,
        },
        {
            "id": "21",
            "revid": "202",
            "url": "https://example/B",
            "title": "B",
            "text": text,
        },
    ]
    chunk.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )
    deduped, duplicates = builder.deduplicate(
        builder.iter_wikiextractor_json(root)
    )
    assert duplicates == 1
    assert len(deduped) == 1
    assert deduped[0].record_id == "THWIKI-20-201"


def test_tatoeba_english_parser_uses_content_group_split(tmp_path):
    path = tmp_path / "eng.tsv.bz2"
    import bz2

    with bz2.open(path, "wt", encoding="utf-8") as stream:
        stream.write("1\teng\tHello world.\n")
        stream.write("2\teng\tAnother sentence.\n")
    rows = list(builder.iter_tatoeba_english(path))
    assert len(rows) == 2
    for row in rows:
        assert row.partition == assign_partition(
            builder.TATOEBA_SOURCE_ID,
            dedup_fingerprint(row.text),
        )


def test_byte_selector_is_deterministic_and_stops_after_target():
    rows = [
        builder.SourceRow(
            record_id=f"R{i}",
            source_id="S",
            source_group_id=f"G{i}",
            text=("x" * (i + 1)),
            language="en",
            license="CC0",
            partition="train",
            source_url="https://example.org",
        )
        for i in range(20)
    ]
    a = builder.select_bytes(rows, 30)
    b = builder.select_bytes(list(reversed(rows)), 30)
    assert [row.record_id for row in a] == [row.record_id for row in b]
    assert sum(row.utf8_bytes for row in a) >= 30
