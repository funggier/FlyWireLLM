from __future__ import annotations

import pytest

from flywire_llm.near_duplicate import (
    NearDuplicateIndexError,
    SQLiteSimhashIndex,
)
from flywire_llm.quality import simhash64


def test_sqlite_simhash_index_adds_and_finds_exact_match(tmp_path):
    path = tmp_path / "near.sqlite"
    text = "the quick brown fox jumps over the lazy dog"
    fingerprint = simhash64(text)
    with SQLiteSimhashIndex(path) as index:
        index.add("R1", fingerprint)
        matches = index.find_matches(fingerprint)
        assert len(matches) == 1
        assert matches[0].record_id == "R1"
        assert matches[0].hamming_distance == 0
        assert index.record_count == 1


def test_sqlite_simhash_index_guarantees_three_bit_candidate_capture(tmp_path):
    base = 0x123456789ABCDEF0
    changed = base ^ (1 << 0) ^ (1 << 16) ^ (1 << 32)
    with SQLiteSimhashIndex(tmp_path / "near.sqlite") as index:
        index.add("BASE", base)
        matches = index.find_matches(changed, max_hamming_distance=3)
        assert [(m.record_id, m.hamming_distance) for m in matches] == [
            ("BASE", 3)
        ]


def test_add_if_unique_does_not_insert_near_duplicate(tmp_path):
    base = 0x123456789ABCDEF0
    near = base ^ 1
    far = 0xFFFFFFFFFFFFFFFF
    with SQLiteSimhashIndex(tmp_path / "near.sqlite") as index:
        inserted, matches = index.add_if_unique("A", base)
        assert inserted is True
        assert matches == ()

        inserted, matches = index.add_if_unique("B", near)
        assert inserted is False
        assert matches[0].record_id == "A"
        assert index.record_count == 1

        inserted, matches = index.add_if_unique("C", far)
        assert inserted is True
        assert matches == ()
        assert index.record_count == 2


def test_duplicate_record_id_fails_closed(tmp_path):
    with SQLiteSimhashIndex(tmp_path / "near.sqlite") as index:
        index.add("R1", 1)
        with pytest.raises(NearDuplicateIndexError, match="duplicate record_id"):
            index.add("R1", 2)


def test_invalid_fingerprint_is_rejected(tmp_path):
    with SQLiteSimhashIndex(tmp_path / "near.sqlite") as index:
        with pytest.raises(
            NearDuplicateIndexError,
            match="unsigned 64-bit",
        ):
            index.add("R1", -1)
