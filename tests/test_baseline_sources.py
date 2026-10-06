from __future__ import annotations

import bz2
from pathlib import Path

from flywire_llm.baseline_sources import (
    TATOEBA_SPLIT_SOURCE,
    iter_tatoeba,
)
from flywire_llm.data_contract import assign_partition, dedup_fingerprint


def test_tatoeba_baseline_rows_deduplicate_before_partition(tmp_path):
    path = tmp_path / "tha.tsv.bz2"
    with bz2.open(path, "wt", encoding="utf-8") as stream:
        stream.write("1\ttha\tภาษาไทยตัวอย่าง\n")
        stream.write("2\ttha\tภาษาไทยตัวอย่าง\n")
        stream.write("3\ttha\tข้อความอีกชุด\n")

    rows = list(
        iter_tatoeba(
            path,
            language="th",
            code="tha",
            artifact_id="fixture-th",
        )
    )
    assert len(rows) == 2
    assert rows[0].record_id == "TATOEBA-THA-1"
    assert rows[0].rights_lane == "release_safe"
    assert rows[0].category == "general_thai"
    fingerprint = dedup_fingerprint(rows[0].text)
    assert rows[0].source_group_id == fingerprint
    assert rows[0].partition == assign_partition(
        TATOEBA_SPLIT_SOURCE,
        fingerprint,
    )
