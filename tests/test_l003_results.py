from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads(
        (ROOT / "results" / "l003" / name).read_text(encoding="utf-8")
    )


def test_acquired_quality_audit_is_post_dedup_and_safe_to_commit():
    result = _load("acquired-quality-audit.json")
    assert result["schema_version"] == 1
    assert result["screening_policy"][
        "raw_flagged_text_persisted_in_report"
    ] is False
    assert result["overall"]["records"] == 2_106_635
    assert result["overall"]["tokens"] == 58_982_391
    assert result["overall"]["status"]["accept"]["tokens"] == 58_840_212
    assert result["overall"]["status"]["quarantine"]["records"] == 26
    assert result["overall"]["status"]["reject"]["records"] == 161
    assert result["screened_train_budget"] == {
        "release_safe_tokens": 20_061_246,
        "release_safe_gap_to_500m": 479_938_754,
        "research_only_tokens": 38_172_496,
        "combined_research_eligible_tokens": 58_233_742,
        "combined_research_gap_to_500m": 441_766_258,
    }


def test_fineweb2_thai_test_calibration_never_authorizes_training():
    result = _load("fineweb2-thai-calibration.json")
    assert result["calibration_only"] is True
    assert result["training_authorized"] is False
    assert result["source_split"] == "test"
    assert result["rows"]["total"] == 24_037
    assert result["content"]["tokens"] == 20_573_344
    assert result["screening"]["status"]["accept"]["tokens"] == 19_837_629
    assert result["screening"]["accepted_token_fraction"] == (
        0.9642394060975211
    )
    assert result["deduplication"]["exact_duplicate_records"] == 0
    assert result["deduplication"][
        "near_duplicate_records_after_exact"
    ] == 0
    assert result["density"]["accepted_tokens_per_compressed_gib"] == (
        415_331_888.65445745
    )
