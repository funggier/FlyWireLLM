from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import scripts.freeze_pretraining_l003 as freezer


TOKENIZER_SHA = (
    "998bc75f058e554d4f50b6cbc77c52898e3446e516c3b25a126b99b113513818"
)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, payload: dict) -> str:
    encoded = (json.dumps(payload, indent=2) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return _sha(encoded)


def _report(tmp_path: Path, *, ready: bool = True) -> Path:
    registry_path = tmp_path / "configs" / "registry.json"
    registry_sha = _write_json(
        registry_path,
        {
            "schema_version": 1,
            "stage": "L003",
            "tokenizer": "base50m-unigram-32000-v1",
            "technical_classifier": "technical-heuristic-v5",
            "near_duplicate_hamming_distance": 3,
            "central_global_manifest_authorization": True,
            "pretraining_authorized": False,
            "inputs": [],
        },
    )
    report = {
        "schema_version": 1,
        "stage": "L003",
        "global_cross_source_dedup_applied": True,
        "pretraining_authorized": False,
        "input_registry": {
            "path": "configs/registry.json",
            "sha256": registry_sha,
        },
        "tokenizer_model": {
            "candidate_id": "base50m-unigram-32000-v1",
            "sha256": TOKENIZER_SHA,
        },
        "decisions": {
            "sha256": "a" * 64,
            "contains_raw_text": False,
        },
        "mixture": {
            "research_only": {
                "ready": ready,
                "thai_gap": 0,
                "english_gap": 0,
                "technical_gap": 0,
                "total_gap": 0,
            }
        },
    }
    path = tmp_path / "results" / "report.json"
    _write_json(path, report)
    return path


def test_build_freeze_pins_report_registry_decisions_and_tokenizer(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(freezer, "ROOT", tmp_path)
    report = _report(tmp_path)
    payload = freezer.build_freeze(report)

    assert payload["checkpoint_lane"] == "research_only"
    assert payload["optimizer_steps_completed"] == 0
    assert payload["source_manifest"]["path"] == "configs/registry.json"
    assert payload["source_manifest"]["frozen"] is True
    assert payload["global_manifest_report"]["path"] == "results/report.json"
    assert payload["global_manifest_report"]["sha256"] == freezer.sha256_file(
        report
    )
    assert payload["evaluation_manifest"]["decisions_sha256"] == "a" * 64
    assert payload["tokenizer"]["model_sha256"] == TOKENIZER_SHA
    assert payload["flywiremodel_automatic_export_allowed"] is False
    assert payload["public_release_eligibility"] == "not_qualified"
    assert payload["pretraining_authorized"] is True


def test_build_freeze_rejects_unready_research_mixture(tmp_path, monkeypatch):
    monkeypatch.setattr(freezer, "ROOT", tmp_path)
    report = _report(tmp_path, ready=False)
    with pytest.raises(RuntimeError, match="research_only mixture is not ready"):
        freezer.build_freeze(report)
