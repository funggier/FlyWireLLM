from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import scripts.materialize_global_manifest_l003 as runner


def test_verify_tokenizer_model_requires_frozen_sha256(tmp_path):
    model = tmp_path / "tokenizer.model"
    model.write_bytes(b"frozen-tokenizer")
    digest = hashlib.sha256(b"frozen-tokenizer").hexdigest()
    assert runner.verify_tokenizer_model(model, expected_sha256=digest) == digest

    with pytest.raises(RuntimeError, match="tokenizer model SHA-256 mismatch"):
        runner.verify_tokenizer_model(model, expected_sha256="0" * 64)


def test_finalize_report_records_external_storage_and_keeps_training_blocked(
    tmp_path,
):
    registry = tmp_path / "inputs.json"
    registry.write_text(
        json.dumps({"schema_version": 1}) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    raw = {
        "schema_version": 1,
        "stage": "L003",
        "train_capacity": {
            "release_safe": {
                "general_thai": 1,
                "general_english": 1,
                "technical_scientific_code": 0,
                "flywire_domain": 0,
            },
            "research_only": {
                "general_thai": 200_000_000,
                "general_english": 200_000_000,
                "technical_scientific_code": 100_000_000,
                "flywire_domain": 0,
            },
        },
        "decisions": {
            "path": "decisions.jsonl",
            "bytes": 123,
            "sha256": "a" * 64,
            "contains_raw_text": False,
        },
        "pretraining_authorized": False,
    }
    mixture = {
        "release_safe": {"ready": False},
        "research_only": {"ready": True},
        "pretraining_authorized": False,
    }
    final = runner.finalize_report(
        raw,
        registry_path=registry,
        registry_display_path="configs/global-manifest-inputs-l003.json",
        output_storage=(
            "external://FlyWireLLM-data/L003/Global/global-manifest-v1"
        ),
        tokenizer_sha256="b" * 64,
        mixture=mixture,
    )
    assert final["global_cross_source_dedup_applied"] is True
    assert final["input_registry"]["path"] == (
        "configs/global-manifest-inputs-l003.json"
    )
    assert final["decisions"]["storage"].endswith("/decisions.jsonl")
    assert final["mixture"]["research_only"]["ready"] is True
    assert final["pretraining_authorized"] is False
    assert "T:\\" not in json.dumps(final)
