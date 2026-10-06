from __future__ import annotations

import json
from pathlib import Path

import pytest

from flywire_llm.calibration import (
    CalibrationArtifactError,
    assert_not_training_artifact,
    load_calibration_artifacts,
    verify_calibration_file,
)
from flywire_llm.data_contract import load_corpus_inventory
from flywire_llm.rights import load_rights_policy


ROOT = Path(__file__).resolve().parents[1]


def _policy():
    inventory = load_corpus_inventory(
        ROOT / "configs" / "corpus-sources-l002.json"
    )
    return load_rights_policy(
        ROOT / "configs" / "rights-lanes-l003.json",
        expected_source_ids={
            source.source_id for source in inventory.sources
        },
    )


def test_fineweb2_thai_test_shard_is_calibration_only():
    artifacts = load_calibration_artifacts(
        ROOT / "configs" / "calibration-artifacts-l003.json",
        rights_policy=_policy(),
    )
    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert artifact.calibration_id == "fineweb2-thai-test-calibration-v1"
    assert artifact.source_id == "fineweb2-thai"
    assert artifact.source_split == "test"
    assert artifact.rights_lane == "research_only"
    assert artifact.never_train is True
    assert artifact.never_validation is True
    assert artifact.never_holdout is True
    assert artifact.bytes == 51_285_472
    assert artifact.sha256 == (
        "34102d27b6068605a62a69e7fae8f7d3f4e67165720afd8fb685a2eb5e85ff1b"
    )


def test_calibration_artifact_can_never_be_materialized_to_model_partition():
    artifact = load_calibration_artifacts(
        ROOT / "configs" / "calibration-artifacts-l003.json",
        rights_policy=_policy(),
    )[0]
    with pytest.raises(
        CalibrationArtifactError,
        match="can never be materialized",
    ):
        assert_not_training_artifact(artifact)


def test_calibration_catalog_rejects_never_train_regression(tmp_path):
    payload = json.loads(
        (ROOT / "configs" / "calibration-artifacts-l003.json").read_text(
            encoding="utf-8"
        )
    )
    payload["artifacts"][0]["never_train"] = False
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        CalibrationArtifactError,
        match="never_train must be true",
    ):
        load_calibration_artifacts(path, rights_policy=_policy())


def test_calibration_file_hash_and_size_verification(tmp_path):
    content = b"parquet calibration fixture"
    path = tmp_path / "fixture.parquet"
    path.write_bytes(content)

    payload = json.loads(
        (ROOT / "configs" / "calibration-artifacts-l003.json").read_text(
            encoding="utf-8"
        )
    )
    import hashlib

    raw = payload["artifacts"][0]
    raw["bytes"] = len(content)
    raw["sha256"] = hashlib.sha256(content).hexdigest()
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps(payload), encoding="utf-8")
    artifact = load_calibration_artifacts(
        catalog,
        rights_policy=_policy(),
    )[0]
    verify_calibration_file(artifact, path)

    path.write_bytes(content + b"x")
    with pytest.raises(
        CalibrationArtifactError,
        match="byte size mismatch",
    ):
        verify_calibration_file(artifact, path)
