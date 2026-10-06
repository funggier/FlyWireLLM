from __future__ import annotations

import json
from pathlib import Path

import pytest

from flywire_llm.artifacts import load_artifact_ledger
from flywire_llm.data_contract import load_corpus_inventory
from flywire_llm.manifest_inputs import (
    ManifestInputError,
    load_global_manifest_inputs,
)
from flywire_llm.rights import load_rights_policy


ROOT = Path(__file__).resolve().parents[1]


def _context():
    inventory = load_corpus_inventory(
        ROOT / "configs" / "corpus-sources-l002.json"
    )
    rights = load_rights_policy(
        ROOT / "configs" / "rights-lanes-l003.json",
        expected_source_ids={
            source.source_id for source in inventory.sources
        },
    )
    ledger = load_artifact_ledger(
        ROOT / "configs" / "source-artifacts-l003.json",
        rights_policy=rights,
    )
    return rights, ledger


def test_global_manifest_input_plan_pins_all_five_screened_corpora():
    rights, ledger = _context()
    plan = load_global_manifest_inputs(
        ROOT / "configs" / "global-manifest-inputs-l003.json",
        repo_root=ROOT,
        rights_policy=rights,
        artifact_ledger=ledger,
    )
    assert plan.technical_classifier == "technical-heuristic-v5"
    assert plan.near_duplicate_hamming_distance == 3
    assert plan.pretraining_authorized is False
    assert len(plan.inputs) == 5
    assert {item.input_id for item in plan.inputs} == {
        "tatoeba-th-screened-v1",
        "tatoeba-en-screened-v1",
        "thwiki-sample-v1-screened-v1",
        "fineweb2-thai-train-005-00002-sample1000000-v1",
        "fineweb-english-10bt-014-00000-sample600000-v1",
    }


def test_global_manifest_inputs_preserve_rights_lanes():
    rights, ledger = _context()
    plan = load_global_manifest_inputs(
        ROOT / "configs" / "global-manifest-inputs-l003.json",
        repo_root=ROOT,
        rights_policy=rights,
        artifact_ledger=ledger,
    )
    by_id = {item.input_id: item for item in plan.inputs}
    assert by_id["tatoeba-th-screened-v1"].rights_lane == "release_safe"
    assert by_id["tatoeba-en-screened-v1"].rights_lane == "release_safe"
    assert by_id["thwiki-sample-v1-screened-v1"].rights_lane == "research_only"
    assert by_id[
        "fineweb2-thai-train-005-00002-sample1000000-v1"
    ].rights_lane == "research_only"
    assert by_id[
        "fineweb-english-10bt-014-00000-sample600000-v1"
    ].rights_lane == "research_only"


def test_global_manifest_input_summary_hash_regression_fails_closed(tmp_path):
    payload = json.loads(
        (
            ROOT / "configs" / "global-manifest-inputs-l003.json"
        ).read_text(encoding="utf-8")
    )
    payload["inputs"][0]["summary_sha256"] = "0" * 64
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    rights, ledger = _context()
    with pytest.raises(ManifestInputError, match="summary SHA-256 mismatch"):
        load_global_manifest_inputs(
            path,
            repo_root=ROOT,
            rights_policy=rights,
            artifact_ledger=ledger,
        )


def test_expanded_global_manifest_input_plan_uses_six_unique_artifacts():
    rights, ledger = _context()
    plan = load_global_manifest_inputs(
        ROOT / "configs" / "global-manifest-inputs-l003-v2.json",
        repo_root=ROOT,
        rights_policy=rights,
        artifact_ledger=ledger,
    )
    assert len(plan.inputs) == 6
    assert len({item.artifact_id for item in plan.inputs}) == 6
    assert {item.input_id for item in plan.inputs} == {
        "tatoeba-th-screened-v1",
        "tatoeba-en-screened-v1",
        "thwiki-sample-v1-screened-v1",
        "fineweb2-thai-train-005-00002-sample1000000-v1",
        "fineweb-english-10bt-013-00000-sample550000-v1",
        "fineweb-english-10bt-014-00000-full-v1",
    }


def test_duplicate_artifact_in_global_manifest_plan_fails_closed(tmp_path):
    payload = json.loads(
        (ROOT / "configs" / "global-manifest-inputs-l003.json")
        .read_text(encoding="utf-8")
    )
    duplicate = dict(payload["inputs"][-1])
    duplicate["input_id"] = "duplicate-artifact-input"
    payload["inputs"].append(duplicate)
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    rights, ledger = _context()
    with pytest.raises(ManifestInputError, match="duplicate artifact_id"):
        load_global_manifest_inputs(
            path,
            repo_root=ROOT,
            rights_policy=rights,
            artifact_ledger=ledger,
        )


def test_global_manifest_input_lane_regression_fails_closed(tmp_path):
    payload = json.loads(
        (
            ROOT / "configs" / "global-manifest-inputs-l003.json"
        ).read_text(encoding="utf-8")
    )
    payload["inputs"][0]["rights_lane"] = "research_only"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    rights, ledger = _context()
    with pytest.raises(ManifestInputError, match="rights lane mismatch"):
        load_global_manifest_inputs(
            path,
            repo_root=ROOT,
            rights_policy=rights,
            artifact_ledger=ledger,
        )
