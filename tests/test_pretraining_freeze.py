from __future__ import annotations

import hashlib
import json

import pytest

from flywire_llm.pretraining_freeze import (
    PretrainingFreezeError,
    load_pretraining_freeze,
)


TOKENIZER_SHA = (
    "998bc75f058e554d4f50b6cbc77c52898e3446e516c3b25a126b99b113513818"
)
DECISIONS_SHA = "a" * 64


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2) + "\n"
    path.write_text(encoded, encoding="utf-8", newline="\n")
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _fixture(tmp_path):
    registry = {
        "schema_version": 1,
        "stage": "L003",
        "tokenizer": "base50m-unigram-32000-v1",
        "technical_classifier": "technical-heuristic-v5",
        "near_duplicate_hamming_distance": 3,
        "central_global_manifest_authorization": True,
        "pretraining_authorized": False,
        "inputs": [
            {
                "input_id": "safe",
                "source_id": "tatoeba-sentences",
                "artifact_id": "safe-artifact",
                "rights_lane": "release_safe",
                "language": "en",
                "summary_path": "safe.json",
                "summary_sha256": "b" * 64,
                "accepted_storage": (
                    "external://FlyWireLLM-data/safe/accepted.jsonl"
                ),
                "accepted_bytes": 1,
                "accepted_sha256": "c" * 64,
            },
            {
                "input_id": "research",
                "source_id": "fineweb-english",
                "artifact_id": "research-artifact",
                "rights_lane": "research_only",
                "language": "en",
                "summary_path": "research.json",
                "summary_sha256": "d" * 64,
                "accepted_storage": (
                    "external://FlyWireLLM-data/research/accepted.jsonl"
                ),
                "accepted_bytes": 1,
                "accepted_sha256": "e" * 64,
            },
        ],
    }
    registry_path = tmp_path / "configs" / "registry.json"
    registry_sha = _write_json(registry_path, registry)

    report = {
        "schema_version": 1,
        "stage": "L003",
        "tokenizer": "base50m-unigram-32000-v1",
        "technical_classifier": "technical-heuristic-v5",
        "near_duplicate_hamming_distance": 3,
        "global_cross_source_dedup_applied": True,
        "input_registry": {
            "path": "configs/registry.json",
            "sha256": registry_sha,
        },
        "tokenizer_model": {
            "candidate_id": "base50m-unigram-32000-v1",
            "sha256": TOKENIZER_SHA,
        },
        "decisions": {
            "storage": (
                "external://FlyWireLLM-data/L003/Global/v2/decisions.jsonl"
            ),
            "sha256": DECISIONS_SHA,
            "contains_raw_text": False,
        },
        "mixture": {
            "release_safe": {"ready": False},
            "research_only": {
                "ready": True,
                "total_available_tokens": 700_000_000,
                "thai_gap": 0,
                "english_gap": 0,
                "technical_gap": 0,
                "total_gap": 0,
            },
            "pretraining_authorized": False,
        },
        "pretraining_authorized": False,
    }
    report_path = tmp_path / "results" / "report.json"
    report_sha = _write_json(report_path, report)

    freeze = {
        "schema_version": 1,
        "stage": "L003",
        "checkpoint_lane": "research_only",
        "optimizer_steps_completed": 0,
        "checkpoint_lane_immutable_after_first_optimizer_step": True,
        "source_manifest": {
            "path": "configs/registry.json",
            "sha256": registry_sha,
            "frozen": True,
        },
        "global_manifest_report": {
            "path": "results/report.json",
            "sha256": report_sha,
            "frozen": True,
        },
        "evaluation_manifest": {
            "validation_partition": "validation",
            "final_holdout_partition": "holdout",
            "pretraining_fit_partitions": ["train"],
            "tokenizer_fit_must_exclude_holdout": True,
            "model_fit_must_exclude_holdout": True,
            "decisions_sha256": DECISIONS_SHA,
            "frozen": True,
        },
        "tokenizer": {
            "candidate_id": "base50m-unigram-32000-v1",
            "model_sha256": TOKENIZER_SHA,
        },
        "flywiremodel_automatic_export_allowed": False,
        "public_release_eligibility": "not_qualified",
        "pretraining_authorized": True,
    }
    freeze_path = tmp_path / "configs" / "freeze.json"
    _write_json(freeze_path, freeze)
    return freeze_path, report_path, registry_path


def test_research_only_freeze_authorizes_only_after_global_mixture_ready(tmp_path):
    freeze_path, _, _ = _fixture(tmp_path)
    result = load_pretraining_freeze(freeze_path, repo_root=tmp_path)
    assert result.checkpoint_lane == "research_only"
    assert result.pretraining_authorized is True
    assert result.optimizer_steps_completed == 0


def test_freeze_rejects_unready_selected_lane(tmp_path):
    freeze_path, report_path, _ = _fixture(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["mixture"]["research_only"]["ready"] = False
    report_sha = _write_json(report_path, report)
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    freeze["global_manifest_report"]["sha256"] = report_sha
    _write_json(freeze_path, freeze)

    with pytest.raises(PretrainingFreezeError, match="mixture is not ready"):
        load_pretraining_freeze(freeze_path, repo_root=tmp_path)


def test_freeze_must_be_declared_before_first_optimizer_step(tmp_path):
    freeze_path, _, _ = _fixture(tmp_path)
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    freeze["optimizer_steps_completed"] = 1
    _write_json(freeze_path, freeze)

    with pytest.raises(PretrainingFreezeError, match="before optimizer step 1"):
        load_pretraining_freeze(freeze_path, repo_root=tmp_path)


def test_release_safe_freeze_rejects_research_only_sources(tmp_path):
    freeze_path, _, _ = _fixture(tmp_path)
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    freeze["checkpoint_lane"] = "release_safe"
    freeze["public_release_eligibility"] = (
        "preserved_pending_final_release_audit"
    )
    _write_json(freeze_path, freeze)

    with pytest.raises(
        PretrainingFreezeError,
        match="release_safe freeze includes research_only",
    ):
        load_pretraining_freeze(freeze_path, repo_root=tmp_path)


def test_freeze_requires_metadata_only_decision_manifest(tmp_path):
    freeze_path, report_path, _ = _fixture(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["decisions"]["contains_raw_text"] = True
    report_sha = _write_json(report_path, report)
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    freeze["global_manifest_report"]["sha256"] = report_sha
    _write_json(freeze_path, freeze)

    with pytest.raises(PretrainingFreezeError, match="must be metadata-only"):
        load_pretraining_freeze(freeze_path, repo_root=tmp_path)
