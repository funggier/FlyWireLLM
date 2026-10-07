from __future__ import annotations

import json
from pathlib import Path

import pytest

from flywire_llm.pretraining_expansion import (
    PretrainingExpansionError,
    build_expansion_state_payload,
    load_expansion_authorization,
    sha256_file,
)


EVIDENCE = "c0966a146a5e2807b0567a3653a73e20bf5aee39"


def _fixture(tmp_path: Path):
    repo = tmp_path / "repo"
    ext = tmp_path / "external"
    repo.mkdir()
    ext.mkdir()
    pin = repo / "pin.json"
    pin.write_text("{}\n", encoding="utf-8")
    result = repo / "result.json"
    result.write_text("{}\n", encoding="utf-8")
    checkpoint = (
        ext / "L004" / "Runs" / "base50m-second-tranche-v1"
        / "checkpoints" / "step-000008.pt"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"step8")
    payload = {
        "schema_version": 1,
        "stage": "L004",
        "authorization_id": "base50m-third-tranche-v1",
        "runner_commit": "1" * 40,
        "evidence_commit": EVIDENCE,
        "checkpoint_lane": "research_only",
        "start_optimizer_step": 8,
        "max_additional_optimizer_updates": 8,
        "end_optimizer_step": 16,
        "target_additional_supervised_tokens": 524288,
        "target_cumulative_supervised_tokens": 1048576,
        "checkpoint_every_updates": 1,
        "source_checkpoint": {
            "result_path": "result.json",
            "result_sha256": sha256_file(result),
            "checkpoint": (
                "external://FlyWireLLM-data/L004/Runs/"
                "base50m-second-tranche-v1/checkpoints/step-000008.pt"
            ),
            "checkpoint_bytes": checkpoint.stat().st_size,
            "checkpoint_sha256": sha256_file(checkpoint),
            "optimizer_step": 8,
            "supervised_tokens_seen": 524288,
            "physical_input_positions_seen": 525712,
            "microbatches_seen": 521,
            "data_state": {
                "block_input_offset": 400,
                "order_position": 513,
                "supervised_tokens_consumed": 524288,
            },
        },
        "run_root": (
            "external://FlyWireLLM-data/L004/Runs/base50m-third-tranche-v1"
        ),
        "repo_file_pins": [{
            "path": "pin.json",
            "sha256": sha256_file(pin),
        }],
        "post_tranche_validation_gate": {
            "evaluation_id": "base50m-validation-v1",
            "baseline_optimizer_step": 8,
            "candidate_optimizer_step": 16,
            "same_validation_pack_required": True,
            "final_holdout_must_remain_untouched": True,
            "combined_loss_must_not_increase_vs_step8": True,
            "max_category_relative_loss_increase_vs_step8": 0.005,
            "require_all_losses_finite": True,
        },
        "pretraining_authorized": True,
        "public_release_eligibility": "not_qualified",
        "automatic_flywiremodel_export_allowed": False,
    }
    path = repo / "auth.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    auth = load_expansion_authorization(
        path, repo_root=repo, external_root=ext
    )
    return repo, ext, path, payload, auth


def test_expansion_is_bounded_to_step16(tmp_path):
    _, _, _, _, auth = _fixture(tmp_path)
    assert auth.start_optimizer_step == 8
    assert auth.end_optimizer_step == 16
    assert auth.max_additional_optimizer_updates == 8
    assert auth.target_cumulative_supervised_tokens == 1048576


def test_expansion_rejects_bound_or_evidence_change(tmp_path):
    repo, ext, path, payload, _ = _fixture(tmp_path)
    payload["max_additional_optimizer_updates"] = 16
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PretrainingExpansionError):
        load_expansion_authorization(path, repo_root=repo, external_root=ext)
    payload["max_additional_optimizer_updates"] = 8
    payload["evidence_commit"] = "9" * 40
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        PretrainingExpansionError, match="step8 evidence commit changed"
    ):
        load_expansion_authorization(path, repo_root=repo, external_root=ext)


def test_expansion_rejects_validation_gate_change(tmp_path):
    repo, ext, path, payload, _ = _fixture(tmp_path)
    payload["post_tranche_validation_gate"][
        "final_holdout_must_remain_untouched"
    ] = False
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        PretrainingExpansionError, match="post-validation gate changed"
    ):
        load_expansion_authorization(path, repo_root=repo, external_root=ext)


def test_expansion_state_rejects_step17(tmp_path):
    _, _, _, _, auth = _fixture(tmp_path)
    checkpoint = auth.run_root / "checkpoint.pt"
    metric = auth.run_root / "metric.json"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"x")
    metric.write_bytes(b"y")
    with pytest.raises(
        PretrainingExpansionError, match="outside third tranche"
    ):
        build_expansion_state_payload(
            auth,
            optimizer_step=17,
            supervised_tokens_seen=1114112,
            physical_input_positions_seen=1115000,
            microbatches_seen=1100,
            data_state={
                "order_position": 1100,
                "block_input_offset": 0,
                "supervised_tokens_consumed": 1114112,
            },
            checkpoint_path=checkpoint,
            checkpoint_sha256=sha256_file(checkpoint),
            metric_path=metric,
            metric_sha256=sha256_file(metric),
        )
