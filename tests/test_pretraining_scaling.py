from __future__ import annotations

import json
from pathlib import Path

import pytest

from flywire_llm.pretraining_scaling import (
    PretrainingScalingError,
    build_scaling_state_payload,
    load_scaling_authorization,
    sha256_file,
)


EVIDENCE = "73d78d18134f7f27b530eaea2eb4d9733435ad61"


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
        ext / "L004" / "Runs" / "base50m-third-tranche-v1"
        / "checkpoints" / "step-000016.pt"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"step16")
    payload = {
        "schema_version": 1,
        "stage": "L004",
        "authorization_id": "base50m-fourth-tranche-v1",
        "runner_commit": "1" * 40,
        "evidence_commit": EVIDENCE,
        "checkpoint_lane": "research_only",
        "start_optimizer_step": 16,
        "max_additional_optimizer_updates": 16,
        "end_optimizer_step": 32,
        "target_additional_supervised_tokens": 1_048_576,
        "target_cumulative_supervised_tokens": 2_097_152,
        "checkpoint_every_updates": 1,
        "source_checkpoint": {
            "result_path": "result.json",
            "result_sha256": sha256_file(result),
            "checkpoint": (
                "external://FlyWireLLM-data/L004/Runs/"
                "base50m-third-tranche-v1/checkpoints/step-000016.pt"
            ),
            "checkpoint_bytes": checkpoint.stat().st_size,
            "checkpoint_sha256": sha256_file(checkpoint),
            "optimizer_step": 16,
            "supervised_tokens_seen": 1_048_576,
            "physical_input_positions_seen": 1_051_555,
            "microbatches_seen": 1_042,
            "data_state": {
                "block_input_offset": 931,
                "order_position": 1026,
                "supervised_tokens_consumed": 1_048_576,
            },
        },
        "run_root": (
            "external://FlyWireLLM-data/L004/Runs/base50m-fourth-tranche-v1"
        ),
        "repo_file_pins": [{
            "path": "pin.json",
            "sha256": sha256_file(pin),
        }],
        "post_tranche_validation_gate": {
            "evaluation_id": "base50m-validation-v1",
            "baseline_optimizer_step": 16,
            "candidate_optimizer_step": 32,
            "same_validation_pack_required": True,
            "final_holdout_must_remain_untouched": True,
            "combined_loss_must_not_increase_vs_step16": True,
            "max_category_relative_loss_increase_vs_step16": 0.005,
            "require_all_losses_finite": True,
        },
        "pretraining_authorized": True,
        "public_release_eligibility": "not_qualified",
        "automatic_flywiremodel_export_allowed": False,
    }
    path = repo / "auth.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    auth = load_scaling_authorization(
        path, repo_root=repo, external_root=ext
    )
    return repo, ext, path, payload, auth


def test_scaling_is_bounded_to_step32(tmp_path):
    _, _, _, _, auth = _fixture(tmp_path)
    assert auth.start_optimizer_step == 16
    assert auth.end_optimizer_step == 32
    assert auth.max_additional_optimizer_updates == 16
    assert auth.target_cumulative_supervised_tokens == 2_097_152


def test_scaling_rejects_bound_or_evidence_change(tmp_path):
    repo, ext, path, payload, _ = _fixture(tmp_path)
    payload["max_additional_optimizer_updates"] = 32
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PretrainingScalingError):
        load_scaling_authorization(path, repo_root=repo, external_root=ext)
    payload["max_additional_optimizer_updates"] = 16
    payload["evidence_commit"] = "9" * 40
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        PretrainingScalingError, match="step16 evidence commit changed"
    ):
        load_scaling_authorization(path, repo_root=repo, external_root=ext)


def test_scaling_rejects_gate_change(tmp_path):
    repo, ext, path, payload, _ = _fixture(tmp_path)
    payload["post_tranche_validation_gate"][
        "final_holdout_must_remain_untouched"
    ] = False
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        PretrainingScalingError, match="post-validation gate changed"
    ):
        load_scaling_authorization(path, repo_root=repo, external_root=ext)


def test_scaling_state_rejects_step33(tmp_path):
    _, _, _, _, auth = _fixture(tmp_path)
    checkpoint = auth.run_root / "checkpoint.pt"
    metric = auth.run_root / "metric.json"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"x")
    metric.write_bytes(b"y")
    with pytest.raises(
        PretrainingScalingError, match="outside fourth tranche"
    ):
        build_scaling_state_payload(
            auth,
            optimizer_step=33,
            supervised_tokens_seen=2_162_688,
            physical_input_positions_seen=2_170_000,
            microbatches_seen=2_100,
            data_state={
                "order_position": 2_100,
                "block_input_offset": 0,
                "supervised_tokens_consumed": 2_162_688,
            },
            checkpoint_path=checkpoint,
            checkpoint_sha256=sha256_file(checkpoint),
            metric_path=metric,
            metric_sha256=sha256_file(metric),
        )
