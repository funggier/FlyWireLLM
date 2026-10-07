from __future__ import annotations

import json
from pathlib import Path

import pytest

from flywire_llm.pretraining_warmup_penultimate import (
    PretrainingWarmupPenultimateError,
    build_warmup_penultimate_state_payload,
    load_warmup_penultimate_authorization,
    prune_prior_checkpoint,
    sha256_file,
)


EVIDENCE = "9f951c0c33b00006e7ffb376626acfcf4c60b30d"


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
        ext / "L004" / "Runs" / "base50m-ninth-tranche-v1"
        / "checkpoints" / "step-000120.pt"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"step120")
    payload = {
        "schema_version": 1,
        "stage": "L004",
        "authorization_id": "base50m-tenth-tranche-v1",
        "runner_commit": "1" * 40,
        "evidence_commit": EVIDENCE,
        "checkpoint_lane": "research_only",
        "start_optimizer_step": 120,
        "max_additional_optimizer_updates": 16,
        "end_optimizer_step": 136,
        "target_additional_supervised_tokens": 1_048_576,
        "target_cumulative_supervised_tokens": 8_912_896,
        "checkpoint_every_updates": 1,
        "rolling_checkpoint_retention": 1,
        "prune_prior_checkpoint_after_state_commit": True,
        "source_checkpoint": {
            "result_path": "result.json",
            "result_sha256": sha256_file(result),
            "checkpoint": (
                "external://FlyWireLLM-data/L004/Runs/"
                "base50m-ninth-tranche-v1/checkpoints/step-000120.pt"
            ),
            "checkpoint_bytes": checkpoint.stat().st_size,
            "checkpoint_sha256": sha256_file(checkpoint),
            "optimizer_step": 120,
            "supervised_tokens_seen": 7_864_320,
            "physical_input_positions_seen": 7_887_185,
            "microbatches_seen": 7_822,
            "data_state": {
                "block_input_offset": 337,
                "order_position": 7702,
                "supervised_tokens_consumed": 7_864_320,
            },
        },
        "run_root": (
            "external://FlyWireLLM-data/L004/Runs/base50m-tenth-tranche-v1"
        ),
        "repo_file_pins": [{
            "path": "pin.json",
            "sha256": sha256_file(pin),
        }],
        "post_tranche_validation_gate": {
            "evaluation_id": "base50m-validation-v1",
            "baseline_optimizer_step": 120,
            "candidate_optimizer_step": 136,
            "same_validation_pack_required": True,
            "final_holdout_must_remain_untouched": True,
            "combined_loss_must_not_increase_vs_step120": True,
            "max_category_relative_loss_increase_vs_step120": 0.005,
            "require_all_losses_finite": True,
        },
        "pretraining_authorized": True,
        "public_release_eligibility": "not_qualified",
        "automatic_flywiremodel_export_allowed": False,
    }
    path = repo / "auth.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    auth = load_warmup_penultimate_authorization(
        path, repo_root=repo, external_root=ext
    )
    return repo, ext, path, payload, auth


def test_warmup_penultimate_authorization_is_bounded(tmp_path):
    _, _, _, _, auth = _fixture(tmp_path)
    assert auth.start_optimizer_step == 120
    assert auth.end_optimizer_step == 136
    assert auth.max_additional_optimizer_updates == 16
    assert auth.target_cumulative_supervised_tokens == 8_912_896


def test_warmup_penultimate_rejects_policy_or_evidence_change(tmp_path):
    repo, ext, path, payload, _ = _fixture(tmp_path)
    payload["rolling_checkpoint_retention"] = 2
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PretrainingWarmupPenultimateError):
        load_warmup_penultimate_authorization(
            path, repo_root=repo, external_root=ext
        )
    payload["rolling_checkpoint_retention"] = 1
    payload["evidence_commit"] = "9" * 40
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        PretrainingWarmupPenultimateError,
        match="step120 evidence commit changed",
    ):
        load_warmup_penultimate_authorization(
            path, repo_root=repo, external_root=ext
        )


def test_warmup_penultimate_prune_is_confined_to_run_root(tmp_path):
    root = tmp_path / "run"
    checkpoints = root / "checkpoints"
    checkpoints.mkdir(parents=True)
    old = checkpoints / "step-000121.pt"
    current = checkpoints / "step-000122.pt"
    old.write_bytes(b"old")
    current.write_bytes(b"new")
    assert prune_prior_checkpoint(root, old, current) is True
    assert not old.exists()
    assert current.exists()
    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"x")
    with pytest.raises(
        PretrainingWarmupPenultimateError, match="escapes run root"
    ):
        prune_prior_checkpoint(root, outside, current)


def test_warmup_penultimate_state_rejects_step137(tmp_path):
    _, _, _, _, auth = _fixture(tmp_path)
    checkpoint = auth.run_root / "checkpoints" / "step-000137.pt"
    metric = auth.run_root / "metrics" / "step-000137.json"
    checkpoint.parent.mkdir(parents=True)
    metric.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"x")
    metric.write_bytes(b"y")
    with pytest.raises(
        PretrainingWarmupPenultimateError, match="outside tenth tranche"
    ):
        build_warmup_penultimate_state_payload(
            auth,
            optimizer_step=137,
            supervised_tokens_seen=8_978_432,
            physical_input_positions_seen=8_990_000,
            microbatches_seen=8_950,
            data_state={
                "order_position": 8_950,
                "block_input_offset": 0,
                "supervised_tokens_consumed": 8_978_432,
            },
            checkpoint_path=checkpoint,
            checkpoint_sha256=sha256_file(checkpoint),
            metric_path=metric,
            metric_sha256=sha256_file(metric),
        )
