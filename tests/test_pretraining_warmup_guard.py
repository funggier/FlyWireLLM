from __future__ import annotations

import json
from pathlib import Path

import pytest

from flywire_llm.pretraining_warmup_guard import (
    PretrainingWarmupGuardError,
    build_warmup_guard_state_payload,
    load_warmup_guard_authorization,
    prune_prior_checkpoint,
    sha256_file,
)


EVIDENCE = "61f969000a330d686ebbabb0d6387c6e4a283fa9"


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
        ext / "L004" / "Runs" / "base50m-seventh-tranche-v1"
        / "checkpoints" / "step-000088.pt"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"step88")
    payload = {
        "schema_version": 1,
        "stage": "L004",
        "authorization_id": "base50m-eighth-tranche-v1",
        "runner_commit": "1" * 40,
        "evidence_commit": EVIDENCE,
        "checkpoint_lane": "research_only",
        "start_optimizer_step": 88,
        "max_additional_optimizer_updates": 16,
        "end_optimizer_step": 104,
        "target_additional_supervised_tokens": 1_048_576,
        "target_cumulative_supervised_tokens": 6_815_744,
        "checkpoint_every_updates": 1,
        "rolling_checkpoint_retention": 1,
        "prune_prior_checkpoint_after_state_commit": True,
        "source_checkpoint": {
            "result_path": "result.json",
            "result_sha256": sha256_file(result),
            "checkpoint": (
                "external://FlyWireLLM-data/L004/Runs/"
                "base50m-seventh-tranche-v1/checkpoints/step-000088.pt"
            ),
            "checkpoint_bytes": checkpoint.stat().st_size,
            "checkpoint_sha256": sha256_file(checkpoint),
            "optimizer_step": 88,
            "supervised_tokens_seen": 5_767_168,
            "physical_input_positions_seen": 5_784_808,
            "microbatches_seen": 5_737,
            "data_state": {
                "block_input_offset": 232,
                "order_position": 5649,
                "supervised_tokens_consumed": 5_767_168,
            },
        },
        "run_root": (
            "external://FlyWireLLM-data/L004/Runs/base50m-eighth-tranche-v1"
        ),
        "repo_file_pins": [{
            "path": "pin.json",
            "sha256": sha256_file(pin),
        }],
        "post_tranche_validation_gate": {
            "evaluation_id": "base50m-validation-v1",
            "baseline_optimizer_step": 88,
            "candidate_optimizer_step": 104,
            "same_validation_pack_required": True,
            "final_holdout_must_remain_untouched": True,
            "combined_loss_must_not_increase_vs_step88": True,
            "max_category_relative_loss_increase_vs_step88": 0.005,
            "require_all_losses_finite": True,
        },
        "pretraining_authorized": True,
        "public_release_eligibility": "not_qualified",
        "automatic_flywiremodel_export_allowed": False,
    }
    path = repo / "auth.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    auth = load_warmup_guard_authorization(
        path, repo_root=repo, external_root=ext
    )
    return repo, ext, path, payload, auth


def test_warmup_guard_authorization_is_bounded(tmp_path):
    _, _, _, _, auth = _fixture(tmp_path)
    assert auth.start_optimizer_step == 88
    assert auth.end_optimizer_step == 104
    assert auth.max_additional_optimizer_updates == 16
    assert auth.target_cumulative_supervised_tokens == 6_815_744


def test_warmup_guard_rejects_policy_or_evidence_change(tmp_path):
    repo, ext, path, payload, _ = _fixture(tmp_path)
    payload["rolling_checkpoint_retention"] = 2
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PretrainingWarmupGuardError):
        load_warmup_guard_authorization(
            path, repo_root=repo, external_root=ext
        )
    payload["rolling_checkpoint_retention"] = 1
    payload["evidence_commit"] = "9" * 40
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        PretrainingWarmupGuardError,
        match="step88 evidence commit changed",
    ):
        load_warmup_guard_authorization(
            path, repo_root=repo, external_root=ext
        )


def test_warmup_guard_prune_is_confined_to_run_root(tmp_path):
    root = tmp_path / "run"
    checkpoints = root / "checkpoints"
    checkpoints.mkdir(parents=True)
    old = checkpoints / "step-000089.pt"
    current = checkpoints / "step-000090.pt"
    old.write_bytes(b"old")
    current.write_bytes(b"new")
    assert prune_prior_checkpoint(root, old, current) is True
    assert not old.exists()
    assert current.exists()
    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"x")
    with pytest.raises(
        PretrainingWarmupGuardError, match="escapes run root"
    ):
        prune_prior_checkpoint(root, outside, current)


def test_warmup_guard_state_rejects_step105(tmp_path):
    _, _, _, _, auth = _fixture(tmp_path)
    checkpoint = auth.run_root / "checkpoints" / "step-000105.pt"
    metric = auth.run_root / "metrics" / "step-000105.json"
    checkpoint.parent.mkdir(parents=True)
    metric.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"x")
    metric.write_bytes(b"y")
    with pytest.raises(
        PretrainingWarmupGuardError, match="outside eighth tranche"
    ):
        build_warmup_guard_state_payload(
            auth,
            optimizer_step=105,
            supervised_tokens_seen=6_881_280,
            physical_input_positions_seen=6_890_000,
            microbatches_seen=6_900,
            data_state={
                "order_position": 6_900,
                "block_input_offset": 0,
                "supervised_tokens_consumed": 6_881_280,
            },
            checkpoint_path=checkpoint,
            checkpoint_sha256=sha256_file(checkpoint),
            metric_path=metric,
            metric_sha256=sha256_file(metric),
        )
