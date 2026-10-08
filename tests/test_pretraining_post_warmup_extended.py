from __future__ import annotations

import json
from pathlib import Path

import pytest

from flywire_llm.pretraining_post_warmup_extended import (
    PretrainingPostWarmupExtendedError,
    build_post_warmup_extended_state_payload,
    load_post_warmup_extended_authorization,
    load_post_warmup_extended_state,
    prune_prior_checkpoint,
    sha256_file,
)


EVIDENCE = "691cb9d6e6640f8e4c7c36f3927ae12cdb6e8121"


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
        ext / "L004" / "Runs" / "base50m-post-warmup-5-v1"
        / "checkpoints" / "step-000425.pt"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"step425")
    payload = {
        "schema_version": 1,
        "stage": "L004",
        "authorization_id": "base50m-post-warmup-6-v1",
        "runner_commit": "1" * 40,
        "evidence_commit": EVIDENCE,
        "checkpoint_lane": "research_only",
        "start_optimizer_step": 425,
        "max_additional_optimizer_updates": 256,
        "end_optimizer_step": 681,
        "target_additional_supervised_tokens": 16_777_216,
        "target_cumulative_supervised_tokens": 44_630_016,
        "checkpoint_every_updates": 1,
        "rolling_checkpoint_retention": 1,
        "prune_prior_checkpoint_after_state_commit": True,
        "source_checkpoint": {
            "result_path": "result.json",
            "result_sha256": sha256_file(result),
            "checkpoint": (
                "external://FlyWireLLM-data/L004/Runs/"
                "base50m-post-warmup-5-v1/checkpoints/step-000425.pt"
            ),
            "checkpoint_bytes": checkpoint.stat().st_size,
            "checkpoint_sha256": sha256_file(checkpoint),
            "optimizer_step": 425,
            "supervised_tokens_seen": 27_852_800,
            "physical_input_positions_seen": 27_928_093,
            "microbatches_seen": 27_697,
            "data_state": {
                "block_input_offset": 541,
                "order_position": 27_273,
                "supervised_tokens_consumed": 27_852_800,
            },
        },
        "run_root": (
            "external://FlyWireLLM-data/L004/Runs/base50m-post-warmup-6-v1"
        ),
        "repo_file_pins": [{
            "path": "pin.json",
            "sha256": sha256_file(pin),
        }],
        "post_tranche_validation_gate": {
            "evaluation_id": "base50m-validation-v1",
            "baseline_optimizer_step": 425,
            "candidate_optimizer_step": 681,
            "same_validation_pack_required": True,
            "final_holdout_must_remain_untouched": True,
            "combined_loss_must_not_increase_vs_step425": True,
            "max_category_relative_loss_increase_vs_step425": 0.005,
            "require_all_losses_finite": True,
        },
        "pretraining_authorized": True,
        "public_release_eligibility": "not_qualified",
        "automatic_flywiremodel_export_allowed": False,
    }
    path = repo / "auth.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    auth = load_post_warmup_extended_authorization(
        path, repo_root=repo, external_root=ext
    )
    return repo, ext, path, payload, auth


def test_post_warmup_extended_authorization_is_bounded(tmp_path):
    _, _, _, _, auth = _fixture(tmp_path)
    assert auth.start_optimizer_step == 425
    assert auth.end_optimizer_step == 681
    assert auth.max_additional_optimizer_updates == 256
    assert auth.target_additional_supervised_tokens == 16_777_216
    assert auth.target_cumulative_supervised_tokens == 44_630_016


def test_post_warmup_extended_rejects_policy_or_evidence_change(tmp_path):
    repo, ext, path, payload, _ = _fixture(tmp_path)
    payload["rolling_checkpoint_retention"] = 2
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PretrainingPostWarmupExtendedError):
        load_post_warmup_extended_authorization(
            path, repo_root=repo, external_root=ext
        )
    payload["rolling_checkpoint_retention"] = 1
    payload["evidence_commit"] = "9" * 40
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        PretrainingPostWarmupExtendedError,
        match="step425 evidence commit changed",
    ):
        load_post_warmup_extended_authorization(
            path, repo_root=repo, external_root=ext
        )


def test_post_warmup_extended_prune_is_confined_to_run_root(tmp_path):
    root = tmp_path / "run"
    checkpoints = root / "checkpoints"
    checkpoints.mkdir(parents=True)
    old = checkpoints / "step-000426.pt"
    current = checkpoints / "step-000427.pt"
    old.write_bytes(b"old")
    current.write_bytes(b"new")
    assert prune_prior_checkpoint(root, old, current) is True
    assert not old.exists()
    assert current.exists()
    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"x")
    with pytest.raises(
        PretrainingPostWarmupExtendedError, match="escapes run root"
    ):
        prune_prior_checkpoint(root, outside, current)


def test_post_warmup_extended_state_rejects_step682(tmp_path):
    _, _, _, _, auth = _fixture(tmp_path)
    checkpoint = auth.run_root / "checkpoints" / "step-000682.pt"
    metric = auth.run_root / "metrics" / "step-000682.json"
    checkpoint.parent.mkdir(parents=True)
    metric.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"x")
    metric.write_bytes(b"y")
    with pytest.raises(
        PretrainingPostWarmupExtendedError,
        match="outside post-warmup extended tranche",
    ):
        build_post_warmup_extended_state_payload(
            auth,
            optimizer_step=682,
            supervised_tokens_seen=44_695_552,
            physical_input_positions_seen=44_800_000,
            microbatches_seen=44_500,
            data_state={
                "order_position": 44_000,
                "block_input_offset": 0,
                "supervised_tokens_consumed": 44_695_552,
            },
            checkpoint_path=checkpoint,
            checkpoint_sha256=sha256_file(checkpoint),
            metric_path=metric,
            metric_sha256=sha256_file(metric),
        )


@pytest.mark.parametrize(
    ("step", "tranche_complete"),
    [(426, False), (681, True)],
)
def test_post_warmup_extended_state_completion_boundary(
    tmp_path, step, tranche_complete
):
    _, _, _, _, auth = _fixture(tmp_path)
    checkpoint = auth.run_root / "checkpoints" / f"step-{step:06d}.pt"
    metric = auth.run_root / "metrics" / f"step-{step:06d}.json"
    checkpoint.parent.mkdir(parents=True)
    metric.parent.mkdir(parents=True)
    checkpoint.write_bytes(f"step{step}".encode())
    metric.write_text("{}\n", encoding="utf-8")
    supervised = step * 65_536
    payload = build_post_warmup_extended_state_payload(
        auth,
        optimizer_step=step,
        supervised_tokens_seen=supervised,
        physical_input_positions_seen=supervised + 1_000,
        microbatches_seen=step * 64,
        data_state={
            "order_position": step * 64,
            "block_input_offset": 0,
            "supervised_tokens_consumed": supervised,
        },
        checkpoint_path=checkpoint,
        checkpoint_sha256=sha256_file(checkpoint),
        metric_path=metric,
        metric_sha256=sha256_file(metric),
    )
    assert payload["tranche_complete"] is tranche_complete
    state_path = auth.run_root / "state.json"
    state_path.write_text(json.dumps(payload), encoding="utf-8")

    state = load_post_warmup_extended_state(
        state_path,
        authorization=auth,
        verify_files=True,
    )

    assert state.optimizer_step == step
    assert state.supervised_tokens_seen == supervised
