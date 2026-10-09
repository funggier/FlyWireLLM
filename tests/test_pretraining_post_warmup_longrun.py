from __future__ import annotations

import json
from pathlib import Path

import pytest

from flywire_llm.pretraining_post_warmup_longrun import (
    PretrainingPostWarmupLongrunError,
    build_post_warmup_longrun_state_payload,
    load_post_warmup_longrun_authorization,
    load_post_warmup_longrun_state,
    prune_prior_checkpoint,
    sha256_file,
)


EVIDENCE = "4d2ba99e9af6048de55c90de6f44217988079115"


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
        ext / "L004" / "Runs" / "base50m-post-warmup-7-v1"
        / "checkpoints" / "step-001193.pt"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"step1193")
    payload = {
        "schema_version": 1,
        "stage": "L004",
        "authorization_id": "base50m-post-warmup-8-v1",
        "runner_commit": "1" * 40,
        "evidence_commit": EVIDENCE,
        "checkpoint_lane": "research_only",
        "start_optimizer_step": 1193,
        "max_additional_optimizer_updates": 1024,
        "end_optimizer_step": 2217,
        "target_additional_supervised_tokens": 67_108_864,
        "target_cumulative_supervised_tokens": 145_293_312,
        "checkpoint_every_updates": 1,
        "rolling_checkpoint_retention": 1,
        "prune_prior_checkpoint_after_state_commit": True,
        "source_checkpoint": {
            "result_path": "result.json",
            "result_sha256": sha256_file(result),
            "checkpoint": (
                "external://FlyWireLLM-data/L004/Runs/"
                "base50m-post-warmup-7-v1/checkpoints/step-001193.pt"
            ),
            "checkpoint_bytes": checkpoint.stat().st_size,
            "checkpoint_sha256": sha256_file(checkpoint),
            "optimizer_step": 1193,
            "supervised_tokens_seen": 78_184_448,
            "physical_input_positions_seen": 78_394_399,
            "microbatches_seen": 77_749,
            "data_state": {
                "block_input_offset": 31,
                "order_position": 76_557,
                "supervised_tokens_consumed": 78_184_448,
            },
        },
        "run_root": (
            "external://FlyWireLLM-data/L004/Runs/base50m-post-warmup-8-v1"
        ),
        "repo_file_pins": [{
            "path": "pin.json",
            "sha256": sha256_file(pin),
        }],
        "post_tranche_validation_gate": {
            "evaluation_id": "base50m-validation-v1",
            "baseline_optimizer_step": 1193,
            "candidate_optimizer_step": 2217,
            "same_validation_pack_required": True,
            "final_holdout_must_remain_untouched": True,
            "combined_loss_must_not_increase_vs_step1193": True,
            "max_category_relative_loss_increase_vs_step1193": 0.005,
            "require_all_losses_finite": True,
        },
        "pretraining_authorized": True,
        "public_release_eligibility": "not_qualified",
        "automatic_flywiremodel_export_allowed": False,
    }
    path = repo / "auth.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    auth = load_post_warmup_longrun_authorization(
        path, repo_root=repo, external_root=ext
    )
    return repo, ext, path, payload, auth


def test_post_warmup_longrun_authorization_is_bounded(tmp_path):
    _, _, _, _, auth = _fixture(tmp_path)
    assert auth.start_optimizer_step == 1193
    assert auth.end_optimizer_step == 2217
    assert auth.max_additional_optimizer_updates == 1024
    assert auth.target_additional_supervised_tokens == 67_108_864
    assert auth.target_cumulative_supervised_tokens == 145_293_312


def test_post_warmup_longrun_rejects_policy_or_evidence_change(tmp_path):
    repo, ext, path, payload, _ = _fixture(tmp_path)
    payload["rolling_checkpoint_retention"] = 2
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PretrainingPostWarmupLongrunError):
        load_post_warmup_longrun_authorization(
            path, repo_root=repo, external_root=ext
        )
    payload["rolling_checkpoint_retention"] = 1
    payload["evidence_commit"] = "9" * 40
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        PretrainingPostWarmupLongrunError,
        match="step1193 evidence commit changed",
    ):
        load_post_warmup_longrun_authorization(
            path, repo_root=repo, external_root=ext
        )


def test_post_warmup_longrun_prune_is_confined_to_run_root(tmp_path):
    root = tmp_path / "run"
    checkpoints = root / "checkpoints"
    checkpoints.mkdir(parents=True)
    old = checkpoints / "step-001194.pt"
    current = checkpoints / "step-001195.pt"
    old.write_bytes(b"old")
    current.write_bytes(b"new")
    assert prune_prior_checkpoint(root, old, current) is True
    assert not old.exists()
    assert current.exists()
    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"x")
    with pytest.raises(
        PretrainingPostWarmupLongrunError, match="escapes run root"
    ):
        prune_prior_checkpoint(root, outside, current)


def test_post_warmup_longrun_state_rejects_step2218(tmp_path):
    _, _, _, _, auth = _fixture(tmp_path)
    checkpoint = auth.run_root / "checkpoints" / "step-002218.pt"
    metric = auth.run_root / "metrics" / "step-002218.json"
    checkpoint.parent.mkdir(parents=True)
    metric.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"x")
    metric.write_bytes(b"y")
    with pytest.raises(
        PretrainingPostWarmupLongrunError,
        match="outside post-warmup longrun tranche",
    ):
        build_post_warmup_longrun_state_payload(
            auth,
            optimizer_step=2218,
            supervised_tokens_seen=78_249_984,
            physical_input_positions_seen=78_350_000,
            microbatches_seen=77_000,
            data_state={
                "order_position": 76_000,
                "block_input_offset": 0,
                "supervised_tokens_consumed": 78_249_984,
            },
            checkpoint_path=checkpoint,
            checkpoint_sha256=sha256_file(checkpoint),
            metric_path=metric,
            metric_sha256=sha256_file(metric),
        )


@pytest.mark.parametrize(
    ("step", "tranche_complete"),
    [(1194, False), (2217, True)],
)
def test_post_warmup_longrun_state_completion_boundary(
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
    payload = build_post_warmup_longrun_state_payload(
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

    state = load_post_warmup_longrun_state(
        state_path,
        authorization=auth,
        verify_files=True,
    )

    assert state.optimizer_step == step
    assert state.supervised_tokens_seen == supervised
