from __future__ import annotations

import json
from pathlib import Path

import pytest

from flywire_llm.pretraining_continuation import (
    PretrainingContinuationError,
    atomic_write_json,
    build_continuation_state_payload,
    load_continuation_authorization,
    load_continuation_state,
    sha256_file,
)


def _fixture(tmp_path: Path):
    repo = tmp_path / "repo"
    external = tmp_path / "external"
    repo.mkdir()
    external.mkdir()
    pinned = repo / "runtime.json"
    pinned.write_text("{}\n", encoding="utf-8", newline="\n")
    result = repo / "result.json"
    result.write_text(
        '{"optimizer_step": 4}\n',
        encoding="utf-8",
        newline="\n",
    )
    checkpoint = (
        external
        / "L004"
        / "Runs"
        / "base50m-first-tranche-v1"
        / "checkpoints"
        / "step-000004.pt"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"source-checkpoint")
    payload = {
        "schema_version": 1,
        "stage": "L004",
        "authorization_id": "base50m-second-tranche-v1",
        "foundation_commit": "1" * 40,
        "runner_commit": "2" * 40,
        "validation_gate_commit": (
            "2151b1199346bbc26da944f731b01b84556e3ed3"
        ),
        "checkpoint_lane": "research_only",
        "start_optimizer_step": 4,
        "max_additional_optimizer_updates": 4,
        "end_optimizer_step": 8,
        "target_additional_supervised_tokens": 262_144,
        "target_cumulative_supervised_tokens": 524_288,
        "checkpoint_every_updates": 1,
        "source_checkpoint": {
            "result_path": "result.json",
            "result_sha256": sha256_file(result),
            "checkpoint": (
                "external://FlyWireLLM-data/L004/Runs/"
                "base50m-first-tranche-v1/checkpoints/step-000004.pt"
            ),
            "checkpoint_bytes": checkpoint.stat().st_size,
            "checkpoint_sha256": sha256_file(checkpoint),
            "optimizer_step": 4,
            "supervised_tokens_seen": 262_144,
            "data_state": {
                "block_input_offset": 767,
                "order_position": 256,
                "supervised_tokens_consumed": 262_144,
            },
        },
        "run_root": (
            "external://FlyWireLLM-data/L004/Runs/"
            "base50m-second-tranche-v1"
        ),
        "repo_file_pins": [
            {
                "path": "runtime.json",
                "sha256": sha256_file(pinned),
            }
        ],
        "pretraining_authorized": True,
        "public_release_eligibility": "not_qualified",
        "automatic_flywiremodel_export_allowed": False,
    }
    auth_path = repo / "authorization.json"
    auth_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    authorization = load_continuation_authorization(
        auth_path,
        repo_root=repo,
        external_root=external,
    )
    return repo, external, auth_path, payload, authorization


def test_continuation_authorization_is_bounded_to_steps_five_through_eight(
    tmp_path,
):
    _, external, _, _, authorization = _fixture(tmp_path)
    assert authorization.start_optimizer_step == 4
    assert authorization.max_additional_optimizer_updates == 4
    assert authorization.end_optimizer_step == 8
    assert authorization.target_additional_supervised_tokens == 262_144
    assert authorization.target_cumulative_supervised_tokens == 524_288
    assert authorization.run_root == (
        external / "L004" / "Runs" / "base50m-second-tranche-v1"
    ).resolve()


def test_continuation_authorization_rejects_expansion(tmp_path):
    repo, external, path, payload, _ = _fixture(tmp_path)
    payload["max_additional_optimizer_updates"] = 8
    path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(
        PretrainingContinuationError,
        match="max_additional_optimizer_updates changed",
    ):
        load_continuation_authorization(
            path,
            repo_root=repo,
            external_root=external,
        )


def test_continuation_authorization_pins_validation_gate(tmp_path):
    repo, external, path, payload, _ = _fixture(tmp_path)
    payload["validation_gate_commit"] = "9" * 40
    path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(
        PretrainingContinuationError,
        match="validation gate commit changed",
    ):
        load_continuation_authorization(
            path,
            repo_root=repo,
            external_root=external,
        )


def test_continuation_state_round_trip(tmp_path):
    _, _, _, _, authorization = _fixture(tmp_path)
    checkpoint = (
        authorization.run_root / "checkpoints" / "step-000005.pt"
    )
    metric = authorization.run_root / "metrics" / "step-000005.json"
    checkpoint.parent.mkdir(parents=True)
    metric.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint-five")
    metric.write_bytes(b"metric-five")
    payload = build_continuation_state_payload(
        authorization,
        optimizer_step=5,
        supervised_tokens_seen=327_680,
        physical_input_positions_seen=328_500,
        microbatches_seen=325,
        data_state={
            "order_position": 320,
            "block_input_offset": 100,
            "supervised_tokens_consumed": 327_680,
        },
        checkpoint_path=checkpoint,
        checkpoint_sha256=sha256_file(checkpoint),
        metric_path=metric,
        metric_sha256=sha256_file(metric),
    )
    state_path = authorization.run_root / "state.json"
    atomic_write_json(state_path, payload)
    state = load_continuation_state(
        state_path,
        authorization=authorization,
        verify_files=True,
    )
    assert state.optimizer_step == 5
    assert state.supervised_tokens_seen == 327_680
    assert state.checkpoint_sha256 == sha256_file(checkpoint)


def test_continuation_state_rejects_step_outside_bound(tmp_path):
    _, _, _, _, authorization = _fixture(tmp_path)
    checkpoint = authorization.run_root / "checkpoint.pt"
    metric = authorization.run_root / "metric.json"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"x")
    metric.write_bytes(b"y")
    with pytest.raises(
        PretrainingContinuationError,
        match="outside second tranche",
    ):
        build_continuation_state_payload(
            authorization,
            optimizer_step=9,
            supervised_tokens_seen=589_824,
            physical_input_positions_seen=590_000,
            microbatches_seen=600,
            data_state={
                "order_position": 600,
                "block_input_offset": 0,
                "supervised_tokens_consumed": 589_824,
            },
            checkpoint_path=checkpoint,
            checkpoint_sha256=sha256_file(checkpoint),
            metric_path=metric,
            metric_sha256=sha256_file(metric),
        )
