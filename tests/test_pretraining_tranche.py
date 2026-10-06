from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from flywire_llm.pretraining_tranche import (
    PretrainingTrancheError,
    atomic_write_json,
    build_state_payload,
    load_bounded_tranche_authorization,
    load_tranche_state,
    sha256_file,
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _authorization_fixture(tmp_path: Path):
    repo = tmp_path / "repo"
    external = tmp_path / "external"
    repo.mkdir()
    external.mkdir()
    pinned = repo / "runtime.json"
    pinned.write_text("{}\n", encoding="utf-8", newline="\n")
    payload = {
        "schema_version": 1,
        "stage": "L004",
        "authorization_id": "base50m-first-tranche-v1",
        "foundation_commit": "1" * 40,
        "runner_commit": "2" * 40,
        "checkpoint_lane": "research_only",
        "start_optimizer_step": 0,
        "max_optimizer_updates": 4,
        "target_supervised_tokens": 262_144,
        "checkpoint_every_updates": 1,
        "run_root": (
            "external://FlyWireLLM-data/L004/Runs/"
            "base50m-first-tranche-v1"
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
    path = repo / "authorization.json"
    path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    authorization = load_bounded_tranche_authorization(
        path,
        repo_root=repo,
        external_root=external,
    )
    return repo, external, path, payload, authorization


def test_bounded_authorization_pins_repo_files_and_limits_four_updates(
    tmp_path,
):
    _, external, _, _, authorization = _authorization_fixture(tmp_path)
    assert authorization.max_optimizer_updates == 4
    assert authorization.target_supervised_tokens == 262_144
    assert authorization.checkpoint_every_updates == 1
    assert authorization.checkpoint_lane == "research_only"
    assert authorization.run_root == (
        external
        / "L004"
        / "Runs"
        / "base50m-first-tranche-v1"
    ).resolve()


def test_bounded_authorization_rejects_repo_pin_tamper(tmp_path):
    repo, external, path, _, _ = _authorization_fixture(tmp_path)
    (repo / "runtime.json").write_text(
        '{"tampered": true}\n',
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(
        PretrainingTrancheError,
        match="SHA-256 mismatch",
    ):
        load_bounded_tranche_authorization(
            path,
            repo_root=repo,
            external_root=external,
        )


def test_bounded_authorization_rejects_larger_tranche(tmp_path):
    repo, external, path, payload, _ = _authorization_fixture(tmp_path)
    payload["max_optimizer_updates"] = 5
    path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(
        PretrainingTrancheError,
        match="limited to 4 updates",
    ):
        load_bounded_tranche_authorization(
            path,
            repo_root=repo,
            external_root=external,
        )


def test_state_round_trip_verifies_checkpoint_metric_and_progress(tmp_path):
    _, _, _, _, authorization = _authorization_fixture(tmp_path)
    checkpoint = authorization.run_root / "checkpoints" / "step-000001.pt"
    metric = authorization.run_root / "metrics" / "step-000001.json"
    checkpoint.parent.mkdir(parents=True)
    metric.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    metric.write_bytes(b"metric")
    data_state = {
        "order_position": 64,
        "block_input_offset": 264,
        "supervised_tokens_consumed": 65_536,
    }
    payload = build_state_payload(
        authorization,
        optimizer_step=1,
        supervised_tokens_seen=65_536,
        physical_input_positions_seen=65_800,
        microbatches_seen=65,
        data_state=data_state,
        checkpoint_path=checkpoint,
        checkpoint_sha256=_sha(b"checkpoint"),
        metric_path=metric,
        metric_sha256=_sha(b"metric"),
    )
    state_path = authorization.run_root / "state.json"
    atomic_write_json(state_path, payload)
    state = load_tranche_state(
        state_path,
        authorization=authorization,
        verify_files=True,
    )
    assert state.optimizer_step == 1
    assert state.supervised_tokens_seen == 65_536
    assert state.physical_input_positions_seen == 65_800
    assert state.checkpoint_file == "checkpoints/step-000001.pt"
    assert state.metric_file == "metrics/step-000001.json"


def test_state_rejects_checkpoint_hash_tamper(tmp_path):
    _, _, _, _, authorization = _authorization_fixture(tmp_path)
    checkpoint = authorization.run_root / "checkpoints" / "step-000001.pt"
    metric = authorization.run_root / "metrics" / "step-000001.json"
    checkpoint.parent.mkdir(parents=True)
    metric.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    metric.write_bytes(b"metric")
    payload = build_state_payload(
        authorization,
        optimizer_step=1,
        supervised_tokens_seen=65_536,
        physical_input_positions_seen=65_800,
        microbatches_seen=65,
        data_state={
            "order_position": 64,
            "block_input_offset": 264,
            "supervised_tokens_consumed": 65_536,
        },
        checkpoint_path=checkpoint,
        checkpoint_sha256=_sha(b"checkpoint"),
        metric_path=metric,
        metric_sha256=_sha(b"metric"),
    )
    state_path = authorization.run_root / "state.json"
    atomic_write_json(state_path, payload)
    checkpoint.write_bytes(b"changed")
    with pytest.raises(
        PretrainingTrancheError,
        match="checkpoint byte size mismatch|checkpoint SHA-256 mismatch",
    ):
        load_tranche_state(
            state_path,
            authorization=authorization,
            verify_files=True,
        )


def test_state_rejects_progress_outside_authorization(tmp_path):
    _, _, _, _, authorization = _authorization_fixture(tmp_path)
    checkpoint = authorization.run_root / "checkpoint.pt"
    metric = authorization.run_root / "metric.json"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"x")
    metric.write_bytes(b"y")
    with pytest.raises(
        PretrainingTrancheError,
        match="outside bounded tranche",
    ):
        build_state_payload(
            authorization,
            optimizer_step=5,
            supervised_tokens_seen=327_680,
            physical_input_positions_seen=328_000,
            microbatches_seen=325,
            data_state={
                "order_position": 320,
                "block_input_offset": 0,
                "supervised_tokens_consumed": 327_680,
            },
            checkpoint_path=checkpoint,
            checkpoint_sha256=_sha(b"x"),
            metric_path=metric,
            metric_sha256=_sha(b"y"),
        )
