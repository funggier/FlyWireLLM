from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from flywire_llm.pretraining_post_warmup_accelerated import sha256_file
from flywire_llm.pretraining_post_warmup_accelerated_evidence import (
    PretrainingPostWarmupAcceleratedEvidenceError,
    load_completed_post_warmup_accelerated_state,
)


def _fixture(tmp_path):
    run_root = tmp_path / "run"
    checkpoint = run_root / "checkpoints" / "step-000425.pt"
    metric = run_root / "metrics" / "step-000425.json"
    checkpoint.parent.mkdir(parents=True)
    metric.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"step425")
    metric.write_text("{}\n", encoding="utf-8")
    authorization = SimpleNamespace(
        authorization_id="base50m-post-warmup-5-v1",
        authorization_sha256="a" * 64,
        checkpoint_lane="research_only",
        run_root=run_root,
    )
    payload = {
        "schema_version": 1,
        "stage": "L004",
        "run_id": authorization.authorization_id,
        "authorization_sha256": authorization.authorization_sha256,
        "checkpoint_lane": "research_only",
        "rolling_checkpoint_retention": 1,
        "prune_prior_checkpoint_after_state_commit": True,
        "source_optimizer_step": 297,
        "optimizer_step": 425,
        "supervised_tokens_seen": 27_852_800,
        "physical_input_positions_seen": 27_928_093,
        "microbatches_seen": 27_697,
        "data_state": {
            "order_position": 27_273,
            "block_input_offset": 541,
            "supervised_tokens_consumed": 27_852_800,
        },
        "checkpoint": {
            "file": "checkpoints/step-000425.pt",
            "bytes": checkpoint.stat().st_size,
            "sha256": sha256_file(checkpoint),
        },
        "metric": {
            "file": "metrics/step-000425.json",
            "sha256": sha256_file(metric),
        },
        "tranche_complete": True,
        "public_release_eligibility": "not_qualified",
        "automatic_flywiremodel_export_allowed": False,
    }
    state_path = run_root / "state.json"
    state_path.write_text(json.dumps(payload), encoding="utf-8")
    return authorization, state_path, payload


def test_completed_accelerated_state_accepts_step425(tmp_path):
    authorization, state_path, _ = _fixture(tmp_path)

    state = load_completed_post_warmup_accelerated_state(
        state_path,
        authorization=authorization,
        verify_files=True,
    )

    assert state.optimizer_step == 425
    assert state.supervised_tokens_seen == 27_852_800
    assert state.checkpoint_file == "checkpoints/step-000425.pt"


def test_completed_accelerated_state_rejects_nonfinal_step(tmp_path):
    authorization, state_path, payload = _fixture(tmp_path)
    payload["optimizer_step"] = 424
    payload["supervised_tokens_seen"] = 27_787_264
    payload["data_state"]["supervised_tokens_consumed"] = 27_787_264
    payload["tranche_complete"] = False
    state_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        PretrainingPostWarmupAcceleratedEvidenceError,
        match="optimizer step 425",
    ):
        load_completed_post_warmup_accelerated_state(
            state_path,
            authorization=authorization,
            verify_files=False,
        )
