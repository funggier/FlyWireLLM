from __future__ import annotations

from pathlib import Path

import pytest
import torch

from flywire_llm.config import BlankLLMConfig
from flywire_llm.model import BlankCausalLM
from flywire_llm.pretraining_checkpoint import (
    PretrainingCheckpointError,
    PretrainingLineage,
    PretrainingProgress,
    load_pretraining_checkpoint,
    restore_pretraining_rng,
    save_pretraining_checkpoint,
)
from flywire_llm.pretraining_training import (
    build_adamw,
    learning_rate_for_update,
    build_primary_schedule,
    load_training_configuration,
    pretraining_update,
)


ROOT = Path(__file__).resolve().parents[1]


def _lineage(**overrides):
    payload = {
        "checkpoint_lane": "research_only",
        "l003_freeze_sha256": "1" * 64,
        "source_manifest_sha256": "2" * 64,
        "global_manifest_report_sha256": "3" * 64,
        "decisions_sha256": "4" * 64,
        "tokenizer_sha256": "5" * 64,
        "model_config_sha256": "6" * 64,
        "training_config_sha256": "7" * 64,
        "data_runtime_sha256": "8" * 64,
    }
    payload.update(overrides)
    return PretrainingLineage(**payload)


def _data_state(tokens: int) -> dict[str, int]:
    return {
        "order_position": 0,
        "block_input_offset": tokens,
        "supervised_tokens_consumed": tokens,
    }


def _training():
    return load_training_configuration(
        ROOT / "configs" / "training-base-50m-v1.json"
    )


def _next_batch(generator: torch.Generator):
    ids = torch.randint(
        4,
        260,
        (1, 16),
        generator=generator,
        dtype=torch.long,
    )
    return ids, ids.roll(-1, dims=1)


def test_pretraining_checkpoint_round_trip_preserves_state(tmp_path):
    torch.manual_seed(41)
    model = BlankCausalLM(BlankLLMConfig.smoke())
    optimizer = build_adamw(model, _training())
    data_generator = torch.Generator().manual_seed(99)
    schedule = build_primary_schedule(_training())

    batch = _next_batch(data_generator)
    pretraining_update(
        model,
        optimizer,
        [batch],
        learning_rate=learning_rate_for_update(schedule, 1),
        clip_norm=1.0,
    )
    progress = PretrainingProgress(
        optimizer_step=1,
        supervised_tokens_seen=16,
        microbatches_seen=1,
        physical_input_positions_seen=16,
    )
    path = tmp_path / "step1.pt"
    save_pretraining_checkpoint(
        path,
        model,
        optimizer,
        lineage=_lineage(),
        progress=progress,
        primary_token_budget=500_000_000,
        primary_seed=1234,
        data_generator=data_generator,
        data_state=_data_state(16),
    )
    loaded = load_pretraining_checkpoint(
        path,
        expected_lineage=_lineage(),
    )
    assert loaded.progress == progress
    assert loaded.data_state == _data_state(16)
    assert loaded.metadata["pretraining_started"] is True
    assert loaded.metadata["pretraining_complete"] is False
    assert loaded.metadata["external_pretrained_weights"] is False
    for name, value in model.state_dict().items():
        torch.testing.assert_close(
            loaded.model.state_dict()[name],
            value,
            rtol=0.0,
            atol=0.0,
        )


def test_resume_matches_continuous_second_update(tmp_path):
    training = _training()
    schedule = build_primary_schedule(training)
    torch.manual_seed(123)
    continuous_model = BlankCausalLM(BlankLLMConfig.smoke())
    continuous_optimizer = build_adamw(continuous_model, training)
    continuous_data = torch.Generator().manual_seed(456)

    first_batch = _next_batch(continuous_data)
    pretraining_update(
        continuous_model,
        continuous_optimizer,
        [first_batch],
        learning_rate=learning_rate_for_update(schedule, 1),
        clip_norm=1.0,
    )
    checkpoint = tmp_path / "resume.pt"
    save_pretraining_checkpoint(
        checkpoint,
        continuous_model,
        continuous_optimizer,
        lineage=_lineage(),
        progress=PretrainingProgress(
            optimizer_step=1,
            supervised_tokens_seen=16,
            microbatches_seen=1,
            physical_input_positions_seen=16,
        ),
        primary_token_budget=500_000_000,
        primary_seed=1234,
        data_generator=continuous_data,
        data_state=_data_state(16),
    )

    second_batch = _next_batch(continuous_data)
    continuous_result = pretraining_update(
        continuous_model,
        continuous_optimizer,
        [second_batch],
        learning_rate=learning_rate_for_update(schedule, 2),
        clip_norm=1.0,
    )

    loaded = load_pretraining_checkpoint(
        checkpoint,
        expected_lineage=_lineage(),
    )
    resumed_optimizer = build_adamw(loaded.model, training)
    resumed_optimizer.load_state_dict(loaded.optimizer_state)
    resumed_data = torch.Generator()
    restore_pretraining_rng(
        loaded,
        data_generator=resumed_data,
    )
    resumed_batch = _next_batch(resumed_data)
    assert torch.equal(second_batch[0], resumed_batch[0])
    assert torch.equal(second_batch[1], resumed_batch[1])

    resumed_result = pretraining_update(
        loaded.model,
        resumed_optimizer,
        [resumed_batch],
        learning_rate=learning_rate_for_update(schedule, 2),
        clip_norm=1.0,
    )
    assert resumed_result.mean_loss == pytest.approx(
        continuous_result.mean_loss,
        rel=0,
        abs=0,
    )
    for name, value in continuous_model.state_dict().items():
        torch.testing.assert_close(
            loaded.model.state_dict()[name],
            value,
            rtol=0.0,
            atol=0.0,
        )


def test_checkpoint_rejects_lineage_change(tmp_path):
    model = BlankCausalLM(BlankLLMConfig.smoke())
    optimizer = build_adamw(model, _training())
    generator = torch.Generator().manual_seed(1)
    path = tmp_path / "lineage.pt"
    save_pretraining_checkpoint(
        path,
        model,
        optimizer,
        lineage=_lineage(),
        progress=PretrainingProgress(0, 0, 0, 0),
        primary_token_budget=500_000_000,
        primary_seed=1234,
        data_generator=generator,
        data_state=_data_state(0),
    )
    with pytest.raises(
        PretrainingCheckpointError,
        match="lineage does not match",
    ):
        load_pretraining_checkpoint(
            path,
            expected_lineage=_lineage(decisions_sha256="9" * 64),
        )
