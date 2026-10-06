from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from flywire_llm.config import BlankLLMConfig
from flywire_llm.model import BlankCausalLM
from flywire_llm.pretraining_checkpoint import (
    PretrainingLineage,
    PretrainingProgress,
    load_pretraining_checkpoint,
    restore_pretraining_rng,
    save_pretraining_checkpoint,
)
from flywire_llm.pretraining_data import (
    PackedDataCursor,
    PackedPretrainingData,
    PretrainingDataContract,
    cursor_from_state_dict,
    cursor_state_dict,
)
from flywire_llm.pretraining_training import (
    build_adamw,
    build_primary_schedule,
    learning_rate_for_update,
    load_training_configuration,
    pretraining_update,
)


ROOT = Path(__file__).resolve().parents[1]


def _training():
    return load_training_configuration(
        ROOT / "configs" / "training-base-50m-v1.json"
    )


def _lineage() -> PretrainingLineage:
    return PretrainingLineage(
        checkpoint_lane="research_only",
        l003_freeze_sha256="1" * 64,
        source_manifest_sha256="2" * 64,
        global_manifest_report_sha256="3" * 64,
        decisions_sha256="4" * 64,
        tokenizer_sha256="5" * 64,
        model_config_sha256="6" * 64,
        training_config_sha256="7" * 64,
        data_runtime_sha256="8" * 64,
    )


def _data_contract(tmp_path: Path) -> PretrainingDataContract:
    tokens = np.array(
        [4, 5, 2, 6, 7, 8, 2, 9, 10],
        dtype="<u2",
    )
    order = np.array([1, 0], dtype="<u4")
    token_path = tmp_path / "tokens.u16"
    order_path = tmp_path / "order.u32"
    tokens.tofile(token_path)
    order.tofile(order_path)
    return PretrainingDataContract(
        data_runtime_id="test",
        packing_summary_sha256="a" * 64,
        block_order_summary_sha256="b" * 64,
        token_stream_path=token_path,
        token_stream_sha256="c" * 64,
        token_stream_bytes=token_path.stat().st_size,
        physical_tokens=9,
        input_positions=8,
        block_order_path=order_path,
        block_order_sha256="d" * 64,
        block_order_bytes=order_path.stat().st_size,
        blocks=2,
        max_physical_sequence_length=4,
        eos_id=2,
        ignore_index=-100,
        global_supervised_tokens_per_update=4,
        final_update_supervised_tokens=2,
        total_supervised_tokens=6,
    )


def _batches(data, cursor, budget):
    packed = list(
        data.iter_update_microbatches(
            cursor,
            batch_size=1,
            max_input_length=4,
            supervised_budget=budget,
        )
    )
    return (
        [(item.input_ids, item.targets) for item in packed],
        packed[-1].cursor,
    )


def test_checkpoint_resume_preserves_packed_data_cursor_mid_block(tmp_path):
    training = _training()
    schedule = build_primary_schedule(training)
    contract = _data_contract(tmp_path)

    torch.manual_seed(91)
    model = BlankCausalLM(BlankLLMConfig.smoke())
    optimizer = build_adamw(model, training)
    data_generator = torch.Generator().manual_seed(92)

    with PackedPretrainingData(contract) as data:
        first_batches, first_cursor = _batches(
            data,
            PackedDataCursor(),
            4,
        )
    pretraining_update(
        model,
        optimizer,
        first_batches,
        learning_rate=learning_rate_for_update(schedule, 1),
        clip_norm=1.0,
    )
    assert first_cursor.block_input_offset == 1
    assert first_cursor.supervised_tokens_consumed == 4

    checkpoint = tmp_path / "packed-resume.pt"
    save_pretraining_checkpoint(
        checkpoint,
        model,
        optimizer,
        lineage=_lineage(),
        progress=PretrainingProgress(
            optimizer_step=1,
            supervised_tokens_seen=4,
            microbatches_seen=len(first_batches),
            physical_input_positions_seen=sum(
                int(ids.numel()) for ids, _ in first_batches
            ),
        ),
        primary_token_budget=6,
        primary_seed=1234,
        data_generator=data_generator,
        data_state=cursor_state_dict(first_cursor),
    )

    with PackedPretrainingData(contract) as data:
        continuous_batches, final_cursor = _batches(
            data,
            first_cursor,
            2,
        )
    continuous_result = pretraining_update(
        model,
        optimizer,
        continuous_batches,
        learning_rate=learning_rate_for_update(schedule, 2),
        clip_norm=1.0,
    )
    assert final_cursor.supervised_tokens_consumed == 6

    loaded = load_pretraining_checkpoint(
        checkpoint,
        expected_lineage=_lineage(),
    )
    resumed_optimizer = build_adamw(loaded.model, training)
    resumed_optimizer.load_state_dict(loaded.optimizer_state)
    resumed_generator = torch.Generator()
    restore_pretraining_rng(
        loaded,
        data_generator=resumed_generator,
    )
    restored_cursor = cursor_from_state_dict(
        loaded.data_state,
        contract=contract,
    )
    assert restored_cursor == first_cursor

    with PackedPretrainingData(contract) as data:
        resumed_batches, resumed_final_cursor = _batches(
            data,
            restored_cursor,
            2,
        )
    for (expected_ids, expected_targets), (
        resumed_ids,
        resumed_targets,
    ) in zip(continuous_batches, resumed_batches):
        assert torch.equal(expected_ids, resumed_ids)
        assert torch.equal(expected_targets, resumed_targets)

    resumed_result = pretraining_update(
        loaded.model,
        resumed_optimizer,
        resumed_batches,
        learning_rate=learning_rate_for_update(schedule, 2),
        clip_norm=1.0,
    )
    assert resumed_final_cursor == final_cursor
    assert resumed_result.mean_loss == continuous_result.mean_loss
    for name, value in model.state_dict().items():
        torch.testing.assert_close(
            loaded.model.state_dict()[name],
            value,
            rtol=0.0,
            atol=0.0,
        )
