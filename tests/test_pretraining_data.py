from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from flywire_llm.pretraining_data import (
    PackedDataCursor,
    PackedPretrainingData,
    PretrainingDataContract,
    PretrainingDataError,
    cursor_from_state_dict,
    cursor_state_dict,
)


def _contract(tmp_path: Path) -> PretrainingDataContract:
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
        packing_summary_sha256="1" * 64,
        block_order_summary_sha256="2" * 64,
        token_stream_path=token_path,
        token_stream_sha256="3" * 64,
        token_stream_bytes=token_path.stat().st_size,
        physical_tokens=9,
        input_positions=8,
        block_order_path=order_path,
        block_order_sha256="4" * 64,
        block_order_bytes=order_path.stat().st_size,
        blocks=2,
        max_physical_sequence_length=4,
        eos_id=2,
        ignore_index=-100,
        global_supervised_tokens_per_update=4,
        final_update_supervised_tokens=2,
        total_supervised_tokens=6,
    )


def test_packed_runtime_follows_block_order_and_masks_eos(tmp_path):
    contract = _contract(tmp_path)
    with PackedPretrainingData(contract) as data:
        first = data.next_sequence(PackedDataCursor())
        assert first.input_ids.tolist() == [7, 8, 2, 9]
        assert first.targets.tolist() == [8, 2, -100, 10]
        assert first.supervised_positions == 3
        assert first.eos_input_positions == 1
        assert first.cursor == PackedDataCursor(
            order_position=1,
            block_input_offset=0,
            supervised_tokens_consumed=3,
        )

        second = data.next_sequence(first.cursor)
        assert second.input_ids.tolist() == [4, 5, 2, 6]
        assert second.targets.tolist() == [5, 2, -100, 7]
        assert second.supervised_positions == 3
        assert second.cursor == PackedDataCursor(
            order_position=2,
            block_input_offset=0,
            supervised_tokens_consumed=6,
        )


def test_update_budget_truncates_and_next_update_resumes_mid_block(tmp_path):
    contract = _contract(tmp_path)
    with PackedPretrainingData(contract) as data:
        first_update = list(
            data.iter_update_microbatches(
                PackedDataCursor(),
                batch_size=1,
                max_input_length=4,
                supervised_budget=4,
            )
        )
        assert sum(
            batch.supervised_positions for batch in first_update
        ) == 4
        assert first_update[-1].cursor == PackedDataCursor(
            order_position=1,
            block_input_offset=1,
            supervised_tokens_consumed=4,
        )

        second_update = list(
            data.iter_update_microbatches(
                first_update[-1].cursor,
                batch_size=1,
                max_input_length=4,
                supervised_budget=2,
            )
        )
        assert sum(
            batch.supervised_positions for batch in second_update
        ) == 2
        assert second_update[-1].cursor == PackedDataCursor(
            order_position=2,
            block_input_offset=0,
            supervised_tokens_consumed=6,
        )


def test_microbatch_padding_masks_targets(tmp_path):
    contract = _contract(tmp_path)
    with PackedPretrainingData(contract) as data:
        batch = data.next_microbatch(
            PackedDataCursor(),
            batch_size=2,
            max_input_length=4,
            max_supervised_positions=4,
        )
    assert batch.input_ids.shape == (2, 4)
    assert batch.targets.shape == (2, 4)
    assert batch.supervised_positions == 4
    assert batch.input_ids[1].tolist() == [4, 0, 0, 0]
    assert batch.targets[1].tolist() == [5, -100, -100, -100]


def test_cursor_state_round_trip(tmp_path):
    contract = _contract(tmp_path)
    cursor = PackedDataCursor(
        order_position=1,
        block_input_offset=2,
        supervised_tokens_consumed=5,
    )
    restored = cursor_from_state_dict(
        cursor_state_dict(cursor),
        contract=contract,
    )
    assert restored == cursor


def test_terminal_cursor_requires_full_supervised_budget(tmp_path):
    contract = _contract(tmp_path)
    with pytest.raises(
        PretrainingDataError,
        match="terminal cursor must consume",
    ):
        cursor_from_state_dict(
            {
                "order_position": 2,
                "block_input_offset": 0,
                "supervised_tokens_consumed": 5,
            },
            contract=contract,
        )


def test_runtime_tensors_are_torch_long(tmp_path):
    contract = _contract(tmp_path)
    with PackedPretrainingData(contract) as data:
        batch = data.next_microbatch(
            PackedDataCursor(),
            batch_size=1,
            max_input_length=4,
            max_supervised_positions=3,
        )
    assert batch.input_ids.dtype == torch.long
    assert batch.targets.dtype == torch.long
