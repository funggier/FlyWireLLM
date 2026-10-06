from __future__ import annotations

import pytest

from flywire_llm.pretraining_packing import (
    PretrainingPackingError,
    iter_packed_blocks,
    truncate_block_to_supervised_budget,
)


def test_packing_inserts_eos_masks_boundary_and_preserves_exact_content_budget():
    blocks = list(
        iter_packed_blocks(
            [
                ([4, 5, 6], 3),
                ([7, 8], 2),
            ],
            max_physical_sequence_length=4,
        )
    )
    assert len(blocks) == 2
    assert blocks[0].input_ids == (4, 5, 6, 2)
    assert blocks[0].targets == (5, 6, 2, -100)
    assert blocks[0].supervised_positions == 3
    assert blocks[0].eos_input_positions == 1

    assert blocks[1].input_ids == (7, 8)
    assert blocks[1].targets == (8, 2)
    assert blocks[1].supervised_positions == 2
    assert blocks[1].eos_input_positions == 0

    assert sum(block.supervised_positions for block in blocks) == 5


def test_packing_truncates_selected_record_prefix_before_eos():
    blocks = list(
        iter_packed_blocks(
            [
                ([10, 11, 12, 13, 14], 3),
                ([20, 21], 2),
            ],
            max_physical_sequence_length=16,
        )
    )
    assert len(blocks) == 1
    assert blocks[0].input_ids == (10, 11, 12, 2, 20, 21)
    assert blocks[0].targets == (11, 12, 2, -100, 21, 2)
    assert blocks[0].supervised_positions == 5


def test_packing_rejects_reserved_special_tokens_in_raw_content():
    with pytest.raises(
        PretrainingPackingError,
        match="reserved special token",
    ):
        list(iter_packed_blocks([([4, 2, 5], 3)]))


def test_truncate_block_hits_exact_supervised_update_budget():
    block = list(
        iter_packed_blocks(
            [
                ([4, 5, 6], 3),
                ([7, 8, 9], 3),
            ],
            max_physical_sequence_length=16,
        )
    )[0]
    first, rest = truncate_block_to_supervised_budget(
        block,
        supervised_budget=4,
    )
    assert first.supervised_positions == 4
    assert rest is not None
    assert rest.supervised_positions == 2
    assert first.input_ids == (4, 5, 6, 2, 7)
    assert first.targets == (5, 6, 2, -100, 8)
    assert rest.input_ids == (8, 9)
    assert rest.targets == (9, 2)


def test_packing_requires_at_least_one_record():
    with pytest.raises(
        PretrainingPackingError,
        match="at least one selected record",
    ):
        list(iter_packed_blocks([]))
