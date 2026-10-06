from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence


class PretrainingPackingError(ValueError):
    pass


@dataclass(frozen=True)
class PackedBlock:
    input_ids: tuple[int, ...]
    targets: tuple[int, ...]
    supervised_positions: int
    eos_input_positions: int


def _validate_content_tokens(
    token_ids: Sequence[int],
    *,
    selected_token_count: int,
    vocab_size: int,
    eos_id: int,
) -> tuple[int, ...]:
    if not isinstance(selected_token_count, int) or selected_token_count <= 0:
        raise PretrainingPackingError(
            "selected_token_count must be positive"
        )
    if selected_token_count > len(token_ids):
        raise PretrainingPackingError(
            "selected_token_count exceeds encoded record length"
        )
    selected = tuple(
        int(token_id)
        for token_id in token_ids[:selected_token_count]
    )
    for token_id in selected:
        if token_id < 0 or token_id >= vocab_size:
            raise PretrainingPackingError(
                "content token is outside tokenizer vocabulary"
            )
        if token_id in {0, 1, 2, 3}:
            raise PretrainingPackingError(
                "raw content unexpectedly emitted a reserved special token"
            )
    if eos_id != 2:
        raise PretrainingPackingError(
            "Base-50M packing contract requires eos_id=2"
        )
    return selected


def _make_block(
    physical_tokens: Sequence[int],
    *,
    eos_id: int,
    ignore_index: int,
) -> PackedBlock:
    if len(physical_tokens) < 2:
        raise PretrainingPackingError(
            "packed block requires input plus one lookahead token"
        )
    input_ids = tuple(int(value) for value in physical_tokens[:-1])
    raw_targets = tuple(int(value) for value in physical_tokens[1:])
    targets = tuple(
        ignore_index if input_id == eos_id else target
        for input_id, target in zip(input_ids, raw_targets)
    )
    eos_inputs = sum(input_id == eos_id for input_id in input_ids)
    supervised = len(input_ids) - eos_inputs
    if supervised <= 0:
        raise PretrainingPackingError(
            "packed block has no supervised content positions"
        )
    return PackedBlock(
        input_ids=input_ids,
        targets=targets,
        supervised_positions=supervised,
        eos_input_positions=eos_inputs,
    )


def iter_packed_blocks(
    records: Iterable[tuple[Sequence[int], int]],
    *,
    max_physical_sequence_length: int = 1024,
    vocab_size: int = 32_000,
    eos_id: int = 2,
    ignore_index: int = -100,
) -> Iterator[PackedBlock]:
    if max_physical_sequence_length < 2:
        raise PretrainingPackingError(
            "max physical sequence length must be at least 2"
        )
    buffer: list[int] = []
    records_seen = 0
    content_tokens = 0

    for token_ids, selected_token_count in records:
        selected = _validate_content_tokens(
            token_ids,
            selected_token_count=selected_token_count,
            vocab_size=vocab_size,
            eos_id=eos_id,
        )
        buffer.extend(selected)
        buffer.append(eos_id)
        records_seen += 1
        content_tokens += len(selected)

        while len(buffer) >= max_physical_sequence_length + 1:
            physical = buffer[: max_physical_sequence_length + 1]
            yield _make_block(
                physical,
                eos_id=eos_id,
                ignore_index=ignore_index,
            )
            del buffer[:max_physical_sequence_length]

    if records_seen == 0:
        raise PretrainingPackingError(
            "packing requires at least one selected record"
        )
    if len(buffer) >= 2:
        yield _make_block(
            buffer,
            eos_id=eos_id,
            ignore_index=ignore_index,
        )

    # The stream always ends in EOS. Therefore every selected content token is
    # an input position exactly once, while every EOS input position is masked.
    # Consumers can assert aggregate supervised positions == content_tokens.


def truncate_block_to_supervised_budget(
    block: PackedBlock,
    *,
    supervised_budget: int,
    ignore_index: int = -100,
) -> tuple[PackedBlock, PackedBlock | None]:
    if not isinstance(supervised_budget, int) or supervised_budget <= 0:
        raise PretrainingPackingError(
            "supervised_budget must be positive"
        )
    if supervised_budget >= block.supervised_positions:
        return block, None

    supervised = 0
    split_index = None
    for index, target in enumerate(block.targets, start=1):
        if target != ignore_index:
            supervised += 1
        if supervised == supervised_budget:
            split_index = index
            break
    if split_index is None:
        raise PretrainingPackingError(
            "unable to locate supervised split boundary"
        )

    first_inputs = block.input_ids[:split_index]
    first_targets = block.targets[:split_index]
    first = PackedBlock(
        input_ids=first_inputs,
        targets=first_targets,
        supervised_positions=supervised_budget,
        eos_input_positions=sum(
            token_id == 2 for token_id in first_inputs
        ),
    )

    remaining_inputs = block.input_ids[split_index:]
    remaining_targets = block.targets[split_index:]
    if not remaining_inputs:
        return first, None
    remaining_supervised = sum(
        target != ignore_index for target in remaining_targets
    )
    if remaining_supervised == 0:
        return first, None
    rest = PackedBlock(
        input_ids=remaining_inputs,
        targets=remaining_targets,
        supervised_positions=remaining_supervised,
        eos_input_positions=sum(
            token_id == 2 for token_id in remaining_inputs
        ),
    )
    return first, rest
