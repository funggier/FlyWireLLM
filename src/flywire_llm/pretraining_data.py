from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .artifacts import resolve_external_storage_uri


class PretrainingDataError(ValueError):
    pass


@dataclass(frozen=True)
class PretrainingDataContract:
    data_runtime_id: str
    packing_summary_sha256: str
    block_order_summary_sha256: str
    token_stream_path: Path
    token_stream_sha256: str
    token_stream_bytes: int
    physical_tokens: int
    input_positions: int
    block_order_path: Path
    block_order_sha256: str
    block_order_bytes: int
    blocks: int
    max_physical_sequence_length: int
    eos_id: int
    ignore_index: int
    global_supervised_tokens_per_update: int
    final_update_supervised_tokens: int
    total_supervised_tokens: int


@dataclass(frozen=True)
class PackedDataCursor:
    order_position: int = 0
    block_input_offset: int = 0
    supervised_tokens_consumed: int = 0


@dataclass(frozen=True)
class PackedSequence:
    input_ids: torch.Tensor
    targets: torch.Tensor
    supervised_positions: int
    eos_input_positions: int
    cursor: PackedDataCursor


@dataclass(frozen=True)
class PackedMicrobatch:
    input_ids: torch.Tensor
    targets: torch.Tensor
    supervised_positions: int
    cursor: PackedDataCursor


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _repo_file(
    repo_root: Path,
    block: object,
    *,
    field: str,
) -> tuple[Path, str]:
    if not isinstance(block, dict):
        raise PretrainingDataError(f"{field} must be an object")
    relative = block.get("path")
    expected = block.get("sha256")
    if not isinstance(relative, str) or not relative:
        raise PretrainingDataError(f"{field} path must be non-empty")
    if not _valid_sha256(expected):
        raise PretrainingDataError(f"{field} SHA-256 is invalid")
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise PretrainingDataError(f"{field} path must be repo-relative")
    root = repo_root.resolve()
    path = (root / candidate).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise PretrainingDataError(
            f"{field} path escapes repository"
        ) from exc
    if not path.is_file():
        raise PretrainingDataError(f"{field} file is missing")
    observed = sha256_file(path)
    if observed != expected:
        raise PretrainingDataError(
            f"{field} SHA-256 mismatch expected={expected} "
            f"observed={observed}"
        )
    return path, expected


def _external_file(
    block: object,
    *,
    external_root: str | Path,
    field: str,
    expected_dtype: str,
    verify_hash: bool,
) -> tuple[Path, str, int]:
    if not isinstance(block, dict):
        raise PretrainingDataError(f"{field} must be an object")
    storage = block.get("storage")
    expected_sha = block.get("sha256")
    expected_bytes = block.get("bytes")
    if block.get("dtype") != expected_dtype:
        raise PretrainingDataError(f"{field} dtype changed")
    if not _valid_sha256(expected_sha):
        raise PretrainingDataError(f"{field} SHA-256 is invalid")
    if not isinstance(expected_bytes, int) or expected_bytes <= 0:
        raise PretrainingDataError(f"{field} bytes must be positive")
    path = resolve_external_storage_uri(
        storage,
        external_root=external_root,
    )
    if verify_hash:
        if not path.is_file():
            raise PretrainingDataError(f"{field} file is missing")
        if path.stat().st_size != expected_bytes:
            raise PretrainingDataError(
                f"{field} byte size mismatch expected={expected_bytes} "
                f"observed={path.stat().st_size}"
            )
        observed = sha256_file(path)
        if observed != expected_sha:
            raise PretrainingDataError(
                f"{field} SHA-256 mismatch expected={expected_sha} "
                f"observed={observed}"
            )
    return path, expected_sha, expected_bytes


def load_pretraining_data_contract(
    path: str | Path,
    *,
    repo_root: str | Path,
    external_root: str | Path,
    verify_external_hashes: bool = True,
) -> PretrainingDataContract:
    repo_root = Path(repo_root)
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PretrainingDataError(
            "pretraining data runtime must be a JSON object"
        )
    if payload.get("schema_version") != 1:
        raise PretrainingDataError(
            "unsupported pretraining data runtime schema"
        )
    if payload.get("stage") != "L004":
        raise PretrainingDataError(
            "pretraining data runtime stage must be L004"
        )
    if payload.get("data_runtime_id") != "base50m-packed-runtime-v1":
        raise PretrainingDataError(
            "pretraining data runtime id changed"
        )
    if payload.get("production_optimizer_steps_recorded") != 0:
        raise PretrainingDataError(
            "data qualification cannot record production optimizer steps"
        )
    if payload.get("pretraining_authorized") is not False:
        raise PretrainingDataError(
            "data runtime cannot self-authorize pretraining"
        )

    packing_path, packing_sha = _repo_file(
        repo_root,
        payload.get("packing_summary"),
        field="packing_summary",
    )
    order_summary_path, order_summary_sha = _repo_file(
        repo_root,
        payload.get("block_order_summary"),
        field="block_order_summary",
    )
    packing = json.loads(packing_path.read_text(encoding="utf-8"))
    order_summary = json.loads(
        order_summary_path.read_text(encoding="utf-8")
    )
    if packing.get("schema_version") != 1 or packing.get("stage") != "L004":
        raise PretrainingDataError("packing summary schema/stage changed")
    if packing.get("packing_id") != "base50m-packed-u16-v1":
        raise PretrainingDataError("packing summary id changed")
    if packing.get("pretraining_authorized") is not False:
        raise PretrainingDataError(
            "packing summary cannot self-authorize"
        )
    if order_summary.get("schema_version") != 1 or (
        order_summary.get("stage") != "L004"
    ):
        raise PretrainingDataError(
            "block-order summary schema/stage changed"
        )
    if order_summary.get("packing_id") != packing.get("packing_id"):
        raise PretrainingDataError(
            "block-order/packing summary id mismatch"
        )
    if order_summary.get("pretraining_authorized") is not False:
        raise PretrainingDataError(
            "block-order summary cannot self-authorize"
        )

    token_block = payload.get("token_stream")
    order_block = payload.get("block_order")
    token_path, token_sha, token_bytes = _external_file(
        token_block,
        external_root=external_root,
        field="token_stream",
        expected_dtype="uint16_le",
        verify_hash=verify_external_hashes,
    )
    order_path, order_sha, order_bytes = _external_file(
        order_block,
        external_root=external_root,
        field="block_order",
        expected_dtype="uint32_le",
        verify_hash=verify_external_hashes,
    )

    if not isinstance(token_block, dict) or not isinstance(order_block, dict):
        raise PretrainingDataError(
            "token_stream and block_order must be objects"
        )
    physical_tokens = token_block.get("physical_tokens")
    input_positions = token_block.get("input_positions")
    blocks = order_block.get("blocks")
    if physical_tokens != 501_358_839:
        raise PretrainingDataError(
            "packed physical token count changed"
        )
    if input_positions != 501_358_838:
        raise PretrainingDataError(
            "packed input-position count changed"
        )
    if blocks != 489_609:
        raise PretrainingDataError("packed block count changed")
    if token_bytes != physical_tokens * 2:
        raise PretrainingDataError(
            "packed token stream byte contract changed"
        )
    if order_bytes != blocks * 4:
        raise PretrainingDataError(
            "block-order byte contract changed"
        )

    packing_stream = packing.get("token_stream")
    packing_index = packing.get("block_index")
    order_meta = order_summary.get("block_order")
    if not isinstance(packing_stream, dict) or not isinstance(
        packing_index, dict
    ) or not isinstance(order_meta, dict):
        raise PretrainingDataError(
            "packing/block-order metadata is incomplete"
        )
    if (
        packing_stream.get("sha256") != token_sha
        or packing_stream.get("bytes") != token_bytes
        or packing_stream.get("physical_tokens") != physical_tokens
    ):
        raise PretrainingDataError(
            "token stream does not match packing summary"
        )
    if (
        order_meta.get("sha256") != order_sha
        or order_meta.get("bytes") != order_bytes
        or order_meta.get("blocks") != blocks
        or order_meta.get("complete_permutation") is not True
    ):
        raise PretrainingDataError(
            "block order does not match summary"
        )
    if packing_index.get("blocks") != blocks:
        raise PretrainingDataError(
            "block index count does not match order"
        )

    max_sequence = payload.get("max_physical_sequence_length")
    eos_id = payload.get("eos_id")
    ignore_index = payload.get("ignore_index")
    global_tokens = payload.get("global_supervised_tokens_per_update")
    final_tokens = payload.get("final_update_supervised_tokens")
    total_tokens = payload.get("total_supervised_tokens")
    if max_sequence != 1024:
        raise PretrainingDataError(
            "max physical sequence length changed"
        )
    if eos_id != 2 or ignore_index != -100:
        raise PretrainingDataError(
            "EOS/loss masking contract changed"
        )
    if global_tokens != 65_536:
        raise PretrainingDataError(
            "global supervised tokens/update changed"
        )
    if final_tokens != 25_856:
        raise PretrainingDataError(
            "final update supervised token count changed"
        )
    if total_tokens != 500_000_000:
        raise PretrainingDataError(
            "total supervised token count changed"
        )
    if packing.get("loss_contract", {}).get(
        "supervised_positions"
    ) != total_tokens:
        raise PretrainingDataError(
            "packing supervised total mismatch"
        )

    return PretrainingDataContract(
        data_runtime_id="base50m-packed-runtime-v1",
        packing_summary_sha256=packing_sha,
        block_order_summary_sha256=order_summary_sha,
        token_stream_path=token_path,
        token_stream_sha256=token_sha,
        token_stream_bytes=token_bytes,
        physical_tokens=physical_tokens,
        input_positions=input_positions,
        block_order_path=order_path,
        block_order_sha256=order_sha,
        block_order_bytes=order_bytes,
        blocks=blocks,
        max_physical_sequence_length=max_sequence,
        eos_id=eos_id,
        ignore_index=ignore_index,
        global_supervised_tokens_per_update=global_tokens,
        final_update_supervised_tokens=final_tokens,
        total_supervised_tokens=total_tokens,
    )


def validate_cursor(
    cursor: PackedDataCursor,
    *,
    contract: PretrainingDataContract,
) -> None:
    if not isinstance(cursor.order_position, int) or (
        cursor.order_position < 0
    ):
        raise PretrainingDataError(
            "cursor order_position must be non-negative"
        )
    if not isinstance(cursor.block_input_offset, int) or (
        cursor.block_input_offset < 0
    ):
        raise PretrainingDataError(
            "cursor block_input_offset must be non-negative"
        )
    if not isinstance(cursor.supervised_tokens_consumed, int) or (
        cursor.supervised_tokens_consumed < 0
    ):
        raise PretrainingDataError(
            "cursor supervised_tokens_consumed must be non-negative"
        )
    if cursor.order_position > contract.blocks:
        raise PretrainingDataError(
            "cursor order_position exceeds block order"
        )
    if cursor.order_position == contract.blocks:
        if cursor.block_input_offset != 0:
            raise PretrainingDataError(
                "terminal cursor cannot have a block offset"
            )
        if (
            cursor.supervised_tokens_consumed
            != contract.total_supervised_tokens
        ):
            raise PretrainingDataError(
                "terminal cursor must consume the full supervised budget"
            )
    elif cursor.block_input_offset >= contract.max_physical_sequence_length:
        raise PretrainingDataError(
            "cursor block offset exceeds maximum sequence length"
        )
    if (
        cursor.supervised_tokens_consumed
        > contract.total_supervised_tokens
    ):
        raise PretrainingDataError(
            "cursor supervised total exceeds training budget"
        )


class PackedPretrainingData:
    def __init__(self, contract: PretrainingDataContract) -> None:
        self.contract = contract
        self._tokens = np.memmap(
            contract.token_stream_path,
            dtype="<u2",
            mode="r",
        )
        self._order = np.memmap(
            contract.block_order_path,
            dtype="<u4",
            mode="r",
        )
        if int(self._tokens.size) != contract.physical_tokens:
            raise PretrainingDataError(
                "token stream element count changed"
            )
        if int(self._order.size) != contract.blocks:
            raise PretrainingDataError(
                "block order element count changed"
            )

    def close(self) -> None:
        token_map = getattr(self._tokens, "_mmap", None)
        if token_map is not None:
            token_map.close()
        order_map = getattr(self._order, "_mmap", None)
        if order_map is not None:
            order_map.close()

    def __enter__(self) -> "PackedPretrainingData":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _block_input_length(self, block_index: int) -> int:
        if block_index < 0 or block_index >= self.contract.blocks:
            raise PretrainingDataError(
                f"invalid block index {block_index}"
            )
        start = (
            block_index
            * self.contract.max_physical_sequence_length
        )
        remaining = self.contract.input_positions - start
        if remaining <= 0:
            raise PretrainingDataError(
                "block starts beyond packed input positions"
            )
        return min(
            self.contract.max_physical_sequence_length,
            remaining,
        )

    def next_sequence(
        self,
        cursor: PackedDataCursor,
        *,
        max_input_length: int | None = None,
        max_supervised_positions: int | None = None,
    ) -> PackedSequence:
        validate_cursor(cursor, contract=self.contract)
        if cursor.order_position >= self.contract.blocks:
            raise StopIteration
        if max_input_length is None:
            max_input_length = (
                self.contract.max_physical_sequence_length
            )
        if not isinstance(max_input_length, int) or (
            max_input_length <= 0
            or max_input_length
            > self.contract.max_physical_sequence_length
        ):
            raise PretrainingDataError(
                "max_input_length is outside the qualified range"
            )
        if max_supervised_positions is not None and (
            not isinstance(max_supervised_positions, int)
            or max_supervised_positions <= 0
        ):
            raise PretrainingDataError(
                "max_supervised_positions must be positive"
            )

        block_index = int(self._order[cursor.order_position])
        block_length = self._block_input_length(block_index)
        offset = cursor.block_input_offset
        if offset >= block_length:
            raise PretrainingDataError(
                "cursor offset exceeds current block length"
            )
        available = min(max_input_length, block_length - offset)
        physical_start = (
            block_index
            * self.contract.max_physical_sequence_length
            + offset
        )
        inputs_np = np.asarray(
            self._tokens[
                physical_start : physical_start + available
            ],
            dtype=np.int64,
        ).copy()
        targets_np = np.asarray(
            self._tokens[
                physical_start + 1 : physical_start + available + 1
            ],
            dtype=np.int64,
        ).copy()
        if inputs_np.size != available or targets_np.size != available:
            raise PretrainingDataError(
                "packed stream ended unexpectedly"
            )
        eos_mask = inputs_np == self.contract.eos_id
        targets_np[eos_mask] = self.contract.ignore_index

        if max_supervised_positions is not None:
            supervised_flags = (
                targets_np != self.contract.ignore_index
            )
            total_supervised = int(supervised_flags.sum())
            if total_supervised > max_supervised_positions:
                cumulative = np.cumsum(
                    supervised_flags,
                    dtype=np.int64,
                )
                positions = np.flatnonzero(
                    cumulative == max_supervised_positions
                )
                if positions.size == 0:
                    raise PretrainingDataError(
                        "unable to truncate sequence to supervised budget"
                    )
                keep = int(positions[0]) + 1
                inputs_np = inputs_np[:keep]
                targets_np = targets_np[:keep]
                eos_mask = eos_mask[:keep]

        supervised = int(
            np.count_nonzero(
                targets_np != self.contract.ignore_index
            )
        )
        if supervised <= 0:
            raise PretrainingDataError(
                "packed sequence has no supervised positions"
            )
        consumed_inputs = int(inputs_np.size)
        next_offset = offset + consumed_inputs
        next_order = cursor.order_position
        if next_offset == block_length:
            next_order += 1
            next_offset = 0
        elif next_offset > block_length:
            raise PretrainingDataError(
                "sequence advanced beyond block boundary"
            )

        next_cursor = PackedDataCursor(
            order_position=next_order,
            block_input_offset=next_offset,
            supervised_tokens_consumed=(
                cursor.supervised_tokens_consumed + supervised
            ),
        )
        validate_cursor(next_cursor, contract=self.contract)
        return PackedSequence(
            input_ids=torch.from_numpy(inputs_np).long(),
            targets=torch.from_numpy(targets_np).long(),
            supervised_positions=supervised,
            eos_input_positions=int(np.count_nonzero(eos_mask)),
            cursor=next_cursor,
        )

    def next_microbatch(
        self,
        cursor: PackedDataCursor,
        *,
        batch_size: int,
        max_input_length: int,
        max_supervised_positions: int,
        pad_id: int = 0,
    ) -> PackedMicrobatch:
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise PretrainingDataError("batch_size must be positive")
        if max_supervised_positions <= 0:
            raise PretrainingDataError(
                "max_supervised_positions must be positive"
            )
        sequences: list[PackedSequence] = []
        current = cursor
        supervised = 0
        while len(sequences) < batch_size and (
            supervised < max_supervised_positions
        ):
            remaining = max_supervised_positions - supervised
            sequence = self.next_sequence(
                current,
                max_input_length=max_input_length,
                max_supervised_positions=remaining,
            )
            sequences.append(sequence)
            current = sequence.cursor
            supervised += sequence.supervised_positions

        width = max(int(seq.input_ids.numel()) for seq in sequences)
        inputs = torch.full(
            (len(sequences), width),
            int(pad_id),
            dtype=torch.long,
        )
        targets = torch.full(
            (len(sequences), width),
            int(self.contract.ignore_index),
            dtype=torch.long,
        )
        for row, sequence in enumerate(sequences):
            length = int(sequence.input_ids.numel())
            inputs[row, :length] = sequence.input_ids
            targets[row, :length] = sequence.targets
        return PackedMicrobatch(
            input_ids=inputs,
            targets=targets,
            supervised_positions=supervised,
            cursor=current,
        )

    def iter_update_microbatches(
        self,
        cursor: PackedDataCursor,
        *,
        batch_size: int = 1,
        max_input_length: int = 1024,
        supervised_budget: int | None = None,
    ):
        validate_cursor(cursor, contract=self.contract)
        remaining_total = (
            self.contract.total_supervised_tokens
            - cursor.supervised_tokens_consumed
        )
        if remaining_total <= 0:
            return
        if supervised_budget is None:
            supervised_budget = min(
                self.contract.global_supervised_tokens_per_update,
                remaining_total,
            )
        if not isinstance(supervised_budget, int) or (
            supervised_budget <= 0
            or supervised_budget > remaining_total
        ):
            raise PretrainingDataError(
                "invalid update supervised budget"
            )

        current = cursor
        emitted = 0
        while emitted < supervised_budget:
            microbatch = self.next_microbatch(
                current,
                batch_size=batch_size,
                max_input_length=max_input_length,
                max_supervised_positions=(
                    supervised_budget - emitted
                ),
            )
            if microbatch.supervised_positions <= 0:
                raise PretrainingDataError(
                    "microbatch made no supervised progress"
                )
            emitted += microbatch.supervised_positions
            current = microbatch.cursor
            yield microbatch

        if emitted != supervised_budget:
            raise PretrainingDataError(
                "update did not hit exact supervised budget"
            )


def cursor_state_dict(cursor: PackedDataCursor) -> dict[str, int]:
    return {
        "order_position": cursor.order_position,
        "block_input_offset": cursor.block_input_offset,
        "supervised_tokens_consumed": cursor.supervised_tokens_consumed,
    }


def cursor_from_state_dict(
    payload: object,
    *,
    contract: PretrainingDataContract,
) -> PackedDataCursor:
    if not isinstance(payload, dict):
        raise PretrainingDataError(
            "checkpoint data cursor must be an object"
        )
    expected = {
        "order_position",
        "block_input_offset",
        "supervised_tokens_consumed",
    }
    if set(payload) != expected:
        raise PretrainingDataError(
            "checkpoint data cursor fields changed"
        )
    cursor = PackedDataCursor(
        order_position=payload["order_position"],
        block_input_offset=payload["block_input_offset"],
        supervised_tokens_consumed=payload[
            "supervised_tokens_consumed"
        ],
    )
    validate_cursor(cursor, contract=contract)
    return cursor
