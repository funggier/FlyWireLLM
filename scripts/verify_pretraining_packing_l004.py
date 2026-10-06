from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np

from flywire_llm.artifacts import resolve_external_storage_uri


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--external-root", required=True)
    parser.add_argument(
        "--summary",
        default=str(
            ROOT / "results" / "l004" / "pretraining-packing-v1.json"
        ),
    )
    parser.add_argument("--threads", type=int, default=12)
    return parser.parse_args()


def _sql_path(path: Path) -> str:
    return str(path).replace("\\", "/").replace("'", "''")


def main() -> None:
    args = parse_args()
    summary = json.loads(
        Path(args.summary).read_text(encoding="utf-8")
    )
    if summary.get("schema_version") != 1 or summary.get("stage") != "L004":
        raise RuntimeError("packing summary schema/stage changed")
    if summary.get("packing_id") != "base50m-packed-u16-v1":
        raise RuntimeError("packing id changed")
    if summary.get("checkpoint_lane") != "research_only":
        raise RuntimeError("packing checkpoint lane changed")
    if summary.get("pretraining_authorized") is not False:
        raise RuntimeError("packing summary cannot self-authorize")
    if summary.get("production_optimizer_steps_recorded") != 0:
        raise RuntimeError(
            "packing qualification cannot record production steps"
        )

    external_root = Path(args.external_root)
    stream_meta = summary.get("token_stream")
    index_meta = summary.get("block_index")
    if not isinstance(stream_meta, dict) or not isinstance(index_meta, dict):
        raise RuntimeError("packing output metadata is incomplete")
    if stream_meta.get("contains_raw_text") is not False:
        raise RuntimeError("packed token stream must be binary-only")
    if stream_meta.get("dtype") != "uint16_le":
        raise RuntimeError("packed token stream dtype changed")

    stream_path = resolve_external_storage_uri(
        stream_meta["storage"],
        external_root=external_root,
    )
    index_path = resolve_external_storage_uri(
        index_meta["storage"],
        external_root=external_root,
    )

    expected_stream_bytes = 1_002_717_678
    if stream_path.stat().st_size != expected_stream_bytes:
        raise RuntimeError("packed stream byte size changed")
    if stream_meta.get("bytes") != expected_stream_bytes:
        raise RuntimeError("packed summary stream size changed")
    observed_stream_sha = sha256_file(stream_path)
    if observed_stream_sha != stream_meta.get("sha256"):
        raise RuntimeError("packed stream SHA-256 mismatch")
    if observed_stream_sha != (
        "07a0faf43c241f3e232a6200bc1732381d18e9ef1b2cd77c185d2fb4c4296a15"
    ):
        raise RuntimeError("packed stream frozen hash changed")
    print(
        f"PASS token_stream bytes={expected_stream_bytes} "
        f"sha256={observed_stream_sha}",
        flush=True,
    )

    tokens = np.memmap(stream_path, dtype="<u2", mode="r")
    if int(tokens.size) != 501_358_839:
        raise RuntimeError("packed physical token count changed")
    if int(tokens[-1]) != 2:
        raise RuntimeError("packed stream must end with EOS")

    chunk_size = 8_000_000
    eos_count = 0
    pad_count = 0
    bos_count = 0
    unk_count = 0
    out_of_range = 0
    for start in range(0, int(tokens.size), chunk_size):
        chunk = np.asarray(tokens[start : start + chunk_size])
        eos_count += int(np.count_nonzero(chunk == 2))
        pad_count += int(np.count_nonzero(chunk == 0))
        bos_count += int(np.count_nonzero(chunk == 1))
        unk_count += int(np.count_nonzero(chunk == 3))
        out_of_range += int(np.count_nonzero(chunk >= 32_000))
    if eos_count != 1_358_839:
        raise RuntimeError(
            f"EOS separator count changed: {eos_count}"
        )
    if pad_count or bos_count or unk_count:
        raise RuntimeError(
            "packed stream contains unexpected reserved tokens "
            f"pad={pad_count} bos={bos_count} unk={unk_count}"
        )
    if out_of_range:
        raise RuntimeError(
            f"packed stream contains {out_of_range} out-of-range tokens"
        )
    content_tokens = int(tokens.size) - eos_count
    if content_tokens != 500_000_000:
        raise RuntimeError("packed content token total is not 500M")
    del tokens
    print(
        "PASS token_scan physical=501358839 content=500000000 "
        "eos=1358839 reserved_other=0 out_of_range=0",
        flush=True,
    )

    if index_path.stat().st_size != int(index_meta["bytes"]):
        raise RuntimeError("block index byte size changed")
    observed_index_sha = sha256_file(index_path)
    if observed_index_sha != index_meta.get("sha256"):
        raise RuntimeError("block index SHA-256 mismatch")
    if observed_index_sha != (
        "dd1ebd08a69eff0f44b0714b16eb5c257feb6d3e9879489d7d5e05f4c8dcf0d9"
    ):
        raise RuntimeError("block index frozen hash changed")
    print(
        f"PASS block_index bytes={index_path.stat().st_size} "
        f"sha256={observed_index_sha}",
        flush=True,
    )

    con = duckdb.connect()
    con.execute(f"SET threads = {int(args.threads)}")
    source = _sql_path(index_path)
    con.execute(
        "CREATE TEMP VIEW blocks AS "
        f"SELECT * FROM read_json_auto('{source}', "
        "format='newline_delimited')"
    )
    shape = con.execute(
        """
        SELECT
            count(*)::BIGINT AS blocks,
            count(DISTINCT block_index)::BIGINT AS unique_blocks,
            count(DISTINCT order_key)::BIGINT AS unique_keys,
            min(block_index)::BIGINT AS min_block,
            max(block_index)::BIGINT AS max_block,
            sum(input_length)::BIGINT AS input_positions,
            sum(supervised_positions)::BIGINT AS supervised_positions,
            sum(eos_input_positions)::BIGINT AS eos_inputs,
            sum(CASE
                    WHEN physical_offset != block_index * 1024
                    THEN 1 ELSE 0 END)::BIGINT AS bad_offsets,
            sum(CASE
                    WHEN supervised_positions !=
                         input_length - eos_input_positions
                    THEN 1 ELSE 0 END)::BIGINT AS bad_supervision,
            sum(CASE
                    WHEN input_length <= 0 OR input_length > 1024
                    THEN 1 ELSE 0 END)::BIGINT AS bad_lengths
        FROM blocks
        """
    ).fetchone()
    (
        blocks,
        unique_blocks,
        unique_keys,
        min_block,
        max_block,
        input_positions,
        supervised_positions,
        eos_inputs,
        bad_offsets,
        bad_supervision,
        bad_lengths,
    ) = map(int, shape)
    if blocks != 489_609 or unique_blocks != blocks:
        raise RuntimeError("block index count/uniqueness changed")
    if unique_keys != blocks:
        raise RuntimeError("block order keys are not unique")
    if min_block != 0 or max_block != 489_608:
        raise RuntimeError("block index range changed")
    if input_positions != 501_358_838:
        raise RuntimeError("block input-position total changed")
    if supervised_positions != 500_000_000:
        raise RuntimeError("block supervised total changed")
    if eos_inputs != 1_358_838:
        raise RuntimeError("block EOS-input total changed")
    if bad_offsets or bad_supervision or bad_lengths:
        raise RuntimeError(
            "block structural validation failed "
            f"offsets={bad_offsets} supervision={bad_supervision} "
            f"lengths={bad_lengths}"
        )

    last = con.execute(
        """
        SELECT input_length, supervised_positions, eos_input_positions
        FROM blocks
        WHERE block_index = 489608
        """
    ).fetchone()
    if tuple(map(int, last)) != (246, 246, 0):
        raise RuntimeError(
            f"final block shape changed: {last}"
        )
    bad_keys = int(
        con.execute(
            """
            SELECT count(*)
            FROM blocks
            WHERE order_key !=
                sha256('block|1234|' || cast(block_index AS VARCHAR))
            """
        ).fetchone()[0]
    )
    if bad_keys:
        raise RuntimeError("block shuffle key recomputation mismatch")
    con.close()

    schedule = summary.get("schedule_contract")
    if schedule != {
        "global_supervised_tokens_per_update": 65_536,
        "optimizer_steps": 7_630,
        "full_updates": 7_629,
        "final_update_supervised_tokens": 25_856,
    }:
        raise RuntimeError("packing schedule contract changed")
    loss = summary.get("loss_contract")
    if not isinstance(loss, dict):
        raise RuntimeError("loss contract missing")
    if loss.get("ignore_index") != -100:
        raise RuntimeError("loss ignore index changed")
    if loss.get("mask_target_at_eos_input_positions") is not True:
        raise RuntimeError("EOS-input masking changed")
    if loss.get("content_last_token_targets_eos") is not True:
        raise RuntimeError("record-end EOS target contract changed")

    print(
        "PASS blocks=489609 input_positions=501358838 "
        "supervised_positions=500000000 eos_inputs=1358838 "
        "final_block_input_length=246",
        flush=True,
    )
    print("pretraining_packing_verification=PASS", flush=True)


if __name__ == "__main__":
    main()
