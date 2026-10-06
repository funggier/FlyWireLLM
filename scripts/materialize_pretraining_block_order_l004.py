from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from array import array
from pathlib import Path

import duckdb

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
        "--packing-config",
        default=str(
            ROOT / "configs" / "pretraining-packing-l004.json"
        ),
    )
    parser.add_argument(
        "--packing-summary",
        default=str(
            ROOT / "results" / "l004" / "pretraining-packing-v1.json"
        ),
    )
    parser.add_argument(
        "--summary",
        default=str(
            ROOT / "results" / "l004" / "pretraining-block-order-v1.json"
        ),
    )
    parser.add_argument("--threads", type=int, default=12)
    return parser.parse_args()


def _sql_path(path: Path) -> str:
    return str(path).replace("\\", "/").replace("'", "''")


def main() -> None:
    if sys.byteorder != "little":
        raise RuntimeError("uint32_le order requires little-endian host")
    args = parse_args()
    packing = json.loads(
        Path(args.packing_config).read_text(encoding="utf-8")
    )
    summary = json.loads(
        Path(args.packing_summary).read_text(encoding="utf-8")
    )
    order = packing.get("block_order")
    if not isinstance(order, dict):
        raise RuntimeError("block_order contract is missing")
    if order.get("seed") != 1234:
        raise RuntimeError("block order seed changed")
    if order.get("key") != (
        "sha256('block|' + seed_decimal + '|' + block_index_decimal)"
    ):
        raise RuntimeError("block order key changed")
    if order.get("dtype") != "uint32_le":
        raise RuntimeError("block order dtype changed")

    index_meta = summary.get("block_index")
    if not isinstance(index_meta, dict):
        raise RuntimeError("packing summary block index missing")
    index_path = resolve_external_storage_uri(
        index_meta["storage"],
        external_root=args.external_root,
    )
    if index_path.stat().st_size != int(index_meta["bytes"]):
        raise RuntimeError("block index byte size changed")
    observed_index_sha = sha256_file(index_path)
    if observed_index_sha != index_meta["sha256"]:
        raise RuntimeError("block index SHA-256 mismatch")
    if observed_index_sha != (
        "dd1ebd08a69eff0f44b0714b16eb5c257feb6d3e9879489d7d5e05f4c8dcf0d9"
    ):
        raise RuntimeError("block index frozen hash changed")
    expected_blocks = int(index_meta["blocks"])
    if expected_blocks != 489_609:
        raise RuntimeError("block count changed")

    output_path = resolve_external_storage_uri(
        order["storage"],
        external_root=args.external_root,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp = output_path.with_suffix(output_path.suffix + ".tmp")
    if temp.exists():
        temp.unlink()

    con = duckdb.connect()
    con.execute(f"SET threads = {int(args.threads)}")
    source = _sql_path(index_path)
    cursor = con.execute(
        "SELECT block_index FROM "
        f"read_json_auto('{source}', format='newline_delimited') "
        "ORDER BY order_key, block_index"
    )

    digest = hashlib.sha256()
    count = 0
    seen = bytearray(expected_blocks)
    with temp.open("wb") as stream:
        while True:
            rows = cursor.fetchmany(50_000)
            if not rows:
                break
            values = array("I")
            for row in rows:
                block_index = int(row[0])
                if block_index < 0 or block_index >= expected_blocks:
                    raise RuntimeError(
                        f"invalid block index {block_index}"
                    )
                if seen[block_index]:
                    raise RuntimeError(
                        f"duplicate block index {block_index}"
                    )
                seen[block_index] = 1
                values.append(block_index)
                count += 1
            encoded = values.tobytes()
            stream.write(encoded)
            digest.update(encoded)
    con.close()

    if count != expected_blocks:
        raise RuntimeError(
            f"block order count mismatch {count} != {expected_blocks}"
        )
    if not all(seen):
        raise RuntimeError("block order is not a complete permutation")
    expected_bytes = expected_blocks * 4
    if temp.stat().st_size != expected_bytes:
        raise RuntimeError("block order byte size mismatch")
    os.replace(temp, output_path)

    result = {
        "schema_version": 1,
        "stage": "L004",
        "order_id": "base50m-block-order-v1",
        "packing_id": "base50m-packed-u16-v1",
        "source_block_index": {
            "storage": index_meta["storage"],
            "bytes": index_path.stat().st_size,
            "sha256": observed_index_sha,
        },
        "block_order": {
            "storage": order["storage"],
            "dtype": "uint32_le",
            "bytes": expected_bytes,
            "sha256": digest.hexdigest(),
            "blocks": expected_blocks,
            "seed": 1234,
            "key": order["key"],
            "complete_permutation": True,
        },
        "production_optimizer_steps_recorded": 0,
        "pretraining_authorized": False,
    }
    output_summary = Path(args.summary)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        f"block_order_blocks={expected_blocks} "
        f"bytes={expected_bytes} sha256={digest.hexdigest()}",
        flush=True,
    )
    print("pretraining_block_order_materialization=PASS", flush=True)


if __name__ == "__main__":
    main()
