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


def _sql_path(path: Path) -> str:
    return str(path).replace("\\", "/").replace("'", "''")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--external-root", required=True)
    parser.add_argument(
        "--summary",
        default=str(
            ROOT
            / "results"
            / "l004"
            / "pretraining-block-order-v1.json"
        ),
    )
    parser.add_argument("--threads", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = json.loads(
        Path(args.summary).read_text(encoding="utf-8")
    )
    if summary.get("schema_version") != 1 or summary.get("stage") != "L004":
        raise RuntimeError("block-order summary schema/stage changed")
    if summary.get("order_id") != "base50m-block-order-v1":
        raise RuntimeError("block-order id changed")
    if summary.get("packing_id") != "base50m-packed-u16-v1":
        raise RuntimeError("block-order packing id changed")
    if summary.get("pretraining_authorized") is not False:
        raise RuntimeError("block order cannot self-authorize training")
    if summary.get("production_optimizer_steps_recorded") != 0:
        raise RuntimeError(
            "block-order qualification cannot record production steps"
        )

    order_meta = summary.get("block_order")
    source_meta = summary.get("source_block_index")
    if not isinstance(order_meta, dict) or not isinstance(source_meta, dict):
        raise RuntimeError("block-order metadata is incomplete")
    if order_meta.get("dtype") != "uint32_le":
        raise RuntimeError("block-order dtype changed")
    if order_meta.get("blocks") != 489_609:
        raise RuntimeError("block-order count changed")
    if order_meta.get("seed") != 1234:
        raise RuntimeError("block-order seed changed")
    if order_meta.get("key") != (
        "sha256('block|' + seed_decimal + '|' + block_index_decimal)"
    ):
        raise RuntimeError("block-order key changed")
    if order_meta.get("complete_permutation") is not True:
        raise RuntimeError("block-order permutation flag changed")

    order_path = resolve_external_storage_uri(
        order_meta["storage"],
        external_root=args.external_root,
    )
    index_path = resolve_external_storage_uri(
        source_meta["storage"],
        external_root=args.external_root,
    )
    if order_path.stat().st_size != int(order_meta["bytes"]):
        raise RuntimeError("block-order byte size changed")
    observed_order_sha = sha256_file(order_path)
    if observed_order_sha != order_meta["sha256"]:
        raise RuntimeError("block-order SHA-256 mismatch")
    if observed_order_sha != (
        "146f12fdccd0d2694f0e81e3f85f2a786cb4d72fc47b4302990d6c8696eecc6d"
    ):
        raise RuntimeError("block-order frozen hash changed")

    order = np.memmap(order_path, dtype="<u4", mode="r")
    if int(order.size) != 489_609:
        raise RuntimeError("block-order element count changed")
    seen = np.zeros(489_609, dtype=np.bool_)
    for start in range(0, int(order.size), 100_000):
        chunk = np.asarray(order[start : start + 100_000])
        if np.any(chunk >= 489_609):
            raise RuntimeError("block-order contains out-of-range index")
        if len(np.unique(chunk)) != len(chunk):
            raise RuntimeError(
                "block-order contains duplicate inside a chunk"
            )
        if np.any(seen[chunk]):
            raise RuntimeError(
                "block-order contains duplicate across chunks"
            )
        seen[chunk] = True
    if not bool(np.all(seen)):
        raise RuntimeError("block-order is not a complete permutation")
    del seen

    if index_path.stat().st_size != int(source_meta["bytes"]):
        raise RuntimeError("source block-index byte size changed")
    observed_index_sha = sha256_file(index_path)
    if observed_index_sha != source_meta["sha256"]:
        raise RuntimeError("source block-index SHA-256 mismatch")

    con = duckdb.connect()
    con.execute(f"SET threads = {int(args.threads)}")
    source = _sql_path(index_path)
    cursor = con.execute(
        "SELECT block_index FROM "
        f"read_json_auto('{source}', format='newline_delimited') "
        "ORDER BY order_key, block_index"
    )
    offset = 0
    while True:
        rows = cursor.fetchmany(50_000)
        if not rows:
            break
        expected = np.fromiter(
            (int(row[0]) for row in rows),
            dtype=np.uint32,
            count=len(rows),
        )
        observed = np.asarray(
            order[offset : offset + len(rows)],
            dtype=np.uint32,
        )
        if not np.array_equal(expected, observed):
            mismatch = int(np.flatnonzero(expected != observed)[0])
            raise RuntimeError(
                "block-order deterministic ordering mismatch at "
                f"position={offset + mismatch}"
            )
        offset += len(rows)
    con.close()
    if offset != 489_609:
        raise RuntimeError(
            f"verified block-order length changed: {offset}"
        )
    del order

    print(
        "PASS block_order blocks=489609 bytes="
        f"{order_meta['bytes']} sha256={observed_order_sha}",
        flush=True,
    )
    print(
        "PASS deterministic_order key=sha256(block|1234|block_index)",
        flush=True,
    )
    print("pretraining_block_order_verification=PASS", flush=True)


if __name__ == "__main__":
    main()
