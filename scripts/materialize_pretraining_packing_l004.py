from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from array import array
from pathlib import Path

import duckdb

from flywire_llm.artifacts import resolve_external_storage_uri
from flywire_llm.pretraining_execution import (
    load_pretraining_execution_contract,
    verify_base50m_tokenizer,
)


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize the L004 selected 500M content tokens into an "
            "external uint16 stream with EOS record separators."
        )
    )
    parser.add_argument("--external-root", required=True)
    parser.add_argument(
        "--packing-config",
        default=str(
            ROOT / "configs" / "pretraining-packing-l004.json"
        ),
    )
    parser.add_argument(
        "--selection-summary",
        default=str(
            ROOT / "results" / "l004" / "pretraining-selection-v1.json"
        ),
    )
    parser.add_argument(
        "--input-registry",
        default=str(
            ROOT / "configs" / "global-manifest-inputs-l003-v2.json"
        ),
    )
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


def _load_selected_for_artifact(
    con: duckdb.DuckDBPyConnection,
    artifact_id: str,
) -> dict[str, tuple[int, int, str]]:
    rows = con.execute(
        """
        SELECT record_id,
               token_count::BIGINT,
               selected_token_count::BIGINT,
               primary_category
        FROM selected
        WHERE artifact_id = ?
        """,
        [artifact_id],
    ).fetchall()
    result: dict[str, tuple[int, int, str]] = {}
    for record_id, token_count, selected_count, category in rows:
        key = str(record_id)
        if key in result:
            raise RuntimeError(
                f"{artifact_id}: duplicate selected record_id {key!r}"
            )
        result[key] = (
            int(token_count),
            int(selected_count),
            str(category),
        )
    return result


def _write_block_index(
    path: Path,
    *,
    physical_tokens: int,
    eos_positions: list[int],
    block_size: int,
    seed: int,
) -> dict[str, object]:
    input_positions = physical_tokens - 1
    if input_positions <= 0:
        raise RuntimeError("packed token stream is empty")
    block_count = math.ceil(input_positions / block_size)
    eos_counts = [0] * block_count
    final_physical_index = physical_tokens - 1
    for position in eos_positions:
        if position == final_physical_index:
            continue
        if position < 0 or position >= final_physical_index:
            raise RuntimeError("invalid EOS position")
        eos_counts[position // block_size] += 1

    digest = hashlib.sha256()
    byte_count = 0
    supervised_total = 0
    eos_input_total = 0
    temp = path.with_suffix(path.suffix + ".tmp")
    if temp.exists():
        temp.unlink()
    with temp.open("wb") as stream:
        for block_index in range(block_count):
            offset = block_index * block_size
            input_length = min(
                block_size,
                input_positions - offset,
            )
            eos_inputs = eos_counts[block_index]
            supervised = input_length - eos_inputs
            if supervised <= 0:
                raise RuntimeError(
                    f"block {block_index} has no supervised positions"
                )
            order_key = hashlib.sha256(
                f"block|{seed}|{block_index}".encode("utf-8")
            ).hexdigest()
            row = {
                "block_index": block_index,
                "physical_offset": offset,
                "input_length": input_length,
                "supervised_positions": supervised,
                "eos_input_positions": eos_inputs,
                "order_key": order_key,
            }
            encoded = (
                json.dumps(
                    row,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode("utf-8")
            stream.write(encoded)
            digest.update(encoded)
            byte_count += len(encoded)
            supervised_total += supervised
            eos_input_total += eos_inputs
    os.replace(temp, path)
    return {
        "blocks": block_count,
        "bytes": byte_count,
        "sha256": digest.hexdigest(),
        "input_positions": input_positions,
        "supervised_positions": supervised_total,
        "eos_input_positions": eos_input_total,
    }


def main() -> None:
    if sys.byteorder != "little":
        raise RuntimeError(
            "uint16_le materialization requires a little-endian host"
        )
    args = parse_args()
    packing = json.loads(
        Path(args.packing_config).read_text(encoding="utf-8")
    )
    selection_summary = json.loads(
        Path(args.selection_summary).read_text(encoding="utf-8")
    )
    registry = json.loads(
        Path(args.input_registry).read_text(encoding="utf-8")
    )

    if packing.get("schema_version") != 1 or packing.get("stage") != "L004":
        raise RuntimeError("packing config schema/stage changed")
    if packing.get("packing_id") != "base50m-packed-u16-v1":
        raise RuntimeError("packing_id changed")
    selection = packing.get("selection")
    if not isinstance(selection, dict):
        raise RuntimeError("packing selection pin is missing")
    expected_selection_sha = (
        "7729f79a0447238289de55a9ada52390a7e0a7291ab49f714b53b14e7da90268"
    )
    if selection.get("sha256") != expected_selection_sha:
        raise RuntimeError("packing selection SHA changed")
    if selection.get("content_tokens") != 500_000_000:
        raise RuntimeError("packing content budget changed")
    if selection.get("records") != 1_358_839:
        raise RuntimeError("packing record count changed")
    if packing.get("record_order") != (
        "global_manifest_registry_order_then_accepted_file_order"
    ):
        raise RuntimeError("packing record order changed")
    block_order = packing.get("block_order")
    if not isinstance(block_order, dict):
        raise RuntimeError("packing block order is missing")
    if block_order.get("seed") != 1234:
        raise RuntimeError("packing block order seed changed")
    if block_order.get("key") != (
        "sha256('block|' + seed_decimal + '|' + block_index_decimal)"
    ):
        raise RuntimeError("packing block order key changed")

    stream_contract = packing.get("stream")
    loss_contract = packing.get("loss")
    block_contract = packing.get("blocks")
    tokenizer_contract = packing.get("tokenizer")
    if not all(
        isinstance(value, dict)
        for value in (
            stream_contract,
            loss_contract,
            block_contract,
            tokenizer_contract,
        )
    ):
        raise RuntimeError("packing contract is incomplete")
    if stream_contract.get("dtype") != "uint16_le":
        raise RuntimeError("packed stream dtype changed")
    if stream_contract.get("record_separator") != "eos":
        raise RuntimeError("record separator changed")
    if stream_contract.get("separator_after_every_record") is not True:
        raise RuntimeError("record separator policy changed")
    if (
        stream_contract.get(
            "separator_tokens_count_toward_content_budget"
        )
        is not False
    ):
        raise RuntimeError("EOS budget policy changed")
    if tokenizer_contract.get("eos_id") != 2:
        raise RuntimeError("EOS id changed")
    if loss_contract.get("ignore_index") != -100:
        raise RuntimeError("loss ignore_index changed")
    if loss_contract.get("mask_target_at_eos_input_positions") is not True:
        raise RuntimeError("EOS loss masking changed")
    if loss_contract.get("content_last_token_targets_eos") is not True:
        raise RuntimeError("record-end target policy changed")
    if block_contract.get("max_physical_sequence_length") != 1024:
        raise RuntimeError("packing block size changed")
    if block_contract.get("global_supervised_tokens_per_update") != 65_536:
        raise RuntimeError("global supervised token budget changed")
    if block_contract.get("optimizer_steps") != 7_630:
        raise RuntimeError("optimizer step plan changed")
    if block_contract.get("full_updates") != 7_629:
        raise RuntimeError("full update count changed")
    if block_contract.get("final_update_supervised_tokens") != 25_856:
        raise RuntimeError("final update token budget changed")
    if packing.get("raw_text_in_packed_stream") is not False:
        raise RuntimeError("packed stream cannot contain raw text")
    if packing.get("production_optimizer_steps_recorded") != 0:
        raise RuntimeError("materialization cannot record optimizer steps")
    if packing.get("pretraining_authorized") is not False:
        raise RuntimeError("packing config cannot self-authorize")

    external_root = Path(args.external_root)
    execution = load_pretraining_execution_contract(
        ROOT / "configs" / "pretraining-execution-l004.json",
        repo_root=ROOT,
        external_root=external_root,
    )
    tokenizer = verify_base50m_tokenizer(
        execution.tokenizer_path,
        expected_sha256=execution.tokenizer_sha256,
        expected_vocab_size=execution.tokenizer_vocab_size,
    )
    if tokenizer.eos_id != 2:
        raise RuntimeError("runtime tokenizer EOS id changed")

    selection_path = resolve_external_storage_uri(
        selection["storage"],
        external_root=external_root,
    )
    if selection_path.stat().st_size != (
        selection_summary["selection_manifest"]["bytes"]
    ):
        raise RuntimeError("selection byte size changed")
    observed_selection_sha = sha256_file(selection_path)
    if observed_selection_sha != expected_selection_sha:
        raise RuntimeError("selection SHA-256 changed")
    if observed_selection_sha != (
        selection_summary["selection_manifest"]["sha256"]
    ):
        raise RuntimeError("selection summary hash mismatch")

    if registry.get("schema_version") != 1 or registry.get("stage") != "L003":
        raise RuntimeError("global input registry schema/stage changed")
    raw_inputs = registry.get("inputs")
    if not isinstance(raw_inputs, list) or not raw_inputs:
        raise RuntimeError("global input registry is empty")

    con = duckdb.connect()
    con.execute(f"SET threads = {int(args.threads)}")
    source = _sql_path(selection_path)
    con.execute(
        "CREATE TEMP TABLE selected AS "
        f"SELECT * FROM read_json_auto('{source}', "
        "format='newline_delimited')"
    )
    selected_count = int(
        con.execute("SELECT count(*) FROM selected").fetchone()[0]
    )
    if selected_count != 1_358_839:
        raise RuntimeError("selected record count changed")

    stream_path = resolve_external_storage_uri(
        stream_contract["storage"],
        external_root=external_root,
    )
    index_path = resolve_external_storage_uri(
        stream_contract["block_index_storage"],
        external_root=external_root,
    )
    stream_path.parent.mkdir(parents=True, exist_ok=True)
    temp_stream = stream_path.with_suffix(stream_path.suffix + ".tmp")
    if temp_stream.exists():
        temp_stream.unlink()

    stream_digest = hashlib.sha256()
    content_tokens_written = 0
    physical_tokens_written = 0
    records_written = 0
    eos_positions: list[int] = []
    selected_by_category: dict[str, int] = {}
    records_by_artifact: dict[str, int] = {}
    source_verification: dict[str, dict[str, object]] = {}
    materialize_start = time.perf_counter()

    with temp_stream.open("wb") as output:
        for item in raw_inputs:
            artifact_id = str(item["artifact_id"])
            selected_rows = _load_selected_for_artifact(
                con,
                artifact_id,
            )
            expected_selected = int(
                selection_summary["selected_records_by_artifact"].get(
                    artifact_id,
                    0,
                )
            )
            if len(selected_rows) != expected_selected:
                raise RuntimeError(
                    f"{artifact_id}: selected mapping count mismatch "
                    f"expected={expected_selected} "
                    f"observed={len(selected_rows)}"
                )
            if not selected_rows:
                continue

            accepted_path = resolve_external_storage_uri(
                item["accepted_storage"],
                external_root=external_root,
            )
            if accepted_path.stat().st_size != int(item["accepted_bytes"]):
                raise RuntimeError(
                    f"{artifact_id}: accepted byte size mismatch"
                )
            artifact_digest = hashlib.sha256()
            found = 0
            artifact_content_tokens = 0
            artifact_start = time.perf_counter()

            with accepted_path.open("rb") as source_stream:
                for raw_line in source_stream:
                    artifact_digest.update(raw_line)
                    if not raw_line.strip():
                        continue
                    payload = json.loads(raw_line)
                    record_id = payload.get("record_id")
                    selected_meta = selected_rows.get(record_id)
                    if selected_meta is None:
                        continue
                    expected_token_count, selected_token_count, category = (
                        selected_meta
                    )
                    if payload.get("artifact_id") != artifact_id:
                        raise RuntimeError(
                            f"{artifact_id}:{record_id}: artifact mismatch"
                        )
                    if payload.get("partition") != "train":
                        raise RuntimeError(
                            f"{artifact_id}:{record_id}: selected non-train row"
                        )
                    text = payload.get("text")
                    if not isinstance(text, str) or not text:
                        raise RuntimeError(
                            f"{artifact_id}:{record_id}: text is missing"
                        )
                    token_ids = tokenizer.encode(text)
                    if len(token_ids) != expected_token_count:
                        raise RuntimeError(
                            f"{artifact_id}:{record_id}: token count mismatch "
                            f"expected={expected_token_count} "
                            f"observed={len(token_ids)}"
                        )
                    if not 0 < selected_token_count <= len(token_ids):
                        raise RuntimeError(
                            f"{artifact_id}:{record_id}: invalid selected count"
                        )
                    selected_ids = token_ids[:selected_token_count]
                    if any(
                        token_id < 4
                        or token_id >= execution.tokenizer_vocab_size
                        for token_id in selected_ids
                    ):
                        raise RuntimeError(
                            f"{artifact_id}:{record_id}: reserved/out-of-range "
                            "token emitted in content"
                        )
                    eos_position = (
                        physical_tokens_written + selected_token_count
                    )
                    eos_positions.append(eos_position)
                    packed = array("H", selected_ids)
                    packed.append(tokenizer.eos_id)
                    encoded = packed.tobytes()
                    output.write(encoded)
                    stream_digest.update(encoded)

                    content_tokens_written += selected_token_count
                    physical_tokens_written += selected_token_count + 1
                    records_written += 1
                    found += 1
                    artifact_content_tokens += selected_token_count
                    selected_by_category[category] = (
                        selected_by_category.get(category, 0)
                        + selected_token_count
                    )
                    records_by_artifact[artifact_id] = (
                        records_by_artifact.get(artifact_id, 0) + 1
                    )

            observed_artifact_sha = artifact_digest.hexdigest()
            if observed_artifact_sha != item["accepted_sha256"]:
                raise RuntimeError(
                    f"{artifact_id}: accepted SHA-256 mismatch "
                    f"expected={item['accepted_sha256']} "
                    f"observed={observed_artifact_sha}"
                )
            if found != expected_selected:
                missing = expected_selected - found
                raise RuntimeError(
                    f"{artifact_id}: selected rows missing from accepted "
                    f"corpus count={missing}"
                )
            elapsed = time.perf_counter() - artifact_start
            source_verification[artifact_id] = {
                "accepted_bytes": int(item["accepted_bytes"]),
                "accepted_sha256": observed_artifact_sha,
                "selected_records": found,
                "selected_content_tokens": artifact_content_tokens,
                "scan_seconds": elapsed,
            }
            print(
                f"PASS artifact={artifact_id} selected_records={found} "
                f"content_tokens={artifact_content_tokens} "
                f"scan_seconds={elapsed:.3f}",
                flush=True,
            )

    con.close()

    expected_category_tokens = {
        "general_english": 200_000_000,
        "general_thai": 200_000_000,
        "technical_scientific_code": 100_000_000,
    }
    if selected_by_category != expected_category_tokens:
        raise RuntimeError(
            "packed category totals changed "
            f"observed={selected_by_category}"
        )
    if content_tokens_written != 500_000_000:
        raise RuntimeError(
            f"packed content token total changed: {content_tokens_written}"
        )
    if records_written != 1_358_839:
        raise RuntimeError(
            f"packed record total changed: {records_written}"
        )
    expected_physical = 501_358_839
    if physical_tokens_written != expected_physical:
        raise RuntimeError(
            "packed physical token count changed "
            f"expected={expected_physical} "
            f"observed={physical_tokens_written}"
        )
    expected_stream_bytes = expected_physical * 2
    if temp_stream.stat().st_size != expected_stream_bytes:
        raise RuntimeError(
            "packed stream byte size mismatch "
            f"expected={expected_stream_bytes} "
            f"observed={temp_stream.stat().st_size}"
        )
    os.replace(temp_stream, stream_path)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_report = _write_block_index(
        index_path,
        physical_tokens=physical_tokens_written,
        eos_positions=eos_positions,
        block_size=1024,
        seed=1234,
    )
    if index_report["supervised_positions"] != 500_000_000:
        raise RuntimeError(
            "block index supervised total is not exactly 500M"
        )
    expected_eos_inputs = records_written - 1
    if index_report["eos_input_positions"] != expected_eos_inputs:
        raise RuntimeError(
            "block index EOS input total changed "
            f"expected={expected_eos_inputs} "
            f"observed={index_report['eos_input_positions']}"
        )
    expected_blocks = math.ceil(
        (physical_tokens_written - 1) / 1024
    )
    if index_report["blocks"] != expected_blocks:
        raise RuntimeError("block count changed")

    elapsed_total = time.perf_counter() - materialize_start
    summary = {
        "schema_version": 1,
        "stage": "L004",
        "packing_id": "base50m-packed-u16-v1",
        "checkpoint_lane": "research_only",
        "selection_manifest": {
            "storage": selection["storage"],
            "bytes": selection_path.stat().st_size,
            "sha256": observed_selection_sha,
            "records": records_written,
            "content_tokens": content_tokens_written,
        },
        "tokenizer": {
            "candidate_id": execution.tokenizer_candidate_id,
            "sha256": execution.tokenizer_sha256,
            "vocab_size": execution.tokenizer_vocab_size,
            "eos_id": tokenizer.eos_id,
        },
        "token_stream": {
            "storage": stream_contract["storage"],
            "dtype": "uint16_le",
            "bytes": stream_path.stat().st_size,
            "sha256": stream_digest.hexdigest(),
            "physical_tokens": physical_tokens_written,
            "content_tokens": content_tokens_written,
            "eos_separator_tokens": records_written,
            "contains_raw_text": False,
        },
        "block_index": {
            "storage": stream_contract["block_index_storage"],
            **index_report,
            "order_seed": 1234,
            "order_key": (
                "sha256('block|' + seed_decimal + '|' + "
                "block_index_decimal)"
            ),
        },
        "loss_contract": {
            "ignore_index": -100,
            "mask_target_at_eos_input_positions": True,
            "content_last_token_targets_eos": True,
            "supervised_positions": 500_000_000,
        },
        "attention_contract": packing["attention"],
        "schedule_contract": {
            "global_supervised_tokens_per_update": 65_536,
            "optimizer_steps": 7_630,
            "full_updates": 7_629,
            "final_update_supervised_tokens": 25_856,
        },
        "selected_tokens_by_category": selected_by_category,
        "selected_records_by_artifact": records_by_artifact,
        "source_verification": source_verification,
        "materialization_seconds": elapsed_total,
        "production_optimizer_steps_recorded": 0,
        "pretraining_authorized": False,
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        f"token_stream_bytes={stream_path.stat().st_size} "
        f"token_stream_sha256={stream_digest.hexdigest()}",
        flush=True,
    )
    print(
        f"blocks={index_report['blocks']} "
        f"block_index_sha256={index_report['sha256']}",
        flush=True,
    )
    print("supervised_positions=500000000", flush=True)
    print(
        f"materialization_seconds={elapsed_total:.3f}",
        flush=True,
    )
    print("pretraining_packing_materialization=PASS", flush=True)


if __name__ == "__main__":
    main()
