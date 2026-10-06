from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from array import array
from collections import defaultdict
from pathlib import Path

import duckdb

from flywire_llm.artifacts import resolve_external_storage_uri
from flywire_llm.pretraining_execution import (
    load_pretraining_execution_contract,
)


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selection_key(seed: int, canonical_id: str) -> str:
    return hashlib.sha256(
        f"{seed}|{canonical_id}".encode("utf-8")
    ).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--external-root", required=True)
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "validation-l004-v1.json"),
    )
    parser.add_argument(
        "--summary",
        default=str(ROOT / "results" / "l004" / "validation-pack-v1.json"),
    )
    parser.add_argument("--threads", type=int, default=12)
    return parser.parse_args()


def _sql_path(path: Path) -> str:
    return str(path).replace("\\", "/").replace("'", "''")


def _load_config(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("stage") != "L004":
        raise RuntimeError("validation config schema/stage changed")
    if payload.get("evaluation_id") != "base50m-validation-v1":
        raise RuntimeError("validation evaluation_id changed")
    if payload.get("evaluation_only") is not True:
        raise RuntimeError("validation must remain evaluation-only")
    if payload.get("partition") != "validation":
        raise RuntimeError("validation partition changed")
    if payload.get("final_holdout_touched") is not False:
        raise RuntimeError("final holdout must remain untouched")
    if payload.get("pretraining_authorized") is not False:
        raise RuntimeError("validation config cannot authorize training")
    if payload.get("production_optimizer_steps_recorded") != 4:
        raise RuntimeError("validation gate must follow production step 4")
    return payload


def main() -> None:
    if sys.byteorder != "little":
        raise RuntimeError("validation uint16 packing requires little-endian host")
    args = parse_args()
    started = time.perf_counter()
    external_root = Path(args.external_root)
    config = _load_config(Path(args.config))

    registry_meta = config["global_input_registry"]
    registry_path = ROOT / registry_meta["path"]
    if sha256_file(registry_path) != registry_meta["sha256"]:
        raise RuntimeError("global input registry SHA-256 mismatch")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))

    decision_meta = config["decision_manifest"]
    decision_path = resolve_external_storage_uri(
        decision_meta["storage"],
        external_root=external_root,
    )
    if sha256_file(decision_path) != decision_meta["sha256"]:
        raise RuntimeError("global decision manifest SHA-256 mismatch")

    selection = config["selection"]
    seed = int(selection["seed"])
    targets = {
        str(key): int(value)
        for key, value in selection["category_targets"].items()
    }
    if targets != {
        "general_english": 100_000,
        "general_thai": 100_000,
        "technical_scientific_code": 100_000,
    }:
        raise RuntimeError("validation category targets changed")

    con = duckdb.connect()
    con.execute(f"SET threads={int(args.threads)}")
    source = _sql_path(decision_path)
    rows = con.execute(
        f"""
        SELECT
            canonical_id,
            source_id,
            artifact_id,
            record_id,
            primary_category,
            token_count::BIGINT,
            rights_lane
        FROM read_json_auto('{source}', format='newline_delimited')
        WHERE status='accept'
          AND partition='validation'
          AND rights_lane IN ('release_safe', 'research_only')
          AND primary_category IN (
              'general_english',
              'general_thai',
              'technical_scientific_code'
          )
        """
    ).fetchall()

    grouped: dict[str, list[tuple[str, tuple]]] = {
        category: [] for category in targets
    }
    available: dict[str, dict[str, int]] = {
        category: {"records": 0, "tokens": 0}
        for category in targets
    }
    for row in rows:
        category = str(row[4])
        canonical_id = str(row[0])
        token_count = int(row[5])
        if token_count <= 0:
            raise RuntimeError("validation candidate has non-positive token count")
        key = selection_key(seed, canonical_id)
        grouped[category].append((key, row))
        available[category]["records"] += 1
        available[category]["tokens"] += token_count

    selected_rows: list[dict] = []
    selected_counts: dict[str, int] = {}
    partial_by_category: dict[str, int] = {}
    for category in sorted(targets):
        target = targets[category]
        consumed = 0
        records = 0
        partial = 0
        for key, row in sorted(
            grouped[category],
            key=lambda item: (item[0], str(item[1][0])),
        ):
            if consumed >= target:
                break
            token_count = int(row[5])
            take = min(token_count, target - consumed)
            if take < token_count:
                partial += 1
            selected_rows.append(
                {
                    "canonical_id": str(row[0]),
                    "source_id": str(row[1]),
                    "artifact_id": str(row[2]),
                    "record_id": str(row[3]),
                    "primary_category": category,
                    "token_count": token_count,
                    "selected_token_count": take,
                    "rights_lane": str(row[6]),
                    "selection_key": key,
                }
            )
            consumed += take
            records += 1
        if consumed != target:
            raise RuntimeError(
                f"insufficient validation {category} capacity "
                f"target={target} selected={consumed}"
            )
        if partial != 1:
            raise RuntimeError(
                f"validation {category} must have exactly one partial record"
            )
        selected_counts[category] = records
        partial_by_category[category] = partial

    selected_rows.sort(
        key=lambda row: (
            row["primary_category"],
            row["selection_key"],
            row["canonical_id"],
        )
    )
    selection_path = resolve_external_storage_uri(
        selection["storage"],
        external_root=external_root,
    )
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    temp_selection = selection_path.with_suffix(".jsonl.tmp")
    digest = hashlib.sha256()
    with temp_selection.open("wb") as stream:
        for row in selected_rows:
            encoded = (
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode("utf-8")
            stream.write(encoded)
            digest.update(encoded)
    os.replace(temp_selection, selection_path)
    selection_sha = digest.hexdigest()

    execution = load_pretraining_execution_contract(
        ROOT / "configs" / "pretraining-execution-l004.json",
        repo_root=ROOT,
        external_root=external_root,
    )
    tokenizer = execution.tokenizer_path
    from flywire_llm.sentencepiece_tokenizer import SentencePieceTokenizer

    sp = SentencePieceTokenizer(tokenizer)
    if sp.vocab_size != 32_000 or sp.eos_id != 2:
        raise RuntimeError("validation tokenizer contract changed")

    selected_by_artifact: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in selected_rows:
        artifact = row["artifact_id"]
        record_id = row["record_id"]
        if record_id in selected_by_artifact[artifact]:
            raise RuntimeError("duplicate selected artifact/record id")
        selected_by_artifact[artifact][record_id] = row

    packed_records: dict[str, list[tuple[str, str, list[int]]]] = {
        category: [] for category in targets
    }
    source_verification: dict[str, dict] = {}
    found = 0
    selection_sql = _sql_path(selection_path)
    con.execute(
        "CREATE OR REPLACE TEMP VIEW validation_selected AS "
        f"SELECT * FROM read_json_auto('{selection_sql}', "
        "format='newline_delimited')"
    )
    for item in registry["inputs"]:
        artifact = item["artifact_id"]
        expected_selected = selected_by_artifact.get(artifact)
        if not expected_selected:
            continue
        accepted_path = resolve_external_storage_uri(
            item["accepted_storage"],
            external_root=external_root,
        )
        if accepted_path.stat().st_size != int(item["accepted_bytes"]):
            raise RuntimeError(f"{artifact}: accepted byte size mismatch")
        observed_sha = sha256_file(accepted_path)
        if observed_sha != item["accepted_sha256"]:
            raise RuntimeError(f"{artifact}: accepted SHA-256 mismatch")

        accepted_sql = _sql_path(accepted_path)
        query = f"""
            SELECT
                a.record_id,
                a.text,
                s.canonical_id,
                s.primary_category,
                s.token_count::BIGINT,
                s.selected_token_count::BIGINT,
                s.selection_key
            FROM read_json_auto(
                '{accepted_sql}',
                format='newline_delimited'
            ) a
            JOIN validation_selected s
              ON s.artifact_id='{artifact.replace("'", "''")}'
             AND s.record_id=a.record_id
        """
        artifact_found = 0
        for (
            record_id,
            text,
            canonical_id,
            category,
            token_count,
            selected_token_count,
            key,
        ) in con.execute(query).fetchall():
            ids = sp.encode(str(text))
            if len(ids) != int(token_count):
                raise RuntimeError(
                    f"{artifact}:{record_id}: tokenizer count mismatch "
                    f"expected={token_count} observed={len(ids)}"
                )
            take = int(selected_token_count)
            ids = ids[:take]
            if any(token_id < 4 or token_id >= 32_000 for token_id in ids):
                raise RuntimeError(
                    f"{artifact}:{record_id}: invalid content token id"
                )
            packed_records[str(category)].append(
                (str(key), str(canonical_id), ids)
            )
            artifact_found += 1
            found += 1
        if artifact_found != len(expected_selected):
            raise RuntimeError(
                f"{artifact}: selected record retrieval mismatch "
                f"expected={len(expected_selected)} found={artifact_found}"
            )
        source_verification[artifact] = {
            "accepted_bytes": int(item["accepted_bytes"]),
            "accepted_sha256": observed_sha,
            "selected_records": artifact_found,
        }

    con.close()
    if found != len(selected_rows):
        raise RuntimeError(
            f"validation selected retrieval mismatch "
            f"expected={len(selected_rows)} found={found}"
        )

    pack_dir = resolve_external_storage_uri(
        config["packing"]["directory"],
        external_root=external_root,
    )
    pack_dir.mkdir(parents=True, exist_ok=True)
    packs: dict[str, dict] = {}
    for category in sorted(targets):
        records = sorted(
            packed_records[category],
            key=lambda item: (item[0], item[1]),
        )
        output = pack_dir / f"{category}.u16"
        temp = output.with_suffix(".u16.tmp")
        content_tokens = 0
        record_count = 0
        with temp.open("wb") as stream:
            for _, _, ids in records:
                values = array("H", ids)
                values.append(2)
                stream.write(values.tobytes())
                content_tokens += len(ids)
                record_count += 1
        os.replace(temp, output)
        if content_tokens != targets[category]:
            raise RuntimeError(
                f"{category}: packed content token mismatch"
            )
        physical_tokens = content_tokens + record_count
        if output.stat().st_size != physical_tokens * 2:
            raise RuntimeError(f"{category}: packed byte size mismatch")
        packs[category] = {
            "storage": (
                f"{config['packing']['directory']}/{category}.u16"
            ),
            "dtype": "uint16_le",
            "bytes": output.stat().st_size,
            "sha256": sha256_file(output),
            "records": record_count,
            "content_tokens": content_tokens,
            "eos_separator_tokens": record_count,
            "physical_tokens": physical_tokens,
            "supervised_positions": content_tokens,
            "contains_raw_text": False,
        }

    summary = {
        "schema_version": 1,
        "stage": "L004",
        "evaluation_id": config["evaluation_id"],
        "evaluation_only": True,
        "partition": "validation",
        "final_holdout_touched": False,
        "seed": seed,
        "available": available,
        "targets": targets,
        "selected_records_by_category": selected_counts,
        "partial_records_by_category": partial_by_category,
        "total_selected_content_tokens": sum(targets.values()),
        "selection_manifest": {
            "storage": selection["storage"],
            "bytes": selection_path.stat().st_size,
            "sha256": selection_sha,
            "records": len(selected_rows),
            "contains_raw_text": False,
        },
        "packs": packs,
        "source_verification": source_verification,
        "tokenizer": {
            "candidate_id": execution.tokenizer_candidate_id,
            "sha256": execution.tokenizer_sha256,
            "vocab_size": execution.tokenizer_vocab_size,
            "eos_id": 2,
        },
        "loss_contract": {
            "ignore_index": -100,
            "mask_target_at_eos_input_positions": True,
            "segment_attention_reset": False,
            "max_physical_sequence_length": 1024,
        },
        "production_optimizer_steps_recorded": 4,
        "pretraining_authorized": False,
        "materialization_seconds": time.perf_counter() - started,
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        f"selection_records={len(selected_rows)} "
        f"selection_sha256={selection_sha}"
    )
    for category in sorted(packs):
        print(
            f"PASS category={category} "
            f"records={packs[category]['records']} "
            f"content_tokens={packs[category]['content_tokens']} "
            f"sha256={packs[category]['sha256']}"
        )
    print("holdout_touched=false")
    print("validation_materialization=PASS")


if __name__ == "__main__":
    main()
