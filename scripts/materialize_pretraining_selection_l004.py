from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import duckdb

from flywire_llm.artifacts import resolve_external_storage_uri


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
            "Materialize the deterministic L004 500M metadata-only "
            "pretraining selection."
        )
    )
    parser.add_argument("--external-root", required=True)
    parser.add_argument(
        "--config",
        default=str(
            ROOT / "configs" / "pretraining-selection-l004.json"
        ),
    )
    parser.add_argument(
        "--summary",
        default=str(
            ROOT / "results" / "l004" / "pretraining-selection-v1.json"
        ),
    )
    parser.add_argument("--threads", type=int, default=12)
    return parser.parse_args()


def _sql_path(path: Path) -> str:
    return str(path).replace("\\", "/").replace("'", "''")


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or config.get("stage") != "L004":
        raise RuntimeError("selection contract schema/stage changed")
    if config.get("selection_id") != "base50m-500m-research-v1":
        raise RuntimeError("selection_id changed")
    if config.get("checkpoint_lane") != "research_only":
        raise RuntimeError("selection checkpoint lane changed")
    if config.get("status") != "accept":
        raise RuntimeError("selection status must remain accept")
    if config.get("partition") != "train":
        raise RuntimeError("selection partition must remain train")
    if config.get("seed") != 1234:
        raise RuntimeError("selection seed changed")
    if config.get("selection_key") != (
        "sha256(seed_decimal + '|' + canonical_id)"
    ):
        raise RuntimeError("selection key algorithm changed")
    if config.get("record_boundary_policy") != (
        "prefix_truncate_final_record"
    ):
        raise RuntimeError("record boundary policy changed")
    if config.get("raw_text_in_selection") is not False:
        raise RuntimeError("selection manifest must remain metadata-only")
    if config.get("pretraining_authorized") is not False:
        raise RuntimeError("selection manifest cannot self-authorize")

    targets = config.get("category_targets")
    expected_targets = {
        "general_thai": 200_000_000,
        "general_english": 200_000_000,
        "technical_scientific_code": 100_000_000,
        "flywire_domain": 0,
    }
    if targets != expected_targets:
        raise RuntimeError("L004 category targets changed")
    if config.get("total_target_tokens") != 500_000_000:
        raise RuntimeError("total target changed")
    if sum(targets.values()) != 500_000_000:
        raise RuntimeError("category targets do not sum to 500M")

    decision = config.get("decision_manifest")
    if not isinstance(decision, dict):
        raise RuntimeError("decision_manifest must be an object")
    expected_decision_sha = decision.get("sha256")
    if expected_decision_sha != (
        "018aa08f347ebfd747f3aa11cada563bbd27918b469c7c5f11abe0289df7770a"
    ):
        raise RuntimeError("decision manifest hash changed")
    decision_path = resolve_external_storage_uri(
        decision["storage"],
        external_root=args.external_root,
    )
    print(
        f"verify_decisions bytes={decision_path.stat().st_size}",
        flush=True,
    )
    observed_decision_sha = sha256_file(decision_path)
    if observed_decision_sha != expected_decision_sha:
        raise RuntimeError(
            "decision manifest SHA-256 mismatch "
            f"expected={expected_decision_sha} "
            f"observed={observed_decision_sha}"
        )
    print(
        f"decision_manifest_sha256={observed_decision_sha}",
        flush=True,
    )

    output_path = resolve_external_storage_uri(
        config["selection_output"],
        external_root=args.external_root,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_output = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )
    if temp_output.exists():
        temp_output.unlink()
    temp_dir = output_path.parent / "duckdb-temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute(f"SET threads = {int(args.threads)}")
    con.execute(
        f"SET temp_directory = '{_sql_path(temp_dir)}'"
    )
    source = _sql_path(decision_path)
    con.execute(
        "CREATE TEMP VIEW decisions AS "
        f"SELECT * FROM read_json_auto('{source}', "
        "format='newline_delimited')"
    )

    invalid = con.execute(
        """
        SELECT count(*)
        FROM decisions
        WHERE status = 'accept'
          AND partition = 'train'
          AND rights_lane NOT IN ('release_safe', 'research_only')
        """
    ).fetchone()[0]
    if invalid:
        raise RuntimeError(
            "accepted train decisions contain invalid rights lanes"
        )

    available_rows = con.execute(
        """
        SELECT primary_category,
               count(*) AS records,
               sum(token_count)::BIGINT AS tokens
        FROM decisions
        WHERE status = 'accept'
          AND partition = 'train'
        GROUP BY primary_category
        ORDER BY primary_category
        """
    ).fetchall()
    available = {
        str(category): {
            "records": int(records),
            "tokens": int(tokens),
        }
        for category, records, tokens in available_rows
    }
    for category, target in targets.items():
        observed = int(
            available.get(category, {}).get("tokens", 0)
        )
        if observed < target:
            raise RuntimeError(
                f"insufficient {category} capacity "
                f"target={target} available={observed}"
            )
        print(
            f"capacity category={category} "
            f"available={observed} target={target}",
            flush=True,
        )

    target_values = ", ".join(
        f"('{category}', {int(target)})"
        for category, target in targets.items()
        if target > 0
    )
    seed = int(config["seed"])
    query = f"""
        WITH targets(primary_category, target_tokens) AS (
            VALUES {target_values}
        ),
        eligible AS (
            SELECT
                canonical_id,
                source_id,
                artifact_id,
                record_id,
                primary_category,
                token_count::BIGINT AS token_count,
                rights_lane,
                sha256('{seed}|' || canonical_id) AS selection_key
            FROM decisions
            WHERE status = 'accept'
              AND partition = 'train'
              AND primary_category IN (
                  SELECT primary_category FROM targets
              )
              AND rights_lane IN ('release_safe', 'research_only')
        ),
        ranked AS (
            SELECT
                eligible.*,
                targets.target_tokens,
                sum(token_count) OVER (
                    PARTITION BY eligible.primary_category
                    ORDER BY selection_key, canonical_id
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ) AS tokens_before
            FROM eligible
            JOIN targets USING (primary_category)
        )
        SELECT
            canonical_id,
            source_id,
            artifact_id,
            record_id,
            primary_category,
            token_count,
            least(
                token_count,
                target_tokens - coalesce(tokens_before, 0)
            )::BIGINT AS selected_token_count,
            rights_lane,
            selection_key
        FROM ranked
        WHERE coalesce(tokens_before, 0) < target_tokens
        ORDER BY primary_category, selection_key, canonical_id
    """

    cursor = con.execute(query)
    digest = hashlib.sha256()
    byte_count = 0
    selected_records = 0
    partial_records = 0
    selected_tokens: dict[str, int] = {}
    selected_records_by_category: dict[str, int] = {}
    selected_records_by_artifact: dict[str, int] = {}
    with temp_output.open("wb") as stream:
        while True:
            batch = cursor.fetchmany(10_000)
            if not batch:
                break
            for (
                canonical_id,
                source_id,
                artifact_id,
                record_id,
                category,
                token_count,
                selected_token_count,
                rights_lane,
                selection_key,
            ) in batch:
                token_count = int(token_count)
                selected_token_count = int(selected_token_count)
                if not 0 < selected_token_count <= token_count:
                    raise RuntimeError(
                        "invalid selected token count"
                    )
                row = {
                    "canonical_id": str(canonical_id),
                    "source_id": str(source_id),
                    "artifact_id": str(artifact_id),
                    "record_id": str(record_id),
                    "primary_category": str(category),
                    "token_count": token_count,
                    "selected_token_count": selected_token_count,
                    "rights_lane": str(rights_lane),
                    "selection_key": str(selection_key),
                }
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
                byte_count += len(encoded)
                selected_records += 1
                if selected_token_count != token_count:
                    partial_records += 1
                selected_tokens[str(category)] = (
                    selected_tokens.get(str(category), 0)
                    + selected_token_count
                )
                selected_records_by_category[str(category)] = (
                    selected_records_by_category.get(
                        str(category),
                        0,
                    )
                    + 1
                )
                selected_records_by_artifact[str(artifact_id)] = (
                    selected_records_by_artifact.get(
                        str(artifact_id),
                        0,
                    )
                    + 1
                )
    con.close()

    expected_selected = {
        category: target
        for category, target in targets.items()
        if target > 0
    }
    if selected_tokens != expected_selected:
        raise RuntimeError(
            "selected token totals do not match exact targets "
            f"expected={expected_selected} observed={selected_tokens}"
        )
    if sum(selected_tokens.values()) != 500_000_000:
        raise RuntimeError("selection total is not exactly 500M")
    if partial_records > len(expected_selected):
        raise RuntimeError(
            "more than one partial final record per category"
        )

    os.replace(temp_output, output_path)
    summary = {
        "schema_version": 1,
        "stage": "L004",
        "selection_id": config["selection_id"],
        "checkpoint_lane": "research_only",
        "seed": seed,
        "selection_key": config["selection_key"],
        "partition": "train",
        "decision_manifest": {
            "storage": decision["storage"],
            "sha256": observed_decision_sha,
        },
        "selection_manifest": {
            "storage": config["selection_output"],
            "bytes": byte_count,
            "sha256": digest.hexdigest(),
            "contains_raw_text": False,
        },
        "available": available,
        "targets": targets,
        "selected_tokens": selected_tokens,
        "selected_records": selected_records,
        "selected_records_by_category": (
            selected_records_by_category
        ),
        "selected_records_by_artifact": (
            selected_records_by_artifact
        ),
        "partial_records": partial_records,
        "record_boundary_policy": config[
            "record_boundary_policy"
        ],
        "total_selected_tokens": sum(selected_tokens.values()),
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
        f"selection_records={selected_records} "
        f"partial_records={partial_records}",
        flush=True,
    )
    print(
        f"selection_bytes={byte_count} "
        f"selection_sha256={digest.hexdigest()}",
        flush=True,
    )
    print("total_selected_tokens=500000000", flush=True)
    print("pretraining_selection_materialization=PASS", flush=True)


if __name__ == "__main__":
    main()
