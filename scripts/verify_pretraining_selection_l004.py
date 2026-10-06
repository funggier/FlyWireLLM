from __future__ import annotations

import argparse
import hashlib
import json
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--external-root", required=True)
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
    summary = json.loads(
        Path(args.summary).read_text(encoding="utf-8")
    )
    if summary.get("schema_version") != 1:
        raise RuntimeError("selection summary schema changed")
    if summary.get("stage") != "L004":
        raise RuntimeError("selection summary stage changed")
    if summary.get("selection_id") != "base50m-500m-research-v1":
        raise RuntimeError("selection summary id changed")
    if summary.get("checkpoint_lane") != "research_only":
        raise RuntimeError("selection checkpoint lane changed")
    if summary.get("partition") != "train":
        raise RuntimeError("selection partition changed")
    if summary.get("seed") != 1234:
        raise RuntimeError("selection seed changed")
    if summary.get("pretraining_authorized") is not False:
        raise RuntimeError("selection cannot self-authorize pretraining")
    if summary.get("production_optimizer_steps_recorded") != 0:
        raise RuntimeError(
            "selection qualification cannot record production steps"
        )

    manifest = summary.get("selection_manifest")
    if not isinstance(manifest, dict):
        raise RuntimeError("selection_manifest is missing")
    if manifest.get("contains_raw_text") is not False:
        raise RuntimeError("selection manifest must be metadata-only")
    selection_path = resolve_external_storage_uri(
        manifest["storage"],
        external_root=args.external_root,
    )
    expected_bytes = manifest.get("bytes")
    if selection_path.stat().st_size != expected_bytes:
        raise RuntimeError(
            "selection byte size mismatch "
            f"expected={expected_bytes} "
            f"observed={selection_path.stat().st_size}"
        )
    observed_sha = sha256_file(selection_path)
    if observed_sha != manifest.get("sha256"):
        raise RuntimeError(
            "selection SHA-256 mismatch "
            f"expected={manifest.get('sha256')} observed={observed_sha}"
        )
    print(
        f"PASS selection_file bytes={expected_bytes} "
        f"sha256={observed_sha}",
        flush=True,
    )

    con = duckdb.connect()
    con.execute(f"SET threads = {int(args.threads)}")
    source = _sql_path(selection_path)
    con.execute(
        "CREATE TEMP VIEW selected AS "
        f"SELECT * FROM read_json_auto('{source}', "
        "format='newline_delimited')"
    )

    shape = con.execute(
        """
        SELECT
            count(*) AS records,
            count(DISTINCT canonical_id) AS unique_canonical,
            count(DISTINCT selection_key) AS unique_keys,
            sum(selected_token_count)::BIGINT AS selected_tokens,
            sum(CASE WHEN selected_token_count < token_count
                     THEN 1 ELSE 0 END)::BIGINT AS partial_records,
            sum(CASE WHEN selected_token_count <= 0
                       OR selected_token_count > token_count
                     THEN 1 ELSE 0 END)::BIGINT AS invalid_counts,
            sum(CASE WHEN rights_lane NOT IN
                       ('release_safe', 'research_only')
                     THEN 1 ELSE 0 END)::BIGINT AS invalid_lanes
        FROM selected
        """
    ).fetchone()
    (
        records,
        unique_canonical,
        unique_keys,
        selected_tokens,
        partial_records,
        invalid_counts,
        invalid_lanes,
    ) = map(int, shape)
    if records != int(summary["selected_records"]):
        raise RuntimeError("selection record count changed")
    if unique_canonical != records:
        raise RuntimeError("selection canonical IDs are not unique")
    if unique_keys != records:
        raise RuntimeError("selection keys are not unique")
    if selected_tokens != 500_000_000:
        raise RuntimeError("selection total is not exactly 500M")
    if invalid_counts:
        raise RuntimeError("selection contains invalid token counts")
    if invalid_lanes:
        raise RuntimeError("selection contains invalid rights lane")
    if partial_records != int(summary["partial_records"]):
        raise RuntimeError("selection partial record count changed")

    key_mismatches = con.execute(
        """
        SELECT count(*)
        FROM selected
        WHERE selection_key != sha256('1234|' || canonical_id)
        """
    ).fetchone()[0]
    if int(key_mismatches):
        raise RuntimeError("selection key recomputation mismatch")

    category_rows = con.execute(
        """
        SELECT
            primary_category,
            count(*)::BIGINT AS records,
            sum(selected_token_count)::BIGINT AS tokens,
            sum(CASE WHEN selected_token_count < token_count
                     THEN 1 ELSE 0 END)::BIGINT AS partial
        FROM selected
        GROUP BY primary_category
        ORDER BY primary_category
        """
    ).fetchall()
    observed_tokens = {
        str(category): int(tokens)
        for category, _, tokens, _ in category_rows
    }
    expected_tokens = {
        "general_english": 200_000_000,
        "general_thai": 200_000_000,
        "technical_scientific_code": 100_000_000,
    }
    if observed_tokens != expected_tokens:
        raise RuntimeError(
            f"category token totals changed: {observed_tokens}"
        )
    for category, records_count, tokens, partial in category_rows:
        if int(partial) != 1:
            raise RuntimeError(
                f"{category}: expected exactly one partial final record"
            )
        print(
            f"PASS category={category} records={int(records_count)} "
            f"tokens={int(tokens)} partial={int(partial)}",
            flush=True,
        )

    artifact_rows = con.execute(
        """
        SELECT artifact_id, count(*)::BIGINT
        FROM selected
        GROUP BY artifact_id
        ORDER BY artifact_id
        """
    ).fetchall()
    observed_artifacts = {
        str(artifact_id): int(count)
        for artifact_id, count in artifact_rows
    }
    if observed_artifacts != summary["selected_records_by_artifact"]:
        raise RuntimeError("artifact selection counts changed")

    columns = {
        row[0]
        for row in con.execute(
            "DESCRIBE SELECT * FROM selected"
        ).fetchall()
    }
    expected_columns = {
        "canonical_id",
        "source_id",
        "artifact_id",
        "record_id",
        "primary_category",
        "token_count",
        "selected_token_count",
        "rights_lane",
        "selection_key",
    }
    if columns != expected_columns:
        raise RuntimeError(
            "selection manifest columns changed "
            f"expected={sorted(expected_columns)} "
            f"observed={sorted(columns)}"
        )
    con.close()

    print(f"records={records}", flush=True)
    print("total_selected_tokens=500000000", flush=True)
    print("selection_manifest_metadata_only=true", flush=True)
    print("pretraining_selection_verification=PASS", flush=True)


if __name__ == "__main__":
    main()
