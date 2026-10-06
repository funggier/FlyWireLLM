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
        "--config",
        default=str(ROOT / "configs" / "validation-l004-v1.json"),
    )
    parser.add_argument(
        "--summary",
        default=str(ROOT / "results" / "l004" / "validation-pack-v1.json"),
    )
    parser.add_argument(
        "--step0",
        default=str(ROOT / "results" / "l004" / "validation-step0-v1.json"),
    )
    parser.add_argument(
        "--step4",
        default=str(ROOT / "results" / "l004" / "validation-step4-v1.json"),
    )
    parser.add_argument(
        "--comparison",
        default=str(
            ROOT / "results" / "l004" / "validation-comparison-v1.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    external_root = Path(args.external_root)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    step0 = json.loads(Path(args.step0).read_text(encoding="utf-8"))
    step4 = json.loads(Path(args.step4).read_text(encoding="utf-8"))
    comparison = json.loads(
        Path(args.comparison).read_text(encoding="utf-8")
    )

    for payload in (config, summary, step0, step4, comparison):
        if payload.get("schema_version") != 1:
            raise RuntimeError("validation schema changed")
        if payload.get("stage") != "L004":
            raise RuntimeError("validation stage changed")
    for payload in (config, summary, step0, step4, comparison):
        if payload.get("evaluation_id") != "base50m-validation-v1":
            raise RuntimeError("validation evaluation id changed")
        if payload.get("partition") != "validation":
            raise RuntimeError("validation partition changed")
        if payload.get("final_holdout_touched") is not False:
            raise RuntimeError("final holdout was touched")

    selection = summary["selection_manifest"]
    selection_path = resolve_external_storage_uri(
        selection["storage"],
        external_root=external_root,
    )
    if selection_path.stat().st_size != int(selection["bytes"]):
        raise RuntimeError("validation selection byte size changed")
    observed_selection_sha = sha256_file(selection_path)
    if observed_selection_sha != selection["sha256"]:
        raise RuntimeError("validation selection SHA-256 mismatch")

    con = duckdb.connect()
    source = str(selection_path).replace("\\", "/").replace("'", "''")
    con.execute(
        "CREATE TEMP VIEW selected AS "
        f"SELECT * FROM read_json_auto('{source}', "
        "format='newline_delimited')"
    )
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
        raise RuntimeError("validation selection columns changed")
    shape = con.execute(
        """
        SELECT
            count(*)::BIGINT,
            count(DISTINCT canonical_id)::BIGINT,
            sum(selected_token_count)::BIGINT,
            sum(CASE WHEN selected_token_count < token_count
                     THEN 1 ELSE 0 END)::BIGINT,
            sum(CASE WHEN selected_token_count <= 0
                       OR selected_token_count > token_count
                     THEN 1 ELSE 0 END)::BIGINT,
            sum(CASE WHEN rights_lane NOT IN
                       ('release_safe','research_only')
                     THEN 1 ELSE 0 END)::BIGINT
        FROM selected
        """
    ).fetchone()
    records, unique_ids, tokens, partial, invalid_counts, invalid_lanes = map(
        int, shape
    )
    if records != int(selection["records"]) or unique_ids != records:
        raise RuntimeError("validation selection record uniqueness changed")
    if tokens != 300_000 or partial != 3:
        raise RuntimeError("validation selection exact-budget contract changed")
    if invalid_counts or invalid_lanes:
        raise RuntimeError("validation selection contains invalid rows")

    rows = con.execute(
        """
        SELECT primary_category,
               count(*)::BIGINT,
               sum(selected_token_count)::BIGINT,
               sum(CASE WHEN selected_token_count < token_count
                        THEN 1 ELSE 0 END)::BIGINT
        FROM selected
        GROUP BY 1
        ORDER BY 1
        """
    ).fetchall()
    con.close()
    observed_categories = {
        str(category): {
            "records": int(count),
            "tokens": int(count_tokens),
            "partial": int(count_partial),
        }
        for category, count, count_tokens, count_partial in rows
    }
    expected_categories = {
        category: {
            "records": int(summary["selected_records_by_category"][category]),
            "tokens": 100_000,
            "partial": 1,
        }
        for category in (
            "general_english",
            "general_thai",
            "technical_scientific_code",
        )
    }
    if observed_categories != expected_categories:
        raise RuntimeError("validation category accounting changed")
    print(
        f"PASS selection records={records} tokens={tokens} "
        f"sha256={observed_selection_sha}"
    )

    for category, meta in sorted(summary["packs"].items()):
        pack = resolve_external_storage_uri(
            meta["storage"],
            external_root=external_root,
        )
        if pack.stat().st_size != int(meta["bytes"]):
            raise RuntimeError(f"{category}: pack byte size changed")
        observed_sha = sha256_file(pack)
        if observed_sha != meta["sha256"]:
            raise RuntimeError(f"{category}: pack SHA-256 mismatch")
        values = np.memmap(pack, dtype="<u2", mode="r")
        if int(values.size) != int(meta["physical_tokens"]):
            raise RuntimeError(f"{category}: physical token count changed")
        eos = int(np.count_nonzero(values == 2))
        invalid = int(
            np.count_nonzero(
                ((values < 4) & (values != 2)) | (values >= 32_000)
            )
        )
        if eos != int(meta["eos_separator_tokens"]):
            raise RuntimeError(f"{category}: EOS count changed")
        if invalid:
            raise RuntimeError(f"{category}: invalid token ids found")
        content = int(values.size) - eos
        if content != 100_000:
            raise RuntimeError(f"{category}: content-token total changed")
        if int(values[-1]) != 2:
            raise RuntimeError(f"{category}: pack must end in EOS")
        del values
        print(
            f"PASS pack={category} content_tokens=100000 "
            f"sha256={observed_sha}"
        )

    checkpoint_meta = config["step4"]
    checkpoint = resolve_external_storage_uri(
        checkpoint_meta["checkpoint"],
        external_root=external_root,
    )
    if checkpoint.stat().st_size != int(checkpoint_meta["bytes"]):
        raise RuntimeError("step4 checkpoint byte size changed")
    checkpoint_sha = sha256_file(checkpoint)
    if checkpoint_sha != checkpoint_meta["sha256"]:
        raise RuntimeError("step4 checkpoint SHA-256 mismatch")

    if step0.get("optimizer_step") != 0:
        raise RuntimeError("step0 evaluation optimizer step changed")
    if step4.get("optimizer_step") != 4:
        raise RuntimeError("step4 evaluation optimizer step changed")
    if step4.get("checkpoint_sha256") != checkpoint_sha:
        raise RuntimeError("step4 evaluation checkpoint pin changed")
    for payload in (step0, step4):
        if payload["combined"]["supervised_tokens"] != 300_000:
            raise RuntimeError("validation evaluated token total changed")

    if comparison.get("gate_pass") is not True:
        raise RuntimeError("validation gate no longer passes")
    if comparison.get("larger_tranche_may_be_considered") is not True:
        raise RuntimeError("validation gate expansion decision changed")
    if comparison["combined"]["step4_loss"] > (
        comparison["combined"]["step0_loss"]
    ):
        raise RuntimeError("combined validation loss regressed")
    for category, row in comparison["categories"].items():
        if row.get("gate_pass") is not True:
            raise RuntimeError(f"{category}: validation category gate failed")

    print(f"PASS step4_checkpoint sha256={checkpoint_sha}")
    print(
        "combined_loss "
        f"step0={comparison['combined']['step0_loss']:.9f} "
        f"step4={comparison['combined']['step4_loss']:.9f} "
        f"relative={comparison['combined']['relative_change']:.6%}"
    )
    print("holdout_touched=false")
    print("validation_gate_verification=PASS")


if __name__ == "__main__":
    main()
