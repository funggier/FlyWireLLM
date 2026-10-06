from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections import Counter
from pathlib import Path

from flywire_llm.artifacts import (
    load_artifact_ledger,
    validate_checkpoint_artifacts,
    verify_artifact_file,
)
from flywire_llm.data_contract import (
    assign_partition,
    dedup_fingerprint,
    load_corpus_inventory,
)
from flywire_llm.near_duplicate import SQLiteSimhashIndex
from flywire_llm.quality import screen_text, simhash64
from flywire_llm.rights import load_rights_policy
from flywire_llm.sampling import (
    stable_ppm_selected,
    stable_source_group_id,
)
from flywire_llm.sentencepiece_tokenizer import SentencePieceTokenizer


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ID = "fineweb-english"
SELECTION_SEED = "flywirellm-l003-fineweb-english-sample-v1"


def _duckdb():
    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError(
            "FineWeb materialization requires optional data dependency duckdb"
        ) from exc
    return duckdb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize deterministic screened research-only FineWeb English "
            "records from one pinned train shard."
        )
    )
    parser.add_argument("--artifact-id", required=True)
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--selection-ppm",
        type=int,
        default=600_000,
        help="Stable SHA-256 selection threshold out of 1,000,000.",
    )
    parser.add_argument(
        "--near-duplicate-distance",
        type=int,
        default=3,
    )
    parser.add_argument("--batch-size", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.selection_ppm <= 1_000_000:
        raise SystemExit("--selection-ppm must be in [1, 1000000]")
    if not 0 <= args.near_duplicate_distance <= 3:
        raise SystemExit(
            "--near-duplicate-distance must remain in [0, 3]"
        )
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")

    inventory = load_corpus_inventory(
        ROOT / "configs" / "corpus-sources-l002.json"
    )
    rights = load_rights_policy(
        ROOT / "configs" / "rights-lanes-l003.json",
        expected_source_ids={
            source.source_id for source in inventory.sources
        },
    )
    ledger = load_artifact_ledger(
        ROOT / "configs" / "source-artifacts-l003.json",
        rights_policy=rights,
    )
    selected_artifacts = validate_checkpoint_artifacts(
        ledger,
        rights_policy=rights,
        checkpoint_lane="research_only",
        artifact_ids=[args.artifact_id],
    )
    artifact = selected_artifacts[0]
    if artifact.source_id != SOURCE_ID:
        raise RuntimeError(
            f"expected {SOURCE_ID!r}, got {artifact.source_id!r}"
        )
    parquet = Path(args.parquet)
    verify_artifact_file(artifact, parquet)

    tokenizer = SentencePieceTokenizer(args.model)
    duckdb = _duckdb()
    con = duckdb.connect()
    con.execute("SET threads=1")
    row_count = int(
        con.execute(
            "SELECT count(*) FROM read_parquet(?)",
            [str(parquet)],
        ).fetchone()[0]
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = output_dir / "accepted.jsonl"
    rejected_meta_path = output_dir / "excluded-metadata.jsonl"

    counters = {
        "rows_total": row_count,
        "rows_processed": 0,
        "rows_selected": 0,
        "rows_not_selected": 0,
        "rows_missing_id": 0,
        "rows_missing_text": 0,
        "exact_duplicates": 0,
        "near_duplicates": 0,
        "accept_records": 0,
        "quarantine_records": 0,
        "reject_records": 0,
    }
    token_counts = {
        "accept": 0,
        "quarantine": 0,
        "reject": 0,
        "by_partition": {
            "train": 0,
            "validation": 0,
            "holdout": 0,
        },
    }
    record_counts_by_partition = {
        "train": 0,
        "validation": 0,
        "holdout": 0,
    }
    findings: Counter[str] = Counter()
    near_distances: Counter[int] = Counter()
    seen_exact: set[str] = set()
    corpus_hash = hashlib.sha256()
    excluded_hash = hashlib.sha256()
    corpus_bytes = 0
    excluded_bytes = 0

    with tempfile.TemporaryDirectory(
        prefix="flywirellm-l003-fineweb-en-neardup-"
    ) as temp_dir, corpus_path.open("wb") as corpus_stream, (
        rejected_meta_path.open("wb")
    ) as excluded_stream, SQLiteSimhashIndex(
        Path(temp_dir) / "simhash.sqlite",
        bands=4,
    ) as near_index:
        cursor = con.execute(
            """
            SELECT
                file_row_number,
                id,
                text,
                url,
                language,
                language_score,
                dump,
                date
            FROM read_parquet(?, file_row_number=true)
            """,
            [str(parquet)],
        )
        while True:
            batch = cursor.fetchmany(args.batch_size)
            if not batch:
                break
            for (
                file_row_number,
                raw_id,
                raw_text,
                raw_url,
                raw_language,
                raw_language_score,
                raw_dump,
                raw_date,
            ) in batch:
                expected_row = counters["rows_processed"]
                if int(file_row_number) != expected_row:
                    raise RuntimeError(
                        "Parquet physical row order changed: "
                        f"expected {expected_row}, got {file_row_number}"
                    )
                counters["rows_processed"] += 1
                if raw_id is None or not str(raw_id).strip():
                    counters["rows_missing_id"] += 1
                    continue
                record_id_raw = str(raw_id).strip()
                if not stable_ppm_selected(
                    record_id_raw,
                    seed=SELECTION_SEED,
                    parts_per_million=args.selection_ppm,
                ):
                    counters["rows_not_selected"] += 1
                    continue
                counters["rows_selected"] += 1
                if raw_text is None:
                    counters["rows_missing_text"] += 1
                    continue
                text = str(raw_text)
                record_id = "FINEWEB-EN-" + record_id_raw
                source_group_id = stable_source_group_id(
                    source_url=(
                        str(raw_url) if raw_url is not None else None
                    ),
                    record_id=record_id_raw,
                )
                partition = assign_partition(
                    SOURCE_ID,
                    source_group_id,
                )

                screening = screen_text(text)
                if screening.disposition.value == "accept":
                    status = "accept"
                elif screening.disposition.value == "review":
                    status = "quarantine"
                else:
                    status = "reject"
                for finding in screening.findings:
                    findings[finding.code] += 1

                exact = dedup_fingerprint(text)
                if exact in seen_exact:
                    counters["exact_duplicates"] += 1
                    continue
                seen_exact.add(exact)

                token_count = len(tokenizer.encode(text))
                if status != "accept":
                    counters[f"{status}_records"] += 1
                    token_counts[status] += token_count
                    meta = {
                        "record_id": record_id,
                        "status": status,
                        "content_sha256": hashlib.sha256(
                            text.encode("utf-8")
                        ).hexdigest(),
                        "dedup_fingerprint": exact,
                        "findings": [
                            finding.code
                            for finding in screening.findings
                        ],
                    }
                    encoded = (
                        json.dumps(
                            meta,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        + "\n"
                    ).encode("utf-8")
                    excluded_stream.write(encoded)
                    excluded_hash.update(encoded)
                    excluded_bytes += len(encoded)
                    continue

                fingerprint = simhash64(text)
                inserted, matches = near_index.add_if_unique(
                    record_id,
                    fingerprint,
                    max_hamming_distance=args.near_duplicate_distance,
                )
                if not inserted:
                    counters["near_duplicates"] += 1
                    near_distances[matches[0].hamming_distance] += 1
                    continue

                counters["accept_records"] += 1
                token_counts["accept"] += token_count
                token_counts["by_partition"][partition] += token_count
                record_counts_by_partition[partition] += 1
                payload = {
                    "record_id": record_id,
                    "text": text,
                    "language": "en",
                    "source_id": SOURCE_ID,
                    "license": "ODC-By-1.0 dataset release; research_only",
                    "partition": partition,
                    "source_group_id": source_group_id,
                    "artifact_id": artifact.artifact_id,
                    "content_sha256": hashlib.sha256(
                        text.encode("utf-8")
                    ).hexdigest(),
                    "dedup_fingerprint": exact,
                    "simhash64": f"{fingerprint:016x}",
                    "source_language": (
                        str(raw_language)
                        if raw_language is not None
                        else None
                    ),
                    "source_language_score": (
                        float(raw_language_score)
                        if raw_language_score is not None
                        else None
                    ),
                    "source_dump": (
                        str(raw_dump) if raw_dump is not None else None
                    ),
                    "source_date": (
                        str(raw_date) if raw_date is not None else None
                    ),
                }
                encoded = (
                    json.dumps(
                        payload,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                ).encode("utf-8")
                corpus_stream.write(encoded)
                corpus_hash.update(encoded)
                corpus_bytes += len(encoded)

    con.close()
    if counters["rows_processed"] != row_count:
        raise RuntimeError("row processing count mismatch")

    summary = {
        "schema_version": 1,
        "stage": "L003",
        "derived_id": (
            f"{artifact.artifact_id}-sample"
            f"{args.selection_ppm}-v1"
        ),
        "source_id": SOURCE_ID,
        "artifact_id": artifact.artifact_id,
        "rights_lane": "research_only",
        "source_local_training_eligible": True,
        "global_manifest_authorized": False,
        "global_cross_source_dedup_pending": True,
        "selection": {
            "algorithm": "sha256_id_bucket_v1",
            "seed": SELECTION_SEED,
            "parts_per_million": args.selection_ppm,
        },
        "canonical_processing_order": {
            "source": "parquet_file_row_number",
            "duckdb_threads": 1,
            "monotonic_sequence_asserted": True,
            "raw_artifact_sha256": artifact.sha256,
        },
        "screening": {
            "policy": "flywire_llm.quality.screen_text",
            "review_action": "exclude_and_quarantine_metadata_only",
            "reject_action": "exclude_metadata_only",
            "raw_excluded_text_persisted": False,
        },
        "deduplication": {
            "exact": "sha256_over_conservative_normalized_text",
            "near": (
                "simhash64+4x16bit-lsh+hamming<="
                f"{args.near_duplicate_distance}"
            ),
            "scope": "within_this_derived_shard",
            "near_distance_counts": {
                str(key): value
                for key, value in sorted(near_distances.items())
            },
        },
        "counts": counters,
        "accepted": {
            "records_by_partition": record_counts_by_partition,
            "tokens_by_partition": token_counts["by_partition"],
            "tokens_total": token_counts["accept"],
        },
        "excluded": {
            "quarantine_tokens": token_counts["quarantine"],
            "reject_tokens": token_counts["reject"],
            "findings": dict(sorted(findings.items())),
        },
        "files": {
            "accepted": {
                "path": corpus_path.name,
                "bytes": corpus_bytes,
                "sha256": corpus_hash.hexdigest(),
            },
            "excluded_metadata": {
                "path": rejected_meta_path.name,
                "bytes": excluded_bytes,
                "sha256": excluded_hash.hexdigest(),
            },
        },
        "pretraining_authorized": False,
        "reason_not_authorized": (
            "Global cross-source exact/near deduplication and final mixture "
            "manifest are still pending."
        ),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "derived_id": summary["derived_id"],
                "selected_rows": counters["rows_selected"],
                "accepted_records": counters["accept_records"],
                "train_tokens": token_counts["by_partition"]["train"],
                "validation_tokens": token_counts["by_partition"]["validation"],
                "holdout_tokens": token_counts["by_partition"]["holdout"],
                "exact_duplicates": counters["exact_duplicates"],
                "near_duplicates": counters["near_duplicates"],
                "quarantine_records": counters["quarantine_records"],
                "reject_records": counters["reject_records"],
                "accepted_sha256": corpus_hash.hexdigest(),
                "global_manifest_authorized": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
