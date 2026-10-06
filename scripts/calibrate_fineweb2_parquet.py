from __future__ import annotations

import argparse
import json
import math
import tempfile
from collections import Counter
from pathlib import Path

from flywire_llm.calibration import (
    load_calibration_artifacts,
    verify_calibration_file,
)
from flywire_llm.data_contract import (
    dedup_fingerprint,
    load_corpus_inventory,
)
from flywire_llm.near_duplicate import SQLiteSimhashIndex
from flywire_llm.quality import screen_text, simhash64
from flywire_llm.rights import load_rights_policy
from flywire_llm.sentencepiece_tokenizer import SentencePieceTokenizer


ROOT = Path(__file__).resolve().parents[1]


def _duckdb():
    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError(
            "FineWeb2 calibration requires optional data dependency duckdb"
        ) from exc
    return duckdb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate FineWeb2 Parquet without admitting it to model partitions."
    )
    parser.add_argument("--calibration-id", required=True)
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--near-duplicate-distance", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if not 0 <= args.near_duplicate_distance <= 3:
        raise SystemExit(
            "calibration near-duplicate distance must stay in [0, 3]"
        )

    inventory = load_corpus_inventory(
        ROOT / "configs" / "corpus-sources-l002.json"
    )
    rights = load_rights_policy(
        ROOT / "configs" / "rights-lanes-l003.json",
        expected_source_ids={
            source.source_id for source in inventory.sources
        },
    )
    artifacts = load_calibration_artifacts(
        ROOT / "configs" / "calibration-artifacts-l003.json",
        rights_policy=rights,
    )
    by_id = {
        artifact.calibration_id: artifact
        for artifact in artifacts
    }
    artifact = by_id.get(args.calibration_id)
    if artifact is None:
        raise RuntimeError(
            f"unknown calibration-id {args.calibration_id!r}"
        )
    if not (
        artifact.never_train
        and artifact.never_validation
        and artifact.never_holdout
    ):
        raise RuntimeError(
            "calibration artifact partition isolation is not frozen"
        )

    parquet = Path(args.parquet)
    verify_calibration_file(artifact, parquet)
    tokenizer = SentencePieceTokenizer(args.model)
    duckdb = _duckdb()
    con = duckdb.connect()

    schema_rows = con.execute(
        "DESCRIBE SELECT * FROM read_parquet(?)",
        [str(parquet)],
    ).fetchall()
    schema = [
        {"name": row[0], "type": row[1], "nullable": row[2]}
        for row in schema_rows
    ]
    required_columns = {"text", "id", "language", "language_score"}
    observed_columns = {item["name"] for item in schema}
    missing = required_columns - observed_columns
    if missing:
        raise RuntimeError(
            f"FineWeb2 calibration schema missing {sorted(missing)}"
        )

    row_count = int(
        con.execute(
            "SELECT count(*) FROM read_parquet(?)",
            [str(parquet)],
        ).fetchone()[0]
    )
    null_text = int(
        con.execute(
            "SELECT count(*) FROM read_parquet(?) WHERE text IS NULL",
            [str(parquet)],
        ).fetchone()[0]
    )
    null_id = int(
        con.execute(
            "SELECT count(*) FROM read_parquet(?) WHERE id IS NULL",
            [str(parquet)],
        ).fetchone()[0]
    )
    language_counts = {
        str(language): int(count)
        for language, count in con.execute(
            """
            SELECT coalesce(language, '<NULL>'), count(*)
            FROM read_parquet(?)
            GROUP BY 1
            ORDER BY 2 DESC, 1
            """,
            [str(parquet)],
        ).fetchall()
    }
    quantiles = con.execute(
        """
        SELECT
            min(language_score),
            approx_quantile(language_score, 0.01),
            approx_quantile(language_score, 0.05),
            approx_quantile(language_score, 0.50),
            approx_quantile(language_score, 0.95),
            approx_quantile(language_score, 0.99),
            max(language_score),
            avg(language_score)
        FROM read_parquet(?)
        """,
        [str(parquet)],
    ).fetchone()
    language_score = {
        key: (float(value) if value is not None else None)
        for key, value in zip(
            ("min", "p01", "p05", "p50", "p95", "p99", "max", "mean"),
            quantiles,
        )
    }

    status = {
        "accept": {"records": 0, "tokens": 0, "utf8_bytes": 0},
        "quarantine": {"records": 0, "tokens": 0, "utf8_bytes": 0},
        "reject": {"records": 0, "tokens": 0, "utf8_bytes": 0},
    }
    findings: Counter[str] = Counter()
    total_tokens = 0
    total_text_bytes = 0
    total_characters = 0
    seen_exact: set[str] = set()
    exact_duplicate_records = 0
    exact_duplicate_tokens = 0
    near_duplicate_records = 0
    near_duplicate_tokens = 0
    near_distance_counts: Counter[int] = Counter()
    rows_processed = 0

    with tempfile.TemporaryDirectory(
        prefix="flywirellm-l003-neardup-"
    ) as temp_dir:
        index_path = Path(temp_dir) / "simhash.sqlite"
        with SQLiteSimhashIndex(index_path, bands=4) as near_index:
            cursor = con.execute(
                """
                SELECT id, text, language, language_score
                FROM read_parquet(?)
                ORDER BY id
                """,
                [str(parquet)],
            )
            while True:
                batch = cursor.fetchmany(args.batch_size)
                if not batch:
                    break
                for raw_id, raw_text, _language, _score in batch:
                    rows_processed += 1
                    if raw_text is None:
                        continue
                    text = str(raw_text)
                    token_count = len(tokenizer.encode(text))
                    utf8_bytes = len(text.encode("utf-8"))
                    total_tokens += token_count
                    total_text_bytes += utf8_bytes
                    total_characters += len(text)

                    screening = screen_text(text)
                    if screening.disposition.value == "accept":
                        disposition = "accept"
                    elif screening.disposition.value == "review":
                        disposition = "quarantine"
                    else:
                        disposition = "reject"
                    bucket = status[disposition]
                    bucket["records"] += 1
                    bucket["tokens"] += token_count
                    bucket["utf8_bytes"] += utf8_bytes
                    for finding in screening.findings:
                        findings[finding.code] += 1

                    exact = dedup_fingerprint(text)
                    if exact in seen_exact:
                        exact_duplicate_records += 1
                        exact_duplicate_tokens += token_count
                        continue
                    seen_exact.add(exact)

                    if disposition != "accept":
                        continue

                    fingerprint = simhash64(text)
                    record_id = (
                        str(raw_id)
                        if raw_id is not None
                        else f"row-{rows_processed:08d}"
                    )
                    inserted, matches = near_index.add_if_unique(
                        record_id,
                        fingerprint,
                        max_hamming_distance=args.near_duplicate_distance,
                    )
                    if not inserted:
                        near_duplicate_records += 1
                        near_duplicate_tokens += token_count
                        near_distance_counts[
                            matches[0].hamming_distance
                        ] += 1

    con.close()
    if rows_processed != row_count:
        raise RuntimeError(
            f"row processing mismatch {rows_processed} != {row_count}"
        )

    accepted_tokens = status["accept"]["tokens"]
    accepted_bytes = status["accept"]["utf8_bytes"]
    parquet_bytes = parquet.stat().st_size
    output = {
        "schema_version": 1,
        "stage": "L003",
        "calibration_id": artifact.calibration_id,
        "source_id": artifact.source_id,
        "source_split": artifact.source_split,
        "purpose": artifact.purpose,
        "partition_isolation": {
            "never_train": artifact.never_train,
            "never_validation": artifact.never_validation,
            "never_holdout": artifact.never_holdout,
        },
        "artifact": {
            "bytes": artifact.bytes,
            "sha256": artifact.sha256,
        },
        "tooling": {
            "duckdb_version": duckdb.__version__,
            "tokenizer": "base50m-unigram-32000-v1",
            "near_duplicate": (
                "simhash64+4x16bit-lsh+hamming<="
                f"{args.near_duplicate_distance}"
            ),
        },
        "schema": schema,
        "rows": {
            "total": row_count,
            "null_text": null_text,
            "null_id": null_id,
            "processed": rows_processed,
        },
        "language": {
            "counts": language_counts,
            "score": language_score,
        },
        "content": {
            "characters": total_characters,
            "utf8_bytes": total_text_bytes,
            "tokens": total_tokens,
            "tokens_per_text_utf8_byte": (
                total_tokens / total_text_bytes
                if total_text_bytes
                else 0.0
            ),
        },
        "screening": {
            "status": status,
            "findings": dict(sorted(findings.items())),
            "accepted_token_fraction": (
                accepted_tokens / total_tokens
                if total_tokens
                else 0.0
            ),
            "accepted_record_fraction": (
                status["accept"]["records"] / row_count
                if row_count
                else 0.0
            ),
        },
        "deduplication": {
            "exact_duplicate_records": exact_duplicate_records,
            "exact_duplicate_tokens": exact_duplicate_tokens,
            "near_duplicate_records_after_exact": near_duplicate_records,
            "near_duplicate_tokens_after_exact": near_duplicate_tokens,
            "near_duplicate_distance_counts": {
                str(key): value
                for key, value in sorted(near_distance_counts.items())
            },
        },
        "density": {
            "accepted_tokens_per_parquet_byte": (
                accepted_tokens / parquet_bytes
                if parquet_bytes
                else 0.0
            ),
            "accepted_tokens_per_compressed_gib": (
                accepted_tokens * (1024**3) / parquet_bytes
                if parquet_bytes
                else 0.0
            ),
            "accepted_tokens_per_text_utf8_byte": (
                accepted_tokens / accepted_bytes
                if accepted_bytes
                else 0.0
            ),
        },
        "training_authorized": False,
        "calibration_only": True,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "calibration_id": artifact.calibration_id,
                "rows": row_count,
                "tokens": total_tokens,
                "accepted_tokens": accepted_tokens,
                "accepted_token_fraction": output["screening"][
                    "accepted_token_fraction"
                ],
                "exact_duplicate_records": exact_duplicate_records,
                "near_duplicate_records_after_exact": near_duplicate_records,
                "accepted_tokens_per_compressed_gib": output["density"][
                    "accepted_tokens_per_compressed_gib"
                ],
                "training_authorized": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
