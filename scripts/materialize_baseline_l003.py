from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

from flywire_llm.baseline_sources import (
    BaselineSourceRow,
    iter_tatoeba,
    iter_thwiki_sample,
)
from flywire_llm.quality import screen_text, simhash64
from flywire_llm.sentencepiece_tokenizer import SentencePieceTokenizer


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize screened L003 baseline accepted corpus."
    )
    parser.add_argument(
        "--kind",
        choices=("tatoeba-th", "tatoeba-en", "thwiki"),
        required=True,
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--accepted-storage", required=True)
    return parser.parse_args()


def rows_for(args: argparse.Namespace) -> Iterable[BaselineSourceRow]:
    if args.kind == "tatoeba-th":
        return iter_tatoeba(
            args.input,
            language="th",
            code="tha",
            artifact_id="tatoeba-tha-sentences-2026-10-03",
        )
    if args.kind == "tatoeba-en":
        return iter_tatoeba(
            args.input,
            language="en",
            code="eng",
            artifact_id="tatoeba-eng-sentences-2026-10-03",
        )
    return iter_thwiki_sample(args.input)


def main() -> None:
    args = parse_args()
    tokenizer = SentencePieceTokenizer(args.model)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    accepted_path = output_dir / "accepted.jsonl"

    counts = Counter()
    findings = Counter()
    tokens_by_partition = Counter()
    bytes_by_partition = Counter()
    source_id = artifact_id = rights_lane = primary_category = None

    with accepted_path.open("w", encoding="utf-8", newline="\n") as output:
        for row in rows_for(args):
            source_id = source_id or row.source_id
            artifact_id = artifact_id or row.artifact_id
            rights_lane = rights_lane or row.rights_lane
            primary_category = primary_category or row.category
            if (
                source_id != row.source_id
                or artifact_id != row.artifact_id
                or rights_lane != row.rights_lane
                or primary_category != row.category
            ):
                raise RuntimeError("materializer input changed source identity")

            counts["input_records"] += 1
            screening = screen_text(row.text)
            for finding in screening.findings:
                findings[finding.code] += 1
            disposition = screening.disposition.value
            if disposition == "review":
                counts["quarantine_records"] += 1
                continue
            if disposition == "reject":
                counts["reject_records"] += 1
                continue

            content_sha = hashlib.sha256(
                row.text.encode("utf-8")
            ).hexdigest()
            token_count = len(tokenizer.encode(row.text))
            payload = {
                "record_id": row.record_id,
                "text": row.text,
                "language": row.language,
                "source_id": row.source_id,
                "license": row.license,
                "partition": row.partition,
                "source_group_id": row.source_group_id,
                "artifact_id": row.artifact_id,
                "content_sha256": content_sha,
                "dedup_fingerprint": row.source_group_id,
                "simhash64": f"{simhash64(row.text):016x}",
            }
            output.write(
                json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            )
            output.write("\n")
            counts["accept_records"] += 1
            tokens_by_partition[row.partition] += token_count
            bytes_by_partition[row.partition] += screening.utf8_bytes

    if not source_id or not artifact_id or not rights_lane or not primary_category:
        raise RuntimeError("input yielded no records")

    accepted_bytes = accepted_path.stat().st_size
    accepted_sha = sha256_file(accepted_path)
    derived_id = {
        "tatoeba-th": "tatoeba-th-screened-v1",
        "tatoeba-en": "tatoeba-en-screened-v1",
        "thwiki": "thwiki-sample-v1-screened-v1",
    }[args.kind]
    summary = {
        "schema_version": 1,
        "stage": "L003",
        "derived_id": derived_id,
        "source_id": source_id,
        "artifact_id": artifact_id,
        "rights_lane": rights_lane,
        "primary_category": primary_category,
        "tokenizer": "base50m-unigram-32000-v1",
        "screening": "quality-v1",
        "counts": dict(counts),
        "findings": dict(sorted(findings.items())),
        "accepted": {
            "tokens_by_partition": {
                name: int(tokens_by_partition[name])
                for name in ("train", "validation", "holdout")
            },
            "text_bytes_by_partition": {
                name: int(bytes_by_partition[name])
                for name in ("train", "validation", "holdout")
            },
        },
        "files": {
            "accepted": {
                "path": "accepted.jsonl",
                "bytes": accepted_bytes,
                "sha256": accepted_sha,
                "storage": args.accepted_storage,
            }
        },
        "global_manifest_authorized": False,
        "pretraining_authorized": False,
        "raw_flagged_text_persisted": False,
    }
    summary_output = Path(args.summary_output)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
