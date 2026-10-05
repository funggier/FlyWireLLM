from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import sentencepiece as spm


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-dir", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pilot = Path(args.pilot_dir)
    summary = json.loads(
        (pilot / "pilot_summary.json").read_text(encoding="utf-8")
    )
    benchmark = json.loads(
        (pilot / "benchmark-8192.json").read_text(encoding="utf-8")
    )

    payload = {
        "schema_version": 1,
        "stage": "L002",
        "pilot_id": "tatoeba-balanced-unigram-8192",
        "source_snapshot": {
            "source_id": summary["source_id"],
            "snapshot_date": summary["snapshot_date"],
            "license": summary["license"],
            "raw_sha256": {
                "th": summary["raw_artifacts"]["thai"]["sha256"],
                "en": summary["raw_artifacts"]["english"]["sha256"],
            },
            "raw_records": {
                "th": summary["raw_artifacts"]["thai"]["raw_records"],
                "en": summary["raw_artifacts"]["english"]["raw_records"],
            },
        },
        "split": {
            "method": "deterministic SHA-256 source-group split",
            "group": "dedup_fingerprint",
            "seed": summary["split_seed"],
            "partition_counts": summary["partition_counts"],
            "holdout_materialized": False,
        },
        "fit": {
            "records": {
                "th": summary["fit_materialization"]["thai_records"],
                "en": summary["fit_materialization"]["english_records"],
            },
            "utf8_bytes": {
                "th": summary["fit_materialization"]["thai_utf8_bytes"],
                "en": summary["fit_materialization"]["english_utf8_bytes"],
            },
            "fit_sha256": summary["fit_sha256"],
        },
        "tokenizer": {
            "implementation": "sentencepiece",
            "version": spm.__version__,
            "model_type": "unigram",
            "requested_vocab_size": 8192,
            "actual_vocab_size": benchmark["vocab_size"],
            "normalization": "identity",
            "byte_fallback": True,
            "model_sha256": sha256_file(
                pilot / "tatoeba-unigram-8192.model"
            ),
            "vocab_sha256": sha256_file(
                pilot / "tatoeba-unigram-8192.vocab"
            ),
            "final_base50m_tokenizer": False,
        },
        "benchmark": {
            "cases": benchmark["cases"],
            "roundtrip_failures": benchmark["overall"][
                "roundtrip_failures"
            ],
            "unk_tokens": benchmark["overall"]["unk_tokens"],
            "compression_vs_byte_overall": benchmark["overall"][
                "compression_vs_byte"
            ],
            "compression_vs_byte_thai": benchmark["by_language"]["th"][
                "compression_vs_byte"
            ],
            "compression_vs_byte_english": benchmark["by_language"]["en"][
                "compression_vs_byte"
            ],
            "compression_vs_byte_mixed": benchmark["by_language"]["mixed"][
                "compression_vs_byte"
            ],
            "validation_sha256": summary["validation_sha256"],
        },
        "decision": (
            "PILOT GREEN / UNIGRAM + IDENTITY NORMALIZATION + BYTE FALLBACK "
            "RETAINED AS BASE-50M TOKENIZER CANDIDATE / 32K FINAL VOCAB "
            "NOT YET QUALIFIED"
        ),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(output)
    print(json.dumps(payload["benchmark"], indent=2))


if __name__ == "__main__":
    main()
