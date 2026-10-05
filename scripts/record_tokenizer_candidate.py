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
    parser = argparse.ArgumentParser(
        description="Record a reproducible L002 final tokenizer candidate."
    )
    parser.add_argument("--corpus-summary", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-vocab-size", type=int, default=32_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    corpus = json.loads(
        Path(args.corpus_summary).read_text(encoding="utf-8")
    )
    benchmark = json.loads(
        Path(args.benchmark).read_text(encoding="utf-8")
    )
    model = Path(args.model)
    vocab = Path(args.vocab)

    if benchmark["vocab_size"] != args.target_vocab_size:
        raise RuntimeError(
            f"expected vocab size {args.target_vocab_size}, "
            f"got {benchmark['vocab_size']}"
        )
    if benchmark["overall"]["roundtrip_failures"] != 0:
        raise RuntimeError("candidate has round-trip failures")
    if benchmark["overall"]["unk_tokens"] != 0:
        raise RuntimeError("candidate produced unknown tokens")

    payload = {
        "schema_version": 1,
        "stage": "L002",
        "candidate_id": "base50m-unigram-32000-v1",
        "corpus": {
            "builder": corpus["builder"],
            "split_seed": corpus["split_seed"],
            "split_group": corpus["split_group"],
            "fit_target_bytes_per_language": corpus[
                "fit_target_bytes_per_language"
            ],
            "validation_target_bytes_per_language": corpus[
                "validation_target_bytes_per_language"
            ],
            "sources": corpus["sources"],
            "materialized": corpus["materialized"],
        },
        "tokenizer": {
            "implementation": "sentencepiece",
            "version": spm.__version__,
            "model_type": "unigram",
            "vocab_size": benchmark["vocab_size"],
            "normalization": "identity",
            "byte_fallback": True,
            "special_tokens": {
                "pad": 0,
                "bos": 1,
                "eos": 2,
                "unk": 3,
            },
            "model_sha256": sha256_file(model),
            "vocab_sha256": sha256_file(vocab),
        },
        "benchmark": {
            "cases": benchmark["cases"],
            "overall": benchmark["overall"],
            "by_language": benchmark["by_language"],
        },
        "qualification": {
            "exact_roundtrip_required": True,
            "zero_unk_required": True,
            "thai_report_required": True,
            "english_report_required": True,
            "mixed_report_required": True,
            "source_license_audit_separate": True,
            "language_quality_claimed": False,
            "pretraining_completed": False,
        },
        "decision": (
            "TOKENIZER ENGINEERING GREEN / BASE-50M 32K UNIGRAM CANDIDATE "
            "REPRODUCIBLE AND LOSSLESS ON L002 BENCHMARK / CORPUS RIGHTS "
            "AND 500M PRETRAINING READINESS REMAIN SEPARATE"
        ),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"output={output}")
    print(f"model_sha256={payload['tokenizer']['model_sha256']}")
    print(f"vocab_sha256={payload['tokenizer']['vocab_sha256']}")
    print(
        "compression_vs_byte="
        f"{payload['benchmark']['overall']['compression_vs_byte']}"
    )


if __name__ == "__main__":
    main()
