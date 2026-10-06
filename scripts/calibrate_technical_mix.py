from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from flywire_llm.sampling import stable_ppm_selected
from flywire_llm.sentencepiece_tokenizer import SentencePieceTokenizer
from flywire_llm.technical import classify_technical_text


SEED = "flywirellm-l003-technical-calibration-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--sample-ppm", type=int, default=10_000)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.sample_ppm <= 1_000_000:
        raise SystemExit("--sample-ppm must be in [1,1000000]")
    tokenizer = SentencePieceTokenizer(args.model)

    records_total = 0
    records_sampled = 0
    token_counts = Counter()
    record_counts = Counter()
    signal_counts = Counter()
    score_histogram = Counter()
    corpus = Path(args.corpus)

    with corpus.open("r", encoding="utf-8") as stream:
        for raw in stream:
            if not raw.strip():
                continue
            row = json.loads(raw)
            records_total += 1
            record_id = str(row["record_id"])
            if not stable_ppm_selected(
                record_id,
                seed=SEED,
                parts_per_million=args.sample_ppm,
            ):
                continue
            records_sampled += 1
            text = str(row["text"])
            classification = classify_technical_text(text)
            tokens = len(tokenizer.encode(text))
            category = classification.category
            record_counts[category] += 1
            token_counts[category] += tokens
            score_histogram[str(classification.score)] += 1
            for signal in classification.signals:
                signal_counts[signal] += 1

    total_tokens = sum(token_counts.values())
    technical_tokens = token_counts["technical_scientific_code"]
    payload = {
        "schema_version": 1,
        "stage": "L003",
        "classifier": "technical-heuristic-v2",
        "sample_seed": SEED,
        "sample_ppm": args.sample_ppm,
        "corpus_path": str(corpus),
        "corpus_sha256": _sha256_file(corpus),
        "records_total": records_total,
        "records_sampled": records_sampled,
        "records_by_category": dict(record_counts),
        "tokens_by_category": dict(token_counts),
        "technical_token_fraction": (
            technical_tokens / total_tokens if total_tokens else 0.0
        ),
        "signals": dict(signal_counts.most_common()),
        "score_histogram": dict(
            sorted(score_histogram.items(), key=lambda item: int(item[0]))
        ),
        "training_authorized": False,
        "calibration_only": True,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(payload, indent=2))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
