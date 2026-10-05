from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from flywire_llm.corpus import iter_jsonl_records
from flywire_llm.sentencepiece_tokenizer import SentencePieceTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count exact tokenizer tokens in corpus JSONL files."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("corpus", nargs="+")
    parser.add_argument("--output-json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = SentencePieceTokenizer(args.model)

    totals = {
        "records": 0,
        "characters": 0,
        "utf8_bytes": 0,
        "tokens": 0,
    }
    by_language: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "records": 0,
            "characters": 0,
            "utf8_bytes": 0,
            "tokens": 0,
        }
    )
    by_partition: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "records": 0,
            "characters": 0,
            "utf8_bytes": 0,
            "tokens": 0,
        }
    )

    for corpus_path in args.corpus:
        for record in iter_jsonl_records(corpus_path):
            token_count = len(tokenizer.encode(record.text))
            chars = len(record.text)
            utf8_bytes = len(record.text.encode("utf-8"))
            for bucket in (
                totals,
                by_language[record.language],
                by_partition[record.partition],
            ):
                bucket["records"] += 1
                bucket["characters"] += chars
                bucket["utf8_bytes"] += utf8_bytes
                bucket["tokens"] += token_count

    payload = {
        "schema_version": 1,
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "corpora": [str(Path(path)) for path in args.corpus],
        "total": totals,
        "by_language": dict(sorted(by_language.items())),
        "by_partition": dict(sorted(by_partition.items())),
    }
    rendered = json.dumps(payload, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output_json:
        Path(args.output_json).write_text(
            rendered + "\n",
            encoding="utf-8",
            newline="\n",
        )


if __name__ == "__main__":
    main()
