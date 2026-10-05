from __future__ import annotations

import argparse
import json

from flywire_llm.sentencepiece_tokenizer import (
    train_sentencepiece_unigram,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the FlyWireLLM SentencePiece Unigram candidate."
    )
    parser.add_argument("corpus", nargs="+")
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--vocab-size", type=int, default=32_000)
    parser.add_argument(
        "--allow-smaller-vocab",
        action="store_true",
        help="Pilot-only: permit smaller realized vocabulary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_sentencepiece_unigram(
        args.corpus,
        output_prefix=args.output_prefix,
        vocab_size=args.vocab_size,
        partitions={"train"},
        require_exact_vocab=not args.allow_smaller_vocab,
    )
    print(
        json.dumps(
            {
                "model_path": str(result.model_path),
                "vocab_path": str(result.vocab_path),
                "requested_vocab_size": result.requested_vocab_size,
                "actual_vocab_size": result.actual_vocab_size,
                "training_records": result.training_records,
                "training_utf8_bytes": result.training_utf8_bytes,
                "languages": result.languages,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
