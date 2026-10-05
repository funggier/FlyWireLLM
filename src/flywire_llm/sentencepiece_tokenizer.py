from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol

from .corpus import TextRecord, iter_jsonl_records
from .tokenizer import UTF8ByteTokenizer


class SentencePieceDependencyError(RuntimeError):
    pass


class SentencePieceContractError(ValueError):
    pass


def _sentencepiece():
    try:
        import sentencepiece as spm
    except ImportError as exc:
        raise SentencePieceDependencyError(
            "SentencePiece support requires the optional 'tokenizer' dependency"
        ) from exc
    return spm


@dataclass(frozen=True)
class SentencePieceTrainingResult:
    model_path: Path
    vocab_path: Path
    requested_vocab_size: int
    actual_vocab_size: int
    training_records: int
    training_utf8_bytes: int
    languages: tuple[str, ...]


class TokenizerLike(Protocol):
    vocab_size: int

    def encode(self, text: str, **kwargs) -> list[int]: ...

    def decode(self, token_ids: Iterable[int], **kwargs) -> str: ...


class SentencePieceTokenizer:
    """SentencePiece Unigram wrapper with FlyWireLLM special-token IDs."""

    pad_id = 0
    bos_id = 1
    eos_id = 2
    unk_id = 3

    def __init__(self, model_path: str | Path) -> None:
        spm = _sentencepiece()
        self.model_path = Path(model_path)
        self._processor = spm.SentencePieceProcessor(
            model_file=str(self.model_path)
        )
        self.vocab_size = int(self._processor.vocab_size())
        expected = {
            self.pad_id: "<pad>",
            self.bos_id: "<s>",
            self.eos_id: "</s>",
            self.unk_id: "<unk>",
        }
        for token_id, piece in expected.items():
            if self._processor.id_to_piece(token_id) != piece:
                raise SentencePieceContractError(
                    f"special token id {token_id} must be {piece!r}"
                )

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be str")
        ids = list(self._processor.encode(text, out_type=int))
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(
        self,
        token_ids: Iterable[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        ids = [int(value) for value in token_ids]
        if skip_special_tokens:
            ids = [
                token_id
                for token_id in ids
                if token_id not in {
                    self.pad_id,
                    self.bos_id,
                    self.eos_id,
                }
            ]
        return str(self._processor.decode(ids))

    def metadata(self) -> dict[str, object]:
        return {
            "name": "sentencepiece-unigram-v1",
            "model_path": self.model_path.as_posix(),
            "vocab_size": self.vocab_size,
            "model_type": "unigram",
            "normalization": "identity",
            "byte_fallback": True,
            "lossless_target": True,
            "special_tokens": {
                "pad": self.pad_id,
                "bos": self.bos_id,
                "eos": self.eos_id,
                "unk": self.unk_id,
            },
        }


def _fit_records(
    corpus_paths: Iterable[str | Path],
    *,
    partitions: set[str],
) -> list[TextRecord]:
    records: list[TextRecord] = []
    seen_ids: set[str] = set()
    for corpus_path in corpus_paths:
        for record in iter_jsonl_records(corpus_path):
            if record.partition not in partitions:
                continue
            if record.record_id in seen_ids:
                raise SentencePieceContractError(
                    f"duplicate record_id across tokenizer corpora: "
                    f"{record.record_id!r}"
                )
            seen_ids.add(record.record_id)
            records.append(record)
    if not records:
        raise SentencePieceContractError(
            "tokenizer fitting requires at least one selected record"
        )
    return records


def train_sentencepiece_unigram(
    corpus_paths: Iterable[str | Path],
    *,
    output_prefix: str | Path,
    vocab_size: int = 32_000,
    partitions: set[str] | None = None,
    require_exact_vocab: bool = True,
    character_coverage: float = 0.99995,
) -> SentencePieceTrainingResult:
    """Train deterministic Unigram SentencePiece from selected corpus partitions.

    Identity normalization, whitespace preservation and byte fallback are fixed
    by contract so identifiers and arbitrary UTF-8 remain recoverable.
    """

    if vocab_size < 512:
        raise SentencePieceContractError(
            "vocab_size must be at least 512 when byte fallback is enabled"
        )
    if not 0.0 < character_coverage <= 1.0:
        raise SentencePieceContractError(
            "character_coverage must be in (0, 1]"
        )
    partitions = set(partitions or {"train"})
    if "holdout" in partitions:
        raise SentencePieceContractError(
            "holdout data cannot be used to fit the tokenizer"
        )

    records = _fit_records(corpus_paths, partitions=partitions)
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    spm = _sentencepiece()

    with tempfile.TemporaryDirectory(
        prefix="flywirellm-tokenizer-"
    ) as temp_dir:
        training_text = Path(temp_dir) / "fit.txt"
        with training_text.open("w", encoding="utf-8", newline="\n") as stream:
            for record in records:
                # SentencePiece is line-oriented. Preserve text content within a
                # record by converting embedded line endings to U+2028 rather
                # than merging records or leaking partition structure.
                text = record.text.replace("\r\n", "\n").replace(
                    "\r", "\n"
                ).replace("\n", "\u2028")
                stream.write(text)
                stream.write("\n")

        spm.SentencePieceTrainer.train(
            input=str(training_text),
            model_prefix=str(output_prefix),
            model_type="unigram",
            vocab_size=int(vocab_size),
            character_coverage=float(character_coverage),
            byte_fallback=True,
            normalization_rule_name="identity",
            add_dummy_prefix=False,
            remove_extra_whitespaces=False,
            hard_vocab_limit=bool(require_exact_vocab),
            pad_id=0,
            bos_id=1,
            eos_id=2,
            unk_id=3,
            input_sentence_size=0,
            shuffle_input_sentence=False,
            num_threads=1,
            minloglevel=2,
        )

    model_path = output_prefix.with_suffix(".model")
    vocab_path = output_prefix.with_suffix(".vocab")
    tokenizer = SentencePieceTokenizer(model_path)
    if require_exact_vocab and tokenizer.vocab_size != vocab_size:
        raise SentencePieceContractError(
            f"expected exact vocab_size={vocab_size}, "
            f"got {tokenizer.vocab_size}"
        )

    return SentencePieceTrainingResult(
        model_path=model_path,
        vocab_path=vocab_path,
        requested_vocab_size=int(vocab_size),
        actual_vocab_size=tokenizer.vocab_size,
        training_records=len(records),
        training_utf8_bytes=sum(
            len(record.text.encode("utf-8")) for record in records
        ),
        languages=tuple(sorted({record.language for record in records})),
    )


@dataclass(frozen=True)
class TokenizerBenchmarkRow:
    case_id: str
    language: str
    category: str
    characters: int
    utf8_bytes: int
    byte_tokens: int
    candidate_tokens: int
    byte_tokens_per_char: float
    candidate_tokens_per_char: float
    compression_vs_byte: float
    exact_roundtrip: bool
    unk_tokens: int


def load_benchmark_cases(path: str | Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, start=1):
            if not raw.strip():
                continue
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise SentencePieceContractError(
                    f"{path}:{line_number}: benchmark row must be an object"
                )
            required = {"case_id", "language", "category", "text"}
            missing = required - set(payload)
            if missing:
                raise SentencePieceContractError(
                    f"{path}:{line_number}: missing {sorted(missing)}"
                )
            case_id = str(payload["case_id"])
            if case_id in seen:
                raise SentencePieceContractError(
                    f"duplicate benchmark case_id {case_id!r}"
                )
            seen.add(case_id)
            rows.append({key: str(payload[key]) for key in required})
    if not rows:
        raise SentencePieceContractError("benchmark cases must not be empty")
    return rows


def benchmark_tokenizer(
    tokenizer: SentencePieceTokenizer,
    cases: Iterable[dict[str, str]],
) -> list[TokenizerBenchmarkRow]:
    byte_tokenizer = UTF8ByteTokenizer()
    rows: list[TokenizerBenchmarkRow] = []
    for case in cases:
        text = case["text"]
        byte_ids = byte_tokenizer.encode(text)
        candidate_ids = tokenizer.encode(text)
        decoded = tokenizer.decode(candidate_ids)
        characters = len(text)
        if characters == 0:
            raise SentencePieceContractError(
                "benchmark cases must contain non-empty text"
            )
        candidate_tokens = len(candidate_ids)
        byte_tokens = len(byte_ids)
        rows.append(
            TokenizerBenchmarkRow(
                case_id=case["case_id"],
                language=case["language"],
                category=case["category"],
                characters=characters,
                utf8_bytes=len(text.encode("utf-8")),
                byte_tokens=byte_tokens,
                candidate_tokens=candidate_tokens,
                byte_tokens_per_char=byte_tokens / characters,
                candidate_tokens_per_char=candidate_tokens / characters,
                compression_vs_byte=(
                    byte_tokens / candidate_tokens
                    if candidate_tokens
                    else float("inf")
                ),
                exact_roundtrip=decoded == text,
                unk_tokens=sum(
                    token_id == tokenizer.unk_id
                    for token_id in candidate_ids
                ),
            )
        )
    return rows
