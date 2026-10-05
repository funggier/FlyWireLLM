from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("sentencepiece")

from flywire_llm.sentencepiece_tokenizer import (
    SentencePieceContractError,
    SentencePieceTokenizer,
    benchmark_tokenizer,
    train_sentencepiece_unigram,
)


def _training_corpus(path: Path) -> Path:
    rows = []
    thai = [
        "โมเดลภาษากำลังเรียนรู้รูปแบบของข้อความภาษาไทย",
        "ข้อมูลฝึกต้องแยกจากข้อมูลตรวจสอบอย่างชัดเจน",
        "การแบ่งคำภาษาไทยไม่จำเป็นต้องอาศัยช่องว่างเสมอไป",
        "ตัวระบุและตัวเลขต้องถอดกลับได้อย่างถูกต้อง",
    ]
    english = [
        "A language model learns statistical structure from training text.",
        "Validation data must remain separate from tokenizer fitting data.",
        "Identifiers and numbers should round trip without corruption.",
        "Causal attention must not read future tokens.",
    ]
    for i in range(120):
        language = "th" if i % 2 == 0 else "en"
        source = thai if language == "th" else english
        text = source[i % len(source)] + f" sample={i:03d}"
        rows.append(
            {
                "record_id": f"R{i:03d}",
                "text": text,
                "language": language,
                "source_id": "synthetic-tokenizer-test",
                "license": "CC0-1.0",
                "partition": "train",
            }
        )
    rows.append(
        {
            "record_id": "HOLDOUT-001",
            "text": "ข้อความนี้เป็น holdout และต้องไม่เข้า tokenizer fit",
            "language": "th",
            "source_id": "synthetic-tokenizer-test",
            "license": "CC0-1.0",
            "partition": "holdout",
        }
    )
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )
    return path


def test_sentencepiece_candidate_roundtrips_thai_english_and_identifiers(tmp_path):
    corpus = _training_corpus(tmp_path / "train.jsonl")
    result = train_sentencepiece_unigram(
        [corpus],
        output_prefix=tmp_path / "candidate",
        vocab_size=512,
        require_exact_vocab=False,
        character_coverage=1.0,
    )
    tokenizer = SentencePieceTokenizer(result.model_path)
    cases = [
        "ภาษาไทยไม่มีช่องว่างทุกคำ",
        "Hello tokenizer.",
        "ไทย + English root_id=720575940630024566",
        "https://example.org/a?x=1&y=2",
        "  repeated   spaces\tand tabs  ",
        "แมลงหวี่ 🪰 αβγ",
    ]
    for text in cases:
        ids = tokenizer.encode(text)
        assert tokenizer.decode(ids) == text
        assert tokenizer.unk_id not in ids


def test_sentencepiece_fit_rejects_holdout_partition(tmp_path):
    corpus = _training_corpus(tmp_path / "train.jsonl")
    with pytest.raises(
        SentencePieceContractError,
        match="holdout data cannot be used",
    ):
        train_sentencepiece_unigram(
            [corpus],
            output_prefix=tmp_path / "bad",
            vocab_size=512,
            partitions={"train", "holdout"},
            require_exact_vocab=False,
        )


def test_candidate_benchmark_reports_compression_without_unknowns(tmp_path):
    corpus = _training_corpus(tmp_path / "train.jsonl")
    result = train_sentencepiece_unigram(
        [corpus],
        output_prefix=tmp_path / "candidate",
        vocab_size=512,
        require_exact_vocab=False,
        character_coverage=1.0,
    )
    tokenizer = SentencePieceTokenizer(result.model_path)
    rows = benchmark_tokenizer(
        tokenizer,
        [
            {
                "case_id": "TH",
                "language": "th",
                "category": "natural",
                "text": "ข้อมูลภาษาไทยสำหรับวัดประสิทธิภาพโทเคไนเซอร์",
            },
            {
                "case_id": "EN",
                "language": "en",
                "category": "natural",
                "text": "Tokenizer efficiency matters for model training.",
            },
        ],
    )
    assert all(row.exact_roundtrip for row in rows)
    assert sum(row.unk_tokens for row in rows) == 0
    assert sum(row.candidate_tokens for row in rows) < sum(
        row.byte_tokens for row in rows
    )
