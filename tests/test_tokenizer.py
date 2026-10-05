from __future__ import annotations

import pytest

from flywire_llm.config import BYTE_VOCAB_SIZE
from flywire_llm.tokenizer import TokenizerError, UTF8ByteTokenizer


@pytest.mark.parametrize(
    "text",
    [
        "สวัสดีครับ นี่คือโมเดลเปล่า",
        "Hello, this is a blank model.",
        "ไทย + English + 123 + 🪰",
        "root_id=720575940630024566; ค่า=3.14",
        "",
    ],
)
def test_utf8_byte_tokenizer_round_trip(text: str):
    tokenizer = UTF8ByteTokenizer()
    token_ids = tokenizer.encode(text)
    assert tokenizer.decode(token_ids) == text
    assert all(4 <= token < BYTE_VOCAB_SIZE for token in token_ids)


def test_special_tokens_can_wrap_text_without_changing_decoded_content():
    tokenizer = UTF8ByteTokenizer()
    text = "FlyWire ภาษาไทย"
    ids = tokenizer.encode(text, add_bos=True, add_eos=True)
    assert ids[0] == tokenizer.bos_id
    assert ids[-1] == tokenizer.eos_id
    assert tokenizer.decode(ids) == text


def test_invalid_token_id_is_rejected():
    tokenizer = UTF8ByteTokenizer()
    with pytest.raises(TokenizerError, match="outside"):
        tokenizer.decode([9999])


def test_metadata_states_tokenizer_is_untrained_and_lossless():
    metadata = UTF8ByteTokenizer().metadata()
    assert metadata["trained"] is False
    assert metadata["lossless_utf8"] is True
    assert metadata["thai_safe"] is True
    assert metadata["english_safe"] is True
