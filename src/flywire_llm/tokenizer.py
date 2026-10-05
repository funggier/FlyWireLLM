from __future__ import annotations

from collections.abc import Iterable

from .config import BYTE_VOCAB_SIZE, SPECIAL_TOKEN_COUNT


SPECIAL_TOKENS = {
    "pad": 0,
    "bos": 1,
    "eos": 2,
    "unk": 3,
}


class TokenizerError(ValueError):
    pass


class UTF8ByteTokenizer:
    """Training-free, lossless UTF-8 byte tokenizer.

    Bytes occupy token ids 4..259. Four ids are reserved for special tokens.
    This is a bootstrap tokenizer, not a claim of tokenization efficiency.
    """

    name = "utf8-byte-v1"
    vocab_size = BYTE_VOCAB_SIZE
    byte_offset = SPECIAL_TOKEN_COUNT
    pad_id = SPECIAL_TOKENS["pad"]
    bos_id = SPECIAL_TOKENS["bos"]
    eos_id = SPECIAL_TOKENS["eos"]
    unk_id = SPECIAL_TOKENS["unk"]

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be str")
        token_ids = [self.byte_offset + value for value in text.encode("utf-8")]
        if add_bos:
            token_ids.insert(0, self.bos_id)
        if add_eos:
            token_ids.append(self.eos_id)
        return token_ids

    def decode(
        self,
        token_ids: Iterable[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        data = bytearray()
        for raw_token in token_ids:
            token = int(raw_token)
            if 0 <= token < self.byte_offset:
                if skip_special_tokens:
                    continue
                raise TokenizerError(
                    "special tokens cannot be rendered as UTF-8 bytes when "
                    "skip_special_tokens=False"
                )
            byte = token - self.byte_offset
            if not (0 <= byte <= 255):
                raise TokenizerError(f"token id {token} is outside the byte vocabulary")
            data.append(byte)
        return data.decode("utf-8", errors="strict")

    def metadata(self) -> dict[str, object]:
        return {
            "name": self.name,
            "trained": False,
            "vocab_size": self.vocab_size,
            "byte_offset": self.byte_offset,
            "special_tokens": dict(SPECIAL_TOKENS),
            "lossless_utf8": True,
            "thai_safe": True,
            "english_safe": True,
        }
