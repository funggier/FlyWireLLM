from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


SPECIAL_TOKEN_COUNT = 4
BYTE_VOCAB_SIZE = SPECIAL_TOKEN_COUNT + 256


class BlankLLMConfigError(ValueError):
    pass


@dataclass(frozen=True)
class BlankLLMConfig:
    vocab_size: int = BYTE_VOCAB_SIZE
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    n_kv_heads: int = 2
    d_ff: int = 352
    max_seq_len: int = 512
    rope_theta: float = 10000.0
    dropout: float = 0.0
    tie_embeddings: bool = True
    init_std: float = 0.02

    def validate(self) -> "BlankLLMConfig":
        if self.vocab_size < BYTE_VOCAB_SIZE:
            raise BlankLLMConfigError(
                f"vocab_size must be >= {BYTE_VOCAB_SIZE} for the UTF-8 byte tokenizer"
            )
        for name in (
            "d_model",
            "n_layers",
            "n_heads",
            "n_kv_heads",
            "d_ff",
            "max_seq_len",
        ):
            if getattr(self, name) <= 0:
                raise BlankLLMConfigError(f"{name} must be positive")
        if self.d_model % self.n_heads:
            raise BlankLLMConfigError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads:
            raise BlankLLMConfigError("n_heads must be divisible by n_kv_heads")
        head_dim = self.d_model // self.n_heads
        if head_dim % 2:
            raise BlankLLMConfigError(
                "attention head dimension must be even for rotary embeddings"
            )
        if not (0.0 <= self.dropout < 1.0):
            raise BlankLLMConfigError("dropout must be in [0, 1)")
        if self.rope_theta <= 0:
            raise BlankLLMConfigError("rope_theta must be positive")
        if self.init_std <= 0:
            raise BlankLLMConfigError("init_std must be positive")
        return self

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BlankLLMConfig":
        return cls(**payload).validate()

    @classmethod
    def smoke(cls) -> "BlankLLMConfig":
        return cls(
            d_model=64,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            d_ff=176,
            max_seq_len=128,
        ).validate()
