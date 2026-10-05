from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .config import BlankLLMConfig


class BlankLLMError(ValueError):
    pass


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * scale * self.weight


def _apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    even = x[..., 0::2]
    odd = x[..., 1::2]
    rotated_even = even * cos - odd * sin
    rotated_odd = even * sin + odd * cos
    return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: BlankLLMConfig) -> None:
        super().__init__()
        config.validate()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.d_model // config.n_heads
        self.kv_repeat = config.n_heads // config.n_kv_heads
        self.scale = self.head_dim**-0.5
        self.dropout = float(config.dropout)
        self.rope_theta = float(config.rope_theta)

        self.q_proj = nn.Linear(
            config.d_model,
            config.n_heads * self.head_dim,
            bias=False,
        )
        self.k_proj = nn.Linear(
            config.d_model,
            config.n_kv_heads * self.head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            config.d_model,
            config.n_kv_heads * self.head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)

    def _rope(
        self,
        length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = 1.0 / (
            self.rope_theta
            ** (
                torch.arange(
                    0,
                    self.head_dim,
                    2,
                    device=device,
                    dtype=torch.float32,
                )
                / self.head_dim
            )
        )
        positions = torch.arange(length, device=device, dtype=torch.float32)
        frequencies = torch.outer(positions, inv_freq)
        cos = frequencies.cos().to(dtype=dtype)[None, None, :, :]
        sin = frequencies.sin().to(dtype=dtype)[None, None, :, :]
        return cos, sin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        q = self.q_proj(x).view(
            batch,
            length,
            self.n_heads,
            self.head_dim,
        ).transpose(1, 2)
        k = self.k_proj(x).view(
            batch,
            length,
            self.n_kv_heads,
            self.head_dim,
        ).transpose(1, 2)
        v = self.v_proj(x).view(
            batch,
            length,
            self.n_kv_heads,
            self.head_dim,
        ).transpose(1, 2)

        cos, sin = self._rope(length, x.device, q.dtype)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        if self.kv_repeat != 1:
            k = k.repeat_interleave(self.kv_repeat, dim=1)
            v = v.repeat_interleave(self.kv_repeat, dim=1)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        future_mask = torch.triu(
            torch.ones(
                length,
                length,
                dtype=torch.bool,
                device=x.device,
            ),
            diagonal=1,
        )
        scores = scores.masked_fill(future_mask[None, None, :, :], float("-inf"))
        probabilities = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        probabilities = F.dropout(
            probabilities,
            p=self.dropout,
            training=self.training,
        )
        output = torch.matmul(probabilities, v)
        output = output.transpose(1, 2).contiguous().view(batch, length, -1)
        return self.o_proj(output)


class SwiGLU(nn.Module):
    def __init__(self, config: BlankLLMConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = float(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.gate_proj(x)) * self.up_proj(x)
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        return self.down_proj(hidden)


class TransformerBlock(nn.Module):
    def __init__(self, config: BlankLLMConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.d_model)
        self.ffn = SwiGLU(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


@dataclass
class CausalLMOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None


class BlankCausalLM(nn.Module):
    model_type = "flywire_blank_causal_lm"

    def __init__(self, config: BlankLLMConfig) -> None:
        super().__init__()
        self.config = config.validate()
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.n_layers)
        )
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.apply(self._init_weights)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=self.config.init_std,
            )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        targets: torch.Tensor | None = None,
    ) -> CausalLMOutput:
        if input_ids.ndim != 2:
            raise BlankLLMError("input_ids must have shape [batch, sequence]")
        if input_ids.dtype != torch.long:
            raise BlankLLMError("input_ids must use torch.long dtype")
        if input_ids.shape[1] > self.config.max_seq_len:
            raise BlankLLMError(
                f"sequence length {input_ids.shape[1]} exceeds "
                f"max_seq_len={self.config.max_seq_len}"
            )
        if bool(torch.any(input_ids < 0)) or bool(
            torch.any(input_ids >= self.config.vocab_size)
        ):
            raise BlankLLMError("input_ids contain token ids outside vocabulary")

        hidden = self.token_embedding(input_ids)
        for block in self.blocks:
            hidden = block(hidden)
        logits = self.lm_head(self.norm(hidden))

        loss = None
        if targets is not None:
            if targets.shape != input_ids.shape:
                raise BlankLLMError(
                    "targets must have the same shape as input_ids"
                )
            if targets.dtype != torch.long:
                raise BlankLLMError("targets must use torch.long dtype")
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                ignore_index=-100,
            )
        return CausalLMOutput(logits=logits, loss=loss)
