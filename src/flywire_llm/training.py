from __future__ import annotations

import torch

from .model import BlankCausalLM, BlankLLMError
from .tokenizer import UTF8ByteTokenizer


def causal_batch_from_text(
    text: str,
    *,
    tokenizer: UTF8ByteTokenizer | None = None,
    max_seq_len: int | None = None,
    add_eos: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokenizer = tokenizer or UTF8ByteTokenizer()
    token_ids = tokenizer.encode(text, add_eos=add_eos)
    if len(token_ids) < 2:
        raise BlankLLMError("text must yield at least two tokens")
    if max_seq_len is not None:
        if max_seq_len < 1:
            raise BlankLLMError("max_seq_len must be positive")
        token_ids = token_ids[: max_seq_len + 1]
    input_ids = torch.tensor(
        token_ids[:-1],
        dtype=torch.long,
    ).unsqueeze(0)
    targets = torch.tensor(
        token_ids[1:],
        dtype=torch.long,
    ).unsqueeze(0)
    return input_ids, targets


def optimizer_step(
    model: BlankCausalLM,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(input_ids, targets=targets)
    if output.loss is None:
        raise RuntimeError("training forward pass did not return a loss")
    if not bool(torch.isfinite(output.loss)):
        raise RuntimeError("training loss is not finite")
    output.loss.backward()
    optimizer.step()
    return float(output.loss.detach().cpu())
