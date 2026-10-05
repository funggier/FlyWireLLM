from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from flywire_llm.checkpoint import save_checkpoint
from flywire_llm.config import BlankLLMConfig
from flywire_llm.corpus import load_partition
from flywire_llm.model import BlankCausalLM
from flywire_llm.training import causal_batch_from_text, optimizer_step


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one deterministic FlyWireLLM engineering smoke step."
    )
    parser.add_argument("--config", default="configs/smoke.json")
    parser.add_argument("--corpus", default="examples/corpus/sample.jsonl")
    parser.add_argument("--output", default="checkpoints/smoke-step1.pt")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = BlankLLMConfig.from_dict(
        json.loads(Path(args.config).read_text(encoding="utf-8"))
    )
    train_records = load_partition(args.corpus, "train")
    if not train_records:
        raise RuntimeError("smoke corpus contains no training records")
    text = "\n".join(record.text for record in train_records)

    torch.manual_seed(args.seed)
    model = BlankCausalLM(config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
    )
    inputs, targets = causal_batch_from_text(
        text,
        max_seq_len=config.max_seq_len,
    )

    with torch.no_grad():
        before = model(inputs, targets=targets).loss
    loss = optimizer_step(model, optimizer, inputs, targets)
    with torch.no_grad():
        after = model(inputs, targets=targets).loss

    save_checkpoint(
        args.output,
        model,
        training_step=1,
        initialization_seed=args.seed,
        optimizer=optimizer,
    )

    print(f"parameters={model.parameter_count}")
    print(f"tokens={inputs.numel() + 1}")
    print(f"loss_before={float(before):.6f}")
    print(f"loss_step={loss:.6f}")
    print(f"loss_after={float(after):.6f}")
    print("quality_claim=false")
    print("training_step=1")
    print(f"output={Path(args.output)}")


if __name__ == "__main__":
    main()
