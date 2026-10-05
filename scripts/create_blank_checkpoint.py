from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from flywire_llm.checkpoint import save_checkpoint
from flywire_llm.config import BlankLLMConfig
from flywire_llm.model import BlankCausalLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a random-initialized FlyWireLLM blank checkpoint."
    )
    parser.add_argument("--config", default="configs/smoke.json")
    parser.add_argument("--output", default="checkpoints/blank-smoke.pt")
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(Path(args.config).read_text(encoding="utf-8"))
    config = BlankLLMConfig.from_dict(payload)

    torch.manual_seed(args.seed)
    model = BlankCausalLM(config)
    save_checkpoint(
        args.output,
        model,
        training_step=0,
        initialization_seed=args.seed,
    )

    print(f"output={Path(args.output)}")
    print(f"seed={args.seed}")
    print(f"parameters={model.parameter_count}")
    print("pretrained=false")
    print("external_pretrained_weights=false")
    print("knowledge_in_weights_claimed=false")
    print("untrained=true")


if __name__ == "__main__":
    main()
