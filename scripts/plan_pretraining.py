from __future__ import annotations

import argparse
import json
from pathlib import Path

from flywire_llm.planning import build_token_budget_plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate Base-50M optimizer schedules for token budgets."
    )
    parser.add_argument(
        "--config",
        default="configs/training-base-50m-v1.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(Path(args.config).read_text(encoding="utf-8"))
    schedule = payload["schedule"]
    for budget in payload["token_budgets"]:
        plan = build_token_budget_plan(
            budget,
            global_tokens_per_update=payload["global_tokens_per_update"],
            warmup_fraction=schedule["warmup_fraction"],
            peak_learning_rate=schedule["peak_learning_rate"],
            min_learning_rate=schedule["min_learning_rate"],
        )
        print(
            f"target_tokens={plan.target_tokens} "
            f"steps={plan.optimizer_steps} "
            f"warmup_steps={plan.warmup_steps} "
            f"scheduled_tokens={plan.scheduled_tokens} "
            f"overshoot_tokens={plan.overshoot_tokens}"
        )


if __name__ == "__main__":
    main()
