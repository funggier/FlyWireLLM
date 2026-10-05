from __future__ import annotations

import math
from dataclasses import dataclass


class TrainingPlanError(ValueError):
    pass


@dataclass(frozen=True)
class TokenBudgetPlan:
    target_tokens: int
    global_tokens_per_update: int
    optimizer_steps: int
    warmup_steps: int
    scheduled_tokens: int
    overshoot_tokens: int
    warmup_fraction: float
    peak_learning_rate: float
    min_learning_rate: float


def build_token_budget_plan(
    target_tokens: int,
    *,
    global_tokens_per_update: int = 65_536,
    warmup_fraction: float = 0.02,
    peak_learning_rate: float = 6e-4,
    min_learning_rate: float = 6e-5,
) -> TokenBudgetPlan:
    if target_tokens <= 0:
        raise TrainingPlanError("target_tokens must be positive")
    if global_tokens_per_update <= 0:
        raise TrainingPlanError("global_tokens_per_update must be positive")
    if not 0.0 < warmup_fraction < 1.0:
        raise TrainingPlanError("warmup_fraction must be between 0 and 1")
    if not 0.0 < min_learning_rate <= peak_learning_rate:
        raise TrainingPlanError(
            "learning rates must satisfy 0 < min <= peak"
        )

    optimizer_steps = math.ceil(target_tokens / global_tokens_per_update)
    warmup_steps = max(1, math.ceil(optimizer_steps * warmup_fraction))
    scheduled_tokens = optimizer_steps * global_tokens_per_update

    return TokenBudgetPlan(
        target_tokens=int(target_tokens),
        global_tokens_per_update=int(global_tokens_per_update),
        optimizer_steps=optimizer_steps,
        warmup_steps=warmup_steps,
        scheduled_tokens=scheduled_tokens,
        overshoot_tokens=scheduled_tokens - target_tokens,
        warmup_fraction=float(warmup_fraction),
        peak_learning_rate=float(peak_learning_rate),
        min_learning_rate=float(min_learning_rate),
    )
