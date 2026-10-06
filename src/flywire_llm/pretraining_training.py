from __future__ import annotations

import json
import math
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch

from .model import BlankCausalLM
from .planning import TokenBudgetPlan, build_token_budget_plan


class PretrainingTrainingError(ValueError):
    pass


@dataclass(frozen=True)
class PretrainingSchedule:
    target_tokens: int
    optimizer_steps: int
    warmup_steps: int
    global_tokens_per_update: int
    peak_learning_rate: float
    min_learning_rate: float


@dataclass(frozen=True)
class PretrainingUpdateResult:
    mean_loss: float
    gradient_norm: float
    learning_rate: float
    microbatches: int
    supervised_tokens: int


def load_training_configuration(
    path: str | Path,
) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PretrainingTrainingError(
            "training configuration must be a JSON object"
        )
    return payload


def build_primary_schedule(
    training: dict[str, object],
) -> PretrainingSchedule:
    if training.get("profile") != "base-50m-v1":
        raise PretrainingTrainingError("training profile changed")
    target_tokens = training.get("primary_token_budget")
    global_tokens = training.get("global_tokens_per_update")
    schedule = training.get("schedule")
    if not isinstance(target_tokens, int) or target_tokens <= 0:
        raise PretrainingTrainingError(
            "primary_token_budget must be positive"
        )
    if not isinstance(global_tokens, int) or global_tokens <= 0:
        raise PretrainingTrainingError(
            "global_tokens_per_update must be positive"
        )
    if not isinstance(schedule, dict):
        raise PretrainingTrainingError("schedule must be an object")

    plan: TokenBudgetPlan = build_token_budget_plan(
        target_tokens,
        global_tokens_per_update=global_tokens,
        warmup_fraction=float(schedule["warmup_fraction"]),
        peak_learning_rate=float(schedule["peak_learning_rate"]),
        min_learning_rate=float(schedule["min_learning_rate"]),
    )
    if schedule.get("decay") != "cosine":
        raise PretrainingTrainingError(
            "only cosine decay is qualified"
        )
    return PretrainingSchedule(
        target_tokens=plan.target_tokens,
        optimizer_steps=plan.optimizer_steps,
        warmup_steps=plan.warmup_steps,
        global_tokens_per_update=plan.global_tokens_per_update,
        peak_learning_rate=plan.peak_learning_rate,
        min_learning_rate=plan.min_learning_rate,
    )


def learning_rate_for_update(
    schedule: PretrainingSchedule,
    update_number: int,
) -> float:
    if update_number < 1 or update_number > schedule.optimizer_steps:
        raise PretrainingTrainingError(
            "update_number must be within the planned optimizer schedule"
        )
    if update_number <= schedule.warmup_steps:
        return (
            schedule.peak_learning_rate
            * update_number
            / schedule.warmup_steps
        )

    decay_updates = schedule.optimizer_steps - schedule.warmup_steps
    if decay_updates <= 0:
        return schedule.min_learning_rate
    progress = (
        update_number - schedule.warmup_steps
    ) / decay_updates
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return schedule.min_learning_rate + (
        schedule.peak_learning_rate - schedule.min_learning_rate
    ) * cosine


def build_adamw(
    model: BlankCausalLM,
    training: dict[str, object],
) -> torch.optim.AdamW:
    optimizer = training.get("optimizer")
    if not isinstance(optimizer, dict):
        raise PretrainingTrainingError(
            "optimizer configuration must be an object"
        )
    if optimizer.get("name") != "AdamW":
        raise PretrainingTrainingError(
            "only AdamW is qualified for Base-50M-v1"
        )
    return torch.optim.AdamW(
        model.parameters(),
        lr=0.0,
        betas=(
            float(optimizer["beta1"]),
            float(optimizer["beta2"]),
        ),
        eps=float(optimizer["eps"]),
        weight_decay=float(optimizer["weight_decay"]),
    )


def gradient_clip_norm(
    training: dict[str, object],
) -> float:
    optimizer = training.get("optimizer")
    if not isinstance(optimizer, dict):
        raise PretrainingTrainingError(
            "optimizer configuration must be an object"
        )
    value = float(optimizer["gradient_clip_norm"])
    if value <= 0:
        raise PretrainingTrainingError(
            "gradient_clip_norm must be positive"
        )
    return value


def pretraining_update(
    model: BlankCausalLM,
    optimizer: torch.optim.Optimizer,
    microbatches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    *,
    learning_rate: float,
    clip_norm: float,
    autocast_dtype: torch.dtype | None = None,
) -> PretrainingUpdateResult:
    batches = list(microbatches)
    if not batches:
        raise PretrainingTrainingError(
            "pretraining update requires at least one microbatch"
        )
    if learning_rate <= 0:
        raise PretrainingTrainingError(
            "learning_rate must be positive"
        )
    if clip_norm <= 0:
        raise PretrainingTrainingError("clip_norm must be positive")

    for group in optimizer.param_groups:
        group["lr"] = float(learning_rate)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    supervised_counts = [
        int(torch.count_nonzero(targets != -100).item())
        for _, targets in batches
    ]
    if any(count <= 0 for count in supervised_counts):
        raise PretrainingTrainingError(
            "every microbatch must contain supervised targets"
        )
    total_supervised = sum(supervised_counts)
    weighted_loss_sum = 0.0
    for (input_ids, targets), supervised_count in zip(
        batches,
        supervised_counts,
    ):
        context = (
            nullcontext()
            if autocast_dtype is None
            else torch.autocast(
                device_type=input_ids.device.type,
                dtype=autocast_dtype,
            )
        )
        with context:
            output = model(input_ids, targets=targets)
            loss = output.loss
        if loss is None:
            raise RuntimeError(
                "pretraining forward pass did not return a loss"
            )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("pretraining loss is not finite")
        weight = supervised_count / total_supervised
        weighted_loss_sum += (
            float(loss.detach().cpu()) * supervised_count
        )
        (loss * weight).backward()

    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=clip_norm,
        error_if_nonfinite=True,
    )
    optimizer.step()
    return PretrainingUpdateResult(
        mean_loss=weighted_loss_sum / total_supervised,
        gradient_norm=float(grad_norm.detach().cpu()),
        learning_rate=float(learning_rate),
        microbatches=len(batches),
        supervised_tokens=total_supervised,
    )
