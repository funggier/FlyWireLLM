from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from flywire_llm.config import BlankLLMConfig
from flywire_llm.model import BlankCausalLM
from flywire_llm.pretraining_training import (
    build_adamw,
    build_primary_schedule,
    gradient_clip_norm,
    learning_rate_for_update,
    load_training_configuration,
    pretraining_update,
)


ROOT = Path(__file__).resolve().parents[1]


def _training():
    return load_training_configuration(
        ROOT / "configs" / "training-base-50m-v1.json"
    )


def test_primary_schedule_matches_frozen_500m_plan():
    schedule = build_primary_schedule(_training())
    assert schedule.target_tokens == 500_000_000
    assert schedule.global_tokens_per_update == 65_536
    assert schedule.optimizer_steps == 7_630
    assert schedule.warmup_steps == 153
    assert learning_rate_for_update(schedule, 1) == pytest.approx(
        0.0006 / 153
    )
    assert learning_rate_for_update(schedule, 153) == pytest.approx(0.0006)
    assert learning_rate_for_update(schedule, 7_630) == pytest.approx(0.00006)


def test_adamw_and_gradient_clip_match_training_contract():
    model = BlankCausalLM(BlankLLMConfig.smoke())
    optimizer = build_adamw(model, _training())
    assert optimizer.param_groups[0]["lr"] == 0.0
    assert optimizer.param_groups[0]["betas"] == (0.9, 0.95)
    assert optimizer.param_groups[0]["eps"] == 1e-8
    assert optimizer.param_groups[0]["weight_decay"] == 0.1
    assert gradient_clip_norm(_training()) == 1.0


def test_pretraining_update_supports_cpu_bf16_autocast():
    torch.manual_seed(29)
    model = BlankCausalLM(BlankLLMConfig.smoke())
    optimizer = build_adamw(model, _training())
    ids = torch.arange(4, 20, dtype=torch.long).reshape(1, 16)
    result = pretraining_update(
        model,
        optimizer,
        [(ids, ids.roll(-1, dims=1))],
        learning_rate=1e-4,
        clip_norm=1.0,
        autocast_dtype=torch.bfloat16,
    )
    assert result.mean_loss > 0
    assert result.gradient_norm > 0


def test_pretraining_update_accumulates_microbatches():
    torch.manual_seed(31)
    model = BlankCausalLM(BlankLLMConfig.smoke())
    optimizer = build_adamw(model, _training())
    batches = []
    for offset in (0, 1):
        ids = (
            torch.arange(0, 16, dtype=torch.long).reshape(1, 16)
            + offset
            + 4
        )
        batches.append((ids, ids.roll(-1, dims=1)))
    result = pretraining_update(
        model,
        optimizer,
        batches,
        learning_rate=1e-4,
        clip_norm=1.0,
    )
    assert result.microbatches == 2
    assert result.supervised_tokens == 32
    assert result.learning_rate == 1e-4
    assert result.mean_loss > 0
    assert result.gradient_norm > 0
    assert optimizer.param_groups[0]["lr"] == 1e-4


def test_pretraining_update_weights_gradient_by_supervised_tokens():
    torch.manual_seed(77)
    base = BlankCausalLM(BlankLLMConfig.smoke())
    accumulated = deepcopy(base)
    combined = deepcopy(base)
    accumulated_optimizer = build_adamw(accumulated, _training())
    combined_optimizer = build_adamw(combined, _training())

    ids_a = torch.arange(4, 12, dtype=torch.long).reshape(1, 8)
    ids_b = torch.arange(12, 20, dtype=torch.long).reshape(1, 8)
    targets_a = ids_a.roll(-1, dims=1)
    targets_b = ids_b.roll(-1, dims=1)
    targets_b[:, 4:] = -100

    result = pretraining_update(
        accumulated,
        accumulated_optimizer,
        [(ids_a, targets_a), (ids_b, targets_b)],
        learning_rate=1e-4,
        clip_norm=1.0,
    )
    assert result.supervised_tokens == 12

    combined_optimizer.zero_grad(set_to_none=True)
    combined_output = combined(
        torch.cat([ids_a, ids_b], dim=0),
        targets=torch.cat([targets_a, targets_b], dim=0),
    )
    assert combined_output.loss is not None
    combined_output.loss.backward()
    torch.nn.utils.clip_grad_norm_(
        combined.parameters(),
        max_norm=1.0,
        error_if_nonfinite=True,
    )
    for group in combined_optimizer.param_groups:
        group["lr"] = 1e-4
    combined_optimizer.step()

    assert result.mean_loss == pytest.approx(
        float(combined_output.loss.detach().cpu()),
        rel=2e-7,
        abs=1e-6,
    )
    for name, value in combined.state_dict().items():
        torch.testing.assert_close(
            accumulated.state_dict()[name],
            value,
            rtol=1e-6,
            atol=1e-6,
        )
