from __future__ import annotations

import json
from pathlib import Path

from flywire_llm.config import BlankLLMConfig
from flywire_llm.model import BlankCausalLM
from flywire_llm.planning import build_token_budget_plan


ROOT = Path(__file__).resolve().parents[1]


def test_base_50m_v1_config_is_exactly_frozen():
    payload = json.loads(
        (ROOT / "configs" / "base-50m-v1.json").read_text(encoding="utf-8")
    )
    config = BlankLLMConfig.from_dict(payload)
    assert config.vocab_size == 32_000
    assert config.d_model == 512
    assert config.n_layers == 12
    assert config.n_heads == 8
    assert config.n_kv_heads == 2
    assert config.d_ff == 1_408
    assert config.max_seq_len == 1_024
    assert config.exact_parameter_count == 50_213_376


def test_parameter_count_formula_matches_real_smoke_model():
    config = BlankLLMConfig.smoke()
    model = BlankCausalLM(config)
    assert config.exact_parameter_count == model.parameter_count


def test_l002_token_budget_schedules_are_exact():
    expected = {
        250_000_000: (3_815, 77, 250_019_840, 19_840),
        500_000_000: (7_630, 153, 500_039_680, 39_680),
        1_000_000_000: (15_259, 306, 1_000_013_824, 13_824),
    }
    for tokens, values in expected.items():
        plan = build_token_budget_plan(tokens)
        assert (
            plan.optimizer_steps,
            plan.warmup_steps,
            plan.scheduled_tokens,
            plan.overshoot_tokens,
        ) == values


def test_training_recipe_primary_target_is_500m():
    payload = json.loads(
        (ROOT / "configs" / "training-base-50m-v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["primary_token_budget"] == 500_000_000
    assert payload["global_tokens_per_update"] == 65_536
    assert payload["optimizer"] == {
        "name": "AdamW",
        "beta1": 0.9,
        "beta2": 0.95,
        "eps": 1e-8,
        "weight_decay": 0.1,
        "gradient_clip_norm": 1.0,
    }
    assert payload["schedule"]["peak_learning_rate"] == 6e-4
    assert payload["schedule"]["warmup_fraction"] == 0.02
    assert payload["schedule"]["min_learning_rate"] == 6e-5
