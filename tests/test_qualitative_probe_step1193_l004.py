from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def _load_probe_module():
    path = ROOT / "scripts" / "run_qualitative_probe_step1193_l004.py"
    spec = importlib.util.spec_from_file_location(
        "run_qualitative_probe_step1193_l004", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_qualitative_probe_cli_defaults(monkeypatch, tmp_path):
    module = _load_probe_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_qualitative_probe_step1193_l004.py",
            "--external-root",
            str(tmp_path),
        ],
    )

    args = module.parse_args()

    assert Path(args.config).name == "qualitative-probe-l004-v2.json"
    assert args.models == ["step681", "step1193"]
    assert Path(args.output_json).name == (
        "qualitative-probe-step681-step1193-v2.json"
    )


def test_qualitative_probe_config_is_synthetic_and_balanced():
    module = _load_probe_module()

    payload = module.load_probe_config(
        ROOT / "configs" / "qualitative-probe-l004-v2.json"
    )

    assert payload["synthetic_prompts_only"] is True
    assert payload["source_partitions_read"] == []
    assert payload["final_holdout_touched"] is False
    assert len(payload["prompts"]) == 12


class _ToyConfig:
    max_seq_len = 16


class _ToyOutput:
    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits


class _ToyModel:
    config = _ToyConfig()

    def __call__(self, input_ids: torch.Tensor) -> _ToyOutput:
        vocab = 8
        logits = torch.full(
            (1, input_ids.shape[1], vocab),
            -10.0,
            dtype=torch.float32,
        )
        next_id = 5 if input_ids.shape[1] < 3 else 2
        logits[0, -1, next_id] = 10.0
        return _ToyOutput(logits)


def test_greedy_generation_is_deterministic_and_stops_on_eos():
    module = _load_probe_module()

    generated, eos_reached = module.greedy_generate_ids(
        _ToyModel(),
        [4, 4],
        eos_id=2,
        max_new_tokens=8,
    )

    assert generated == [5, 2]
    assert eos_reached is True
