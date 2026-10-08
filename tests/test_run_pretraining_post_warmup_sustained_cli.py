from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_runner_module():
    path = ROOT / "scripts" / "run_pretraining_post_warmup_sustained_l004.py"
    spec = importlib.util.spec_from_file_location(
        "run_pretraining_post_warmup_sustained_l004", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sustained_runner_defaults_to_v18(monkeypatch):
    module = _load_runner_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_pretraining_post_warmup_sustained_l004.py",
            "--external-root",
            r"T:\external",
            "--validate-only",
        ],
    )

    args = module.parse_args()

    assert Path(args.authorization).name == "pretraining-tranche-l004-v18.json"
    assert args.validate_only is True
