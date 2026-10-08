from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_evaluator_module():
    path = ROOT / "scripts" / "evaluate_validation_l004.py"
    spec = importlib.util.spec_from_file_location("evaluate_validation_l004", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_validation_evaluator_accepts_step425(monkeypatch, tmp_path):
    module = _load_evaluator_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_validation_l004.py",
            "--external-root",
            str(tmp_path),
            "--model",
            "step425",
            "--output-json",
            str(tmp_path / "validation-step425.json"),
        ],
    )

    args = module.parse_args()

    assert args.model == "step425"
