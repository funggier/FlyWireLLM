from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_comparer_module():
    path = ROOT / "scripts" / "compare_validation_step1193_l004.py"
    spec = importlib.util.spec_from_file_location(
        "compare_validation_step1193_l004", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_step1193_comparer_defaults_to_v18(monkeypatch):
    module = _load_comparer_module()
    monkeypatch.setattr(sys, "argv", ["compare_validation_step1193_l004.py"])

    args = module.parse_args()

    assert Path(args.authorization).name == "pretraining-tranche-l004-v18.json"
    assert Path(args.step681).name == "validation-step681-v1.json"
    assert Path(args.step1193).name == "validation-step1193-v1.json"
