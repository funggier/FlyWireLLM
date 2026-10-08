from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_comparer_module():
    path = ROOT / "scripts" / "compare_validation_step425_l004.py"
    spec = importlib.util.spec_from_file_location(
        "compare_validation_step425_l004", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_step425_comparer_defaults_to_v16(monkeypatch):
    module = _load_comparer_module()
    monkeypatch.setattr(sys, "argv", ["compare_validation_step425_l004.py"])

    args = module.parse_args()

    assert Path(args.authorization).name == "pretraining-tranche-l004-v16.json"
    assert Path(args.step297).name == "validation-step297-v1.json"
    assert Path(args.step425).name == "validation-step425-v1.json"
