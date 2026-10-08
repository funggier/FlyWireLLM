from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_verifier_module():
    path = ROOT / "scripts" / "verify_pretraining_post_warmup_accelerated_l004.py"
    spec = importlib.util.spec_from_file_location(
        "verify_pretraining_post_warmup_accelerated_l004", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_accelerated_verifier_defaults_to_v16(monkeypatch, tmp_path):
    module = _load_verifier_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_pretraining_post_warmup_accelerated_l004.py",
            "--external-root",
            str(tmp_path),
        ],
    )

    args = module.parse_args()

    assert Path(args.authorization).name == "pretraining-tranche-l004-v16.json"
