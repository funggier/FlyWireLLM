from __future__ import annotations

import json
import tomllib
from pathlib import Path

from flywire_llm import __version__
from flywire_llm.config import BlankLLMConfig


ROOT = Path(__file__).resolve().parents[1]


def test_repository_uses_mit_license_consistently():
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert license_text.startswith("MIT License\n")
    pyproject = tomllib.loads(
        (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    assert pyproject["project"]["license"]["text"] == "MIT"


def test_tracked_model_profiles_are_valid_and_scalable():
    smoke = BlankLLMConfig.from_dict(
        json.loads((ROOT / "configs" / "smoke.json").read_text(encoding="utf-8"))
    )
    small = BlankLLMConfig.from_dict(
        json.loads((ROOT / "configs" / "small.json").read_text(encoding="utf-8"))
    )
    assert smoke.vocab_size == 260
    assert small.vocab_size == 260
    assert small.d_model > smoke.d_model
    assert small.n_layers > smoke.n_layers
    assert small.max_seq_len > smoke.max_seq_len


def test_package_version_matches_release_metadata():
    pyproject = tomllib.loads(
        (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    assert pyproject["project"]["version"] == __version__ == "0.2.0"
