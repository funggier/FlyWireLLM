from __future__ import annotations

import json
from pathlib import Path

import pytest

import flywire_llm.pretraining_execution as execution
from flywire_llm.pretraining_execution import (
    PretrainingExecutionError,
    load_pretraining_execution_contract,
    verify_base50m_tokenizer,
)


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_TOKENIZER_SHA = (
    "998bc75f058e554d4f50b6cbc77c52898e3446e516c3b25a126b99b113513818"
)


class _FakeTokenizer:
    vocab_size = 32_000


def test_l004_execution_contract_pins_l003_and_base50m(monkeypatch, tmp_path):
    fake_model = tmp_path / "tokenizer.model"
    fake_model.write_bytes(b"test-only")
    monkeypatch.setattr(
        execution,
        "resolve_external_storage_uri",
        lambda *args, **kwargs: fake_model,
    )
    monkeypatch.setattr(
        execution,
        "verify_base50m_tokenizer",
        lambda *args, **kwargs: _FakeTokenizer(),
    )

    contract = load_pretraining_execution_contract(
        ROOT / "configs" / "pretraining-execution-l004.json",
        repo_root=ROOT,
        external_root=tmp_path,
    )
    assert contract.profile == "base-50m-v1"
    assert contract.checkpoint_lane == "research_only"
    assert contract.model_config.exact_parameter_count == 50_213_376
    assert contract.tokenizer_sha256 == EXPECTED_TOKENIZER_SHA
    assert contract.tokenizer_vocab_size == 32_000
    assert contract.primary_token_budget == 500_000_000
    assert contract.global_tokens_per_update == 65_536
    assert contract.primary_seed == 1234
    assert contract.data_runtime_sha256 == (
        "2d99c61c84fbfffb0ba5f23434bf6e8b92a92b3f588d7716238c0ebc76ee424b"
    )
    assert contract.source_manifest_sha256 == (
        "5304c4cc6b302a06650d090d6731f5b0e610fd99ec257c3fd9cc671cf62f79aa"
    )
    assert contract.global_manifest_report_sha256 == (
        "30a792d9fd26d1d4a6a1760aae7a43ae293778926be9044ebda632ec7c5054d5"
    )
    assert contract.decisions_sha256 == (
        "018aa08f347ebfd747f3aa11cada563bbd27918b469c7c5f11abe0289df7770a"
    )


def test_l004_execution_contract_rejects_model_hash_tamper(
    monkeypatch, tmp_path
):
    fake_model = tmp_path / "tokenizer.model"
    fake_model.write_bytes(b"test-only")
    monkeypatch.setattr(
        execution,
        "resolve_external_storage_uri",
        lambda *args, **kwargs: fake_model,
    )
    monkeypatch.setattr(
        execution,
        "verify_base50m_tokenizer",
        lambda *args, **kwargs: _FakeTokenizer(),
    )
    payload = json.loads(
        (ROOT / "configs" / "pretraining-execution-l004.json").read_text(
            encoding="utf-8"
        )
    )
    payload["model_config"]["sha256"] = "0" * 64
    path = tmp_path / "tampered.json"
    path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(
        PretrainingExecutionError,
        match="model_config SHA-256 mismatch",
    ):
        load_pretraining_execution_contract(
            path,
            repo_root=ROOT,
            external_root=tmp_path,
        )


def test_verified_tokenizer_rejects_hash_mismatch(tmp_path):
    path = tmp_path / "tokenizer.model"
    path.write_bytes(b"not-the-frozen-tokenizer")
    with pytest.raises(
        PretrainingExecutionError,
        match="tokenizer SHA-256 mismatch",
    ):
        verify_base50m_tokenizer(
            path,
            expected_sha256=EXPECTED_TOKENIZER_SHA,
        )
