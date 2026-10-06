from __future__ import annotations

from pathlib import Path

import pytest

from flywire_llm.mixing import (
    MixtureContractError,
    evaluate_mixture_readiness,
    load_mixture_contract,
)


ROOT = Path(__file__).resolve().parents[1]


def _contract():
    return load_mixture_contract(ROOT / "configs" / "mixture-l003.json")


def test_l003_mixture_contract_freezes_bilingual_minima():
    contract = _contract()
    assert contract.primary_target_tokens == 500_000_000
    assert contract.thai_minimum_tokens == 200_000_000
    assert contract.english_minimum_tokens == 200_000_000
    assert contract.technical_base_tokens == 50_000_000
    assert contract.flywire_target_tokens == 50_000_000


def test_total_500m_is_not_enough_when_language_mix_is_wrong():
    result = evaluate_mixture_readiness(
        _contract(),
        available_tokens={
            "general_thai": 450_000_000,
            "general_english": 50_000_000,
            "technical_scientific_code": 0,
            "flywire_domain": 0,
        },
    )
    assert result.total_gap == 0
    assert result.english_gap == 150_000_000
    assert result.technical_gap == 100_000_000
    assert result.ready is False


def test_no_flywire_requires_100m_technical_fallback():
    result = evaluate_mixture_readiness(
        _contract(),
        available_tokens={
            "general_thai": 200_000_000,
            "general_english": 200_000_000,
            "technical_scientific_code": 100_000_000,
            "flywire_domain": 0,
        },
    )
    assert result.flywire_used_tokens == 0
    assert result.technical_required_tokens == 100_000_000
    assert result.technical_gap == 0
    assert result.ready is True


def test_flywire_share_reduces_but_never_exceeds_technical_fallback():
    result = evaluate_mixture_readiness(
        _contract(),
        available_tokens={
            "general_thai": 200_000_000,
            "general_english": 200_000_000,
            "technical_scientific_code": 50_000_000,
            "flywire_domain": 80_000_000,
        },
    )
    assert result.flywire_used_tokens == 50_000_000
    assert result.technical_required_tokens == 50_000_000
    assert result.ready is True


def test_missing_category_accounting_fails_closed():
    with pytest.raises(MixtureContractError, match="exactly the four"):
        evaluate_mixture_readiness(
            _contract(),
            available_tokens={
                "general_thai": 200_000_000,
                "general_english": 200_000_000,
                "technical_scientific_code": 100_000_000,
            },
        )
