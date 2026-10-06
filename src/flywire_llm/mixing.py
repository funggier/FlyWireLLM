from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


class MixtureContractError(ValueError):
    pass


@dataclass(frozen=True)
class MixtureContract:
    primary_target_tokens: int
    thai_minimum_tokens: int
    english_minimum_tokens: int
    technical_base_tokens: int
    flywire_target_tokens: int
    flywire_maximum_tokens: int


@dataclass(frozen=True)
class MixtureReadiness:
    ready: bool
    total_available_tokens: int
    thai_gap: int
    english_gap: int
    technical_required_tokens: int
    technical_gap: int
    flywire_used_tokens: int
    total_gap: int


_REQUIRED_CATEGORIES = {
    "general_thai",
    "general_english",
    "technical_scientific_code",
    "flywire_domain",
}


def load_mixture_contract(path: str | Path) -> MixtureContract:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise MixtureContractError("mixture contract must be an object")
    if payload.get("schema_version") != 1:
        raise MixtureContractError("unsupported mixture contract schema")
    if payload.get("stage") != "L003":
        raise MixtureContractError("mixture contract stage must be L003")
    if payload.get("primary_target_tokens") != 500_000_000:
        raise MixtureContractError("primary target must remain 500M")

    categories = payload.get("categories")
    if not isinstance(categories, dict) or set(categories) != _REQUIRED_CATEGORIES:
        raise MixtureContractError("mixture category set changed")

    thai = categories["general_thai"]
    english = categories["general_english"]
    technical = categories["technical_scientific_code"]
    flywire = categories["flywire_domain"]

    expected = {
        "thai_target": thai.get("target_tokens"),
        "thai_min": thai.get("minimum_tokens"),
        "english_target": english.get("target_tokens"),
        "english_min": english.get("minimum_tokens"),
        "technical_base": technical.get("base_target_tokens"),
        "flywire_target": flywire.get("target_tokens"),
        "flywire_max": flywire.get("maximum_tokens"),
    }
    if expected != {
        "thai_target": 200_000_000,
        "thai_min": 200_000_000,
        "english_target": 200_000_000,
        "english_min": 200_000_000,
        "technical_base": 50_000_000,
        "flywire_target": 50_000_000,
        "flywire_max": 50_000_000,
    }:
        raise MixtureContractError("L003 mixture targets changed")

    fallback = payload.get("fallback_policy")
    if not isinstance(fallback, dict):
        raise MixtureContractError("fallback_policy must be an object")
    if fallback.get("unused_flywire_domain_moves_to") != (
        "technical_scientific_code"
    ):
        raise MixtureContractError(
            "unused FlyWire share must move to technical/scientific/code"
        )
    if (
        fallback.get("general_language_overage_cannot_substitute_other_categories")
        is not True
    ):
        raise MixtureContractError(
            "general-language overage substitution must remain forbidden"
        )
    if fallback.get("category_assignment_must_be_single_primary_category") is not True:
        raise MixtureContractError(
            "records must use one primary mixture category"
        )

    authorization = payload.get("authorization")
    if not isinstance(authorization, dict):
        raise MixtureContractError("authorization must be an object")
    required_flags = {
        "total_target_must_be_met": True,
        "thai_minimum_must_be_met": True,
        "english_minimum_must_be_met": True,
    }
    for key, expected_value in required_flags.items():
        if authorization.get(key) is not expected_value:
            raise MixtureContractError(f"{key} must remain true")
    if authorization.get("technical_plus_flywire_must_total") != 100_000_000:
        raise MixtureContractError(
            "technical + FlyWire allocation must remain 100M"
        )

    return MixtureContract(
        primary_target_tokens=500_000_000,
        thai_minimum_tokens=200_000_000,
        english_minimum_tokens=200_000_000,
        technical_base_tokens=50_000_000,
        flywire_target_tokens=50_000_000,
        flywire_maximum_tokens=50_000_000,
    )


def evaluate_mixture_readiness(
    contract: MixtureContract,
    *,
    available_tokens: dict[str, int],
) -> MixtureReadiness:
    if set(available_tokens) != _REQUIRED_CATEGORIES:
        raise MixtureContractError(
            "available token accounting must contain exactly the four "
            "L003 mixture categories"
        )
    for category, value in available_tokens.items():
        if not isinstance(value, int) or value < 0:
            raise MixtureContractError(
                f"{category} token count must be a non-negative integer"
            )

    thai = available_tokens["general_thai"]
    english = available_tokens["general_english"]
    technical = available_tokens["technical_scientific_code"]
    flywire_available = available_tokens["flywire_domain"]

    flywire_used = min(
        flywire_available,
        contract.flywire_maximum_tokens,
    )
    technical_required = (
        contract.technical_base_tokens
        + contract.flywire_target_tokens
        - flywire_used
    )

    thai_gap = max(0, contract.thai_minimum_tokens - thai)
    english_gap = max(0, contract.english_minimum_tokens - english)
    technical_gap = max(0, technical_required - technical)
    total_available = sum(available_tokens.values())
    total_gap = max(0, contract.primary_target_tokens - total_available)

    ready = (
        thai_gap == 0
        and english_gap == 0
        and technical_gap == 0
        and total_gap == 0
    )
    return MixtureReadiness(
        ready=ready,
        total_available_tokens=total_available,
        thai_gap=thai_gap,
        english_gap=english_gap,
        technical_required_tokens=technical_required,
        technical_gap=technical_gap,
        flywire_used_tokens=flywire_used,
        total_gap=total_gap,
    )
