from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class CorpusInventoryError(ValueError):
    pass


_ALLOWED_CATEGORIES = {
    "general_thai",
    "general_english",
    "technical_scientific_code",
    "flywire_domain",
}
_ALLOWED_AUDIT_STATUS = {"approved", "conditional", "blocked"}
_ALLOWED_ACQUISITION_STATUS = {
    "available",
    "pilot_acquired",
    "not_acquired",
    "manual_or_gated",
}


@dataclass(frozen=True)
class CorpusSource:
    source_id: str
    name: str
    categories: tuple[str, ...]
    languages: tuple[str, ...]
    dataset_url: str
    license_name: str
    license_url: str
    license_audit_status: str
    acquisition_status: str
    source_grouping_key: str
    content_rights_notes: str
    pii_risk: str
    intended_role: str


@dataclass(frozen=True)
class CorpusInventory:
    target_tokens: int
    minimum_tokens: int
    stretch_tokens: int
    target_mix: dict[str, float]
    split: dict[str, float]
    sources: tuple[CorpusSource, ...]


def normalize_for_dedup(text: str) -> str:
    """Conservative duplicate-key normalization, not training-text rewriting."""
    if not isinstance(text, str):
        raise TypeError("text must be str")
    value = text.replace("\r\n", "\n").replace("\r", "\n")
    value = unicodedata.normalize("NFC", value)
    return " ".join(value.split())


def dedup_fingerprint(text: str) -> str:
    normalized = normalize_for_dedup(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def assign_partition(
    source_id: str,
    source_group_id: str,
    *,
    seed: str = "flywirellm-l002-split-v1",
) -> str:
    """Assign one source group to a frozen 99/0.5/0.5 partition."""
    if not source_id or not source_group_id:
        raise CorpusInventoryError(
            "source_id and source_group_id must be non-empty"
        )
    key = f"{seed}\0{source_id}\0{source_group_id}".encode("utf-8")
    bucket = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % 10_000
    if bucket < 9_900:
        return "train"
    if bucket < 9_950:
        return "validation"
    return "holdout"


def load_corpus_inventory(path: str | Path) -> CorpusInventory:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CorpusInventoryError("inventory must be a JSON object")
    if payload.get("schema_version") != 1:
        raise CorpusInventoryError("unsupported corpus inventory schema")
    if payload.get("stage") != "L002":
        raise CorpusInventoryError("corpus inventory stage must be L002")

    budgets = payload.get("token_budgets")
    if budgets != {
        "minimum": 250_000_000,
        "primary": 500_000_000,
        "stretch": 1_000_000_000,
    }:
        raise CorpusInventoryError("L002 token budgets changed")

    target_mix = payload.get("target_mix")
    if not isinstance(target_mix, dict):
        raise CorpusInventoryError("target_mix must be an object")
    if set(target_mix) != _ALLOWED_CATEGORIES:
        raise CorpusInventoryError("target_mix category set changed")
    if abs(sum(float(v) for v in target_mix.values()) - 1.0) > 1e-12:
        raise CorpusInventoryError("target_mix must sum to 1.0")
    if float(target_mix["flywire_domain"]) > 0.10:
        raise CorpusInventoryError(
            "flywire_domain target must not exceed 10 percent"
        )

    split = payload.get("split")
    if split != {
        "train": 0.99,
        "validation": 0.005,
        "holdout": 0.005,
    }:
        raise CorpusInventoryError("L002 split must remain 99/0.5/0.5")

    raw_sources = payload.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise CorpusInventoryError("sources must be a non-empty list")

    sources: list[CorpusSource] = []
    seen: set[str] = set()
    for raw in raw_sources:
        if not isinstance(raw, dict):
            raise CorpusInventoryError("source must be an object")
        required = {
            "source_id",
            "name",
            "categories",
            "languages",
            "dataset_url",
            "license_name",
            "license_url",
            "license_audit_status",
            "acquisition_status",
            "source_grouping_key",
            "content_rights_notes",
            "pii_risk",
            "intended_role",
        }
        missing = required - set(raw)
        if missing:
            raise CorpusInventoryError(
                "source missing fields: " + ", ".join(sorted(missing))
            )
        source_id = str(raw["source_id"]).strip()
        if not source_id:
            raise CorpusInventoryError("source_id must be non-empty")
        if source_id in seen:
            raise CorpusInventoryError(f"duplicate source_id {source_id!r}")
        seen.add(source_id)

        categories = tuple(str(x) for x in raw["categories"])
        if not categories or not set(categories) <= _ALLOWED_CATEGORIES:
            raise CorpusInventoryError(
                f"{source_id} has unsupported categories"
            )
        languages = tuple(str(x) for x in raw["languages"])
        if not languages:
            raise CorpusInventoryError(
                f"{source_id} must declare languages"
            )
        audit = str(raw["license_audit_status"])
        if audit not in _ALLOWED_AUDIT_STATUS:
            raise CorpusInventoryError(
                f"{source_id} has invalid license_audit_status"
            )
        acquisition = str(raw["acquisition_status"])
        if acquisition not in _ALLOWED_ACQUISITION_STATUS:
            raise CorpusInventoryError(
                f"{source_id} has invalid acquisition_status"
            )

        for field in (
            "name",
            "dataset_url",
            "license_name",
            "license_url",
            "source_grouping_key",
            "content_rights_notes",
            "pii_risk",
            "intended_role",
        ):
            if not isinstance(raw[field], str) or not raw[field].strip():
                raise CorpusInventoryError(
                    f"{source_id}.{field} must be non-empty"
                )

        sources.append(
            CorpusSource(
                source_id=source_id,
                name=str(raw["name"]),
                categories=categories,
                languages=languages,
                dataset_url=str(raw["dataset_url"]),
                license_name=str(raw["license_name"]),
                license_url=str(raw["license_url"]),
                license_audit_status=audit,
                acquisition_status=acquisition,
                source_grouping_key=str(raw["source_grouping_key"]),
                content_rights_notes=str(raw["content_rights_notes"]),
                pii_risk=str(raw["pii_risk"]),
                intended_role=str(raw["intended_role"]),
            )
        )

    return CorpusInventory(
        target_tokens=budgets["primary"],
        minimum_tokens=budgets["minimum"],
        stretch_tokens=budgets["stretch"],
        target_mix={k: float(v) for k, v in target_mix.items()},
        split={k: float(v) for k, v in split.items()},
        sources=tuple(sources),
    )


def eligible_source_ids(
    inventory: CorpusInventory,
    *,
    include_conditional: bool = False,
) -> tuple[str, ...]:
    allowed = {"approved"}
    if include_conditional:
        allowed.add("conditional")
    return tuple(
        source.source_id
        for source in inventory.sources
        if source.license_audit_status in allowed
    )
