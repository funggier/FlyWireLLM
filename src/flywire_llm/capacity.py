from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .derived import DerivedCorpus
from .mixing import (
    MixtureContract,
    MixtureReadiness,
    evaluate_mixture_readiness,
)
from .rights import RightsPolicy


class CapacityError(ValueError):
    pass


_CATEGORIES = {
    "general_thai",
    "general_english",
    "technical_scientific_code",
    "flywire_domain",
}


@dataclass(frozen=True)
class BaselineCapacity:
    capacity_id: str
    source_id: str
    rights_lane: str
    primary_category: str
    train_tokens: int


@dataclass(frozen=True)
class CapacityPlan:
    baseline: tuple[BaselineCapacity, ...]
    cross_source_dedup_applied: bool


@dataclass(frozen=True)
class LaneCapacity:
    checkpoint_lane: str
    by_category: dict[str, int]
    total_train_tokens: int
    mixture: MixtureReadiness
    planning_only: bool


def load_capacity_plan(
    path: str | Path,
    *,
    rights_policy: RightsPolicy,
) -> CapacityPlan:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CapacityError("capacity plan must be an object")
    if payload.get("schema_version") != 1:
        raise CapacityError("unsupported capacity plan schema")
    if payload.get("stage") != "L003":
        raise CapacityError("capacity plan stage must be L003")
    if payload.get("tokenizer") != "base50m-unigram-32000-v1":
        raise CapacityError("capacity tokenizer changed")
    if payload.get("quality_audit") != (
        "results/l003/acquired-quality-audit.json"
    ):
        raise CapacityError("capacity quality audit path changed")
    if payload.get("quality_audit_sha256") != (
        "f7e6c3ed4e3d5e915ecc319fc7cec00ea0132e40c9a1ea31769397126f309988"
    ):
        raise CapacityError("capacity quality audit hash changed")
    if payload.get("global_cross_source_dedup_applied") is not False:
        raise CapacityError(
            "L003 planning capacity must not claim global dedup is complete"
        )

    raw_baseline = payload.get("baseline")
    if not isinstance(raw_baseline, list) or not raw_baseline:
        raise CapacityError("baseline capacity must be non-empty")
    result: list[BaselineCapacity] = []
    seen: set[str] = set()
    for raw in raw_baseline:
        if not isinstance(raw, dict):
            raise CapacityError("capacity row must be an object")
        required = {
            "capacity_id",
            "source_id",
            "rights_lane",
            "primary_category",
            "train_tokens",
        }
        if set(raw) != required:
            raise CapacityError("capacity row field set changed")
        capacity_id = raw["capacity_id"]
        if not isinstance(capacity_id, str) or not capacity_id.strip():
            raise CapacityError("capacity_id must be non-empty")
        if capacity_id in seen:
            raise CapacityError(
                f"duplicate capacity_id {capacity_id!r}"
            )
        seen.add(capacity_id)

        source_id = raw["source_id"]
        source = rights_policy.sources.get(source_id)
        if source is None:
            raise CapacityError(
                f"{capacity_id}: unknown source_id {source_id!r}"
            )
        rights_lane = raw["rights_lane"]
        if rights_lane != source.lane:
            raise CapacityError(
                f"{capacity_id}: rights lane does not match policy"
            )
        if rights_lane == "blocked":
            raise CapacityError(
                f"{capacity_id}: blocked source has capacity"
            )
        category = raw["primary_category"]
        if category not in _CATEGORIES:
            raise CapacityError(
                f"{capacity_id}: invalid primary category"
            )
        tokens = raw["train_tokens"]
        if not isinstance(tokens, int) or tokens < 0:
            raise CapacityError(
                f"{capacity_id}: train_tokens must be non-negative"
            )
        result.append(
            BaselineCapacity(
                capacity_id=capacity_id,
                source_id=source_id,
                rights_lane=rights_lane,
                primary_category=category,
                train_tokens=tokens,
            )
        )

    return CapacityPlan(
        baseline=tuple(result),
        cross_source_dedup_applied=False,
    )


def calculate_lane_capacity(
    plan: CapacityPlan,
    derived_corpora: tuple[DerivedCorpus, ...],
    *,
    checkpoint_lane: str,
    mixture_contract: MixtureContract,
) -> LaneCapacity:
    if checkpoint_lane not in {"release_safe", "research_only"}:
        raise CapacityError("invalid checkpoint lane")
    totals = {category: 0 for category in _CATEGORIES}

    def lane_allowed(source_lane: str) -> bool:
        if checkpoint_lane == "release_safe":
            return source_lane == "release_safe"
        return source_lane in {"release_safe", "research_only"}

    for row in plan.baseline:
        if lane_allowed(row.rights_lane):
            totals[row.primary_category] += row.train_tokens
    for corpus in derived_corpora:
        if lane_allowed(corpus.rights_lane):
            totals[corpus.primary_category] += corpus.train_tokens

    mixture = evaluate_mixture_readiness(
        mixture_contract,
        available_tokens=totals,
    )
    return LaneCapacity(
        checkpoint_lane=checkpoint_lane,
        by_category=totals,
        total_train_tokens=sum(totals.values()),
        mixture=mixture,
        planning_only=not plan.cross_source_dedup_applied,
    )
