from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Mapping


class PretrainingSelectionError(ValueError):
    pass


@dataclass(frozen=True)
class SelectionCandidate:
    canonical_id: str
    source_id: str
    artifact_id: str
    record_id: str
    primary_category: str
    token_count: int
    rights_lane: str
    partition: str
    status: str


@dataclass(frozen=True)
class SelectedRecord:
    canonical_id: str
    source_id: str
    artifact_id: str
    record_id: str
    primary_category: str
    token_count: int
    selected_token_count: int
    rights_lane: str
    selection_key: str


def deterministic_selection_key(
    seed: int,
    canonical_id: str,
) -> str:
    if not isinstance(seed, int) or seed < 0:
        raise PretrainingSelectionError(
            "selection seed must be a non-negative integer"
        )
    if not isinstance(canonical_id, str) or not canonical_id:
        raise PretrainingSelectionError(
            "canonical_id must be non-empty"
        )
    return hashlib.sha256(
        f"{seed}|{canonical_id}".encode("utf-8")
    ).hexdigest()


def _validate_candidate(candidate: SelectionCandidate) -> None:
    for field in (
        "canonical_id",
        "source_id",
        "artifact_id",
        "record_id",
        "primary_category",
        "rights_lane",
        "partition",
        "status",
    ):
        value = getattr(candidate, field)
        if not isinstance(value, str) or not value:
            raise PretrainingSelectionError(
                f"{field} must be non-empty"
            )
    if candidate.token_count <= 0:
        raise PretrainingSelectionError(
            "candidate token_count must be positive"
        )


def select_exact_category_budgets(
    candidates: Iterable[SelectionCandidate],
    *,
    category_targets: Mapping[str, int],
    seed: int,
) -> list[SelectedRecord]:
    if not category_targets:
        raise PretrainingSelectionError(
            "category_targets must not be empty"
        )
    normalized_targets: dict[str, int] = {}
    for category, target in category_targets.items():
        if not isinstance(category, str) or not category:
            raise PretrainingSelectionError(
                "category target name must be non-empty"
            )
        if not isinstance(target, int) or target < 0:
            raise PretrainingSelectionError(
                "category target must be a non-negative integer"
            )
        normalized_targets[category] = target

    grouped: dict[str, list[tuple[str, SelectionCandidate]]] = {
        category: [] for category in normalized_targets
    }
    seen: set[str] = set()
    for candidate in candidates:
        _validate_candidate(candidate)
        if candidate.canonical_id in seen:
            raise PretrainingSelectionError(
                f"duplicate canonical_id {candidate.canonical_id!r}"
            )
        seen.add(candidate.canonical_id)

        if candidate.status != "accept":
            continue
        if candidate.partition != "train":
            continue
        if candidate.rights_lane not in {
            "release_safe",
            "research_only",
        }:
            raise PretrainingSelectionError(
                "accepted training candidate has invalid rights lane"
            )
        if candidate.primary_category not in normalized_targets:
            raise PretrainingSelectionError(
                "accepted training candidate has unconfigured category "
                f"{candidate.primary_category!r}"
            )
        if normalized_targets[candidate.primary_category] == 0:
            continue
        grouped[candidate.primary_category].append(
            (
                deterministic_selection_key(
                    seed,
                    candidate.canonical_id,
                ),
                candidate,
            )
        )

    selected: list[SelectedRecord] = []
    for category in sorted(normalized_targets):
        target = normalized_targets[category]
        if target == 0:
            continue
        rows = sorted(
            grouped[category],
            key=lambda item: (item[0], item[1].canonical_id),
        )
        consumed = 0
        for key, candidate in rows:
            if consumed >= target:
                break
            take = min(candidate.token_count, target - consumed)
            selected.append(
                SelectedRecord(
                    canonical_id=candidate.canonical_id,
                    source_id=candidate.source_id,
                    artifact_id=candidate.artifact_id,
                    record_id=candidate.record_id,
                    primary_category=category,
                    token_count=candidate.token_count,
                    selected_token_count=take,
                    rights_lane=candidate.rights_lane,
                    selection_key=key,
                )
            )
            consumed += take
        if consumed != target:
            raise PretrainingSelectionError(
                f"insufficient {category} tokens "
                f"target={target} selected={consumed}"
            )

    return selected


def selected_token_totals(
    rows: Iterable[SelectedRecord],
) -> dict[str, int]:
    totals: dict[str, int] = {}
    for row in rows:
        totals[row.primary_category] = (
            totals.get(row.primary_category, 0)
            + row.selected_token_count
        )
    return totals
