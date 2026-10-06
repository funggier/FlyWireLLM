from __future__ import annotations

import pytest

from flywire_llm.pretraining_selection import (
    PretrainingSelectionError,
    SelectionCandidate,
    deterministic_selection_key,
    select_exact_category_budgets,
    selected_token_totals,
)


def _candidate(
    name: str,
    *,
    category: str,
    tokens: int,
    partition: str = "train",
    status: str = "accept",
    lane: str = "research_only",
) -> SelectionCandidate:
    return SelectionCandidate(
        canonical_id=f"canonical::{name}",
        source_id="source",
        artifact_id="artifact",
        record_id=name,
        primary_category=category,
        token_count=tokens,
        rights_lane=lane,
        partition=partition,
        status=status,
    )


def test_selection_key_is_stable_and_seeded():
    a = deterministic_selection_key(1234, "canonical::a")
    b = deterministic_selection_key(1234, "canonical::a")
    c = deterministic_selection_key(1235, "canonical::a")
    assert a == b
    assert a != c
    assert len(a) == 64


def test_exact_category_selection_truncates_only_final_record():
    candidates = [
        _candidate("th-a", category="general_thai", tokens=7),
        _candidate("th-b", category="general_thai", tokens=7),
        _candidate("th-c", category="general_thai", tokens=7),
        _candidate("en-a", category="general_english", tokens=9),
        _candidate("en-b", category="general_english", tokens=9),
        _candidate(
            "ignored-validation",
            category="general_english",
            tokens=1000,
            partition="validation",
        ),
        _candidate(
            "ignored-duplicate",
            category="general_english",
            tokens=1000,
            status="exclude",
        ),
    ]
    selected = select_exact_category_budgets(
        candidates,
        category_targets={
            "general_thai": 10,
            "general_english": 12,
            "technical_scientific_code": 0,
            "flywire_domain": 0,
        },
        seed=1234,
    )
    assert selected_token_totals(selected) == {
        "general_english": 12,
        "general_thai": 10,
    }
    partial = [
        row
        for row in selected
        if row.selected_token_count < row.token_count
    ]
    assert len(partial) == 2
    assert all(row.selected_token_count > 0 for row in selected)


def test_selection_is_independent_of_input_order():
    candidates = [
        _candidate(f"r{i}", category="general_english", tokens=5)
        for i in range(10)
    ]
    first = select_exact_category_budgets(
        candidates,
        category_targets={"general_english": 25},
        seed=1234,
    )
    second = select_exact_category_budgets(
        reversed(candidates),
        category_targets={"general_english": 25},
        seed=1234,
    )
    assert [
        (row.canonical_id, row.selected_token_count)
        for row in first
    ] == [
        (row.canonical_id, row.selected_token_count)
        for row in second
    ]


def test_selection_rejects_insufficient_category_capacity():
    with pytest.raises(
        PretrainingSelectionError,
        match="insufficient general_thai tokens",
    ):
        select_exact_category_budgets(
            [
                _candidate(
                    "th-only",
                    category="general_thai",
                    tokens=5,
                )
            ],
            category_targets={"general_thai": 10},
            seed=1234,
        )


def test_selection_rejects_blocked_accepted_train_record():
    with pytest.raises(
        PretrainingSelectionError,
        match="invalid rights lane",
    ):
        select_exact_category_budgets(
            [
                _candidate(
                    "blocked",
                    category="general_english",
                    tokens=5,
                    lane="blocked",
                )
            ],
            category_targets={"general_english": 5},
            seed=1234,
        )
