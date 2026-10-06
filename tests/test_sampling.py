from __future__ import annotations

import pytest

from flywire_llm.sampling import (
    SamplingContractError,
    stable_ppm_selected,
    stable_source_group_id,
)


def test_stable_ppm_selection_is_deterministic():
    kwargs = {
        "seed": "flywirellm-l003-fineweb2-thai-sample-v1",
        "parts_per_million": 600_000,
    }
    observed = [
        stable_ppm_selected(f"doc-{index}", **kwargs)
        for index in range(100)
    ]
    repeated = [
        stable_ppm_selected(f"doc-{index}", **kwargs)
        for index in range(100)
    ]
    assert observed == repeated
    assert 40 <= sum(observed) <= 80


def test_full_ppm_selection_accepts_every_record():
    assert all(
        stable_ppm_selected(
            f"doc-{index}",
            seed="seed",
            parts_per_million=1_000_000,
        )
        for index in range(100)
    )


def test_selection_seed_changes_membership():
    left = [
        stable_ppm_selected(
            f"doc-{index}",
            seed="seed-a",
            parts_per_million=500_000,
        )
        for index in range(100)
    ]
    right = [
        stable_ppm_selected(
            f"doc-{index}",
            seed="seed-b",
            parts_per_million=500_000,
        )
        for index in range(100)
    ]
    assert left != right


def test_source_group_prefers_url_over_record_id():
    a = stable_source_group_id(
        source_url="https://example.org/page",
        record_id="A",
    )
    b = stable_source_group_id(
        source_url="https://example.org/page",
        record_id="B",
    )
    assert a == b


def test_source_group_falls_back_to_record_id():
    a = stable_source_group_id(source_url=None, record_id="A")
    b = stable_source_group_id(source_url=None, record_id="B")
    assert a != b
    assert len(a) == 64


def test_invalid_sampling_fraction_fails_closed():
    with pytest.raises(SamplingContractError):
        stable_ppm_selected(
            "doc",
            seed="seed",
            parts_per_million=0,
        )
