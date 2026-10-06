from __future__ import annotations

from pathlib import Path

import pytest

from flywire_llm.data_contract import load_corpus_inventory
from flywire_llm.rights import (
    RightsPolicyError,
    can_promote_checkpoint,
    load_rights_policy,
    token_gap,
    training_budget_ready,
    validate_checkpoint_sources,
)


ROOT = Path(__file__).resolve().parents[1]


def _policy():
    inventory = load_corpus_inventory(
        ROOT / "configs" / "corpus-sources-l002.json"
    )
    return load_rights_policy(
        ROOT / "configs" / "rights-lanes-l003.json",
        expected_source_ids={
            source.source_id for source in inventory.sources
        },
    )


def test_rights_policy_covers_exact_l002_source_inventory():
    policy = _policy()
    assert set(policy.sources) == {
        "tatoeba-sentences",
        "mozilla-common-voice",
        "wikimedia-thwiki",
        "wikimedia-enwiki",
        "fineweb2-thai",
        "fineweb-english",
        "project-gutenberg",
        "flywiremodel-curated-export",
    }


def test_release_safe_checkpoint_accepts_only_release_safe_sources():
    policy = _policy()
    assert validate_checkpoint_sources(
        policy,
        checkpoint_lane="release_safe",
        source_ids=["tatoeba-sentences", "mozilla-common-voice"],
    ) == ("tatoeba-sentences", "mozilla-common-voice")

    with pytest.raises(
        RightsPolicyError,
        match="release_safe checkpoint cannot use research_only",
    ):
        validate_checkpoint_sources(
            policy,
            checkpoint_lane="release_safe",
            source_ids=["tatoeba-sentences", "wikimedia-thwiki"],
        )


def test_research_checkpoint_may_mix_safe_and_research_sources():
    policy = _policy()
    assert validate_checkpoint_sources(
        policy,
        checkpoint_lane="research_only",
        source_ids=[
            "tatoeba-sentences",
            "wikimedia-thwiki",
            "fineweb2-thai",
        ],
    ) == (
        "tatoeba-sentences",
        "wikimedia-thwiki",
        "fineweb2-thai",
    )


def test_blocked_flywire_export_is_rejected_in_all_lanes():
    policy = _policy()
    for lane in ("release_safe", "research_only"):
        with pytest.raises(RightsPolicyError, match="blocked source"):
            validate_checkpoint_sources(
                policy,
                checkpoint_lane=lane,
                source_ids=["flywiremodel-curated-export"],
            )


def test_research_lineage_cannot_promote_to_release_safe():
    assert can_promote_checkpoint(
        from_lane="release_safe",
        to_lane="research_only",
    )
    assert not can_promote_checkpoint(
        from_lane="research_only",
        to_lane="release_safe",
    )


def test_known_l003_budget_preserves_baseline_and_uses_final_global_capacity():
    policy = _policy()
    accounting = policy.known_token_accounting
    assert accounting["pre_screen_train"]["release_safe"]["total"] == (
        20_062_981
    )
    assert accounting["pre_screen_train"]["research_only"]["total"] == (
        38_312_940
    )
    assert accounting["screened_train"]["release_safe"]["total"] == (
        20_061_246
    )
    assert accounting["screened_train"]["research_only"]["total"] == (
        38_172_496
    )
    qualified = accounting["qualified_global_train"]
    assert qualified["release_safe"]["total"] == 20_053_464
    assert qualified["research_only"]["total"] == 969_701_220
    assert not training_budget_ready(policy, checkpoint_lane="release_safe")
    assert training_budget_ready(policy, checkpoint_lane="research_only")
    assert token_gap(policy, checkpoint_lane="release_safe") == 479_946_536
    assert token_gap(policy, checkpoint_lane="research_only") == 0
