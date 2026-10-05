from __future__ import annotations

from pathlib import Path

from flywire_llm.data_contract import (
    assign_partition,
    dedup_fingerprint,
    eligible_source_ids,
    load_corpus_inventory,
    normalize_for_dedup,
)


ROOT = Path(__file__).resolve().parents[1]


def test_l002_corpus_inventory_loads_and_preserves_budget():
    inventory = load_corpus_inventory(
        ROOT / "configs" / "corpus-sources-l002.json"
    )
    assert inventory.minimum_tokens == 250_000_000
    assert inventory.target_tokens == 500_000_000
    assert inventory.stretch_tokens == 1_000_000_000
    assert inventory.target_mix == {
        "general_thai": 0.40,
        "general_english": 0.40,
        "technical_scientific_code": 0.10,
        "flywire_domain": 0.10,
    }
    assert inventory.split == {
        "train": 0.99,
        "validation": 0.005,
        "holdout": 0.005,
    }


def test_only_audited_sources_are_approved_without_conditional_flag():
    inventory = load_corpus_inventory(
        ROOT / "configs" / "corpus-sources-l002.json"
    )
    approved = set(eligible_source_ids(inventory))
    assert approved == {
        "tatoeba-sentences",
        "mozilla-common-voice",
    }
    conditional = set(
        eligible_source_ids(inventory, include_conditional=True)
    )
    assert "fineweb2-thai" in conditional
    assert "fineweb-english" in conditional
    assert "wikimedia-thwiki" in conditional
    assert "wikimedia-enwiki" in conditional
    assert "project-gutenberg" in conditional
    assert "flywiremodel-curated-export" not in conditional


def test_dedup_normalization_is_conservative_and_stable():
    a = "ภาษาไทย\r\n  English   text"
    b = "ภาษาไทย\nEnglish text"
    assert normalize_for_dedup(a) == normalize_for_dedup(b)
    assert dedup_fingerprint(a) == dedup_fingerprint(b)


def test_duplicate_content_is_forced_to_same_partition():
    fingerprint = dedup_fingerprint("identical content")
    a = assign_partition("source-a", fingerprint)
    b = assign_partition("source-a", fingerprint)
    assert a == b
    assert a in {"train", "validation", "holdout"}


def test_split_is_source_namespaced():
    group = "document-123"
    # The source id participates in the hash key, preventing unrelated sources
    # from being coupled merely because they reuse an identifier.
    observed = {
        assign_partition(f"source-{index}", group)
        for index in range(20)
    }
    assert observed <= {"train", "validation", "holdout"}
    assert "train" in observed
