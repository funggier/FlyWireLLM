from __future__ import annotations

from pathlib import Path

from flywire_llm.artifacts import load_artifact_ledger
from flywire_llm.capacity import (
    calculate_lane_capacity,
    load_capacity_plan,
)
from flywire_llm.data_contract import load_corpus_inventory
from flywire_llm.derived import load_derived_corpora
from flywire_llm.mixing import load_mixture_contract
from flywire_llm.rights import load_rights_policy


ROOT = Path(__file__).resolve().parents[1]


def _context():
    inventory = load_corpus_inventory(
        ROOT / "configs" / "corpus-sources-l002.json"
    )
    rights = load_rights_policy(
        ROOT / "configs" / "rights-lanes-l003.json",
        expected_source_ids={
            source.source_id for source in inventory.sources
        },
    )
    ledger = load_artifact_ledger(
        ROOT / "configs" / "source-artifacts-l003.json",
        rights_policy=rights,
    )
    derived = load_derived_corpora(
        ROOT / "configs" / "derived-corpora-l003.json",
        repo_root=ROOT,
        rights_policy=rights,
        artifact_ledger=ledger,
    )
    plan = load_capacity_plan(
        ROOT / "configs" / "capacity-l003.json",
        rights_policy=rights,
    )
    mixture = load_mixture_contract(
        ROOT / "configs" / "mixture-l003.json"
    )
    return plan, derived, mixture


def test_release_safe_capacity_excludes_all_research_only_data():
    plan, derived, mixture = _context()
    capacity = calculate_lane_capacity(
        plan,
        derived,
        checkpoint_lane="release_safe",
        mixture_contract=mixture,
    )
    assert capacity.by_category == {
        "general_thai": 41_836,
        "general_english": 20_019_410,
        "technical_scientific_code": 0,
        "flywire_domain": 0,
    }
    assert capacity.total_train_tokens == 20_061_246
    assert capacity.mixture.ready is False
    assert capacity.planning_only is True


def test_research_capacity_includes_thai_and_english_derived_but_is_not_ready():
    plan, derived, mixture = _context()
    capacity = calculate_lane_capacity(
        plan,
        derived,
        checkpoint_lane="research_only",
        mixture_contract=mixture,
    )
    assert capacity.by_category["general_thai"] == 332_040_366
    assert capacity.by_category["general_english"] == 140_994_851
    assert capacity.by_category["technical_scientific_code"] == 0
    assert capacity.by_category["flywire_domain"] == 0
    assert capacity.total_train_tokens == 473_035_217
    assert capacity.mixture.thai_gap == 0
    assert capacity.mixture.english_gap == 59_005_149
    assert capacity.mixture.technical_gap == 100_000_000
    assert capacity.mixture.ready is False
    assert capacity.planning_only is True


def test_thai_overage_does_not_reduce_english_or_technical_gap():
    plan, derived, mixture = _context()
    capacity = calculate_lane_capacity(
        plan,
        derived,
        checkpoint_lane="research_only",
        mixture_contract=mixture,
    )
    assert capacity.by_category["general_thai"] > 200_000_000
    assert capacity.mixture.english_gap == 59_005_149
    assert capacity.mixture.technical_gap == 100_000_000
