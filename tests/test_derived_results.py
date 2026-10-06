from __future__ import annotations

from pathlib import Path

from flywire_llm.artifacts import load_artifact_ledger
from flywire_llm.data_contract import load_corpus_inventory
from flywire_llm.derived import load_derived_corpora
from flywire_llm.rights import load_rights_policy


ROOT = Path(__file__).resolve().parents[1]


def test_tracked_web_derived_registry_is_frozen():
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
    corpora = load_derived_corpora(
        ROOT / "configs" / "derived-corpora-l003.json",
        repo_root=ROOT,
        rights_policy=rights,
        artifact_ledger=ledger,
    )
    assert len(corpora) == 2
    by_id = {corpus.derived_id: corpus for corpus in corpora}

    thai = by_id[
        "fineweb2-thai-train-005-00002-sample1000000-v1"
    ]
    assert thai.source_id == "fineweb2-thai"
    assert thai.rights_lane == "research_only"
    assert thai.primary_category == "general_thai"
    assert thai.summary_sha256 == (
        "b450aad95eadbe0e82c0787147dcc08ddd4dcb6cbeb7a46accff8fde068c7020"
    )
    assert thai.accepted_sha256 == (
        "7b22cccde21f195ee3fc6585af2abd8ba8c3a4f63aa3ed082930a07d9ee0e633"
    )
    assert thai.accepted_bytes == 3_072_938_059
    assert thai.train_tokens == 293_826_034
    assert thai.validation_tokens == 1_449_071
    assert thai.holdout_tokens == 1_479_220
    assert thai.accepted_records == 339_058

    english = by_id[
        "fineweb-english-10bt-014-00000-sample600000-v1"
    ]
    assert english.source_id == "fineweb-english"
    assert english.rights_lane == "research_only"
    assert english.primary_category == "general_english"
    assert english.summary_sha256 == (
        "e23be2af32cb8ddc7e4b0db417e16acc0f29945f164a2258e9108d92dccd1b81"
    )
    assert english.accepted_sha256 == (
        "13526d486bf90e06b6a3f2cdfb6ae0f653541f6c55876da0ca052f3d3d4ad803"
    )
    assert english.accepted_bytes == 603_547_565
    assert english.train_tokens == 120_975_441
    assert english.validation_tokens == 539_655
    assert english.holdout_tokens == 685_017
    assert english.accepted_records == 163_676

    for corpus in corpora:
        assert corpus.global_manifest_authorized is False
        assert corpus.pretraining_authorized is False
