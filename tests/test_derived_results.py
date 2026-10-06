from __future__ import annotations

from pathlib import Path

from flywire_llm.artifacts import load_artifact_ledger
from flywire_llm.data_contract import load_corpus_inventory
from flywire_llm.derived import load_derived_corpora
from flywire_llm.rights import load_rights_policy


ROOT = Path(__file__).resolve().parents[1]


def test_tracked_thai_fineweb2_derived_registry_is_frozen():
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
    assert len(corpora) == 1
    corpus = corpora[0]
    assert corpus.derived_id == (
        "fineweb2-thai-train-005-00002-sample1000000-v1"
    )
    assert corpus.source_id == "fineweb2-thai"
    assert corpus.rights_lane == "research_only"
    assert corpus.primary_category == "general_thai"
    assert corpus.summary_sha256 == (
        "b450aad95eadbe0e82c0787147dcc08ddd4dcb6cbeb7a46accff8fde068c7020"
    )
    assert corpus.accepted_sha256 == (
        "7b22cccde21f195ee3fc6585af2abd8ba8c3a4f63aa3ed082930a07d9ee0e633"
    )
    assert corpus.accepted_bytes == 3_072_938_059
    assert corpus.train_tokens == 293_826_034
    assert corpus.validation_tokens == 1_449_071
    assert corpus.holdout_tokens == 1_479_220
    assert corpus.accepted_records == 339_058
    assert corpus.global_manifest_authorized is False
    assert corpus.pretraining_authorized is False
