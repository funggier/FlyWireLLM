from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads(
        (ROOT / "results" / "l002" / name).read_text(encoding="utf-8")
    )


def test_tatoeba_8k_pilot_is_lossless_and_not_final():
    result = _load("tatoeba-unigram-8192-pilot.json")
    assert result["tokenizer"]["actual_vocab_size"] == 8_192
    assert result["tokenizer"]["final_base50m_tokenizer"] is False
    assert result["benchmark"]["cases"] == 115
    assert result["benchmark"]["roundtrip_failures"] == 0
    assert result["benchmark"]["unk_tokens"] == 0
    assert result["benchmark"]["compression_vs_byte_thai"] > 10
    assert result["benchmark"]["compression_vs_byte_english"] > 3


def test_base50m_32k_candidate_is_lossless_on_frozen_benchmark():
    result = _load("base50m-unigram-32000.json")
    assert result["candidate_id"] == "base50m-unigram-32000-v1"
    assert result["tokenizer"]["vocab_size"] == 32_000
    assert result["tokenizer"]["normalization"] == "identity"
    assert result["tokenizer"]["byte_fallback"] is True
    assert result["benchmark"]["cases"] == 10_150
    assert result["benchmark"]["overall"]["roundtrip_failures"] == 0
    assert result["benchmark"]["overall"]["unk_tokens"] == 0
    assert result["benchmark"]["by_language"]["th"][
        "compression_vs_byte"
    ] > 10
    assert result["benchmark"]["by_language"]["en"][
        "compression_vs_byte"
    ] > 4
    assert result["benchmark"]["by_language"]["mixed"][
        "compression_vs_byte"
    ] > 2


def test_base50m_candidate_corpus_never_materializes_holdout_text():
    result = _load("base50m-unigram-32000.json")
    materialized = result["corpus"]["materialized"]
    assert materialized["holdout_text"] is False
    assert materialized["fit"]["sha256"] == (
        "01f51cec4eb6926e8654fe8d920d498e342351dbc4b07f4b50f0ec4c6be690ad"
    )
    assert materialized["validation"]["sha256"] == (
        "ebb2dfa1c8b4f47e1608d39bb5c5b05c92096a31fd1838345016e5287288cf6b"
    )


def test_acquired_source_inventory_is_only_11_percent_of_500m_target():
    result = _load("source-token-budget-32000.json")
    assert result["by_partition"]["train"]["tokens"] == 58_334_085
    assert result["primary_target_tokens"] == 500_000_000
    assert result["train_gap_to_primary_target"] == 441_665_915
    assert result["train_fraction_of_primary_target"] == 0.11666817


def test_thwiki_sample_is_pinned_but_rights_remain_conditional():
    result = _load("thwiki-source-sample-v1.json")
    assert result["source"]["raw_dump_sha256"] == (
        "1995cb37f27a20349918f8e234694fe700ae748a0b4f4dcc26babfd3cc540522"
    )
    assert result["source"]["license_status"] == "conditional"
    assert result["sample_policy"]["use_full_extraction"] is False
    assert result["sample_policy"]["pretraining_rights_qualified"] is False
    assert len(result["chunks"]) == 8
    assert sum(item["bytes"] for item in result["chunks"]) == 838_467_053
