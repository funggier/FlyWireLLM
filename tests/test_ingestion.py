from __future__ import annotations

from pathlib import Path

import pytest

from flywire_llm.artifacts import load_artifact_ledger
from flywire_llm.data_contract import load_corpus_inventory
from flywire_llm.ingestion import (
    IngestionCandidate,
    IngestionError,
    evaluate_candidate,
)
from flywire_llm.rights import load_rights_policy


ROOT = Path(__file__).resolve().parents[1]


def _context():
    inventory = load_corpus_inventory(
        ROOT / "configs" / "corpus-sources-l002.json"
    )
    policy = load_rights_policy(
        ROOT / "configs" / "rights-lanes-l003.json",
        expected_source_ids={source.source_id for source in inventory.sources},
    )
    ledger = load_artifact_ledger(
        ROOT / "configs" / "source-artifacts-l003.json",
        rights_policy=policy,
    )
    return policy, ledger


def _candidate(text: str) -> IngestionCandidate:
    return IngestionCandidate(
        record_id="TATOEBA-TH-1",
        text=text,
        language="th",
        category="general_thai",
        source_id="tatoeba-sentences",
        artifact_id="tatoeba-tha-sentences-2026-10-03",
        source_group_id="group-1",
        partition="train",
    )


def test_clean_release_safe_record_is_training_eligible():
    policy, ledger = _context()
    result = evaluate_candidate(
        _candidate("ภาษาไทยสำหรับทดสอบ FlyWireLLM root_id=720575940630024566"),
        rights_policy=policy,
        artifact_ledger=ledger,
        checkpoint_lane="release_safe",
    )
    assert result.status == "accept"
    assert result.training_eligible is True
    assert result.source_lane == "release_safe"
    assert len(result.content_sha256) == 64
    assert len(result.dedup_fingerprint) == 64
    assert len(result.simhash64_hex) == 16


def test_review_finding_is_quarantined_not_silently_trained():
    policy, ledger = _context()
    result = evaluate_candidate(
        _candidate("ติดต่อ alice@example.org สำหรับข้อมูลเพิ่มเติม"),
        rights_policy=policy,
        artifact_ledger=ledger,
        checkpoint_lane="release_safe",
    )
    assert result.status == "quarantine"
    assert result.training_eligible is False
    assert "possible_email" in result.screening_findings


def test_reject_finding_is_not_training_eligible():
    policy, ledger = _context()
    result = evaluate_candidate(
        _candidate("api_key=abcdefghijklmnopqrstuvwxyz012345"),
        rights_policy=policy,
        artifact_ledger=ledger,
        checkpoint_lane="release_safe",
    )
    assert result.status == "reject"
    assert result.training_eligible is False
    assert "possible_secret" in result.screening_findings


def test_release_safe_ingestion_rejects_research_only_artifact():
    policy, ledger = _context()
    candidate = IngestionCandidate(
        record_id="THWIKI-10",
        text="บทความตัวอย่าง",
        language="th",
        category="general_thai",
        source_id="wikimedia-thwiki",
        artifact_id="thwiki-pages-articles-2026-10-01",
        source_group_id="page-10",
        partition="train",
    )
    with pytest.raises(
        IngestionError,
        match="release_safe checkpoint cannot use research_only",
    ):
        evaluate_candidate(
            candidate,
            rights_policy=policy,
            artifact_ledger=ledger,
            checkpoint_lane="release_safe",
        )


def test_artifact_source_mismatch_fails_closed():
    policy, ledger = _context()
    candidate = IngestionCandidate(
        record_id="BAD-1",
        text="some clean text",
        language="en",
        category="general_english",
        source_id="mozilla-common-voice",
        artifact_id="tatoeba-eng-sentences-2026-10-03",
        source_group_id="group-1",
        partition="train",
    )
    with pytest.raises(IngestionError, match="does not match artifact source"):
        evaluate_candidate(
            candidate,
            rights_policy=policy,
            artifact_ledger=ledger,
            checkpoint_lane="release_safe",
        )
