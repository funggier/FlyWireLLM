from __future__ import annotations

import json
from pathlib import Path

import pytest

from flywire_llm.data_contract import load_corpus_inventory
from flywire_llm.rights import load_rights_policy
from flywire_llm.source_candidates import (
    SourceCandidateError,
    load_source_candidates,
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


def test_common_voice_27_candidates_are_pinned_release_safe_metadata():
    candidates = load_source_candidates(
        ROOT / "configs" / "source-candidates-l003.json",
        rights_policy=_policy(),
    )
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    assert set(by_id) == {"common-voice-27-th", "common-voice-27-en"}
    assert by_id["common-voice-27-th"].dataset_id == (
        "cmu5x5tj000f2nq0755h8qcy5"
    )
    assert by_id["common-voice-27-en"].dataset_id == (
        "cmu5jplf300nwmh07iqvk9leo"
    )
    assert all(candidate.license == "CC0-1.0" for candidate in candidates)
    assert all(
        candidate.rights_lane == "release_safe"
        for candidate in candidates
    )
    assert all(
        candidate.raw_redistribution == "do_not_mirror"
        for candidate in candidates
    )


def test_common_voice_scope_is_sentence_text_only():
    candidates = load_source_candidates(
        ROOT / "configs" / "source-candidates-l003.json",
        rights_policy=_policy(),
    )
    for candidate in candidates:
        assert candidate.ingestion_scope == {
            "validated_sentences_tsv": True,
            "audio": False,
            "clip_rows": False,
            "client_id": False,
            "speaker_demographics": False,
            "unvalidated_sentences": False,
        }


def test_common_voice_audio_ingestion_regression_fails_closed(tmp_path):
    payload = json.loads(
        (ROOT / "configs" / "source-candidates-l003.json").read_text(
            encoding="utf-8"
        )
    )
    payload["candidates"][0]["ingestion_scope"]["audio"] = True
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        SourceCandidateError,
        match="validated sentence text only",
    ):
        load_source_candidates(path, rights_policy=_policy())


def test_common_voice_raw_mirroring_regression_fails_closed(tmp_path):
    payload = json.loads(
        (ROOT / "configs" / "source-candidates-l003.json").read_text(
            encoding="utf-8"
        )
    )
    payload["candidates"][1]["raw_redistribution"] = "allowed_with_notice"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SourceCandidateError, match="must not be mirrored"):
        load_source_candidates(path, rights_policy=_policy())
