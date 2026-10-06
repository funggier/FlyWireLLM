from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from flywire_llm.artifacts import (
    ArtifactLedgerError,
    load_artifact_ledger,
    resolve_external_storage_uri,
    validate_checkpoint_artifacts,
    verify_artifact_file,
)
from flywire_llm.data_contract import load_corpus_inventory
from flywire_llm.rights import load_rights_policy


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


def test_l003_artifact_ledger_matches_rights_policy():
    policy = _policy()
    ledger = load_artifact_ledger(
        ROOT / "configs" / "source-artifacts-l003.json",
        rights_policy=policy,
    )
    assert [artifact.artifact_id for artifact in ledger.artifacts] == [
        "tatoeba-tha-sentences-2026-10-03",
        "tatoeba-eng-sentences-2026-10-03",
        "thwiki-pages-articles-2026-10-01",
        "fineweb2-thai-train-005-00002",
        "fineweb2-thai-train-005-00001",
        "fineweb-english-10bt-014-00000",
    ]


def test_release_safe_artifacts_reject_research_only_wikipedia():
    policy = _policy()
    ledger = load_artifact_ledger(
        ROOT / "configs" / "source-artifacts-l003.json",
        rights_policy=policy,
    )
    selected = validate_checkpoint_artifacts(
        ledger,
        rights_policy=policy,
        checkpoint_lane="release_safe",
        artifact_ids=[
            "tatoeba-tha-sentences-2026-10-03",
            "tatoeba-eng-sentences-2026-10-03",
        ],
    )
    assert len(selected) == 2

    with pytest.raises(
        ArtifactLedgerError,
        match="release_safe checkpoint cannot use research_only",
    ):
        validate_checkpoint_artifacts(
            ledger,
            rights_policy=policy,
            checkpoint_lane="release_safe",
            artifact_ids=["thwiki-pages-articles-2026-10-01"],
        )


def test_artifact_file_verification_checks_size_and_sha256(tmp_path):
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"immutable source bytes")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    policy = _policy()
    payload = {
        "schema_version": 1,
        "stage": "L003",
        "artifacts": [
            {
                "artifact_id": "fixture",
                "source_id": "tatoeba-sentences",
                "snapshot_date": "2026-10-06",
                "artifact_role": "raw_source",
                "origin_url": "https://example.org/source",
                "acquisition_method": "test fixture",
                "sha256": digest,
                "bytes": path.stat().st_size,
                "rights_lane": "release_safe",
                "raw_redistribution": "allowed_with_attribution",
                "local_storage": "external://fixture",
                "immutable": True,
            }
        ],
    }
    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_text(json.dumps(payload), encoding="utf-8")
    artifact = load_artifact_ledger(
        ledger_path,
        rights_policy=policy,
    ).artifacts[0]
    verify_artifact_file(artifact, path)

    path.write_bytes(b"tampered")
    with pytest.raises(ArtifactLedgerError, match="byte size mismatch"):
        verify_artifact_file(artifact, path)


def test_artifact_lane_mismatch_fails_closed(tmp_path):
    payload = json.loads(
        (ROOT / "configs" / "source-artifacts-l003.json").read_text(
            encoding="utf-8"
        )
    )
    payload["artifacts"][0]["rights_lane"] = "research_only"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ArtifactLedgerError, match="does not match source policy"):
        load_artifact_ledger(path, rights_policy=_policy())

def test_external_storage_uri_resolves_under_declared_root(tmp_path):
    path = resolve_external_storage_uri(
        "external://FlyWireLLM-data/L002/Tatoeba/eng_sentences.tsv.bz2",
        external_root=tmp_path,
    )
    assert path == (
        tmp_path / "L002" / "Tatoeba" / "eng_sentences.tsv.bz2"
    ).resolve()


def test_external_storage_uri_rejects_traversal(tmp_path):
    with pytest.raises(ArtifactLedgerError, match="unsafe"):
        resolve_external_storage_uri(
            "external://FlyWireLLM-data/../secret.txt",
            external_root=tmp_path,
        )

