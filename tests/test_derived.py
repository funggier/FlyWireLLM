from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from flywire_llm.artifacts import load_artifact_ledger
from flywire_llm.data_contract import load_corpus_inventory
from flywire_llm.derived import (
    DerivedCorpusError,
    load_derived_corpora,
    verify_derived_corpus_file,
)
from flywire_llm.rights import load_rights_policy


ROOT = Path(__file__).resolve().parents[1]


def _context():
    inventory = load_corpus_inventory(
        ROOT / "configs" / "corpus-sources-l002.json"
    )
    policy = load_rights_policy(
        ROOT / "configs" / "rights-lanes-l003.json",
        expected_source_ids={
            source.source_id for source in inventory.sources
        },
    )
    ledger = load_artifact_ledger(
        ROOT / "configs" / "source-artifacts-l003.json",
        rights_policy=policy,
    )
    return policy, ledger


def _fixture(tmp_path: Path):
    policy, ledger = _context()
    external_root = tmp_path / "external"
    accepted = external_root / "derived" / "accepted.jsonl"
    accepted.parent.mkdir(parents=True)
    accepted.write_text(
        '{"record_id":"R1","text":"hello"}\n',
        encoding="utf-8",
        newline="\n",
    )
    accepted_hash = hashlib.sha256(accepted.read_bytes()).hexdigest()

    summary = {
        "schema_version": 1,
        "stage": "L003",
        "derived_id": "fixture-derived-v1",
        "source_id": "fineweb2-thai",
        "artifact_id": "fineweb2-thai-train-005-00002",
        "rights_lane": "research_only",
        "global_manifest_authorized": False,
        "counts": {"accept_records": 1},
        "accepted": {
            "records_by_partition": {
                "train": 1,
                "validation": 0,
                "holdout": 0,
            },
            "tokens_by_partition": {
                "train": 7,
                "validation": 0,
                "holdout": 0,
            },
            "tokens_total": 7,
        },
        "files": {
            "accepted": {
                "path": "accepted.jsonl",
                "bytes": accepted.stat().st_size,
                "sha256": accepted_hash,
            }
        },
        "pretraining_authorized": False,
    }
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    summary_hash = hashlib.sha256(summary_path.read_bytes()).hexdigest()

    registry = {
        "schema_version": 1,
        "stage": "L003",
        "corpora": [
            {
                "derived_id": "fixture-derived-v1",
                "source_id": "fineweb2-thai",
                "artifact_id": "fineweb2-thai-train-005-00002",
                "rights_lane": "research_only",
                "primary_category": "general_thai",
                "summary_path": summary_path.name,
                "summary_sha256": summary_hash,
                "accepted_storage": (
                    "external://FlyWireLLM-data/derived/accepted.jsonl"
                ),
                "accepted_bytes": accepted.stat().st_size,
                "accepted_sha256": accepted_hash,
                "train_tokens": 7,
                "validation_tokens": 0,
                "holdout_tokens": 0,
                "accepted_records": 1,
                "global_manifest_authorized": False,
                "pretraining_authorized": False,
            }
        ],
    }
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    corpora = load_derived_corpora(
        registry_path,
        repo_root=tmp_path,
        rights_policy=policy,
        artifact_ledger=ledger,
    )
    return policy, ledger, external_root, registry_path, corpora


def test_derived_registry_validates_summary_and_external_file(tmp_path):
    _, _, external_root, _, corpora = _fixture(tmp_path)
    assert len(corpora) == 1
    corpus = corpora[0]
    assert corpus.train_tokens == 7
    assert corpus.rights_lane == "research_only"
    path = verify_derived_corpus_file(
        corpus,
        external_root=external_root,
    )
    assert path.name == "accepted.jsonl"


def test_derived_external_file_tamper_fails_closed(tmp_path):
    _, _, external_root, _, corpora = _fixture(tmp_path)
    path = external_root / "derived" / "accepted.jsonl"
    path.write_text("tampered", encoding="utf-8")
    with pytest.raises(
        DerivedCorpusError,
        match="byte size mismatch|SHA-256 mismatch",
    ):
        verify_derived_corpus_file(
            corpora[0],
            external_root=external_root,
        )


def test_derived_summary_cannot_self_authorize_pretraining(tmp_path):
    policy, ledger, _, registry_path, _ = _fixture(tmp_path)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    summary_path = tmp_path / registry["corpora"][0]["summary_path"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["pretraining_authorized"] = True
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    registry["corpora"][0]["summary_sha256"] = hashlib.sha256(
        summary_path.read_bytes()
    ).hexdigest()
    registry_path.write_text(json.dumps(registry), encoding="utf-8")

    with pytest.raises(
        DerivedCorpusError,
        match="summary pretraining flag changed",
    ):
        load_derived_corpora(
            registry_path,
            repo_root=tmp_path,
            rights_policy=policy,
            artifact_ledger=ledger,
        )
