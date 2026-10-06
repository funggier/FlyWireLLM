from __future__ import annotations

import argparse
from pathlib import Path

from flywire_llm.artifacts import (
    load_artifact_ledger,
    resolve_external_storage_uri,
    verify_artifact_file,
)
from flywire_llm.data_contract import load_corpus_inventory
from flywire_llm.rights import load_rights_policy


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify every acquired L003 raw artifact against its ledger."
    )
    parser.add_argument(
        "--external-root",
        required=True,
        help="Filesystem path corresponding to external://FlyWireLLM-data/",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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

    for artifact in ledger.artifacts:
        path = resolve_external_storage_uri(
            artifact.local_storage,
            external_root=args.external_root,
        )
        verify_artifact_file(artifact, path)
        print(
            f"PASS artifact={artifact.artifact_id} "
            f"bytes={artifact.bytes} sha256={artifact.sha256}"
        )
    print(f"verified_artifacts={len(ledger.artifacts)}")


if __name__ == "__main__":
    main()
