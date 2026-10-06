from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .artifacts import (
    ArtifactLedger,
    ArtifactLedgerError,
    resolve_external_storage_uri,
)
from .rights import RightsPolicy


class DerivedCorpusError(ValueError):
    pass


@dataclass(frozen=True)
class DerivedCorpus:
    derived_id: str
    source_id: str
    artifact_id: str
    rights_lane: str
    primary_category: str
    summary_path: str
    summary_sha256: str
    accepted_storage: str
    accepted_bytes: int
    accepted_sha256: str
    train_tokens: int
    validation_tokens: int
    holdout_tokens: int
    accepted_records: int
    global_manifest_authorized: bool
    pretraining_authorized: bool


def _valid_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        char in "0123456789abcdef" for char in value
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_derived_corpora(
    path: str | Path,
    *,
    repo_root: str | Path,
    rights_policy: RightsPolicy,
    artifact_ledger: ArtifactLedger,
) -> tuple[DerivedCorpus, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DerivedCorpusError("derived corpus registry must be an object")
    if payload.get("schema_version") != 1:
        raise DerivedCorpusError("unsupported derived corpus registry schema")
    if payload.get("stage") != "L003":
        raise DerivedCorpusError("derived corpus registry stage must be L003")
    raw_items = payload.get("corpora")
    if not isinstance(raw_items, list) or not raw_items:
        raise DerivedCorpusError("corpora must be a non-empty list")

    by_artifact = {
        artifact.artifact_id: artifact
        for artifact in artifact_ledger.artifacts
    }
    result: list[DerivedCorpus] = []
    seen: set[str] = set()
    repo_root = Path(repo_root)

    for raw in raw_items:
        if not isinstance(raw, dict):
            raise DerivedCorpusError("derived corpus entry must be an object")
        required = {
            "derived_id",
            "source_id",
            "artifact_id",
            "rights_lane",
            "primary_category",
            "summary_path",
            "summary_sha256",
            "accepted_storage",
            "accepted_bytes",
            "accepted_sha256",
            "train_tokens",
            "validation_tokens",
            "holdout_tokens",
            "accepted_records",
            "global_manifest_authorized",
            "pretraining_authorized",
        }
        missing = required - set(raw)
        if missing:
            raise DerivedCorpusError(
                "derived corpus missing fields: "
                + ", ".join(sorted(missing))
            )

        derived_id = raw["derived_id"]
        if not isinstance(derived_id, str) or not derived_id.strip():
            raise DerivedCorpusError("derived_id must be non-empty")
        derived_id = derived_id.strip()
        if derived_id in seen:
            raise DerivedCorpusError(
                f"duplicate derived_id {derived_id!r}"
            )
        seen.add(derived_id)

        artifact_id = raw["artifact_id"]
        artifact = by_artifact.get(artifact_id)
        if artifact is None:
            raise DerivedCorpusError(
                f"{derived_id}: unknown artifact_id {artifact_id!r}"
            )

        source_id = raw["source_id"]
        if source_id != artifact.source_id:
            raise DerivedCorpusError(
                f"{derived_id}: source_id does not match raw artifact"
            )
        source_rights = rights_policy.sources.get(source_id)
        if source_rights is None:
            raise DerivedCorpusError(
                f"{derived_id}: source missing from rights policy"
            )
        rights_lane = raw["rights_lane"]
        if rights_lane != artifact.rights_lane:
            raise DerivedCorpusError(
                f"{derived_id}: rights lane does not match raw artifact"
            )
        if rights_lane != source_rights.lane:
            raise DerivedCorpusError(
                f"{derived_id}: rights lane does not match source policy"
            )
        if rights_lane == "blocked":
            raise DerivedCorpusError(
                f"{derived_id}: blocked source cannot have derived corpus"
            )

        primary_category = raw["primary_category"]
        if primary_category not in {
            "general_thai",
            "general_english",
            "technical_scientific_code",
            "flywire_domain",
        }:
            raise DerivedCorpusError(
                f"{derived_id}: invalid primary_category {primary_category!r}"
            )

        summary_path = raw["summary_path"]
        if not isinstance(summary_path, str) or not summary_path.strip():
            raise DerivedCorpusError(
                f"{derived_id}: summary_path must be non-empty"
            )
        summary_file = repo_root / summary_path
        if not summary_file.is_file():
            raise DerivedCorpusError(
                f"{derived_id}: summary file is missing"
            )
        summary_sha = str(raw["summary_sha256"]).lower()
        if not _valid_sha256(summary_sha):
            raise DerivedCorpusError(
                f"{derived_id}: invalid summary SHA-256"
            )
        observed_summary_sha = sha256_file(summary_file)
        if observed_summary_sha != summary_sha:
            raise DerivedCorpusError(
                f"{derived_id}: summary SHA-256 mismatch"
            )

        accepted_sha = str(raw["accepted_sha256"]).lower()
        if not _valid_sha256(accepted_sha):
            raise DerivedCorpusError(
                f"{derived_id}: invalid accepted SHA-256"
            )
        accepted_bytes = raw["accepted_bytes"]
        if not isinstance(accepted_bytes, int) or accepted_bytes <= 0:
            raise DerivedCorpusError(
                f"{derived_id}: accepted_bytes must be positive"
            )

        numeric_fields = (
            "train_tokens",
            "validation_tokens",
            "holdout_tokens",
            "accepted_records",
        )
        numeric: dict[str, int] = {}
        for field in numeric_fields:
            value = raw[field]
            if not isinstance(value, int) or value < 0:
                raise DerivedCorpusError(
                    f"{derived_id}: {field} must be non-negative integer"
                )
            numeric[field] = value

        if raw["global_manifest_authorized"] is not False:
            raise DerivedCorpusError(
                f"{derived_id}: local derived corpus cannot authorize "
                "the global manifest"
            )
        if raw["pretraining_authorized"] is not False:
            raise DerivedCorpusError(
                f"{derived_id}: local derived corpus cannot authorize "
                "pretraining"
            )

        summary = json.loads(summary_file.read_text(encoding="utf-8"))
        if summary.get("derived_id") != derived_id:
            raise DerivedCorpusError(
                f"{derived_id}: summary derived_id mismatch"
            )
        if summary.get("source_id") != source_id:
            raise DerivedCorpusError(
                f"{derived_id}: summary source_id mismatch"
            )
        if summary.get("artifact_id") != artifact_id:
            raise DerivedCorpusError(
                f"{derived_id}: summary artifact_id mismatch"
            )
        if summary.get("rights_lane") != rights_lane:
            raise DerivedCorpusError(
                f"{derived_id}: summary rights lane mismatch"
            )
        if summary.get("global_manifest_authorized") is not False:
            raise DerivedCorpusError(
                f"{derived_id}: summary globally authorizes itself"
            )
        if summary.get("pretraining_authorized") is not False:
            raise DerivedCorpusError(
                f"{derived_id}: summary pretraining flag changed"
            )

        accepted = summary.get("files", {}).get("accepted", {})
        if accepted.get("bytes") != accepted_bytes:
            raise DerivedCorpusError(
                f"{derived_id}: accepted byte count mismatch"
            )
        if accepted.get("sha256") != accepted_sha:
            raise DerivedCorpusError(
                f"{derived_id}: accepted SHA-256 mismatch"
            )
        tokens = summary.get("accepted", {}).get("tokens_by_partition", {})
        if tokens != {
            "train": numeric["train_tokens"],
            "validation": numeric["validation_tokens"],
            "holdout": numeric["holdout_tokens"],
        }:
            raise DerivedCorpusError(
                f"{derived_id}: partition token accounting mismatch"
            )
        if summary.get("counts", {}).get("accept_records") != numeric[
            "accepted_records"
        ]:
            raise DerivedCorpusError(
                f"{derived_id}: accepted record count mismatch"
            )

        accepted_storage = raw["accepted_storage"]
        if not isinstance(accepted_storage, str) or not accepted_storage.strip():
            raise DerivedCorpusError(
                f"{derived_id}: accepted_storage must be non-empty"
            )

        result.append(
            DerivedCorpus(
                derived_id=derived_id,
                source_id=source_id,
                artifact_id=artifact_id,
                rights_lane=rights_lane,
                primary_category=primary_category,
                summary_path=summary_path,
                summary_sha256=summary_sha,
                accepted_storage=accepted_storage.strip(),
                accepted_bytes=accepted_bytes,
                accepted_sha256=accepted_sha,
                train_tokens=numeric["train_tokens"],
                validation_tokens=numeric["validation_tokens"],
                holdout_tokens=numeric["holdout_tokens"],
                accepted_records=numeric["accepted_records"],
                global_manifest_authorized=False,
                pretraining_authorized=False,
            )
        )
    return tuple(result)


def verify_derived_corpus_file(
    corpus: DerivedCorpus,
    *,
    external_root: str | Path,
) -> Path:
    try:
        path = resolve_external_storage_uri(
            corpus.accepted_storage,
            external_root=external_root,
        )
    except ArtifactLedgerError as exc:
        raise DerivedCorpusError(str(exc)) from exc
    if not path.is_file():
        raise DerivedCorpusError(
            f"{corpus.derived_id}: accepted corpus file missing"
        )
    if path.stat().st_size != corpus.accepted_bytes:
        raise DerivedCorpusError(
            f"{corpus.derived_id}: accepted corpus byte size mismatch"
        )
    if sha256_file(path) != corpus.accepted_sha256:
        raise DerivedCorpusError(
            f"{corpus.derived_id}: accepted corpus SHA-256 mismatch"
        )
    return path
