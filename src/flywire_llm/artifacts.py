from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .rights import RightsPolicy, RightsPolicyError


class ArtifactLedgerError(ValueError):
    pass


@dataclass(frozen=True)
class SourceArtifact:
    artifact_id: str
    source_id: str
    snapshot_date: str
    artifact_role: str
    origin_url: str
    acquisition_method: str
    sha256: str
    bytes: int
    rights_lane: str
    raw_redistribution: str
    local_storage: str
    immutable: bool


@dataclass(frozen=True)
class ArtifactLedger:
    artifacts: tuple[SourceArtifact, ...]


_ALLOWED_ROLES = {
    "raw_source",
    "raw_source_subset",
    "derived_corpus",
    "tokenizer_artifact",
    "metadata",
}
_ALLOWED_REDISTRIBUTION = {
    "allowed_with_notice",
    "allowed_with_attribution",
    "do_not_mirror",
    "not_qualified",
    "internal_only",
}
_ALLOWED_LANES = {"release_safe", "research_only", "blocked"}


def _valid_sha256(value: str) -> bool:
    if len(value) != 64:
        return False
    return all(char in "0123456789abcdef" for char in value)


def load_artifact_ledger(
    path: str | Path,
    *,
    rights_policy: RightsPolicy,
) -> ArtifactLedger:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ArtifactLedgerError("artifact ledger must be a JSON object")
    if payload.get("schema_version") != 1:
        raise ArtifactLedgerError("unsupported artifact ledger schema")
    if payload.get("stage") != "L003":
        raise ArtifactLedgerError("artifact ledger stage must be L003")

    raw_artifacts = payload.get("artifacts")
    if not isinstance(raw_artifacts, list) or not raw_artifacts:
        raise ArtifactLedgerError("artifacts must be a non-empty list")

    artifacts: list[SourceArtifact] = []
    seen_ids: set[str] = set()
    for raw in raw_artifacts:
        if not isinstance(raw, dict):
            raise ArtifactLedgerError("artifact must be an object")
        required = {
            "artifact_id",
            "source_id",
            "snapshot_date",
            "artifact_role",
            "origin_url",
            "acquisition_method",
            "sha256",
            "bytes",
            "rights_lane",
            "raw_redistribution",
            "local_storage",
            "immutable",
        }
        missing = required - set(raw)
        if missing:
            raise ArtifactLedgerError(
                "artifact missing fields: " + ", ".join(sorted(missing))
            )

        artifact_id = raw["artifact_id"]
        if not isinstance(artifact_id, str) or not artifact_id.strip():
            raise ArtifactLedgerError("artifact_id must be non-empty")
        artifact_id = artifact_id.strip()
        if artifact_id in seen_ids:
            raise ArtifactLedgerError(
                f"duplicate artifact_id {artifact_id!r}"
            )
        seen_ids.add(artifact_id)

        source_id = raw["source_id"]
        if not isinstance(source_id, str) or source_id not in rights_policy.sources:
            raise ArtifactLedgerError(
                f"{artifact_id}: unknown source_id {source_id!r}"
            )

        rights_lane = raw["rights_lane"]
        if rights_lane not in _ALLOWED_LANES:
            raise ArtifactLedgerError(
                f"{artifact_id}: invalid rights_lane {rights_lane!r}"
            )
        expected_lane = rights_policy.sources[source_id].lane
        if rights_lane != expected_lane:
            raise ArtifactLedgerError(
                f"{artifact_id}: rights lane {rights_lane!r} does not "
                f"match source policy {expected_lane!r}"
            )

        artifact_role = raw["artifact_role"]
        if artifact_role not in _ALLOWED_ROLES:
            raise ArtifactLedgerError(
                f"{artifact_id}: invalid artifact_role {artifact_role!r}"
            )
        redistribution = raw["raw_redistribution"]
        if redistribution not in _ALLOWED_REDISTRIBUTION:
            raise ArtifactLedgerError(
                f"{artifact_id}: invalid raw_redistribution "
                f"{redistribution!r}"
            )
        if rights_lane == "blocked" and redistribution not in {
            "not_qualified",
            "internal_only",
        }:
            raise ArtifactLedgerError(
                f"{artifact_id}: blocked source cannot be redistributable"
            )

        sha256 = raw["sha256"]
        if not isinstance(sha256, str):
            raise ArtifactLedgerError(
                f"{artifact_id}: sha256 must be a string"
            )
        sha256 = sha256.lower()
        if not _valid_sha256(sha256):
            raise ArtifactLedgerError(
                f"{artifact_id}: invalid SHA-256"
            )

        byte_count = raw["bytes"]
        if not isinstance(byte_count, int) or byte_count <= 0:
            raise ArtifactLedgerError(
                f"{artifact_id}: bytes must be a positive integer"
            )
        if raw["immutable"] is not True:
            raise ArtifactLedgerError(
                f"{artifact_id}: acquired artifacts must be immutable"
            )

        text_fields = (
            "snapshot_date",
            "origin_url",
            "acquisition_method",
            "local_storage",
        )
        for field in text_fields:
            value = raw[field]
            if not isinstance(value, str) or not value.strip():
                raise ArtifactLedgerError(
                    f"{artifact_id}: {field} must be non-empty"
                )

        artifacts.append(
            SourceArtifact(
                artifact_id=artifact_id,
                source_id=source_id,
                snapshot_date=raw["snapshot_date"].strip(),
                artifact_role=artifact_role,
                origin_url=raw["origin_url"].strip(),
                acquisition_method=raw["acquisition_method"].strip(),
                sha256=sha256,
                bytes=byte_count,
                rights_lane=rights_lane,
                raw_redistribution=redistribution,
                local_storage=raw["local_storage"].strip(),
                immutable=True,
            )
        )

    return ArtifactLedger(artifacts=tuple(artifacts))


def validate_checkpoint_artifacts(
    ledger: ArtifactLedger,
    *,
    rights_policy: RightsPolicy,
    checkpoint_lane: str,
    artifact_ids: Iterable[str],
) -> tuple[SourceArtifact, ...]:
    by_id = {artifact.artifact_id: artifact for artifact in ledger.artifacts}
    artifact_ids = tuple(artifact_ids)
    if not artifact_ids:
        raise ArtifactLedgerError("artifact_ids must not be empty")
    if len(artifact_ids) != len(set(artifact_ids)):
        raise ArtifactLedgerError("artifact_ids must be unique")

    selected: list[SourceArtifact] = []
    for artifact_id in artifact_ids:
        artifact = by_id.get(artifact_id)
        if artifact is None:
            raise ArtifactLedgerError(
                f"unknown artifact_id {artifact_id!r}"
            )
        try:
            from .rights import validate_checkpoint_sources

            validate_checkpoint_sources(
                rights_policy,
                checkpoint_lane=checkpoint_lane,
                source_ids=[artifact.source_id],
            )
        except RightsPolicyError as exc:
            raise ArtifactLedgerError(str(exc)) from exc
        selected.append(artifact)
    return tuple(selected)

def verify_artifact_file(
    artifact: SourceArtifact,
    path: str | Path,
) -> None:
    import hashlib

    path = Path(path)
    if not path.is_file():
        raise ArtifactLedgerError(
            f"{artifact.artifact_id}: artifact file does not exist: {path}"
        )
    observed_bytes = path.stat().st_size
    if observed_bytes != artifact.bytes:
        raise ArtifactLedgerError(
            f"{artifact.artifact_id}: byte size mismatch "
            f"expected={artifact.bytes} observed={observed_bytes}"
        )
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    observed_sha256 = digest.hexdigest()
    if observed_sha256 != artifact.sha256:
        raise ArtifactLedgerError(
            f"{artifact.artifact_id}: SHA-256 mismatch "
            f"expected={artifact.sha256} observed={observed_sha256}"
        )

def resolve_external_storage_uri(
    uri: str,
    *,
    external_root: str | Path,
    namespace: str = "FlyWireLLM-data",
) -> Path:
    prefix = f"external://{namespace}/"
    if not isinstance(uri, str) or not uri.startswith(prefix):
        raise ArtifactLedgerError(
            f"external storage URI must start with {prefix!r}"
        )
    relative = uri[len(prefix):]
    parts = tuple(part for part in relative.split("/") if part)
    if not parts or any(part in {".", ".."} for part in parts):
        raise ArtifactLedgerError("unsafe external storage URI")
    root = Path(external_root).resolve()
    resolved = root.joinpath(*parts).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ArtifactLedgerError(
            "external storage URI escapes external_root"
        ) from exc
    return resolved
