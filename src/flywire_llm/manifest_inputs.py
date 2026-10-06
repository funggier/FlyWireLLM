from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .artifacts import ArtifactLedger, resolve_external_storage_uri
from .rights import RightsPolicy


class ManifestInputError(ValueError):
    pass


@dataclass(frozen=True)
class GlobalManifestInput:
    input_id: str
    source_id: str
    artifact_id: str
    rights_lane: str
    language: str
    summary_path: str
    summary_sha256: str
    accepted_storage: str
    accepted_bytes: int
    accepted_sha256: str


@dataclass(frozen=True)
class GlobalManifestInputPlan:
    tokenizer: str
    technical_classifier: str
    near_duplicate_hamming_distance: int
    inputs: tuple[GlobalManifestInput, ...]
    pretraining_authorized: bool


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        char in "0123456789abcdef" for char in value
    )


def load_global_manifest_inputs(
    path: str | Path,
    *,
    repo_root: str | Path,
    rights_policy: RightsPolicy,
    artifact_ledger: ArtifactLedger,
) -> GlobalManifestInputPlan:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ManifestInputError("global manifest input plan must be an object")
    if payload.get("schema_version") != 1:
        raise ManifestInputError("unsupported global manifest input schema")
    if payload.get("stage") != "L003":
        raise ManifestInputError("global manifest input stage must be L003")
    if payload.get("tokenizer") != "base50m-unigram-32000-v1":
        raise ManifestInputError("global manifest tokenizer changed")
    if payload.get("technical_classifier") != "technical-heuristic-v5":
        raise ManifestInputError("global manifest technical classifier changed")
    if payload.get("near_duplicate_hamming_distance") != 3:
        raise ManifestInputError(
            "near-duplicate Hamming distance must remain 3"
        )
    if payload.get("central_global_manifest_authorization") is not True:
        raise ManifestInputError(
            "central global manifest authorization must be explicit"
        )
    if payload.get("pretraining_authorized") is not False:
        raise ManifestInputError(
            "input plan cannot authorize pretraining"
        )

    raw_inputs = payload.get("inputs")
    if not isinstance(raw_inputs, list) or not raw_inputs:
        raise ManifestInputError("global manifest inputs must be non-empty")

    repo_root = Path(repo_root)
    artifacts = {
        artifact.artifact_id: artifact
        for artifact in artifact_ledger.artifacts
    }
    seen: set[str] = set()
    seen_artifacts: set[str] = set()
    result: list[GlobalManifestInput] = []

    for raw in raw_inputs:
        if not isinstance(raw, dict):
            raise ManifestInputError("manifest input must be an object")
        required = {
            "input_id",
            "source_id",
            "artifact_id",
            "rights_lane",
            "language",
            "summary_path",
            "summary_sha256",
            "accepted_storage",
            "accepted_bytes",
            "accepted_sha256",
        }
        if set(raw) != required:
            missing = required - set(raw)
            extra = set(raw) - required
            raise ManifestInputError(
                "manifest input field set changed "
                f"missing={sorted(missing)} extra={sorted(extra)}"
            )

        input_id = raw["input_id"]
        if not isinstance(input_id, str) or not input_id.strip():
            raise ManifestInputError("input_id must be non-empty")
        input_id = input_id.strip()
        if input_id in seen:
            raise ManifestInputError(
                f"duplicate input_id {input_id!r}"
            )
        seen.add(input_id)

        source_id = raw["source_id"]
        rights = rights_policy.sources.get(source_id)
        if rights is None:
            raise ManifestInputError(
                f"{input_id}: unknown source_id {source_id!r}"
            )
        artifact_id = raw["artifact_id"]
        if artifact_id in seen_artifacts:
            raise ManifestInputError(
                f"{input_id}: duplicate artifact_id {artifact_id!r}"
            )
        seen_artifacts.add(artifact_id)
        artifact = artifacts.get(artifact_id)
        if artifact is None:
            raise ManifestInputError(
                f"{input_id}: unknown artifact_id {artifact_id!r}"
            )
        if artifact.source_id != source_id:
            raise ManifestInputError(
                f"{input_id}: artifact/source mismatch"
            )

        rights_lane = raw["rights_lane"]
        if rights_lane != rights.lane or rights_lane != artifact.rights_lane:
            raise ManifestInputError(
                f"{input_id}: rights lane mismatch"
            )
        if rights_lane not in {"release_safe", "research_only"}:
            raise ManifestInputError(
                f"{input_id}: blocked source cannot enter global manifest"
            )

        language = raw["language"]
        if language not in {"th", "en"}:
            raise ManifestInputError(
                f"{input_id}: language must be th or en"
            )

        summary_path = raw["summary_path"]
        if not isinstance(summary_path, str) or not summary_path.strip():
            raise ManifestInputError(
                f"{input_id}: summary_path must be non-empty"
            )
        summary_file = repo_root / summary_path
        if not summary_file.is_file():
            raise ManifestInputError(
                f"{input_id}: summary file is missing"
            )
        summary_sha = str(raw["summary_sha256"]).lower()
        if not _valid_sha256(summary_sha):
            raise ManifestInputError(
                f"{input_id}: invalid summary SHA-256"
            )
        if _sha256_file(summary_file) != summary_sha:
            raise ManifestInputError(
                f"{input_id}: summary SHA-256 mismatch"
            )

        summary = json.loads(summary_file.read_text(encoding="utf-8"))
        if summary.get("source_id") != source_id:
            raise ManifestInputError(
                f"{input_id}: summary source_id mismatch"
            )
        if summary.get("artifact_id") != artifact_id:
            raise ManifestInputError(
                f"{input_id}: summary artifact_id mismatch"
            )
        if summary.get("rights_lane") != rights_lane:
            raise ManifestInputError(
                f"{input_id}: summary rights_lane mismatch"
            )
        if summary.get("global_manifest_authorized") is not False:
            raise ManifestInputError(
                f"{input_id}: source summary cannot self-authorize globally"
            )
        if summary.get("pretraining_authorized") is not False:
            raise ManifestInputError(
                f"{input_id}: source summary cannot authorize pretraining"
            )

        accepted = summary.get("files", {}).get("accepted", {})
        accepted_storage = raw["accepted_storage"]
        if (
            not isinstance(accepted_storage, str)
            or not accepted_storage.startswith("external://")
        ):
            raise ManifestInputError(
                f"{input_id}: accepted storage must be external://"
            )
        if accepted.get("storage", accepted_storage) != accepted_storage:
            raise ManifestInputError(
                f"{input_id}: accepted storage mismatch"
            )

        accepted_bytes = raw["accepted_bytes"]
        if not isinstance(accepted_bytes, int) or accepted_bytes <= 0:
            raise ManifestInputError(
                f"{input_id}: accepted_bytes must be positive"
            )
        if accepted.get("bytes") != accepted_bytes:
            raise ManifestInputError(
                f"{input_id}: accepted byte count mismatch"
            )

        accepted_sha = str(raw["accepted_sha256"]).lower()
        if not _valid_sha256(accepted_sha):
            raise ManifestInputError(
                f"{input_id}: invalid accepted SHA-256"
            )
        if accepted.get("sha256") != accepted_sha:
            raise ManifestInputError(
                f"{input_id}: accepted SHA-256 mismatch"
            )

        result.append(
            GlobalManifestInput(
                input_id=input_id,
                source_id=source_id,
                artifact_id=artifact_id,
                rights_lane=rights_lane,
                language=language,
                summary_path=summary_path,
                summary_sha256=summary_sha,
                accepted_storage=accepted_storage,
                accepted_bytes=accepted_bytes,
                accepted_sha256=accepted_sha,
            )
        )

    return GlobalManifestInputPlan(
        tokenizer="base50m-unigram-32000-v1",
        technical_classifier="technical-heuristic-v5",
        near_duplicate_hamming_distance=3,
        inputs=tuple(result),
        pretraining_authorized=False,
    )


def verify_global_manifest_input_file(
    item: GlobalManifestInput,
    *,
    external_root: str | Path,
) -> Path:
    path = resolve_external_storage_uri(
        item.accepted_storage,
        external_root=external_root,
    )
    if not path.is_file():
        raise ManifestInputError(
            f"{item.input_id}: accepted corpus file is missing"
        )
    if path.stat().st_size != item.accepted_bytes:
        raise ManifestInputError(
            f"{item.input_id}: accepted byte size mismatch"
        )
    if _sha256_file(path) != item.accepted_sha256:
        raise ManifestInputError(
            f"{item.input_id}: accepted SHA-256 mismatch"
        )
    return path
