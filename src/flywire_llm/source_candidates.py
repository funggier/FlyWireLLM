from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .rights import RightsPolicy


class SourceCandidateError(ValueError):
    pass


@dataclass(frozen=True)
class SourceCandidate:
    candidate_id: str
    source_id: str
    rights_lane: str
    dataset_id: str
    name: str
    locale: str
    release_date: str
    license: str
    size_bytes: int
    archive_sha256: str
    filename: str
    dataset_url: str
    access_status: str
    raw_redistribution: str
    ingestion_scope: dict[str, bool]
    privacy_constraints: tuple[str, ...]


def _valid_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        char in "0123456789abcdef" for char in value
    )


def load_source_candidates(
    path: str | Path,
    *,
    rights_policy: RightsPolicy,
) -> tuple[SourceCandidate, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SourceCandidateError("candidate catalog must be an object")
    if payload.get("schema_version") != 1:
        raise SourceCandidateError("unsupported candidate catalog schema")
    if payload.get("stage") != "L003":
        raise SourceCandidateError("candidate catalog stage must be L003")

    raw_candidates = payload.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise SourceCandidateError("candidates must be non-empty")

    candidates: list[SourceCandidate] = []
    seen: set[str] = set()
    for raw in raw_candidates:
        if not isinstance(raw, dict):
            raise SourceCandidateError("candidate must be an object")
        candidate_id = raw.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            raise SourceCandidateError("candidate_id must be non-empty")
        candidate_id = candidate_id.strip()
        if candidate_id in seen:
            raise SourceCandidateError(
                f"duplicate candidate_id {candidate_id!r}"
            )
        seen.add(candidate_id)

        source_id = raw.get("source_id")
        if source_id not in rights_policy.sources:
            raise SourceCandidateError(
                f"{candidate_id}: unknown source_id {source_id!r}"
            )
        rights_lane = raw.get("rights_lane")
        expected_lane = rights_policy.sources[source_id].lane
        if rights_lane != expected_lane:
            raise SourceCandidateError(
                f"{candidate_id}: rights lane {rights_lane!r} "
                f"does not match policy {expected_lane!r}"
            )
        if rights_lane == "blocked":
            raise SourceCandidateError(
                f"{candidate_id}: blocked source cannot be an "
                "acquisition candidate"
            )

        size_bytes = raw.get("size_bytes")
        if not isinstance(size_bytes, int) or size_bytes <= 0:
            raise SourceCandidateError(
                f"{candidate_id}: size_bytes must be positive"
            )
        sha256 = raw.get("archive_sha256")
        if not isinstance(sha256, str) or not _valid_sha256(
            sha256.lower()
        ):
            raise SourceCandidateError(
                f"{candidate_id}: invalid archive SHA-256"
            )

        scope = raw.get("ingestion_scope")
        if not isinstance(scope, dict):
            raise SourceCandidateError(
                f"{candidate_id}: ingestion_scope must be an object"
            )
        privacy = raw.get("privacy_constraints")
        if (
            not isinstance(privacy, list)
            or not privacy
            or not all(isinstance(item, str) and item.strip() for item in privacy)
        ):
            raise SourceCandidateError(
                f"{candidate_id}: privacy_constraints must be non-empty"
            )

        text_fields = (
            "dataset_id",
            "name",
            "locale",
            "release_date",
            "license",
            "filename",
            "dataset_url",
            "access_status",
            "raw_redistribution",
        )
        values: dict[str, str] = {}
        for field in text_fields:
            value = raw.get(field)
            if not isinstance(value, str) or not value.strip():
                raise SourceCandidateError(
                    f"{candidate_id}: {field} must be non-empty"
                )
            values[field] = value.strip()

        if source_id == "mozilla-common-voice":
            _validate_common_voice_scope(
                candidate_id,
                license_name=values["license"],
                raw_redistribution=values["raw_redistribution"],
                scope=scope,
            )

        candidates.append(
            SourceCandidate(
                candidate_id=candidate_id,
                source_id=source_id,
                rights_lane=rights_lane,
                dataset_id=values["dataset_id"],
                name=values["name"],
                locale=values["locale"],
                release_date=values["release_date"],
                license=values["license"],
                size_bytes=size_bytes,
                archive_sha256=sha256.lower(),
                filename=values["filename"],
                dataset_url=values["dataset_url"],
                access_status=values["access_status"],
                raw_redistribution=values["raw_redistribution"],
                ingestion_scope={
                    str(key): bool(value)
                    for key, value in scope.items()
                },
                privacy_constraints=tuple(
                    item.strip() for item in privacy
                ),
            )
        )

    return tuple(candidates)


def _validate_common_voice_scope(
    candidate_id: str,
    *,
    license_name: str,
    raw_redistribution: str,
    scope: dict,
) -> None:
    if license_name != "CC0-1.0":
        raise SourceCandidateError(
            f"{candidate_id}: Common Voice candidate must be CC0-1.0"
        )
    if raw_redistribution != "do_not_mirror":
        raise SourceCandidateError(
            f"{candidate_id}: Common Voice raw archive must not be mirrored"
        )
    expected = {
        "validated_sentences_tsv": True,
        "audio": False,
        "clip_rows": False,
        "client_id": False,
        "speaker_demographics": False,
        "unvalidated_sentences": False,
    }
    if scope != expected:
        raise SourceCandidateError(
            f"{candidate_id}: Common Voice ingestion scope must remain "
            "validated sentence text only"
        )
