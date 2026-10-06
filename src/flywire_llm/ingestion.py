from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any

from .artifacts import (
    ArtifactLedger,
    ArtifactLedgerError,
    SourceArtifact,
    validate_checkpoint_artifacts,
)
from .data_contract import dedup_fingerprint
from .quality import ScreeningDisposition, screen_text, simhash64
from .rights import RightsPolicy


class IngestionError(ValueError):
    pass


@dataclass(frozen=True)
class IngestionCandidate:
    record_id: str
    text: str
    language: str
    category: str
    source_id: str
    artifact_id: str
    source_group_id: str
    partition: str


@dataclass(frozen=True)
class IngestionManifestEntry:
    record_id: str
    language: str
    category: str
    source_id: str
    artifact_id: str
    source_group_id: str
    partition: str
    checkpoint_lane: str
    source_lane: str
    status: str
    content_sha256: str
    dedup_fingerprint: str
    simhash64_hex: str
    characters: int
    utf8_bytes: int
    screening_findings: tuple[str, ...]

    @property
    def training_eligible(self) -> bool:
        return self.status == "accept"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["training_eligible"] = self.training_eligible
        payload["screening_findings"] = list(self.screening_findings)
        return payload


_ALLOWED_LANGUAGES = {"th", "en", "mixed", "other"}
_ALLOWED_CATEGORIES = {
    "general_thai",
    "general_english",
    "technical_scientific_code",
    "flywire_domain",
}
_ALLOWED_PARTITIONS = {"train", "validation", "holdout"}


def _validate_candidate(candidate: IngestionCandidate) -> None:
    text_fields = {
        "record_id": candidate.record_id,
        "text": candidate.text,
        "language": candidate.language,
        "category": candidate.category,
        "source_id": candidate.source_id,
        "artifact_id": candidate.artifact_id,
        "source_group_id": candidate.source_group_id,
        "partition": candidate.partition,
    }
    for field, value in text_fields.items():
        if not isinstance(value, str) or not value.strip():
            raise IngestionError(f"{field} must be a non-empty string")
    if candidate.language not in _ALLOWED_LANGUAGES:
        raise IngestionError(
            f"unsupported language {candidate.language!r}"
        )
    if candidate.category not in _ALLOWED_CATEGORIES:
        raise IngestionError(
            f"unsupported category {candidate.category!r}"
        )
    if candidate.partition not in _ALLOWED_PARTITIONS:
        raise IngestionError(
            f"unsupported partition {candidate.partition!r}"
        )


def evaluate_candidate(
    candidate: IngestionCandidate,
    *,
    rights_policy: RightsPolicy,
    artifact_ledger: ArtifactLedger,
    checkpoint_lane: str,
    min_characters: int = 1,
    max_characters: int = 2_000_000,
) -> IngestionManifestEntry:
    _validate_candidate(candidate)

    try:
        selected = validate_checkpoint_artifacts(
            artifact_ledger,
            rights_policy=rights_policy,
            checkpoint_lane=checkpoint_lane,
            artifact_ids=[candidate.artifact_id],
        )
    except ArtifactLedgerError as exc:
        raise IngestionError(str(exc)) from exc
    artifact: SourceArtifact = selected[0]
    if artifact.source_id != candidate.source_id:
        raise IngestionError(
            f"record source {candidate.source_id!r} does not match "
            f"artifact source {artifact.source_id!r}"
        )

    screening = screen_text(
        candidate.text,
        min_characters=min_characters,
        max_characters=max_characters,
    )
    if screening.disposition == ScreeningDisposition.ACCEPT:
        status = "accept"
    elif screening.disposition == ScreeningDisposition.REVIEW:
        status = "quarantine"
    else:
        status = "reject"

    content_bytes = candidate.text.encode("utf-8")
    content_sha256 = hashlib.sha256(content_bytes).hexdigest()
    simhash = simhash64(candidate.text)

    return IngestionManifestEntry(
        record_id=candidate.record_id,
        language=candidate.language,
        category=candidate.category,
        source_id=candidate.source_id,
        artifact_id=candidate.artifact_id,
        source_group_id=candidate.source_group_id,
        partition=candidate.partition,
        checkpoint_lane=checkpoint_lane,
        source_lane=artifact.rights_lane,
        status=status,
        content_sha256=content_sha256,
        dedup_fingerprint=dedup_fingerprint(candidate.text),
        simhash64_hex=f"{simhash:016x}",
        characters=screening.characters,
        utf8_bytes=screening.utf8_bytes,
        screening_findings=tuple(
            finding.code for finding in screening.findings
        ),
    )
