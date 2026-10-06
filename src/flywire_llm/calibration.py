from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .rights import RightsPolicy


class CalibrationArtifactError(ValueError):
    pass


@dataclass(frozen=True)
class CalibrationArtifact:
    calibration_id: str
    source_id: str
    rights_lane: str
    dataset_repo: str
    source_path: str
    source_split: str
    purpose: str
    never_train: bool
    never_validation: bool
    never_holdout: bool
    bytes: int
    sha256: str
    acquisition_method: str
    local_storage: str
    deletion_after_calibration: bool
    notes: str
    report_path: str
    report_sha256: str
    observed_rows: int
    observed_tokens: int
    accepted_tokens: int
    accepted_token_fraction: float
    accepted_tokens_per_compressed_gib: float


def load_calibration_artifacts(
    path: str | Path,
    *,
    rights_policy: RightsPolicy,
) -> tuple[CalibrationArtifact, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CalibrationArtifactError(
            "calibration catalog must be a JSON object"
        )
    if payload.get("schema_version") != 1:
        raise CalibrationArtifactError(
            "unsupported calibration catalog schema"
        )
    if payload.get("stage") != "L003":
        raise CalibrationArtifactError(
            "calibration catalog stage must be L003"
        )
    raw_items = payload.get("artifacts")
    if not isinstance(raw_items, list) or not raw_items:
        raise CalibrationArtifactError(
            "calibration artifacts must be non-empty"
        )

    result: list[CalibrationArtifact] = []
    seen: set[str] = set()
    for raw in raw_items:
        if not isinstance(raw, dict):
            raise CalibrationArtifactError(
                "calibration artifact must be an object"
            )
        calibration_id = raw.get("calibration_id")
        if not isinstance(calibration_id, str) or not calibration_id.strip():
            raise CalibrationArtifactError(
                "calibration_id must be non-empty"
            )
        calibration_id = calibration_id.strip()
        if calibration_id in seen:
            raise CalibrationArtifactError(
                f"duplicate calibration_id {calibration_id!r}"
            )
        seen.add(calibration_id)

        source_id = raw.get("source_id")
        if source_id not in rights_policy.sources:
            raise CalibrationArtifactError(
                f"{calibration_id}: unknown source_id {source_id!r}"
            )
        rights_lane = raw.get("rights_lane")
        if rights_lane != rights_policy.sources[source_id].lane:
            raise CalibrationArtifactError(
                f"{calibration_id}: rights lane does not match policy"
            )
        if rights_lane == "blocked":
            raise CalibrationArtifactError(
                f"{calibration_id}: blocked source cannot be calibrated"
            )

        for field in ("never_train", "never_validation", "never_holdout"):
            if raw.get(field) is not True:
                raise CalibrationArtifactError(
                    f"{calibration_id}: {field} must be true"
                )

        source_split = raw.get("source_split")
        if not isinstance(source_split, str) or not source_split.strip():
            raise CalibrationArtifactError(
                f"{calibration_id}: source_split must be non-empty"
            )
        if source_split == "test" and raw.get("never_train") is not True:
            raise CalibrationArtifactError(
                f"{calibration_id}: test split can never enter training"
            )

        size = raw.get("bytes")
        if not isinstance(size, int) or size <= 0:
            raise CalibrationArtifactError(
                f"{calibration_id}: bytes must be positive"
            )
        sha256 = raw.get("sha256")
        if (
            not isinstance(sha256, str)
            or len(sha256) != 64
            or any(c not in "0123456789abcdef" for c in sha256.lower())
        ):
            raise CalibrationArtifactError(
                f"{calibration_id}: invalid SHA-256"
            )

        text_fields = (
            "dataset_repo",
            "source_path",
            "purpose",
            "acquisition_method",
            "local_storage",
            "notes",
            "report_path",
        )
        values: dict[str, str] = {}
        for field in text_fields:
            value = raw.get(field)
            if not isinstance(value, str) or not value.strip():
                raise CalibrationArtifactError(
                    f"{calibration_id}: {field} must be non-empty"
                )
            values[field] = value.strip()

        deletion = raw.get("deletion_after_calibration")
        if not isinstance(deletion, bool):
            raise CalibrationArtifactError(
                f"{calibration_id}: deletion_after_calibration must be bool"
            )
        report_sha256 = raw.get("report_sha256")
        if (
            not isinstance(report_sha256, str)
            or len(report_sha256) != 64
            or any(
                c not in "0123456789abcdef"
                for c in report_sha256.lower()
            )
        ):
            raise CalibrationArtifactError(
                f"{calibration_id}: invalid report SHA-256"
            )
        integer_metrics = (
            "observed_rows",
            "observed_tokens",
            "accepted_tokens",
        )
        for field in integer_metrics:
            if not isinstance(raw.get(field), int) or raw[field] < 0:
                raise CalibrationArtifactError(
                    f"{calibration_id}: {field} must be non-negative int"
                )
        accepted_fraction = raw.get("accepted_token_fraction")
        density = raw.get("accepted_tokens_per_compressed_gib")
        if not isinstance(accepted_fraction, (int, float)) or not (
            0.0 <= float(accepted_fraction) <= 1.0
        ):
            raise CalibrationArtifactError(
                f"{calibration_id}: invalid accepted_token_fraction"
            )
        if not isinstance(density, (int, float)) or float(density) <= 0:
            raise CalibrationArtifactError(
                f"{calibration_id}: invalid token density"
            )
        if raw["accepted_tokens"] > raw["observed_tokens"]:
            raise CalibrationArtifactError(
                f"{calibration_id}: accepted tokens exceed observed tokens"
            )

        result.append(
            CalibrationArtifact(
                calibration_id=calibration_id,
                source_id=source_id,
                rights_lane=rights_lane,
                dataset_repo=values["dataset_repo"],
                source_path=values["source_path"],
                source_split=source_split.strip(),
                purpose=values["purpose"],
                never_train=True,
                never_validation=True,
                never_holdout=True,
                bytes=size,
                sha256=sha256.lower(),
                acquisition_method=values["acquisition_method"],
                local_storage=values["local_storage"],
                deletion_after_calibration=deletion,
                notes=values["notes"],
                report_path=values["report_path"],
                report_sha256=report_sha256.lower(),
                observed_rows=raw["observed_rows"],
                observed_tokens=raw["observed_tokens"],
                accepted_tokens=raw["accepted_tokens"],
                accepted_token_fraction=float(accepted_fraction),
                accepted_tokens_per_compressed_gib=float(density),
            )
        )
    return tuple(result)


def verify_calibration_file(
    artifact: CalibrationArtifact,
    path: str | Path,
) -> None:
    path = Path(path)
    if not path.is_file():
        raise CalibrationArtifactError(
            f"{artifact.calibration_id}: file does not exist"
        )
    if path.stat().st_size != artifact.bytes:
        raise CalibrationArtifactError(
            f"{artifact.calibration_id}: byte size mismatch"
        )
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != artifact.sha256:
        raise CalibrationArtifactError(
            f"{artifact.calibration_id}: SHA-256 mismatch"
        )


def assert_not_training_artifact(
    artifact: CalibrationArtifact,
) -> None:
    if not (
        artifact.never_train
        and artifact.never_validation
        and artifact.never_holdout
    ):
        raise CalibrationArtifactError(
            "calibration artifact partition isolation failed"
        )
    raise CalibrationArtifactError(
        f"{artifact.calibration_id}: calibration artifacts can never be "
        "materialized into FlyWireLLM model partitions"
    )
