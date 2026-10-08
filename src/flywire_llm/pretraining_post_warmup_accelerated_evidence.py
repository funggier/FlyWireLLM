from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .pretraining_post_warmup_accelerated import (
    PostWarmupAcceleratedState,
    sha256_file,
)


class PretrainingPostWarmupAcceleratedEvidenceError(ValueError):
    pass


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _artifact_path(run_root: Path, relative: object, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise PretrainingPostWarmupAcceleratedEvidenceError(
            f"{label} file is invalid"
        )
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise PretrainingPostWarmupAcceleratedEvidenceError(
            f"{label} file escapes run root"
        )
    root = run_root.resolve()
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise PretrainingPostWarmupAcceleratedEvidenceError(
            f"{label} file escapes run root"
        ) from exc
    return resolved


def load_completed_post_warmup_accelerated_state(
    path: str | Path,
    *,
    authorization: Any,
    verify_files: bool = True,
) -> PostWarmupAcceleratedState:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PretrainingPostWarmupAcceleratedEvidenceError(
            "completed accelerated state must be a JSON object"
        )

    expected_static = {
        "schema_version": 1,
        "stage": "L004",
        "run_id": authorization.authorization_id,
        "authorization_sha256": authorization.authorization_sha256,
        "checkpoint_lane": "research_only",
        "rolling_checkpoint_retention": 1,
        "prune_prior_checkpoint_after_state_commit": True,
        "source_optimizer_step": 297,
        "public_release_eligibility": "not_qualified",
        "automatic_flywiremodel_export_allowed": False,
    }
    for key, expected in expected_static.items():
        if payload.get(key) != expected:
            raise PretrainingPostWarmupAcceleratedEvidenceError(
                f"completed accelerated state {key} mismatch"
            )

    step = payload.get("optimizer_step")
    if step != 425:
        raise PretrainingPostWarmupAcceleratedEvidenceError(
            "completed accelerated state must be optimizer step 425"
        )
    if payload.get("tranche_complete") is not True:
        raise PretrainingPostWarmupAcceleratedEvidenceError(
            "completed accelerated tranche completion flag mismatch"
        )
    supervised = payload.get("supervised_tokens_seen")
    if supervised != 27_852_800:
        raise PretrainingPostWarmupAcceleratedEvidenceError(
            "completed accelerated supervised-token progress mismatch"
        )

    physical = payload.get("physical_input_positions_seen")
    microbatches = payload.get("microbatches_seen")
    if not isinstance(physical, int) or physical < supervised:
        raise PretrainingPostWarmupAcceleratedEvidenceError(
            "completed accelerated physical progress is invalid"
        )
    if not isinstance(microbatches, int) or microbatches <= 0:
        raise PretrainingPostWarmupAcceleratedEvidenceError(
            "completed accelerated microbatch progress is invalid"
        )

    data_state = payload.get("data_state")
    if (
        not isinstance(data_state, dict)
        or data_state.get("supervised_tokens_consumed") != supervised
    ):
        raise PretrainingPostWarmupAcceleratedEvidenceError(
            "completed accelerated data cursor mismatch"
        )
    for key in ("order_position", "block_input_offset", "supervised_tokens_consumed"):
        if not isinstance(data_state.get(key), int) or data_state[key] < 0:
            raise PretrainingPostWarmupAcceleratedEvidenceError(
                f"completed accelerated data cursor {key} is invalid"
            )

    checkpoint = payload.get("checkpoint")
    metric = payload.get("metric")
    if not isinstance(checkpoint, dict) or not isinstance(metric, dict):
        raise PretrainingPostWarmupAcceleratedEvidenceError(
            "completed accelerated artifact metadata is missing"
        )
    checkpoint_file = checkpoint.get("file")
    metric_file = metric.get("file")
    checkpoint_sha = checkpoint.get("sha256")
    metric_sha = metric.get("sha256")
    checkpoint_bytes = checkpoint.get("bytes")
    if not _valid_sha256(checkpoint_sha) or not _valid_sha256(metric_sha):
        raise PretrainingPostWarmupAcceleratedEvidenceError(
            "completed accelerated artifact SHA-256 is invalid"
        )
    if not isinstance(checkpoint_bytes, int) or checkpoint_bytes <= 0:
        raise PretrainingPostWarmupAcceleratedEvidenceError(
            "completed accelerated checkpoint byte size is invalid"
        )

    checkpoint_path = _artifact_path(
        authorization.run_root, checkpoint_file, "checkpoint"
    )
    metric_path = _artifact_path(authorization.run_root, metric_file, "metric")
    if verify_files:
        if not checkpoint_path.is_file():
            raise PretrainingPostWarmupAcceleratedEvidenceError(
                "completed accelerated checkpoint is missing"
            )
        if checkpoint_path.stat().st_size != checkpoint_bytes:
            raise PretrainingPostWarmupAcceleratedEvidenceError(
                "completed accelerated checkpoint byte size mismatch"
            )
        if sha256_file(checkpoint_path) != checkpoint_sha:
            raise PretrainingPostWarmupAcceleratedEvidenceError(
                "completed accelerated checkpoint SHA-256 mismatch"
            )
        if not metric_path.is_file():
            raise PretrainingPostWarmupAcceleratedEvidenceError(
                "completed accelerated metric is missing"
            )
        if sha256_file(metric_path) != metric_sha:
            raise PretrainingPostWarmupAcceleratedEvidenceError(
                "completed accelerated metric SHA-256 mismatch"
            )

    return PostWarmupAcceleratedState(
        authorization_id=authorization.authorization_id,
        authorization_sha256=authorization.authorization_sha256,
        optimizer_step=425,
        supervised_tokens_seen=27_852_800,
        physical_input_positions_seen=physical,
        microbatches_seen=microbatches,
        data_state={key: int(value) for key, value in data_state.items()},
        checkpoint_file=str(checkpoint_file),
        checkpoint_bytes=checkpoint_bytes,
        checkpoint_sha256=str(checkpoint_sha),
        metric_file=str(metric_file),
        metric_sha256=str(metric_sha),
    )
