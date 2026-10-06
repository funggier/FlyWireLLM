from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import resolve_external_storage_uri


class PretrainingContinuationError(ValueError):
    pass


@dataclass(frozen=True)
class ContinuationRepoPin:
    path: str
    sha256: str


@dataclass(frozen=True)
class ContinuationSource:
    result_path: str
    result_sha256: str
    checkpoint: Path
    checkpoint_storage: str
    checkpoint_bytes: int
    checkpoint_sha256: str
    optimizer_step: int
    supervised_tokens_seen: int
    data_state: dict[str, int]


@dataclass(frozen=True)
class ContinuationAuthorization:
    authorization_id: str
    foundation_commit: str
    runner_commit: str
    validation_gate_commit: str
    checkpoint_lane: str
    start_optimizer_step: int
    max_additional_optimizer_updates: int
    end_optimizer_step: int
    target_additional_supervised_tokens: int
    target_cumulative_supervised_tokens: int
    checkpoint_every_updates: int
    source: ContinuationSource
    run_root: Path
    run_root_storage: str
    repo_file_pins: tuple[ContinuationRepoPin, ...]
    authorization_sha256: str


@dataclass(frozen=True)
class ContinuationState:
    authorization_id: str
    authorization_sha256: str
    checkpoint_lane: str
    optimizer_step: int
    supervised_tokens_seen: int
    physical_input_positions_seen: int
    microbatches_seen: int
    data_state: dict[str, int]
    checkpoint_file: str
    checkpoint_bytes: int
    checkpoint_sha256: str
    metric_file: str
    metric_sha256: str


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _valid_git_sha(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(char in "0123456789abcdef" for char in value)
    )


def _repo_file(repo_root: Path, relative: object) -> Path:
    if not isinstance(relative, str) or not relative:
        raise PretrainingContinuationError("repo path must be non-empty")
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise PretrainingContinuationError("repo path must be relative")
    root = repo_root.resolve()
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise PretrainingContinuationError(
            "repo path escapes repository"
        ) from exc
    if not resolved.is_file():
        raise PretrainingContinuationError(
            f"repo file is missing: {relative}"
        )
    return resolved


def _parse_repo_pins(
    raw: object,
    *,
    repo_root: Path,
) -> tuple[ContinuationRepoPin, ...]:
    if not isinstance(raw, list) or not raw:
        raise PretrainingContinuationError(
            "repo_file_pins must be a non-empty list"
        )
    pins: list[ContinuationRepoPin] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            raise PretrainingContinuationError(
                "repo_file_pins entries must be objects"
            )
        relative = item.get("path")
        expected = item.get("sha256")
        if not isinstance(relative, str) or not relative:
            raise PretrainingContinuationError(
                "repo pin path is invalid"
            )
        if relative in seen:
            raise PretrainingContinuationError(
                f"duplicate repo pin {relative!r}"
            )
        seen.add(relative)
        if not _valid_sha256(expected):
            raise PretrainingContinuationError(
                f"{relative}: invalid repo pin SHA-256"
            )
        path = _repo_file(repo_root, relative)
        observed = sha256_file(path)
        if observed != expected:
            raise PretrainingContinuationError(
                f"{relative}: SHA-256 mismatch expected={expected} "
                f"observed={observed}"
            )
        pins.append(
            ContinuationRepoPin(path=relative, sha256=expected)
        )
    return tuple(pins)


def _parse_source(
    raw: object,
    *,
    repo_root: Path,
    external_root: Path,
) -> ContinuationSource:
    if not isinstance(raw, dict):
        raise PretrainingContinuationError(
            "source_checkpoint must be an object"
        )
    result_path = raw.get("result_path")
    result_sha = raw.get("result_sha256")
    if not _valid_sha256(result_sha):
        raise PretrainingContinuationError(
            "source result SHA-256 is invalid"
        )
    result = _repo_file(repo_root, result_path)
    if sha256_file(result) != result_sha:
        raise PretrainingContinuationError(
            "source result SHA-256 mismatch"
        )
    storage = raw.get("checkpoint")
    checkpoint = resolve_external_storage_uri(
        storage,
        external_root=external_root,
    )
    checkpoint_bytes = raw.get("checkpoint_bytes")
    checkpoint_sha = raw.get("checkpoint_sha256")
    if not isinstance(checkpoint_bytes, int) or checkpoint_bytes <= 0:
        raise PretrainingContinuationError(
            "source checkpoint byte size is invalid"
        )
    if not _valid_sha256(checkpoint_sha):
        raise PretrainingContinuationError(
            "source checkpoint SHA-256 is invalid"
        )
    if not checkpoint.is_file():
        raise PretrainingContinuationError(
            "source checkpoint is missing"
        )
    if checkpoint.stat().st_size != checkpoint_bytes:
        raise PretrainingContinuationError(
            "source checkpoint byte size mismatch"
        )
    if sha256_file(checkpoint) != checkpoint_sha:
        raise PretrainingContinuationError(
            "source checkpoint SHA-256 mismatch"
        )
    if raw.get("optimizer_step") != 4:
        raise PretrainingContinuationError(
            "second tranche source must be optimizer step 4"
        )
    if raw.get("supervised_tokens_seen") != 262_144:
        raise PretrainingContinuationError(
            "second tranche source token progress changed"
        )
    data_state = raw.get("data_state")
    expected_data_state = {
        "block_input_offset": 767,
        "order_position": 256,
        "supervised_tokens_consumed": 262_144,
    }
    if data_state != expected_data_state:
        raise PretrainingContinuationError(
            "second tranche source data cursor changed"
        )
    return ContinuationSource(
        result_path=str(result_path),
        result_sha256=str(result_sha),
        checkpoint=checkpoint,
        checkpoint_storage=str(storage),
        checkpoint_bytes=checkpoint_bytes,
        checkpoint_sha256=str(checkpoint_sha),
        optimizer_step=4,
        supervised_tokens_seen=262_144,
        data_state={
            key: int(value)
            for key, value in expected_data_state.items()
        },
    )


def load_continuation_authorization(
    path: str | Path,
    *,
    repo_root: str | Path,
    external_root: str | Path,
) -> ContinuationAuthorization:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PretrainingContinuationError(
            "continuation authorization must be a JSON object"
        )
    expected_static = {
        "schema_version": 1,
        "stage": "L004",
        "authorization_id": "base50m-second-tranche-v1",
        "checkpoint_lane": "research_only",
        "start_optimizer_step": 4,
        "max_additional_optimizer_updates": 4,
        "end_optimizer_step": 8,
        "target_additional_supervised_tokens": 262_144,
        "target_cumulative_supervised_tokens": 524_288,
        "checkpoint_every_updates": 1,
        "pretraining_authorized": True,
        "public_release_eligibility": "not_qualified",
        "automatic_flywiremodel_export_allowed": False,
    }
    for key, expected in expected_static.items():
        if payload.get(key) != expected:
            raise PretrainingContinuationError(
                f"continuation authorization {key} changed"
            )
    foundation = payload.get("foundation_commit")
    runner = payload.get("runner_commit")
    validation = payload.get("validation_gate_commit")
    for name, value in (
        ("foundation_commit", foundation),
        ("runner_commit", runner),
        ("validation_gate_commit", validation),
    ):
        if not _valid_git_sha(value):
            raise PretrainingContinuationError(
                f"{name} must be a full Git SHA"
            )
    if validation != "2151b1199346bbc26da944f731b01b84556e3ed3":
        raise PretrainingContinuationError(
            "second tranche validation gate commit changed"
        )
    root = Path(repo_root)
    ext = Path(external_root)
    source = _parse_source(
        payload.get("source_checkpoint"),
        repo_root=root,
        external_root=ext,
    )
    run_root_storage = payload.get("run_root")
    run_root = resolve_external_storage_uri(
        run_root_storage,
        external_root=ext,
    )
    pins = _parse_repo_pins(
        payload.get("repo_file_pins"),
        repo_root=root,
    )
    return ContinuationAuthorization(
        authorization_id="base50m-second-tranche-v1",
        foundation_commit=str(foundation),
        runner_commit=str(runner),
        validation_gate_commit=str(validation),
        checkpoint_lane="research_only",
        start_optimizer_step=4,
        max_additional_optimizer_updates=4,
        end_optimizer_step=8,
        target_additional_supervised_tokens=262_144,
        target_cumulative_supervised_tokens=524_288,
        checkpoint_every_updates=1,
        source=source,
        run_root=run_root,
        run_root_storage=str(run_root_storage),
        repo_file_pins=pins,
        authorization_sha256=sha256_file(path),
    )


def checkpoint_filename(step: int) -> str:
    if step < 5 or step > 8:
        raise PretrainingContinuationError(
            "second tranche checkpoint step must be in [5, 8]"
        )
    return f"checkpoints/step-{step:06d}.pt"


def metric_filename(step: int) -> str:
    if step < 5 or step > 8:
        raise PretrainingContinuationError(
            "second tranche metric step must be in [5, 8]"
        )
    return f"metrics/step-{step:06d}.json"


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if temp.exists():
        temp.unlink()
    try:
        temp.write_text(
            rendered,
            encoding="utf-8",
            newline="\n",
        )
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def build_continuation_state_payload(
    authorization: ContinuationAuthorization,
    *,
    optimizer_step: int,
    supervised_tokens_seen: int,
    physical_input_positions_seen: int,
    microbatches_seen: int,
    data_state: dict[str, int],
    checkpoint_path: Path,
    checkpoint_sha256: str,
    metric_path: Path,
    metric_sha256: str,
) -> dict[str, Any]:
    if optimizer_step < 5 or optimizer_step > 8:
        raise PretrainingContinuationError(
            "optimizer step is outside second tranche"
        )
    expected_tokens = optimizer_step * 65_536
    if supervised_tokens_seen != expected_tokens:
        raise PretrainingContinuationError(
            "continuation supervised-token progress mismatch"
        )
    if data_state.get("supervised_tokens_consumed") != expected_tokens:
        raise PretrainingContinuationError(
            "continuation data cursor/token mismatch"
        )
    if physical_input_positions_seen < supervised_tokens_seen:
        raise PretrainingContinuationError(
            "continuation physical progress is invalid"
        )
    if microbatches_seen <= 0:
        raise PretrainingContinuationError(
            "continuation microbatch progress is invalid"
        )
    if not _valid_sha256(checkpoint_sha256):
        raise PretrainingContinuationError(
            "continuation checkpoint SHA-256 is invalid"
        )
    if not _valid_sha256(metric_sha256):
        raise PretrainingContinuationError(
            "continuation metric SHA-256 is invalid"
        )
    run_root = authorization.run_root.resolve()
    try:
        checkpoint_relative = checkpoint_path.resolve().relative_to(run_root)
        metric_relative = metric_path.resolve().relative_to(run_root)
    except ValueError as exc:
        raise PretrainingContinuationError(
            "continuation artifacts must remain in run root"
        ) from exc
    return {
        "schema_version": 1,
        "stage": "L004",
        "run_id": authorization.authorization_id,
        "authorization_sha256": authorization.authorization_sha256,
        "checkpoint_lane": authorization.checkpoint_lane,
        "source_optimizer_step": authorization.start_optimizer_step,
        "optimizer_step": optimizer_step,
        "supervised_tokens_seen": supervised_tokens_seen,
        "physical_input_positions_seen": physical_input_positions_seen,
        "microbatches_seen": microbatches_seen,
        "data_state": data_state,
        "checkpoint": {
            "file": checkpoint_relative.as_posix(),
            "bytes": checkpoint_path.stat().st_size,
            "sha256": checkpoint_sha256,
        },
        "metric": {
            "file": metric_relative.as_posix(),
            "sha256": metric_sha256,
        },
        "tranche_complete": optimizer_step == 8,
        "public_release_eligibility": "not_qualified",
        "automatic_flywiremodel_export_allowed": False,
    }


def load_continuation_state(
    path: str | Path,
    *,
    authorization: ContinuationAuthorization,
    verify_files: bool = True,
) -> ContinuationState:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PretrainingContinuationError(
            "continuation state must be a JSON object"
        )
    expected_static = {
        "schema_version": 1,
        "stage": "L004",
        "run_id": authorization.authorization_id,
        "authorization_sha256": authorization.authorization_sha256,
        "checkpoint_lane": "research_only",
        "source_optimizer_step": 4,
        "public_release_eligibility": "not_qualified",
        "automatic_flywiremodel_export_allowed": False,
    }
    for key, expected in expected_static.items():
        if payload.get(key) != expected:
            raise PretrainingContinuationError(
                f"continuation state {key} mismatch"
            )
    step = payload.get("optimizer_step")
    if not isinstance(step, int) or step < 5 or step > 8:
        raise PretrainingContinuationError(
            "continuation state optimizer step is invalid"
        )
    supervised = payload.get("supervised_tokens_seen")
    if supervised != step * 65_536:
        raise PretrainingContinuationError(
            "continuation state token progress mismatch"
        )
    physical = payload.get("physical_input_positions_seen")
    microbatches = payload.get("microbatches_seen")
    if not isinstance(physical, int) or physical < supervised:
        raise PretrainingContinuationError(
            "continuation state physical progress is invalid"
        )
    if not isinstance(microbatches, int) or microbatches <= 0:
        raise PretrainingContinuationError(
            "continuation state microbatch progress is invalid"
        )
    data_state = payload.get("data_state")
    if (
        not isinstance(data_state, dict)
        or data_state.get("supervised_tokens_consumed") != supervised
    ):
        raise PretrainingContinuationError(
            "continuation state data cursor mismatch"
        )
    checkpoint = payload.get("checkpoint")
    metric = payload.get("metric")
    if not isinstance(checkpoint, dict) or not isinstance(metric, dict):
        raise PretrainingContinuationError(
            "continuation state artifact metadata is missing"
        )
    checkpoint_file = checkpoint.get("file")
    metric_file = metric.get("file")
    checkpoint_sha = checkpoint.get("sha256")
    metric_sha = metric.get("sha256")
    checkpoint_bytes = checkpoint.get("bytes")
    if (
        not isinstance(checkpoint_file, str)
        or not isinstance(metric_file, str)
        or not _valid_sha256(checkpoint_sha)
        or not _valid_sha256(metric_sha)
        or not isinstance(checkpoint_bytes, int)
        or checkpoint_bytes <= 0
    ):
        raise PretrainingContinuationError(
            "continuation state artifact metadata is invalid"
        )
    if verify_files:
        checkpoint_path = authorization.run_root / checkpoint_file
        metric_path = authorization.run_root / metric_file
        if checkpoint_path.stat().st_size != checkpoint_bytes:
            raise PretrainingContinuationError(
                "continuation checkpoint byte size mismatch"
            )
        if sha256_file(checkpoint_path) != checkpoint_sha:
            raise PretrainingContinuationError(
                "continuation checkpoint SHA-256 mismatch"
            )
        if sha256_file(metric_path) != metric_sha:
            raise PretrainingContinuationError(
                "continuation metric SHA-256 mismatch"
            )
    if payload.get("tranche_complete") is not (step == 8):
        raise PretrainingContinuationError(
            "continuation tranche completion flag mismatch"
        )
    return ContinuationState(
        authorization_id=authorization.authorization_id,
        authorization_sha256=authorization.authorization_sha256,
        checkpoint_lane="research_only",
        optimizer_step=step,
        supervised_tokens_seen=supervised,
        physical_input_positions_seen=physical,
        microbatches_seen=microbatches,
        data_state={key: int(value) for key, value in data_state.items()},
        checkpoint_file=checkpoint_file,
        checkpoint_bytes=checkpoint_bytes,
        checkpoint_sha256=str(checkpoint_sha),
        metric_file=metric_file,
        metric_sha256=str(metric_sha),
    )
