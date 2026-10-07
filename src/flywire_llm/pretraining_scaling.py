from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import resolve_external_storage_uri


class PretrainingScalingError(ValueError):
    pass


@dataclass(frozen=True)
class ScalingSource:
    result_path: str
    result_sha256: str
    checkpoint: Path
    checkpoint_storage: str
    checkpoint_bytes: int
    checkpoint_sha256: str
    optimizer_step: int
    supervised_tokens_seen: int
    physical_input_positions_seen: int
    microbatches_seen: int
    data_state: dict[str, int]


@dataclass(frozen=True)
class ScalingAuthorization:
    authorization_id: str
    runner_commit: str
    evidence_commit: str
    checkpoint_lane: str
    start_optimizer_step: int
    max_additional_optimizer_updates: int
    end_optimizer_step: int
    target_additional_supervised_tokens: int
    target_cumulative_supervised_tokens: int
    checkpoint_every_updates: int
    source: ScalingSource
    run_root: Path
    run_root_storage: str
    repo_file_pins: tuple[tuple[str, str], ...]
    post_validation_gate: dict[str, Any]
    authorization_sha256: str


@dataclass(frozen=True)
class ScalingState:
    authorization_id: str
    authorization_sha256: str
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
        raise PretrainingScalingError("repo path must be non-empty")
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise PretrainingScalingError("repo path must be relative")
    root = repo_root.resolve()
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise PretrainingScalingError(
            "repo path escapes repository"
        ) from exc
    if not resolved.is_file():
        raise PretrainingScalingError(
            f"repo file is missing: {relative}"
        )
    return resolved


def _parse_repo_pins(
    raw: object,
    *,
    repo_root: Path,
) -> tuple[tuple[str, str], ...]:
    if not isinstance(raw, list) or not raw:
        raise PretrainingScalingError(
            "repo_file_pins must be a non-empty list"
        )
    pins: list[tuple[str, str]] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            raise PretrainingScalingError(
                "repo pin entries must be objects"
            )
        relative = item.get("path")
        expected = item.get("sha256")
        if not isinstance(relative, str) or not relative:
            raise PretrainingScalingError("repo pin path is invalid")
        if relative in seen:
            raise PretrainingScalingError(
                f"duplicate repo pin {relative!r}"
            )
        seen.add(relative)
        if not _valid_sha256(expected):
            raise PretrainingScalingError(
                f"{relative}: repo pin SHA-256 is invalid"
            )
        observed = sha256_file(_repo_file(repo_root, relative))
        if observed != expected:
            raise PretrainingScalingError(
                f"{relative}: SHA-256 mismatch expected={expected} "
                f"observed={observed}"
            )
        pins.append((relative, str(expected)))
    return tuple(pins)


def _parse_source(
    raw: object,
    *,
    repo_root: Path,
    external_root: Path,
) -> ScalingSource:
    if not isinstance(raw, dict):
        raise PretrainingScalingError(
            "source_checkpoint must be an object"
        )
    expected_static = {
        "optimizer_step": 16,
        "supervised_tokens_seen": 1_048_576,
        "physical_input_positions_seen": 1_051_555,
        "microbatches_seen": 1_042,
        "data_state": {
            "block_input_offset": 931,
            "order_position": 1026,
            "supervised_tokens_consumed": 1_048_576,
        },
    }
    for key, expected in expected_static.items():
        if raw.get(key) != expected:
            raise PretrainingScalingError(
                f"fourth tranche source {key} changed"
            )

    result_path = raw.get("result_path")
    result_sha = raw.get("result_sha256")
    if not _valid_sha256(result_sha):
        raise PretrainingScalingError(
            "source result SHA-256 is invalid"
        )
    result = _repo_file(repo_root, result_path)
    if sha256_file(result) != result_sha:
        raise PretrainingScalingError(
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
        raise PretrainingScalingError(
            "source checkpoint byte size is invalid"
        )
    if not _valid_sha256(checkpoint_sha):
        raise PretrainingScalingError(
            "source checkpoint SHA-256 is invalid"
        )
    if checkpoint.stat().st_size != checkpoint_bytes:
        raise PretrainingScalingError(
            "source checkpoint byte size mismatch"
        )
    if sha256_file(checkpoint) != checkpoint_sha:
        raise PretrainingScalingError(
            "source checkpoint SHA-256 mismatch"
        )
    return ScalingSource(
        result_path=str(result_path),
        result_sha256=str(result_sha),
        checkpoint=checkpoint,
        checkpoint_storage=str(storage),
        checkpoint_bytes=checkpoint_bytes,
        checkpoint_sha256=str(checkpoint_sha),
        optimizer_step=16,
        supervised_tokens_seen=1_048_576,
        physical_input_positions_seen=1_051_555,
        microbatches_seen=1_042,
        data_state={
            "block_input_offset": 931,
            "order_position": 1026,
            "supervised_tokens_consumed": 1_048_576,
        },
    )


def load_scaling_authorization(
    path: str | Path,
    *,
    repo_root: str | Path,
    external_root: str | Path,
) -> ScalingAuthorization:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PretrainingScalingError(
            "scaling authorization must be a JSON object"
        )
    expected_static = {
        "schema_version": 1,
        "stage": "L004",
        "authorization_id": "base50m-fourth-tranche-v1",
        "checkpoint_lane": "research_only",
        "start_optimizer_step": 16,
        "max_additional_optimizer_updates": 16,
        "end_optimizer_step": 32,
        "target_additional_supervised_tokens": 1_048_576,
        "target_cumulative_supervised_tokens": 2_097_152,
        "checkpoint_every_updates": 1,
        "pretraining_authorized": True,
        "public_release_eligibility": "not_qualified",
        "automatic_flywiremodel_export_allowed": False,
    }
    for key, expected in expected_static.items():
        if payload.get(key) != expected:
            raise PretrainingScalingError(
                f"scaling authorization {key} changed"
            )
    runner_commit = payload.get("runner_commit")
    evidence_commit = payload.get("evidence_commit")
    if not _valid_git_sha(runner_commit):
        raise PretrainingScalingError(
            "runner_commit must be a full Git SHA"
        )
    if evidence_commit != "73d78d18134f7f27b530eaea2eb4d9733435ad61":
        raise PretrainingScalingError(
            "step16 evidence commit changed"
        )
    source = _parse_source(
        payload.get("source_checkpoint"),
        repo_root=Path(repo_root),
        external_root=Path(external_root),
    )
    run_root_storage = payload.get("run_root")
    run_root = resolve_external_storage_uri(
        run_root_storage,
        external_root=external_root,
    )
    pins = _parse_repo_pins(
        payload.get("repo_file_pins"),
        repo_root=Path(repo_root),
    )
    gate = payload.get("post_tranche_validation_gate")
    expected_gate = {
        "evaluation_id": "base50m-validation-v1",
        "baseline_optimizer_step": 16,
        "candidate_optimizer_step": 32,
        "same_validation_pack_required": True,
        "final_holdout_must_remain_untouched": True,
        "combined_loss_must_not_increase_vs_step16": True,
        "max_category_relative_loss_increase_vs_step16": 0.005,
        "require_all_losses_finite": True,
    }
    if gate != expected_gate:
        raise PretrainingScalingError(
            "fourth-tranche post-validation gate changed"
        )
    return ScalingAuthorization(
        authorization_id="base50m-fourth-tranche-v1",
        runner_commit=str(runner_commit),
        evidence_commit=str(evidence_commit),
        checkpoint_lane="research_only",
        start_optimizer_step=16,
        max_additional_optimizer_updates=16,
        end_optimizer_step=32,
        target_additional_supervised_tokens=1_048_576,
        target_cumulative_supervised_tokens=2_097_152,
        checkpoint_every_updates=1,
        source=source,
        run_root=run_root,
        run_root_storage=str(run_root_storage),
        repo_file_pins=pins,
        post_validation_gate=expected_gate,
        authorization_sha256=sha256_file(path),
    )


def checkpoint_filename(step: int) -> str:
    if step < 17 or step > 32:
        raise PretrainingScalingError(
            "fourth tranche checkpoint step must be in [17, 32]"
        )
    return f"checkpoints/step-{step:06d}.pt"


def metric_filename(step: int) -> str:
    if step < 17 or step > 32:
        raise PretrainingScalingError(
            "fourth tranche metric step must be in [17, 32]"
        )
    return f"metrics/step-{step:06d}.json"


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    if temp.exists():
        temp.unlink()
    try:
        temp.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def build_scaling_state_payload(
    authorization: ScalingAuthorization,
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
    if optimizer_step < 17 or optimizer_step > 32:
        raise PretrainingScalingError(
            "optimizer step is outside fourth tranche"
        )
    expected_tokens = optimizer_step * 65_536
    if supervised_tokens_seen != expected_tokens:
        raise PretrainingScalingError(
            "scaling supervised-token progress mismatch"
        )
    if data_state.get("supervised_tokens_consumed") != expected_tokens:
        raise PretrainingScalingError(
            "scaling data cursor/token mismatch"
        )
    if physical_input_positions_seen < supervised_tokens_seen:
        raise PretrainingScalingError(
            "scaling physical progress is invalid"
        )
    if microbatches_seen <= 0:
        raise PretrainingScalingError(
            "scaling microbatch progress is invalid"
        )
    if not _valid_sha256(checkpoint_sha256):
        raise PretrainingScalingError(
            "scaling checkpoint SHA-256 is invalid"
        )
    if not _valid_sha256(metric_sha256):
        raise PretrainingScalingError(
            "scaling metric SHA-256 is invalid"
        )
    root = authorization.run_root.resolve()
    try:
        checkpoint_relative = checkpoint_path.resolve().relative_to(root)
        metric_relative = metric_path.resolve().relative_to(root)
    except ValueError as exc:
        raise PretrainingScalingError(
            "scaling artifacts must remain in run root"
        ) from exc
    return {
        "schema_version": 1,
        "stage": "L004",
        "run_id": authorization.authorization_id,
        "authorization_sha256": authorization.authorization_sha256,
        "checkpoint_lane": authorization.checkpoint_lane,
        "source_optimizer_step": 16,
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
        "tranche_complete": optimizer_step == 32,
        "public_release_eligibility": "not_qualified",
        "automatic_flywiremodel_export_allowed": False,
    }


def load_scaling_state(
    path: str | Path,
    *,
    authorization: ScalingAuthorization,
    verify_files: bool = True,
) -> ScalingState:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PretrainingScalingError(
            "scaling state must be a JSON object"
        )
    expected_static = {
        "schema_version": 1,
        "stage": "L004",
        "run_id": authorization.authorization_id,
        "authorization_sha256": authorization.authorization_sha256,
        "checkpoint_lane": "research_only",
        "source_optimizer_step": 16,
        "public_release_eligibility": "not_qualified",
        "automatic_flywiremodel_export_allowed": False,
    }
    for key, expected in expected_static.items():
        if payload.get(key) != expected:
            raise PretrainingScalingError(
                f"scaling state {key} mismatch"
            )
    step = payload.get("optimizer_step")
    if not isinstance(step, int) or step < 17 or step > 32:
        raise PretrainingScalingError(
            "scaling state optimizer step is invalid"
        )
    supervised = payload.get("supervised_tokens_seen")
    if supervised != step * 65_536:
        raise PretrainingScalingError(
            "scaling state supervised-token progress mismatch"
        )
    physical = payload.get("physical_input_positions_seen")
    microbatches = payload.get("microbatches_seen")
    if not isinstance(physical, int) or physical < supervised:
        raise PretrainingScalingError(
            "scaling state physical progress is invalid"
        )
    if not isinstance(microbatches, int) or microbatches <= 0:
        raise PretrainingScalingError(
            "scaling state microbatch progress is invalid"
        )
    data_state = payload.get("data_state")
    if (
        not isinstance(data_state, dict)
        or data_state.get("supervised_tokens_consumed") != supervised
    ):
        raise PretrainingScalingError(
            "scaling state data cursor mismatch"
        )
    checkpoint = payload.get("checkpoint")
    metric = payload.get("metric")
    if not isinstance(checkpoint, dict) or not isinstance(metric, dict):
        raise PretrainingScalingError(
            "scaling state artifact metadata is missing"
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
        raise PretrainingScalingError(
            "scaling state artifact metadata is invalid"
        )
    if verify_files:
        checkpoint_path = authorization.run_root / checkpoint_file
        metric_path = authorization.run_root / metric_file
        if checkpoint_path.stat().st_size != checkpoint_bytes:
            raise PretrainingScalingError(
                "scaling checkpoint byte size mismatch"
            )
        if sha256_file(checkpoint_path) != checkpoint_sha:
            raise PretrainingScalingError(
                "scaling checkpoint SHA-256 mismatch"
            )
        if sha256_file(metric_path) != metric_sha:
            raise PretrainingScalingError(
                "scaling metric SHA-256 mismatch"
            )
    if payload.get("tranche_complete") is not (step == 32):
        raise PretrainingScalingError(
            "scaling tranche completion flag mismatch"
        )
    return ScalingState(
        authorization_id=authorization.authorization_id,
        authorization_sha256=authorization.authorization_sha256,
        optimizer_step=step,
        supervised_tokens_seen=supervised,
        physical_input_positions_seen=int(physical),
        microbatches_seen=int(microbatches),
        data_state={key: int(value) for key, value in data_state.items()},
        checkpoint_file=checkpoint_file,
        checkpoint_bytes=checkpoint_bytes,
        checkpoint_sha256=str(checkpoint_sha),
        metric_file=metric_file,
        metric_sha256=str(metric_sha),
    )
