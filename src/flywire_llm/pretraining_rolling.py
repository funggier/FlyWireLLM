from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import resolve_external_storage_uri


class PretrainingRollingError(ValueError):
    pass


@dataclass(frozen=True)
class RollingSource:
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
class RollingAuthorization:
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
    source: RollingSource
    run_root: Path
    run_root_storage: str
    repo_file_pins: tuple[tuple[str, str], ...]
    post_validation_gate: dict[str, Any]
    authorization_sha256: str


@dataclass(frozen=True)
class RollingState:
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
        raise PretrainingRollingError("repo path must be non-empty")
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise PretrainingRollingError("repo path must be relative")
    root = repo_root.resolve()
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise PretrainingRollingError(
            "repo path escapes repository"
        ) from exc
    if not resolved.is_file():
        raise PretrainingRollingError(
            f"repo file is missing: {relative}"
        )
    return resolved


def _parse_repo_pins(
    raw: object,
    *,
    repo_root: Path,
) -> tuple[tuple[str, str], ...]:
    if not isinstance(raw, list) or not raw:
        raise PretrainingRollingError(
            "repo_file_pins must be a non-empty list"
        )
    pins: list[tuple[str, str]] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            raise PretrainingRollingError(
                "repo pin entries must be objects"
            )
        relative = item.get("path")
        expected = item.get("sha256")
        if not isinstance(relative, str) or not relative:
            raise PretrainingRollingError("repo pin path is invalid")
        if relative in seen:
            raise PretrainingRollingError(
                f"duplicate repo pin {relative!r}"
            )
        seen.add(relative)
        if not _valid_sha256(expected):
            raise PretrainingRollingError(
                f"{relative}: repo pin SHA-256 is invalid"
            )
        observed = sha256_file(_repo_file(repo_root, relative))
        if observed != expected:
            raise PretrainingRollingError(
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
) -> RollingSource:
    if not isinstance(raw, dict):
        raise PretrainingRollingError(
            "source_checkpoint must be an object"
        )
    expected_static = {
        "optimizer_step": 32,
        "supervised_tokens_seen": 2_097_152,
        "physical_input_positions_seen": 2_103_242,
        "microbatches_seen": 2_085,
        "data_state": {
            "block_input_offset": 970,
            "order_position": 2053,
            "supervised_tokens_consumed": 2_097_152,
        },
    }
    for key, expected in expected_static.items():
        if raw.get(key) != expected:
            raise PretrainingRollingError(
                f"fifth tranche source {key} changed"
            )

    result_path = raw.get("result_path")
    result_sha = raw.get("result_sha256")
    if not _valid_sha256(result_sha):
        raise PretrainingRollingError(
            "source result SHA-256 is invalid"
        )
    result = _repo_file(repo_root, result_path)
    if sha256_file(result) != result_sha:
        raise PretrainingRollingError(
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
        raise PretrainingRollingError(
            "source checkpoint byte size is invalid"
        )
    if not _valid_sha256(checkpoint_sha):
        raise PretrainingRollingError(
            "source checkpoint SHA-256 is invalid"
        )
    if checkpoint.stat().st_size != checkpoint_bytes:
        raise PretrainingRollingError(
            "source checkpoint byte size mismatch"
        )
    if sha256_file(checkpoint) != checkpoint_sha:
        raise PretrainingRollingError(
            "source checkpoint SHA-256 mismatch"
        )
    return RollingSource(
        result_path=str(result_path),
        result_sha256=str(result_sha),
        checkpoint=checkpoint,
        checkpoint_storage=str(storage),
        checkpoint_bytes=checkpoint_bytes,
        checkpoint_sha256=str(checkpoint_sha),
        optimizer_step=32,
        supervised_tokens_seen=2_097_152,
        physical_input_positions_seen=2_103_242,
        microbatches_seen=2_085,
        data_state={
            "block_input_offset": 970,
            "order_position": 2053,
            "supervised_tokens_consumed": 2_097_152,
        },
    )


def load_rolling_authorization(
    path: str | Path,
    *,
    repo_root: str | Path,
    external_root: str | Path,
) -> RollingAuthorization:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PretrainingRollingError(
            "rolling authorization must be a JSON object"
        )
    expected_static = {
        "schema_version": 1,
        "stage": "L004",
        "authorization_id": "base50m-fifth-tranche-v1",
        "checkpoint_lane": "research_only",
        "start_optimizer_step": 32,
        "max_additional_optimizer_updates": 8,
        "end_optimizer_step": 40,
        "target_additional_supervised_tokens": 524_288,
        "target_cumulative_supervised_tokens": 2_621_440,
        "checkpoint_every_updates": 1,
        "rolling_checkpoint_retention": 1,
        "prune_prior_checkpoint_after_state_commit": True,
        "pretraining_authorized": True,
        "public_release_eligibility": "not_qualified",
        "automatic_flywiremodel_export_allowed": False,
    }
    for key, expected in expected_static.items():
        if payload.get(key) != expected:
            raise PretrainingRollingError(
                f"rolling authorization {key} changed"
            )
    runner_commit = payload.get("runner_commit")
    evidence_commit = payload.get("evidence_commit")
    if not _valid_git_sha(runner_commit):
        raise PretrainingRollingError(
            "runner_commit must be a full Git SHA"
        )
    if evidence_commit != "8fa74f72c9ae025b64728b1d6717eab3a7452136":
        raise PretrainingRollingError(
            "step32 evidence commit changed"
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
        "baseline_optimizer_step": 32,
        "candidate_optimizer_step": 40,
        "same_validation_pack_required": True,
        "final_holdout_must_remain_untouched": True,
        "combined_loss_must_not_increase_vs_step32": True,
        "max_category_relative_loss_increase_vs_step32": 0.005,
        "require_all_losses_finite": True,
    }
    if gate != expected_gate:
        raise PretrainingRollingError(
            "fifth-tranche post-validation gate changed"
        )
    return RollingAuthorization(
        authorization_id="base50m-fifth-tranche-v1",
        runner_commit=str(runner_commit),
        evidence_commit=str(evidence_commit),
        checkpoint_lane="research_only",
        start_optimizer_step=32,
        max_additional_optimizer_updates=8,
        end_optimizer_step=40,
        target_additional_supervised_tokens=524_288,
        target_cumulative_supervised_tokens=2_621_440,
        checkpoint_every_updates=1,
        source=source,
        run_root=run_root,
        run_root_storage=str(run_root_storage),
        repo_file_pins=pins,
        post_validation_gate=expected_gate,
        authorization_sha256=sha256_file(path),
    )


def checkpoint_filename(step: int) -> str:
    if step < 33 or step > 40:
        raise PretrainingRollingError(
            "fifth tranche checkpoint step must be in [33, 40]"
        )
    return f"checkpoints/step-{step:06d}.pt"


def metric_filename(step: int) -> str:
    if step < 33 or step > 40:
        raise PretrainingRollingError(
            "fifth tranche metric step must be in [33, 40]"
        )
    return f"metrics/step-{step:06d}.json"


def prune_prior_checkpoint(
    run_root: str | Path,
    previous_checkpoint: str | Path,
    current_checkpoint: str | Path,
) -> bool:
    root = Path(run_root).resolve()
    previous = Path(previous_checkpoint).resolve()
    current = Path(current_checkpoint).resolve()
    try:
        previous.relative_to(root)
        current.relative_to(root)
    except ValueError as exc:
        raise PretrainingRollingError(
            "rolling checkpoint prune target escapes run root"
        ) from exc
    if previous == current:
        return False
    if previous.parent.name != "checkpoints":
        raise PretrainingRollingError(
            "rolling checkpoint prune target must be in checkpoints"
        )
    if previous.is_file():
        previous.unlink()
        return True
    return False


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


def build_rolling_state_payload(
    authorization: RollingAuthorization,
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
    if optimizer_step < 33 or optimizer_step > 40:
        raise PretrainingRollingError(
            "optimizer step is outside fifth tranche"
        )
    expected_tokens = optimizer_step * 65_536
    if supervised_tokens_seen != expected_tokens:
        raise PretrainingRollingError(
            "rolling supervised-token progress mismatch"
        )
    if data_state.get("supervised_tokens_consumed") != expected_tokens:
        raise PretrainingRollingError(
            "rolling data cursor/token mismatch"
        )
    if physical_input_positions_seen < supervised_tokens_seen:
        raise PretrainingRollingError(
            "rolling physical progress is invalid"
        )
    if microbatches_seen <= 0:
        raise PretrainingRollingError(
            "rolling microbatch progress is invalid"
        )
    if not _valid_sha256(checkpoint_sha256):
        raise PretrainingRollingError(
            "rolling checkpoint SHA-256 is invalid"
        )
    if not _valid_sha256(metric_sha256):
        raise PretrainingRollingError(
            "rolling metric SHA-256 is invalid"
        )
    root = authorization.run_root.resolve()
    try:
        checkpoint_relative = checkpoint_path.resolve().relative_to(root)
        metric_relative = metric_path.resolve().relative_to(root)
    except ValueError as exc:
        raise PretrainingRollingError(
            "rolling artifacts must remain in run root"
        ) from exc
    return {
        "schema_version": 1,
        "stage": "L004",
        "run_id": authorization.authorization_id,
        "authorization_sha256": authorization.authorization_sha256,
        "checkpoint_lane": authorization.checkpoint_lane,
        "rolling_checkpoint_retention": 1,
        "prune_prior_checkpoint_after_state_commit": True,
        "source_optimizer_step": 32,
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
        "tranche_complete": optimizer_step == 40,
        "public_release_eligibility": "not_qualified",
        "automatic_flywiremodel_export_allowed": False,
    }


def load_rolling_state(
    path: str | Path,
    *,
    authorization: RollingAuthorization,
    verify_files: bool = True,
) -> RollingState:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PretrainingRollingError(
            "rolling state must be a JSON object"
        )
    expected_static = {
        "schema_version": 1,
        "stage": "L004",
        "run_id": authorization.authorization_id,
        "authorization_sha256": authorization.authorization_sha256,
        "checkpoint_lane": "research_only",
        "rolling_checkpoint_retention": 1,
        "prune_prior_checkpoint_after_state_commit": True,
        "source_optimizer_step": 32,
        "public_release_eligibility": "not_qualified",
        "automatic_flywiremodel_export_allowed": False,
    }
    for key, expected in expected_static.items():
        if payload.get(key) != expected:
            raise PretrainingRollingError(
                f"rolling state {key} mismatch"
            )
    step = payload.get("optimizer_step")
    if not isinstance(step, int) or step < 33 or step > 40:
        raise PretrainingRollingError(
            "rolling state optimizer step is invalid"
        )
    supervised = payload.get("supervised_tokens_seen")
    if supervised != step * 65_536:
        raise PretrainingRollingError(
            "rolling state supervised-token progress mismatch"
        )
    physical = payload.get("physical_input_positions_seen")
    microbatches = payload.get("microbatches_seen")
    if not isinstance(physical, int) or physical < supervised:
        raise PretrainingRollingError(
            "rolling state physical progress is invalid"
        )
    if not isinstance(microbatches, int) or microbatches <= 0:
        raise PretrainingRollingError(
            "rolling state microbatch progress is invalid"
        )
    data_state = payload.get("data_state")
    if (
        not isinstance(data_state, dict)
        or data_state.get("supervised_tokens_consumed") != supervised
    ):
        raise PretrainingRollingError(
            "rolling state data cursor mismatch"
        )
    checkpoint = payload.get("checkpoint")
    metric = payload.get("metric")
    if not isinstance(checkpoint, dict) or not isinstance(metric, dict):
        raise PretrainingRollingError(
            "rolling state artifact metadata is missing"
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
        raise PretrainingRollingError(
            "rolling state artifact metadata is invalid"
        )
    if verify_files:
        checkpoint_path = authorization.run_root / checkpoint_file
        metric_path = authorization.run_root / metric_file
        if checkpoint_path.stat().st_size != checkpoint_bytes:
            raise PretrainingRollingError(
                "rolling checkpoint byte size mismatch"
            )
        if sha256_file(checkpoint_path) != checkpoint_sha:
            raise PretrainingRollingError(
                "rolling checkpoint SHA-256 mismatch"
            )
        if sha256_file(metric_path) != metric_sha:
            raise PretrainingRollingError(
                "rolling metric SHA-256 mismatch"
            )
    if payload.get("tranche_complete") is not (step == 40):
        raise PretrainingRollingError(
            "scaling tranche completion flag mismatch"
        )
    return RollingState(
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
