from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import resolve_external_storage_uri


class PretrainingPostWarmupLongrunError(ValueError):
    pass


@dataclass(frozen=True)
class PostWarmupLongrunSource:
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
class PostWarmupLongrunAuthorization:
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
    source: PostWarmupLongrunSource
    run_root: Path
    run_root_storage: str
    repo_file_pins: tuple[tuple[str, str], ...]
    post_validation_gate: dict[str, Any]
    authorization_sha256: str


@dataclass(frozen=True)
class PostWarmupLongrunState:
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
        raise PretrainingPostWarmupLongrunError("repo path must be non-empty")
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise PretrainingPostWarmupLongrunError("repo path must be relative")
    root = repo_root.resolve()
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise PretrainingPostWarmupLongrunError(
            "repo path escapes repository"
        ) from exc
    if not resolved.is_file():
        raise PretrainingPostWarmupLongrunError(
            f"repo file is missing: {relative}"
        )
    return resolved


def _parse_repo_pins(
    raw: object,
    *,
    repo_root: Path,
) -> tuple[tuple[str, str], ...]:
    if not isinstance(raw, list) or not raw:
        raise PretrainingPostWarmupLongrunError(
            "repo_file_pins must be a non-empty list"
        )
    pins: list[tuple[str, str]] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            raise PretrainingPostWarmupLongrunError(
                "repo pin entries must be objects"
            )
        relative = item.get("path")
        expected = item.get("sha256")
        if not isinstance(relative, str) or not relative:
            raise PretrainingPostWarmupLongrunError("repo pin path is invalid")
        if relative in seen:
            raise PretrainingPostWarmupLongrunError(
                f"duplicate repo pin {relative!r}"
            )
        seen.add(relative)
        if not _valid_sha256(expected):
            raise PretrainingPostWarmupLongrunError(
                f"{relative}: repo pin SHA-256 is invalid"
            )
        observed = sha256_file(_repo_file(repo_root, relative))
        if observed != expected:
            raise PretrainingPostWarmupLongrunError(
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
) -> PostWarmupLongrunSource:
    if not isinstance(raw, dict):
        raise PretrainingPostWarmupLongrunError(
            "source_checkpoint must be an object"
        )
    expected_static = {
        "optimizer_step": 1193,
        "supervised_tokens_seen": 78_184_448,
        "physical_input_positions_seen": 78_394_399,
        "microbatches_seen": 77_749,
        "data_state": {
            "block_input_offset": 31,
            "order_position": 76_557,
            "supervised_tokens_consumed": 78_184_448,
        },
    }
    for key, expected in expected_static.items():
        if raw.get(key) != expected:
            raise PretrainingPostWarmupLongrunError(
                f"post-warmup longrun tranche source {key} changed"
            )

    result_path = raw.get("result_path")
    result_sha = raw.get("result_sha256")
    if not _valid_sha256(result_sha):
        raise PretrainingPostWarmupLongrunError(
            "source result SHA-256 is invalid"
        )
    result = _repo_file(repo_root, result_path)
    if sha256_file(result) != result_sha:
        raise PretrainingPostWarmupLongrunError(
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
        raise PretrainingPostWarmupLongrunError(
            "source checkpoint byte size is invalid"
        )
    if not _valid_sha256(checkpoint_sha):
        raise PretrainingPostWarmupLongrunError(
            "source checkpoint SHA-256 is invalid"
        )
    if checkpoint.stat().st_size != checkpoint_bytes:
        raise PretrainingPostWarmupLongrunError(
            "source checkpoint byte size mismatch"
        )
    if sha256_file(checkpoint) != checkpoint_sha:
        raise PretrainingPostWarmupLongrunError(
            "source checkpoint SHA-256 mismatch"
        )
    return PostWarmupLongrunSource(
        result_path=str(result_path),
        result_sha256=str(result_sha),
        checkpoint=checkpoint,
        checkpoint_storage=str(storage),
        checkpoint_bytes=checkpoint_bytes,
        checkpoint_sha256=str(checkpoint_sha),
        optimizer_step=1193,
        supervised_tokens_seen=78_184_448,
        physical_input_positions_seen=78_394_399,
        microbatches_seen=77_749,
        data_state={
            "block_input_offset": 31,
            "order_position": 76_557,
            "supervised_tokens_consumed": 78_184_448,
        },
    )


def load_post_warmup_longrun_authorization(
    path: str | Path,
    *,
    repo_root: str | Path,
    external_root: str | Path,
) -> PostWarmupLongrunAuthorization:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PretrainingPostWarmupLongrunError(
            "rolling authorization must be a JSON object"
        )
    expected_static = {
        "schema_version": 1,
        "stage": "L004",
        "authorization_id": "base50m-post-warmup-8-v1",
        "checkpoint_lane": "research_only",
        "start_optimizer_step": 1193,
        "max_additional_optimizer_updates": 1024,
        "end_optimizer_step": 2217,
        "target_additional_supervised_tokens": 67_108_864,
        "target_cumulative_supervised_tokens": 145_293_312,
        "checkpoint_every_updates": 1,
        "rolling_checkpoint_retention": 1,
        "prune_prior_checkpoint_after_state_commit": True,
        "pretraining_authorized": True,
        "public_release_eligibility": "not_qualified",
        "automatic_flywiremodel_export_allowed": False,
    }
    for key, expected in expected_static.items():
        if payload.get(key) != expected:
            raise PretrainingPostWarmupLongrunError(
                f"rolling authorization {key} changed"
            )
    runner_commit = payload.get("runner_commit")
    evidence_commit = payload.get("evidence_commit")
    if not _valid_git_sha(runner_commit):
        raise PretrainingPostWarmupLongrunError(
            "runner_commit must be a full Git SHA"
        )
    if evidence_commit != "4d2ba99e9af6048de55c90de6f44217988079115":
        raise PretrainingPostWarmupLongrunError(
            "step1193 evidence commit changed"
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
        "baseline_optimizer_step": 1193,
        "candidate_optimizer_step": 2217,
        "same_validation_pack_required": True,
        "final_holdout_must_remain_untouched": True,
        "combined_loss_must_not_increase_vs_step1193": True,
        "max_category_relative_loss_increase_vs_step1193": 0.005,
        "require_all_losses_finite": True,
    }
    if gate != expected_gate:
        raise PretrainingPostWarmupLongrunError(
            "post-warmup post-validation gate changed"
        )
    return PostWarmupLongrunAuthorization(
        authorization_id="base50m-post-warmup-8-v1",
        runner_commit=str(runner_commit),
        evidence_commit=str(evidence_commit),
        checkpoint_lane="research_only",
        start_optimizer_step=1193,
        max_additional_optimizer_updates=1024,
        end_optimizer_step=2217,
        target_additional_supervised_tokens=67_108_864,
        target_cumulative_supervised_tokens=145_293_312,
        checkpoint_every_updates=1,
        source=source,
        run_root=run_root,
        run_root_storage=str(run_root_storage),
        repo_file_pins=pins,
        post_validation_gate=expected_gate,
        authorization_sha256=sha256_file(path),
    )


def checkpoint_filename(step: int) -> str:
    if step < 1194 or step > 2217:
        raise PretrainingPostWarmupLongrunError(
            "post-warmup longrun tranche checkpoint step must be in [1194, 2217]"
        )
    return f"checkpoints/step-{step:06d}.pt"


def metric_filename(step: int) -> str:
    if step < 1194 or step > 2217:
        raise PretrainingPostWarmupLongrunError(
            "post-warmup longrun tranche metric step must be in [1194, 2217]"
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
        raise PretrainingPostWarmupLongrunError(
            "rolling checkpoint prune target escapes run root"
        ) from exc
    if previous == current:
        return False
    if previous.parent.name != "checkpoints":
        raise PretrainingPostWarmupLongrunError(
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


def build_post_warmup_longrun_state_payload(
    authorization: PostWarmupLongrunAuthorization,
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
    if optimizer_step < 1194 or optimizer_step > 2217:
        raise PretrainingPostWarmupLongrunError(
            "optimizer step is outside post-warmup longrun tranche"
        )
    expected_tokens = optimizer_step * 65_536
    if supervised_tokens_seen != expected_tokens:
        raise PretrainingPostWarmupLongrunError(
            "rolling supervised-token progress mismatch"
        )
    if data_state.get("supervised_tokens_consumed") != expected_tokens:
        raise PretrainingPostWarmupLongrunError(
            "rolling data cursor/token mismatch"
        )
    if physical_input_positions_seen < supervised_tokens_seen:
        raise PretrainingPostWarmupLongrunError(
            "rolling physical progress is invalid"
        )
    if microbatches_seen <= 0:
        raise PretrainingPostWarmupLongrunError(
            "rolling microbatch progress is invalid"
        )
    if not _valid_sha256(checkpoint_sha256):
        raise PretrainingPostWarmupLongrunError(
            "rolling checkpoint SHA-256 is invalid"
        )
    if not _valid_sha256(metric_sha256):
        raise PretrainingPostWarmupLongrunError(
            "rolling metric SHA-256 is invalid"
        )
    root = authorization.run_root.resolve()
    try:
        checkpoint_relative = checkpoint_path.resolve().relative_to(root)
        metric_relative = metric_path.resolve().relative_to(root)
    except ValueError as exc:
        raise PretrainingPostWarmupLongrunError(
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
        "source_optimizer_step": 1193,
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
        "tranche_complete": optimizer_step == 2217,
        "public_release_eligibility": "not_qualified",
        "automatic_flywiremodel_export_allowed": False,
    }


def load_post_warmup_longrun_state(
    path: str | Path,
    *,
    authorization: PostWarmupLongrunAuthorization,
    verify_files: bool = True,
) -> PostWarmupLongrunState:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PretrainingPostWarmupLongrunError(
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
        "source_optimizer_step": 1193,
        "public_release_eligibility": "not_qualified",
        "automatic_flywiremodel_export_allowed": False,
    }
    for key, expected in expected_static.items():
        if payload.get(key) != expected:
            raise PretrainingPostWarmupLongrunError(
                f"rolling state {key} mismatch"
            )
    step = payload.get("optimizer_step")
    if not isinstance(step, int) or step < 1194 or step > 2217:
        raise PretrainingPostWarmupLongrunError(
            "rolling state optimizer step is invalid"
        )
    supervised = payload.get("supervised_tokens_seen")
    if supervised != step * 65_536:
        raise PretrainingPostWarmupLongrunError(
            "rolling state supervised-token progress mismatch"
        )
    physical = payload.get("physical_input_positions_seen")
    microbatches = payload.get("microbatches_seen")
    if not isinstance(physical, int) or physical < supervised:
        raise PretrainingPostWarmupLongrunError(
            "rolling state physical progress is invalid"
        )
    if not isinstance(microbatches, int) or microbatches <= 0:
        raise PretrainingPostWarmupLongrunError(
            "rolling state microbatch progress is invalid"
        )
    data_state = payload.get("data_state")
    if (
        not isinstance(data_state, dict)
        or data_state.get("supervised_tokens_consumed") != supervised
    ):
        raise PretrainingPostWarmupLongrunError(
            "rolling state data cursor mismatch"
        )
    checkpoint = payload.get("checkpoint")
    metric = payload.get("metric")
    if not isinstance(checkpoint, dict) or not isinstance(metric, dict):
        raise PretrainingPostWarmupLongrunError(
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
        raise PretrainingPostWarmupLongrunError(
            "rolling state artifact metadata is invalid"
        )
    if verify_files:
        checkpoint_path = authorization.run_root / checkpoint_file
        metric_path = authorization.run_root / metric_file
        if checkpoint_path.stat().st_size != checkpoint_bytes:
            raise PretrainingPostWarmupLongrunError(
                "rolling checkpoint byte size mismatch"
            )
        if sha256_file(checkpoint_path) != checkpoint_sha:
            raise PretrainingPostWarmupLongrunError(
                "rolling checkpoint SHA-256 mismatch"
            )
        if sha256_file(metric_path) != metric_sha:
            raise PretrainingPostWarmupLongrunError(
                "rolling metric SHA-256 mismatch"
            )
    if payload.get("tranche_complete") is not (step == 2217):
        raise PretrainingPostWarmupLongrunError(
            "longrun tranche completion flag mismatch"
        )
    return PostWarmupLongrunState(
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
