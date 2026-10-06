from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import resolve_external_storage_uri


class PretrainingTrancheError(ValueError):
    pass


@dataclass(frozen=True)
class RepoFilePin:
    path: str
    sha256: str


@dataclass(frozen=True)
class BoundedTrancheAuthorization:
    authorization_id: str
    foundation_commit: str
    runner_commit: str
    checkpoint_lane: str
    start_optimizer_step: int
    max_optimizer_updates: int
    target_supervised_tokens: int
    checkpoint_every_updates: int
    run_root: Path
    run_root_storage: str
    repo_file_pins: tuple[RepoFilePin, ...]
    authorization_sha256: str


@dataclass(frozen=True)
class TrancheState:
    authorization_id: str
    authorization_sha256: str
    foundation_commit: str
    runner_commit: str
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


def _repo_file(
    repo_root: Path,
    relative: object,
    *,
    field: str,
) -> Path:
    if not isinstance(relative, str) or not relative:
        raise PretrainingTrancheError(f"{field} path must be non-empty")
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise PretrainingTrancheError(f"{field} path must be repo-relative")
    root = repo_root.resolve()
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise PretrainingTrancheError(
            f"{field} path escapes repository"
        ) from exc
    if not resolved.is_file():
        raise PretrainingTrancheError(f"{field} file is missing")
    return resolved


def _parse_repo_pins(
    raw: object,
    *,
    repo_root: Path,
) -> tuple[RepoFilePin, ...]:
    if not isinstance(raw, list) or not raw:
        raise PretrainingTrancheError(
            "repo_file_pins must be a non-empty list"
        )
    pins: list[RepoFilePin] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise PretrainingTrancheError(
                f"repo_file_pins[{index}] must be an object"
            )
        relative = item.get("path")
        expected = item.get("sha256")
        if not isinstance(relative, str) or not relative:
            raise PretrainingTrancheError(
                f"repo_file_pins[{index}] path is invalid"
            )
        if relative in seen:
            raise PretrainingTrancheError(
                f"duplicate repo file pin {relative!r}"
            )
        seen.add(relative)
        if not _valid_sha256(expected):
            raise PretrainingTrancheError(
                f"{relative}: invalid pinned SHA-256"
            )
        path = _repo_file(
            repo_root,
            relative,
            field=f"repo_file_pins[{index}]",
        )
        observed = sha256_file(path)
        if observed != expected:
            raise PretrainingTrancheError(
                f"{relative}: SHA-256 mismatch expected={expected} "
                f"observed={observed}"
            )
        pins.append(
            RepoFilePin(
                path=relative,
                sha256=expected,
            )
        )
    return tuple(pins)


def load_bounded_tranche_authorization(
    path: str | Path,
    *,
    repo_root: str | Path,
    external_root: str | Path,
) -> BoundedTrancheAuthorization:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PretrainingTrancheError(
            "bounded tranche authorization must be a JSON object"
        )
    if payload.get("schema_version") != 1:
        raise PretrainingTrancheError(
            "unsupported bounded tranche authorization schema"
        )
    if payload.get("stage") != "L004":
        raise PretrainingTrancheError(
            "bounded tranche authorization stage must be L004"
        )
    if payload.get("authorization_id") != "base50m-first-tranche-v1":
        raise PretrainingTrancheError(
            "bounded tranche authorization id changed"
        )
    foundation = payload.get("foundation_commit")
    runner = payload.get("runner_commit")
    if not _valid_git_sha(foundation):
        raise PretrainingTrancheError(
            "foundation_commit must be a full Git SHA"
        )
    if not _valid_git_sha(runner):
        raise PretrainingTrancheError(
            "runner_commit must be a full Git SHA"
        )
    if payload.get("checkpoint_lane") != "research_only":
        raise PretrainingTrancheError(
            "bounded tranche lane must remain research_only"
        )
    if payload.get("start_optimizer_step") != 0:
        raise PretrainingTrancheError(
            "first bounded tranche must start at optimizer step 0"
        )
    if payload.get("max_optimizer_updates") != 4:
        raise PretrainingTrancheError(
            "first bounded tranche must remain limited to 4 updates"
        )
    if payload.get("target_supervised_tokens") != 262_144:
        raise PretrainingTrancheError(
            "first bounded tranche must remain limited to 262144 tokens"
        )
    if payload.get("checkpoint_every_updates") != 1:
        raise PretrainingTrancheError(
            "first bounded tranche must checkpoint every update"
        )
    if payload.get("pretraining_authorized") is not True:
        raise PretrainingTrancheError(
            "bounded tranche is not authorized"
        )
    if payload.get("public_release_eligibility") != "not_qualified":
        raise PretrainingTrancheError(
            "public release eligibility changed"
        )
    if payload.get("automatic_flywiremodel_export_allowed") is not False:
        raise PretrainingTrancheError(
            "automatic FlyWireModel export must remain blocked"
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
    return BoundedTrancheAuthorization(
        authorization_id="base50m-first-tranche-v1",
        foundation_commit=foundation,
        runner_commit=runner,
        checkpoint_lane="research_only",
        start_optimizer_step=0,
        max_optimizer_updates=4,
        target_supervised_tokens=262_144,
        checkpoint_every_updates=1,
        run_root=run_root,
        run_root_storage=run_root_storage,
        repo_file_pins=pins,
        authorization_sha256=sha256_file(path),
    )


def checkpoint_filename(step: int) -> str:
    if not isinstance(step, int) or step <= 0:
        raise PretrainingTrancheError(
            "checkpoint step must be a positive integer"
        )
    return f"checkpoints/step-{step:06d}.pt"


def metric_filename(step: int) -> str:
    if not isinstance(step, int) or step <= 0:
        raise PretrainingTrancheError(
            "metric step must be a positive integer"
        )
    return f"metrics/step-{step:06d}.json"


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
    ) + "\n"
    temp = path.with_suffix(path.suffix + ".tmp")
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


def build_state_payload(
    authorization: BoundedTrancheAuthorization,
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
    if optimizer_step < 1 or (
        optimizer_step > authorization.max_optimizer_updates
    ):
        raise PretrainingTrancheError(
            "optimizer step is outside bounded tranche"
        )
    expected_tokens = optimizer_step * 65_536
    if supervised_tokens_seen != expected_tokens:
        raise PretrainingTrancheError(
            "bounded tranche supervised-token progress mismatch"
        )
    if optimizer_step == authorization.max_optimizer_updates and (
        supervised_tokens_seen
        != authorization.target_supervised_tokens
    ):
        raise PretrainingTrancheError(
            "bounded tranche final token target mismatch"
        )
    if (
        not isinstance(physical_input_positions_seen, int)
        or physical_input_positions_seen < supervised_tokens_seen
    ):
        raise PretrainingTrancheError(
            "physical input progress is invalid"
        )
    if not isinstance(microbatches_seen, int) or microbatches_seen <= 0:
        raise PretrainingTrancheError(
            "microbatch progress is invalid"
        )
    if (
        not isinstance(data_state, dict)
        or data_state.get("supervised_tokens_consumed")
        != supervised_tokens_seen
    ):
        raise PretrainingTrancheError(
            "data cursor/token progress mismatch"
        )
    if not _valid_sha256(checkpoint_sha256):
        raise PretrainingTrancheError(
            "checkpoint SHA-256 is invalid"
        )
    if not _valid_sha256(metric_sha256):
        raise PretrainingTrancheError(
            "metric SHA-256 is invalid"
        )

    run_root = authorization.run_root.resolve()
    checkpoint_resolved = checkpoint_path.resolve()
    metric_resolved = metric_path.resolve()
    try:
        checkpoint_relative = checkpoint_resolved.relative_to(run_root)
        metric_relative = metric_resolved.relative_to(run_root)
    except ValueError as exc:
        raise PretrainingTrancheError(
            "tranche artifacts must remain inside the authorized run root"
        ) from exc

    return {
        "schema_version": 1,
        "stage": "L004",
        "run_id": authorization.authorization_id,
        "authorization_sha256": authorization.authorization_sha256,
        "foundation_commit": authorization.foundation_commit,
        "runner_commit": authorization.runner_commit,
        "checkpoint_lane": authorization.checkpoint_lane,
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
        "tranche_complete": (
            optimizer_step == authorization.max_optimizer_updates
        ),
        "public_release_eligibility": "not_qualified",
        "automatic_flywiremodel_export_allowed": False,
    }


def load_tranche_state(
    path: str | Path,
    *,
    authorization: BoundedTrancheAuthorization,
    verify_files: bool = True,
) -> TrancheState:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PretrainingTrancheError(
            "tranche state must be a JSON object"
        )
    expected_static = {
        "schema_version": 1,
        "stage": "L004",
        "run_id": authorization.authorization_id,
        "authorization_sha256": authorization.authorization_sha256,
        "foundation_commit": authorization.foundation_commit,
        "runner_commit": authorization.runner_commit,
        "checkpoint_lane": authorization.checkpoint_lane,
        "public_release_eligibility": "not_qualified",
        "automatic_flywiremodel_export_allowed": False,
    }
    for key, expected in expected_static.items():
        if payload.get(key) != expected:
            raise PretrainingTrancheError(
                f"tranche state {key} mismatch"
            )

    step = payload.get("optimizer_step")
    supervised = payload.get("supervised_tokens_seen")
    physical = payload.get("physical_input_positions_seen")
    microbatches = payload.get("microbatches_seen")
    if not isinstance(step, int) or (
        step < 1 or step > authorization.max_optimizer_updates
    ):
        raise PretrainingTrancheError(
            "tranche state optimizer step is invalid"
        )
    if supervised != step * 65_536:
        raise PretrainingTrancheError(
            "tranche state supervised-token progress mismatch"
        )
    if not isinstance(physical, int) or physical < supervised:
        raise PretrainingTrancheError(
            "tranche state physical progress is invalid"
        )
    if not isinstance(microbatches, int) or microbatches <= 0:
        raise PretrainingTrancheError(
            "tranche state microbatch progress is invalid"
        )
    data_state = payload.get("data_state")
    if (
        not isinstance(data_state, dict)
        or data_state.get("supervised_tokens_consumed") != supervised
    ):
        raise PretrainingTrancheError(
            "tranche state data cursor mismatch"
        )

    checkpoint = payload.get("checkpoint")
    metric = payload.get("metric")
    if not isinstance(checkpoint, dict) or not isinstance(metric, dict):
        raise PretrainingTrancheError(
            "tranche state artifact metadata is missing"
        )
    checkpoint_file = checkpoint.get("file")
    metric_file = metric.get("file")
    checkpoint_sha = checkpoint.get("sha256")
    metric_sha = metric.get("sha256")
    checkpoint_bytes = checkpoint.get("bytes")
    if not isinstance(checkpoint_file, str) or not checkpoint_file:
        raise PretrainingTrancheError(
            "checkpoint file metadata is invalid"
        )
    if not isinstance(metric_file, str) or not metric_file:
        raise PretrainingTrancheError(
            "metric file metadata is invalid"
        )
    if not _valid_sha256(checkpoint_sha) or not _valid_sha256(metric_sha):
        raise PretrainingTrancheError(
            "state artifact SHA-256 is invalid"
        )
    if not isinstance(checkpoint_bytes, int) or checkpoint_bytes <= 0:
        raise PretrainingTrancheError(
            "checkpoint byte size is invalid"
        )
    if verify_files:
        checkpoint_path = authorization.run_root / checkpoint_file
        metric_path = authorization.run_root / metric_file
        if not checkpoint_path.is_file():
            raise PretrainingTrancheError(
                "state checkpoint file is missing"
            )
        if checkpoint_path.stat().st_size != checkpoint_bytes:
            raise PretrainingTrancheError(
                "state checkpoint byte size mismatch"
            )
        if sha256_file(checkpoint_path) != checkpoint_sha:
            raise PretrainingTrancheError(
                "state checkpoint SHA-256 mismatch"
            )
        if not metric_path.is_file():
            raise PretrainingTrancheError(
                "state metric file is missing"
            )
        if sha256_file(metric_path) != metric_sha:
            raise PretrainingTrancheError(
                "state metric SHA-256 mismatch"
            )

    expected_complete = step == authorization.max_optimizer_updates
    if payload.get("tranche_complete") is not expected_complete:
        raise PretrainingTrancheError(
            "tranche completion flag mismatch"
        )

    return TrancheState(
        authorization_id=authorization.authorization_id,
        authorization_sha256=authorization.authorization_sha256,
        foundation_commit=authorization.foundation_commit,
        runner_commit=authorization.runner_commit,
        checkpoint_lane=authorization.checkpoint_lane,
        optimizer_step=step,
        supervised_tokens_seen=supervised,
        physical_input_positions_seen=physical,
        microbatches_seen=microbatches,
        data_state={key: int(value) for key, value in data_state.items()},
        checkpoint_file=checkpoint_file,
        checkpoint_bytes=checkpoint_bytes,
        checkpoint_sha256=checkpoint_sha,
        metric_file=metric_file,
        metric_sha256=metric_sha,
    )
