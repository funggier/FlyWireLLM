from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


class PretrainingFreezeError(ValueError):
    pass


@dataclass(frozen=True)
class PretrainingFreeze:
    checkpoint_lane: str
    optimizer_steps_completed: int
    source_manifest_sha256: str
    global_manifest_report_sha256: str
    decisions_sha256: str
    tokenizer_sha256: str
    pretraining_authorized: bool


_ALLOWED_LANES = {"release_safe", "research_only"}
_TOKENIZER_ID = "base50m-unigram-32000-v1"
_TOKENIZER_SHA256 = (
    "998bc75f058e554d4f50b6cbc77c52898e3446e516c3b25a126b99b113513818"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _repo_file(repo_root: Path, relative: object, *, field: str) -> Path:
    if not isinstance(relative, str) or not relative.strip():
        raise PretrainingFreezeError(f"{field} path must be non-empty")
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise PretrainingFreezeError(f"{field} path must be repo-relative")
    root = repo_root.resolve()
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise PretrainingFreezeError(
            f"{field} path escapes repository"
        ) from exc
    if not resolved.is_file():
        raise PretrainingFreezeError(f"{field} file is missing")
    return resolved


def _pinned_repo_file(
    block: object,
    *,
    repo_root: Path,
    field: str,
) -> tuple[Path, str]:
    if not isinstance(block, dict):
        raise PretrainingFreezeError(f"{field} must be an object")
    if block.get("frozen") is not True:
        raise PretrainingFreezeError(f"{field} must be frozen")
    expected = block.get("sha256")
    if not _valid_sha256(expected):
        raise PretrainingFreezeError(f"{field} SHA-256 is invalid")
    path = _repo_file(repo_root, block.get("path"), field=field)
    observed = _sha256_file(path)
    if observed != expected:
        raise PretrainingFreezeError(
            f"{field} SHA-256 mismatch expected={expected} observed={observed}"
        )
    return path, expected


def load_pretraining_freeze(
    path: str | Path,
    *,
    repo_root: str | Path,
) -> PretrainingFreeze:
    repo_root = Path(repo_root)
    freeze_path = Path(path)
    payload = json.loads(freeze_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PretrainingFreezeError("pretraining freeze must be an object")
    if payload.get("schema_version") != 1:
        raise PretrainingFreezeError(
            "unsupported pretraining freeze schema"
        )
    if payload.get("stage") != "L003":
        raise PretrainingFreezeError("pretraining freeze stage must be L003")

    checkpoint_lane = payload.get("checkpoint_lane")
    if checkpoint_lane not in _ALLOWED_LANES:
        raise PretrainingFreezeError("invalid checkpoint_lane")
    optimizer_steps = payload.get("optimizer_steps_completed")
    if optimizer_steps != 0:
        raise PretrainingFreezeError(
            "pretraining freeze must be declared before optimizer step 1"
        )
    if (
        payload.get(
            "checkpoint_lane_immutable_after_first_optimizer_step"
        )
        is not True
    ):
        raise PretrainingFreezeError(
            "checkpoint lane immutability must remain enabled"
        )

    registry_path, registry_sha = _pinned_repo_file(
        payload.get("source_manifest"),
        repo_root=repo_root,
        field="source_manifest",
    )
    report_path, report_sha = _pinned_repo_file(
        payload.get("global_manifest_report"),
        repo_root=repo_root,
        field="global_manifest_report",
    )

    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if registry.get("schema_version") != 1 or registry.get("stage") != "L003":
        raise PretrainingFreezeError("source manifest is not L003 schema 1")
    if registry.get("tokenizer") != _TOKENIZER_ID:
        raise PretrainingFreezeError("source manifest tokenizer changed")
    if registry.get("technical_classifier") != "technical-heuristic-v5":
        raise PretrainingFreezeError(
            "source manifest technical classifier changed"
        )
    if registry.get("near_duplicate_hamming_distance") != 3:
        raise PretrainingFreezeError(
            "source manifest near-duplicate contract changed"
        )
    if registry.get("pretraining_authorized") is not False:
        raise PretrainingFreezeError(
            "source manifest cannot self-authorize pretraining"
        )
    raw_inputs = registry.get("inputs")
    if not isinstance(raw_inputs, list) or not raw_inputs:
        raise PretrainingFreezeError("source manifest inputs are empty")
    lanes = {row.get("rights_lane") for row in raw_inputs if isinstance(row, dict)}
    if checkpoint_lane == "release_safe" and lanes != {"release_safe"}:
        raise PretrainingFreezeError(
            "release_safe freeze includes research_only source"
        )
    if checkpoint_lane == "research_only" and not lanes <= _ALLOWED_LANES:
        raise PretrainingFreezeError(
            "research_only freeze contains invalid source lane"
        )

    tokenizer = payload.get("tokenizer")
    if not isinstance(tokenizer, dict):
        raise PretrainingFreezeError("tokenizer freeze must be an object")
    if tokenizer.get("candidate_id") != _TOKENIZER_ID:
        raise PretrainingFreezeError("tokenizer candidate changed")
    tokenizer_sha = tokenizer.get("model_sha256")
    if tokenizer_sha != _TOKENIZER_SHA256:
        raise PretrainingFreezeError("tokenizer model SHA-256 changed")

    evaluation = payload.get("evaluation_manifest")
    if not isinstance(evaluation, dict):
        raise PretrainingFreezeError(
            "evaluation_manifest must be an object"
        )
    if evaluation.get("frozen") is not True:
        raise PretrainingFreezeError("evaluation_manifest must be frozen")
    if evaluation.get("validation_partition") != "validation":
        raise PretrainingFreezeError("validation partition changed")
    if evaluation.get("final_holdout_partition") != "holdout":
        raise PretrainingFreezeError("final holdout partition changed")
    if evaluation.get("pretraining_fit_partitions") != ["train"]:
        raise PretrainingFreezeError(
            "pretraining fit partitions must remain train-only"
        )
    if evaluation.get("tokenizer_fit_must_exclude_holdout") is not True:
        raise PretrainingFreezeError(
            "tokenizer fit must exclude holdout"
        )
    if evaluation.get("model_fit_must_exclude_holdout") is not True:
        raise PretrainingFreezeError("model fit must exclude holdout")
    decisions_sha = evaluation.get("decisions_sha256")
    if not _valid_sha256(decisions_sha):
        raise PretrainingFreezeError(
            "evaluation decisions SHA-256 is invalid"
        )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema_version") != 1 or report.get("stage") != "L003":
        raise PretrainingFreezeError("global manifest report is not L003")
    if report.get("global_cross_source_dedup_applied") is not True:
        raise PretrainingFreezeError(
            "global cross-source dedup is not complete"
        )
    if report.get("pretraining_authorized") is not False:
        raise PretrainingFreezeError(
            "global report must not self-authorize pretraining"
        )

    report_registry = report.get("input_registry")
    if not isinstance(report_registry, dict):
        raise PretrainingFreezeError(
            "global report missing input registry pin"
        )
    source_relative = payload["source_manifest"]["path"]
    if report_registry.get("path") != source_relative:
        raise PretrainingFreezeError(
            "global report/source manifest path mismatch"
        )
    if report_registry.get("sha256") != registry_sha:
        raise PretrainingFreezeError(
            "global report/source manifest hash mismatch"
        )

    report_tokenizer = report.get("tokenizer_model")
    if not isinstance(report_tokenizer, dict):
        raise PretrainingFreezeError(
            "global report missing tokenizer pin"
        )
    if (
        report_tokenizer.get("candidate_id") != _TOKENIZER_ID
        or report_tokenizer.get("sha256") != tokenizer_sha
    ):
        raise PretrainingFreezeError(
            "global report/tokenizer freeze mismatch"
        )

    decisions = report.get("decisions")
    if not isinstance(decisions, dict):
        raise PretrainingFreezeError(
            "global report missing decision manifest"
        )
    if decisions.get("contains_raw_text") is not False:
        raise PretrainingFreezeError(
            "global decision manifest must be metadata-only"
        )
    storage = decisions.get("storage")
    if (
        not isinstance(storage, str)
        or not storage.startswith("external://FlyWireLLM-data/")
    ):
        raise PretrainingFreezeError(
            "global decision manifest storage must be external"
        )
    if decisions.get("sha256") != decisions_sha:
        raise PretrainingFreezeError(
            "evaluation/global decisions SHA-256 mismatch"
        )

    mixture = report.get("mixture")
    if not isinstance(mixture, dict):
        raise PretrainingFreezeError("global report missing mixture")
    lane_mixture = mixture.get(checkpoint_lane)
    if not isinstance(lane_mixture, dict):
        raise PretrainingFreezeError(
            f"global report missing {checkpoint_lane} mixture"
        )
    if lane_mixture.get("ready") is not True:
        raise PretrainingFreezeError(
            f"{checkpoint_lane} mixture is not ready"
        )
    for gap in ("thai_gap", "english_gap", "technical_gap", "total_gap"):
        if lane_mixture.get(gap, 0) != 0:
            raise PretrainingFreezeError(
                f"{checkpoint_lane} mixture has non-zero {gap}"
            )
    if mixture.get("pretraining_authorized") is not False:
        raise PretrainingFreezeError(
            "mixture evaluation must not self-authorize pretraining"
        )

    if payload.get("flywiremodel_automatic_export_allowed") is not False:
        raise PretrainingFreezeError(
            "automatic FlyWireModel export must remain blocked"
        )
    expected_release = (
        "preserved_pending_final_release_audit"
        if checkpoint_lane == "release_safe"
        else "not_qualified"
    )
    if payload.get("public_release_eligibility") != expected_release:
        raise PretrainingFreezeError(
            "public release eligibility does not match checkpoint lane"
        )
    if payload.get("pretraining_authorized") is not True:
        raise PretrainingFreezeError(
            "final freeze must explicitly authorize pretraining"
        )

    return PretrainingFreeze(
        checkpoint_lane=checkpoint_lane,
        optimizer_steps_completed=optimizer_steps,
        source_manifest_sha256=registry_sha,
        global_manifest_report_sha256=report_sha,
        decisions_sha256=decisions_sha,
        tokenizer_sha256=tokenizer_sha,
        pretraining_authorized=True,
    )
