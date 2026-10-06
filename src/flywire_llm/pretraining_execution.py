from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import resolve_external_storage_uri
from .config import BlankLLMConfig
from .pretraining_freeze import PretrainingFreeze, load_pretraining_freeze
from .sentencepiece_tokenizer import SentencePieceTokenizer


class PretrainingExecutionError(ValueError):
    pass


@dataclass(frozen=True)
class PretrainingExecutionContract:
    profile: str
    checkpoint_lane: str
    checkpoint_format: str
    l003_freeze_sha256: str
    source_manifest_sha256: str
    global_manifest_report_sha256: str
    decisions_sha256: str
    model_config_sha256: str
    training_config_sha256: str
    data_runtime_sha256: str
    tokenizer_candidate_id: str
    tokenizer_sha256: str
    tokenizer_vocab_size: int
    tokenizer_path: Path
    model_config: BlankLLMConfig
    primary_token_budget: int
    global_tokens_per_update: int
    primary_seed: int


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _repo_path(
    repo_root: Path,
    relative: object,
    *,
    field: str,
) -> Path:
    if not isinstance(relative, str) or not relative.strip():
        raise PretrainingExecutionError(f"{field} path must be non-empty")
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise PretrainingExecutionError(f"{field} path must be repo-relative")
    root = repo_root.resolve()
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise PretrainingExecutionError(
            f"{field} path escapes repository"
        ) from exc
    if not resolved.is_file():
        raise PretrainingExecutionError(f"{field} file is missing")
    return resolved


def _pinned_repo_file(
    block: object,
    *,
    repo_root: Path,
    field: str,
) -> tuple[Path, str]:
    if not isinstance(block, dict):
        raise PretrainingExecutionError(f"{field} must be an object")
    expected = block.get("sha256")
    if not _valid_sha256(expected):
        raise PretrainingExecutionError(f"{field} SHA-256 is invalid")
    path = _repo_path(repo_root, block.get("path"), field=field)
    observed = sha256_file(path)
    if observed != expected:
        raise PretrainingExecutionError(
            f"{field} SHA-256 mismatch expected={expected} "
            f"observed={observed}"
        )
    return path, expected


def _load_json_object(path: Path, *, field: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PretrainingExecutionError(f"{field} must be a JSON object")
    return payload


def verify_base50m_tokenizer(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_vocab_size: int = 32_000,
) -> SentencePieceTokenizer:
    path = Path(path)
    if not path.is_file():
        raise PretrainingExecutionError("tokenizer model file is missing")
    if not _valid_sha256(expected_sha256):
        raise PretrainingExecutionError("tokenizer SHA-256 is invalid")
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise PretrainingExecutionError(
            "tokenizer SHA-256 mismatch "
            f"expected={expected_sha256} observed={observed}"
        )
    tokenizer = SentencePieceTokenizer(path)
    if tokenizer.vocab_size != expected_vocab_size:
        raise PretrainingExecutionError(
            "tokenizer vocabulary mismatch "
            f"expected={expected_vocab_size} observed={tokenizer.vocab_size}"
        )
    return tokenizer


def _validate_training_contract(
    training: dict[str, Any],
    declared: object,
) -> tuple[int, int, int]:
    if not isinstance(declared, dict):
        raise PretrainingExecutionError(
            "training_contract must be an object"
        )
    expected_pairs = {
        "objective": "causal_next_token_cross_entropy",
        "primary_token_budget": 500_000_000,
        "global_tokens_per_update": 65_536,
        "primary_seed": 1234,
        "fit_partitions": ["train"],
        "validation_partition": "validation",
        "final_holdout_partition": "holdout",
    }
    for key, expected in expected_pairs.items():
        if declared.get(key) != expected:
            raise PretrainingExecutionError(
                f"training contract {key} changed"
            )

    if training.get("profile") != "base-50m-v1":
        raise PretrainingExecutionError("training profile changed")
    if training.get("objective") != declared["objective"]:
        raise PretrainingExecutionError("training objective changed")
    if training.get("primary_token_budget") != declared["primary_token_budget"]:
        raise PretrainingExecutionError(
            "training primary token budget changed"
        )
    if (
        training.get("global_tokens_per_update")
        != declared["global_tokens_per_update"]
    ):
        raise PretrainingExecutionError(
            "training global tokens/update changed"
        )
    if training.get("primary_seed") != declared["primary_seed"]:
        raise PretrainingExecutionError("training primary seed changed")

    optimizer = training.get("optimizer")
    if not isinstance(optimizer, dict) or optimizer.get("name") != "AdamW":
        raise PretrainingExecutionError("training optimizer must be AdamW")
    expected_optimizer = {
        "beta1": 0.9,
        "beta2": 0.95,
        "eps": 1e-8,
        "weight_decay": 0.1,
        "gradient_clip_norm": 1,
    }
    for key, expected in expected_optimizer.items():
        if optimizer.get(key) != expected:
            raise PretrainingExecutionError(
                f"training optimizer {key} changed"
            )

    schedule = training.get("schedule")
    if not isinstance(schedule, dict):
        raise PretrainingExecutionError("training schedule is missing")
    expected_schedule = {
        "peak_learning_rate": 0.0006,
        "warmup_fraction": 0.02,
        "decay": "cosine",
        "min_learning_rate": 0.00006,
    }
    for key, expected in expected_schedule.items():
        if schedule.get(key) != expected:
            raise PretrainingExecutionError(
                f"training schedule {key} changed"
            )

    return (
        int(declared["primary_token_budget"]),
        int(declared["global_tokens_per_update"]),
        int(declared["primary_seed"]),
    )


def load_pretraining_execution_contract(
    path: str | Path,
    *,
    repo_root: str | Path,
    external_root: str | Path,
) -> PretrainingExecutionContract:
    repo_root = Path(repo_root)
    payload = _load_json_object(Path(path), field="execution contract")
    if payload.get("schema_version") != 1:
        raise PretrainingExecutionError(
            "unsupported pretraining execution schema"
        )
    if payload.get("stage") != "L004":
        raise PretrainingExecutionError(
            "pretraining execution stage must be L004"
        )
    if payload.get("profile") != "base-50m-v1":
        raise PretrainingExecutionError(
            "pretraining execution profile changed"
        )
    if payload.get("checkpoint_lane") != "research_only":
        raise PretrainingExecutionError(
            "L004 checkpoint lane must remain research_only"
        )
    if payload.get("checkpoint_format") != (
        "flywire-base50m-pretraining-v1"
    ):
        raise PretrainingExecutionError(
            "pretraining checkpoint format changed"
        )
    if payload.get("public_release_eligibility") != "not_qualified":
        raise PretrainingExecutionError(
            "public release eligibility changed"
        )

    freeze_path, freeze_sha = _pinned_repo_file(
        payload.get("l003_freeze"),
        repo_root=repo_root,
        field="l003_freeze",
    )
    freeze: PretrainingFreeze = load_pretraining_freeze(
        freeze_path,
        repo_root=repo_root,
    )
    if freeze.checkpoint_lane != "research_only":
        raise PretrainingExecutionError(
            "L003 freeze checkpoint lane changed"
        )
    if freeze.optimizer_steps_completed != 0:
        raise PretrainingExecutionError(
            "L003 freeze must precede optimizer step 1"
        )
    if freeze.pretraining_authorized is not True:
        raise PretrainingExecutionError(
            "L003 freeze does not authorize pretraining"
        )

    model_path, model_sha = _pinned_repo_file(
        payload.get("model_config"),
        repo_root=repo_root,
        field="model_config",
    )
    model_payload = _load_json_object(
        model_path,
        field="model_config",
    )
    model_config = BlankLLMConfig.from_dict(model_payload)
    model_block = payload["model_config"]
    expected_parameters = model_block.get("exact_parameters")
    if expected_parameters != 50_213_376:
        raise PretrainingExecutionError(
            "Base-50M exact parameter contract changed"
        )
    if model_config.exact_parameter_count != expected_parameters:
        raise PretrainingExecutionError(
            "Base-50M model configuration parameter count mismatch"
        )

    training_path, training_sha = _pinned_repo_file(
        payload.get("training_config"),
        repo_root=repo_root,
        field="training_config",
    )
    training = _load_json_object(
        training_path,
        field="training_config",
    )
    primary_budget, tokens_per_update, seed = _validate_training_contract(
        training,
        payload.get("training_contract"),
    )

    _, data_runtime_sha = _pinned_repo_file(
        payload.get("data_runtime"),
        repo_root=repo_root,
        field="data_runtime",
    )

    tokenizer_block = payload.get("tokenizer")
    if not isinstance(tokenizer_block, dict):
        raise PretrainingExecutionError("tokenizer must be an object")
    if tokenizer_block.get("candidate_id") != (
        "base50m-unigram-32000-v1"
    ):
        raise PretrainingExecutionError(
            "Base-50M tokenizer candidate changed"
        )
    tokenizer_sha = tokenizer_block.get("sha256")
    if tokenizer_sha != freeze.tokenizer_sha256:
        raise PretrainingExecutionError(
            "execution tokenizer hash does not match L003 freeze"
        )
    expected_vocab = tokenizer_block.get("vocab_size")
    if expected_vocab != 32_000:
        raise PretrainingExecutionError(
            "Base-50M tokenizer vocabulary contract changed"
        )
    storage = tokenizer_block.get("storage")
    tokenizer_path = resolve_external_storage_uri(
        storage,
        external_root=external_root,
    )
    verify_base50m_tokenizer(
        tokenizer_path,
        expected_sha256=tokenizer_sha,
        expected_vocab_size=expected_vocab,
    )

    return PretrainingExecutionContract(
        profile="base-50m-v1",
        checkpoint_lane="research_only",
        checkpoint_format="flywire-base50m-pretraining-v1",
        l003_freeze_sha256=freeze_sha,
        source_manifest_sha256=freeze.source_manifest_sha256,
        global_manifest_report_sha256=freeze.global_manifest_report_sha256,
        decisions_sha256=freeze.decisions_sha256,
        model_config_sha256=model_sha,
        training_config_sha256=training_sha,
        data_runtime_sha256=data_runtime_sha,
        tokenizer_candidate_id="base50m-unigram-32000-v1",
        tokenizer_sha256=tokenizer_sha,
        tokenizer_vocab_size=expected_vocab,
        tokenizer_path=tokenizer_path,
        model_config=model_config,
        primary_token_budget=primary_budget,
        global_tokens_per_update=tokens_per_update,
        primary_seed=seed,
    )
