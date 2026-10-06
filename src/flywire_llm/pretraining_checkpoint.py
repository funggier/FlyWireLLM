from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from .config import BlankLLMConfig
from .model import BlankCausalLM
from .pretraining_execution import PretrainingExecutionContract


class PretrainingCheckpointError(ValueError):
    pass


FORMAT_VERSION = 1
CHECKPOINT_TYPE = "flywire-base50m-pretraining-v1"


@dataclass(frozen=True)
class PretrainingLineage:
    checkpoint_lane: str
    l003_freeze_sha256: str
    source_manifest_sha256: str
    global_manifest_report_sha256: str
    decisions_sha256: str
    tokenizer_sha256: str
    model_config_sha256: str
    training_config_sha256: str
    data_runtime_sha256: str


@dataclass(frozen=True)
class PretrainingProgress:
    optimizer_step: int
    supervised_tokens_seen: int
    microbatches_seen: int
    physical_input_positions_seen: int


@dataclass
class LoadedPretrainingCheckpoint:
    model: BlankCausalLM
    optimizer_state: dict[str, Any]
    progress: PretrainingProgress
    torch_rng_state: torch.Tensor
    data_rng_state: torch.Tensor
    data_state: dict[str, int]
    metadata: dict[str, Any]


def lineage_from_contract(
    contract: PretrainingExecutionContract,
) -> PretrainingLineage:
    return PretrainingLineage(
        checkpoint_lane=contract.checkpoint_lane,
        l003_freeze_sha256=contract.l003_freeze_sha256,
        source_manifest_sha256=contract.source_manifest_sha256,
        global_manifest_report_sha256=(
            contract.global_manifest_report_sha256
        ),
        decisions_sha256=contract.decisions_sha256,
        tokenizer_sha256=contract.tokenizer_sha256,
        model_config_sha256=contract.model_config_sha256,
        training_config_sha256=contract.training_config_sha256,
        data_runtime_sha256=contract.data_runtime_sha256,
    )


def _validate_sha256(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise PretrainingCheckpointError(
            f"{field} must be a lowercase SHA-256"
        )
    return value


def validate_lineage(lineage: PretrainingLineage) -> None:
    if lineage.checkpoint_lane != "research_only":
        raise PretrainingCheckpointError(
            "L004 checkpoint lane must remain research_only"
        )
    for field in (
        "l003_freeze_sha256",
        "source_manifest_sha256",
        "global_manifest_report_sha256",
        "decisions_sha256",
        "tokenizer_sha256",
        "model_config_sha256",
        "training_config_sha256",
        "data_runtime_sha256",
    ):
        _validate_sha256(getattr(lineage, field), field=field)


def validate_progress(progress: PretrainingProgress) -> None:
    for field in (
        "optimizer_step",
        "supervised_tokens_seen",
        "microbatches_seen",
        "physical_input_positions_seen",
    ):
        value = getattr(progress, field)
        if not isinstance(value, int) or value < 0:
            raise PretrainingCheckpointError(
                f"{field} must be a non-negative integer"
            )
    if (
        progress.optimizer_step == 0
        and progress.supervised_tokens_seen != 0
    ):
        raise PretrainingCheckpointError(
            "step-0 checkpoint cannot claim consumed training tokens"
        )


def _validate_data_state(
    payload: object,
    *,
    progress: PretrainingProgress,
) -> dict[str, int]:
    if not isinstance(payload, dict):
        raise PretrainingCheckpointError(
            "checkpoint data_state must be an object"
        )
    expected = {
        "order_position",
        "block_input_offset",
        "supervised_tokens_consumed",
    }
    if set(payload) != expected:
        raise PretrainingCheckpointError(
            "checkpoint data_state fields changed"
        )
    result: dict[str, int] = {}
    for field in sorted(expected):
        value = payload[field]
        if not isinstance(value, int) or value < 0:
            raise PretrainingCheckpointError(
                f"checkpoint data_state {field} must be non-negative"
            )
        result[field] = value
    if (
        result["supervised_tokens_consumed"]
        != progress.supervised_tokens_seen
    ):
        raise PretrainingCheckpointError(
            "checkpoint data cursor/token progress mismatch"
        )
    return result


def _metadata(
    model: BlankCausalLM,
    *,
    lineage: PretrainingLineage,
    progress: PretrainingProgress,
    primary_token_budget: int,
    primary_seed: int,
) -> dict[str, Any]:
    validate_lineage(lineage)
    validate_progress(progress)
    if primary_token_budget <= 0:
        raise PretrainingCheckpointError(
            "primary_token_budget must be positive"
        )
    if primary_seed < 0:
        raise PretrainingCheckpointError(
            "primary_seed must be non-negative"
        )
    return {
        "format_version": FORMAT_VERSION,
        "checkpoint_type": CHECKPOINT_TYPE,
        "model_type": model.model_type,
        "profile": "base-50m-v1",
        "initialization_origin": "random",
        "external_pretrained_weights": False,
        "pretraining_started": progress.optimizer_step > 0,
        "pretraining_complete": (
            progress.supervised_tokens_seen >= primary_token_budget
        ),
        "public_release_eligibility": "not_qualified",
        "automatic_flywiremodel_export_allowed": False,
        "primary_token_budget": int(primary_token_budget),
        "primary_seed": int(primary_seed),
        "lineage": asdict(lineage),
        "progress": asdict(progress),
        "model_config": model.config.to_dict(),
        "framework": {
            "name": "pytorch",
            "version": str(torch.__version__),
        },
        "rng_contract": {
            "torch_cpu": True,
            "data_generator": True,
        },
    }


def save_pretraining_checkpoint(
    path: str | Path,
    model: BlankCausalLM,
    optimizer: torch.optim.Optimizer,
    *,
    lineage: PretrainingLineage,
    progress: PretrainingProgress,
    primary_token_budget: int,
    primary_seed: int,
    data_generator: torch.Generator,
    data_state: dict[str, int],
) -> None:
    metadata = _metadata(
        model,
        lineage=lineage,
        progress=progress,
        primary_token_budget=primary_token_budget,
        primary_seed=primary_seed,
    )
    validated_data_state = _validate_data_state(
        data_state,
        progress=progress,
    )
    payload: dict[str, Any] = {
        "metadata": metadata,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "data_rng_state": data_generator.get_state(),
        "data_state": validated_data_state,
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    if temp.exists():
        temp.unlink()
    try:
        torch.save(payload, temp)
        os.replace(temp, output)
    finally:
        if temp.exists():
            temp.unlink()


def _lineage_from_metadata(raw: object) -> PretrainingLineage:
    if not isinstance(raw, dict):
        raise PretrainingCheckpointError(
            "checkpoint lineage is missing"
        )
    required = {
        "checkpoint_lane",
        "l003_freeze_sha256",
        "source_manifest_sha256",
        "global_manifest_report_sha256",
        "decisions_sha256",
        "tokenizer_sha256",
        "model_config_sha256",
        "training_config_sha256",
        "data_runtime_sha256",
    }
    if set(raw) != required:
        raise PretrainingCheckpointError(
            "checkpoint lineage fields changed"
        )
    lineage = PretrainingLineage(**raw)
    validate_lineage(lineage)
    return lineage


def _progress_from_metadata(raw: object) -> PretrainingProgress:
    if not isinstance(raw, dict):
        raise PretrainingCheckpointError(
            "checkpoint progress is missing"
        )
    required = {
        "optimizer_step",
        "supervised_tokens_seen",
        "microbatches_seen",
        "physical_input_positions_seen",
    }
    if set(raw) != required:
        raise PretrainingCheckpointError(
            "checkpoint progress fields changed"
        )
    progress = PretrainingProgress(**raw)
    validate_progress(progress)
    return progress


def load_pretraining_checkpoint(
    path: str | Path,
    *,
    expected_lineage: PretrainingLineage,
    map_location: str | torch.device = "cpu",
) -> LoadedPretrainingCheckpoint:
    validate_lineage(expected_lineage)
    payload = torch.load(
        Path(path),
        map_location=map_location,
        weights_only=True,
    )
    if not isinstance(payload, dict):
        raise PretrainingCheckpointError(
            "checkpoint payload must be a mapping"
        )
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise PretrainingCheckpointError(
            "checkpoint metadata is missing"
        )
    if metadata.get("format_version") != FORMAT_VERSION:
        raise PretrainingCheckpointError(
            "unsupported pretraining checkpoint format"
        )
    if metadata.get("checkpoint_type") != CHECKPOINT_TYPE:
        raise PretrainingCheckpointError(
            "pretraining checkpoint type changed"
        )
    if metadata.get("profile") != "base-50m-v1":
        raise PretrainingCheckpointError(
            "pretraining checkpoint profile changed"
        )
    if metadata.get("model_type") != BlankCausalLM.model_type:
        raise PretrainingCheckpointError(
            "pretraining checkpoint model_type changed"
        )
    if metadata.get("initialization_origin") != "random":
        raise PretrainingCheckpointError(
            "pretraining checkpoint lost random origin"
        )
    if metadata.get("external_pretrained_weights") is not False:
        raise PretrainingCheckpointError(
            "external pretrained weights are forbidden"
        )
    if metadata.get("public_release_eligibility") != "not_qualified":
        raise PretrainingCheckpointError(
            "public release eligibility changed"
        )
    if (
        metadata.get("automatic_flywiremodel_export_allowed")
        is not False
    ):
        raise PretrainingCheckpointError(
            "automatic FlyWireModel export must remain blocked"
        )

    observed_lineage = _lineage_from_metadata(
        metadata.get("lineage")
    )
    if observed_lineage != expected_lineage:
        raise PretrainingCheckpointError(
            "checkpoint lineage does not match execution contract"
        )
    progress = _progress_from_metadata(
        metadata.get("progress")
    )
    if metadata.get("pretraining_started") != (
        progress.optimizer_step > 0
    ):
        raise PretrainingCheckpointError(
            "pretraining_started/progress mismatch"
        )
    primary_budget = metadata.get("primary_token_budget")
    if not isinstance(primary_budget, int) or primary_budget <= 0:
        raise PretrainingCheckpointError(
            "primary token budget is invalid"
        )
    if metadata.get("pretraining_complete") != (
        progress.supervised_tokens_seen >= primary_budget
    ):
        raise PretrainingCheckpointError(
            "pretraining_complete/progress mismatch"
        )

    config_payload = metadata.get("model_config")
    if not isinstance(config_payload, dict):
        raise PretrainingCheckpointError(
            "checkpoint model_config is missing"
        )
    model = BlankCausalLM(
        BlankLLMConfig.from_dict(config_payload)
    )
    model_state = payload.get("model_state")
    if not isinstance(model_state, dict):
        raise PretrainingCheckpointError(
            "checkpoint model_state is missing"
        )
    model.load_state_dict(model_state)

    optimizer_state = payload.get("optimizer_state")
    if not isinstance(optimizer_state, dict):
        raise PretrainingCheckpointError(
            "checkpoint optimizer_state is missing"
        )
    torch_rng_state = payload.get("torch_rng_state")
    data_rng_state = payload.get("data_rng_state")
    data_state = _validate_data_state(
        payload.get("data_state"),
        progress=progress,
    )
    if not isinstance(torch_rng_state, torch.Tensor):
        raise PretrainingCheckpointError(
            "checkpoint torch RNG state is missing"
        )
    if not isinstance(data_rng_state, torch.Tensor):
        raise PretrainingCheckpointError(
            "checkpoint data RNG state is missing"
        )

    return LoadedPretrainingCheckpoint(
        model=model,
        optimizer_state=optimizer_state,
        progress=progress,
        torch_rng_state=torch_rng_state,
        data_rng_state=data_rng_state,
        data_state=data_state,
        metadata=metadata,
    )


def restore_pretraining_rng(
    loaded: LoadedPretrainingCheckpoint,
    *,
    data_generator: torch.Generator,
) -> None:
    torch.set_rng_state(loaded.torch_rng_state.cpu())
    data_generator.set_state(loaded.data_rng_state.cpu())
