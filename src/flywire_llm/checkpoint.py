from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .config import BlankLLMConfig
from .model import BlankCausalLM, BlankLLMError
from .tokenizer import UTF8ByteTokenizer


FORMAT_VERSION = 1


def checkpoint_metadata(
    model: BlankCausalLM,
    *,
    training_step: int,
    initialization_seed: int | None,
) -> dict[str, Any]:
    if training_step < 0:
        raise BlankLLMError("training_step must be non-negative")
    if initialization_seed is not None and initialization_seed < 0:
        raise BlankLLMError("initialization_seed must be non-negative or None")
    return {
        "format_version": FORMAT_VERSION,
        "model_type": model.model_type,
        "pretrained": False,
        "external_pretrained_weights": False,
        "initialization_origin": "random",
        "random_initialization_origin": True,
        "initialization_seed": initialization_seed,
        "knowledge_in_weights_claimed": False,
        "untrained": training_step == 0,
        "training_step": int(training_step),
        "framework": {
            "name": "pytorch",
            "version": str(torch.__version__),
        },
        "config": model.config.to_dict(),
        "tokenizer": UTF8ByteTokenizer().metadata(),
    }


def save_checkpoint(
    path: str | Path,
    model: BlankCausalLM,
    *,
    training_step: int = 0,
    initialization_seed: int | None = None,
    optimizer: torch.optim.Optimizer | None = None,
) -> None:
    payload: dict[str, Any] = {
        "metadata": checkpoint_metadata(
            model,
            training_step=training_step,
            initialization_seed=initialization_seed,
        ),
        "model_state": model.state_dict(),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[BlankCausalLM, dict[str, Any]]:
    payload = torch.load(
        Path(path),
        map_location=map_location,
        weights_only=True,
    )
    if not isinstance(payload, dict):
        raise BlankLLMError("checkpoint payload must be a mapping")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise BlankLLMError("checkpoint metadata is missing")
    if metadata.get("format_version") != FORMAT_VERSION:
        raise BlankLLMError("unsupported checkpoint format version")
    if metadata.get("model_type") != BlankCausalLM.model_type:
        raise BlankLLMError("checkpoint model_type is invalid")
    if metadata.get("pretrained") is not False:
        raise BlankLLMError("checkpoint must declare pretrained=false")
    if metadata.get("external_pretrained_weights") is not False:
        raise BlankLLMError(
            "checkpoint cannot contain imported pretrained weights"
        )
    if metadata.get("initialization_origin") != "random":
        raise BlankLLMError("checkpoint initialization origin must be random")
    if metadata.get("random_initialization_origin") is not True:
        raise BlankLLMError(
            "checkpoint must retain random-initialization provenance"
        )
    seed = metadata.get("initialization_seed")
    if seed is not None and (not isinstance(seed, int) or seed < 0):
        raise BlankLLMError("checkpoint initialization_seed is invalid")
    if metadata.get("knowledge_in_weights_claimed") is not False:
        raise BlankLLMError(
            "blank checkpoint cannot claim learned FlyWire knowledge"
        )

    framework = metadata.get("framework")
    if not isinstance(framework, dict) or framework.get("name") != "pytorch":
        raise BlankLLMError("checkpoint framework metadata is invalid")

    config_payload = metadata.get("config")
    if not isinstance(config_payload, dict):
        raise BlankLLMError("checkpoint config is missing")
    model = BlankCausalLM(BlankLLMConfig.from_dict(config_payload))

    state = payload.get("model_state")
    if not isinstance(state, dict):
        raise BlankLLMError("checkpoint model_state is missing")
    model.load_state_dict(state)
    return model, metadata
