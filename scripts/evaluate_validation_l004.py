from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from flywire_llm.artifacts import resolve_external_storage_uri
from flywire_llm.model import BlankCausalLM
from flywire_llm.pretraining_checkpoint import (
    lineage_from_contract,
    load_pretraining_checkpoint,
)
from flywire_llm.pretraining_execution import (
    load_pretraining_execution_contract,
)


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--external-root", required=True)
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "validation-l004-v1.json"),
    )
    parser.add_argument(
        "--summary",
        default=str(ROOT / "results" / "l004" / "validation-pack-v1.json"),
    )
    parser.add_argument(
        "--model",
        choices=("step0", "step4"),
        required=True,
    )
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def _load_contracts(
    config_path: Path,
    summary_path: Path,
) -> tuple[dict, dict]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or config.get("stage") != "L004":
        raise RuntimeError("validation config schema/stage changed")
    if summary.get("schema_version") != 1 or summary.get("stage") != "L004":
        raise RuntimeError("validation summary schema/stage changed")
    for payload in (config, summary):
        if payload.get("evaluation_id") != "base50m-validation-v1":
            raise RuntimeError("validation evaluation id changed")
        if payload.get("evaluation_only") is not True:
            raise RuntimeError("validation must remain evaluation-only")
        if payload.get("partition") != "validation":
            raise RuntimeError("validation partition changed")
        if payload.get("final_holdout_touched") is not False:
            raise RuntimeError("final holdout must remain untouched")
    if summary.get("total_selected_content_tokens") != 300_000:
        raise RuntimeError("validation token total changed")
    if summary.get("production_optimizer_steps_recorded") != 4:
        raise RuntimeError("validation summary production step changed")
    return config, summary


def _load_model(
    model_name: str,
    *,
    config: dict,
    execution,
    external_root: Path,
) -> tuple[BlankCausalLM, dict]:
    if model_name == "step0":
        step0 = config["step0"]
        if step0 != {"seed": 1234, "optimizer_steps": 0}:
            raise RuntimeError("step0 contract changed")
        torch.manual_seed(1234)
        model = BlankCausalLM(execution.model_config)
        return model, {
            "model": "step0",
            "optimizer_step": 0,
            "supervised_tokens_seen": 0,
            "checkpoint_sha256": None,
        }

    step4 = config["step4"]
    if step4.get("optimizer_steps") != 4:
        raise RuntimeError("step4 optimizer-step contract changed")
    if step4.get("supervised_tokens_seen") != 262_144:
        raise RuntimeError("step4 supervised-token contract changed")
    checkpoint = resolve_external_storage_uri(
        step4["checkpoint"],
        external_root=external_root,
    )
    if checkpoint.stat().st_size != int(step4["bytes"]):
        raise RuntimeError("step4 checkpoint byte size mismatch")
    observed_sha = sha256_file(checkpoint)
    if observed_sha != step4["sha256"]:
        raise RuntimeError("step4 checkpoint SHA-256 mismatch")
    loaded = load_pretraining_checkpoint(
        checkpoint,
        expected_lineage=lineage_from_contract(execution),
    )
    if loaded.progress.optimizer_step != 4:
        raise RuntimeError("loaded checkpoint is not optimizer step 4")
    if loaded.progress.supervised_tokens_seen != 262_144:
        raise RuntimeError("loaded checkpoint supervised-token count changed")
    return loaded.model, {
        "model": "step4",
        "optimizer_step": 4,
        "supervised_tokens_seen": 262_144,
        "checkpoint_sha256": observed_sha,
    }


def _evaluate_pack(
    model: BlankCausalLM,
    path: Path,
    *,
    meta: dict,
    batch_size: int,
    max_length: int,
    eos_id: int,
    ignore_index: int,
) -> dict:
    if path.stat().st_size != int(meta["bytes"]):
        raise RuntimeError("validation pack byte size mismatch")
    observed_sha = sha256_file(path)
    if observed_sha != meta["sha256"]:
        raise RuntimeError("validation pack SHA-256 mismatch")
    tokens = np.memmap(path, dtype="<u2", mode="r")
    if int(tokens.size) != int(meta["physical_tokens"]):
        raise RuntimeError("validation pack physical-token count changed")
    if int(tokens[-1]) != eos_id:
        raise RuntimeError("validation pack must end in EOS")

    blocks: list[tuple[torch.Tensor, torch.Tensor, int]] = []
    total_loss = 0.0
    total_supervised = 0
    total_input_positions = 0
    forward_batches = 0

    def flush() -> None:
        nonlocal total_loss, total_supervised, forward_batches
        if not blocks:
            return
        lengths = {int(item[0].shape[0]) for item in blocks}
        if len(lengths) != 1:
            raise RuntimeError("validation batch contains mixed sequence lengths")
        input_ids = torch.stack([item[0] for item in blocks], dim=0)
        targets = torch.stack([item[1] for item in blocks], dim=0)
        supervised = sum(item[2] for item in blocks)
        with torch.inference_mode():
            output = model(input_ids, targets=targets)
        if output.loss is None or not bool(torch.isfinite(output.loss)):
            raise RuntimeError("validation loss is not finite")
        total_loss += float(output.loss.detach().cpu()) * supervised
        total_supervised += supervised
        forward_batches += 1
        blocks.clear()

    input_positions = int(tokens.size) - 1
    offset = 0
    while offset < input_positions:
        length = min(max_length, input_positions - offset)
        input_np = np.asarray(tokens[offset : offset + length])
        target_np = np.asarray(tokens[offset + 1 : offset + length + 1])
        input_ids = torch.tensor(input_np, dtype=torch.long)
        targets = torch.tensor(target_np, dtype=torch.long)
        targets[input_ids == eos_id] = ignore_index
        supervised = int(torch.count_nonzero(targets != ignore_index).item())
        if supervised <= 0:
            raise RuntimeError("validation block has no supervised positions")
        if blocks and int(blocks[0][0].shape[0]) != length:
            flush()
        blocks.append((input_ids, targets, supervised))
        total_input_positions += length
        if len(blocks) >= batch_size:
            flush()
        offset += length
    flush()
    del tokens

    if total_supervised != int(meta["supervised_positions"]):
        raise RuntimeError(
            "validation supervised total mismatch "
            f"expected={meta['supervised_positions']} "
            f"observed={total_supervised}"
        )
    mean_loss = total_loss / total_supervised
    return {
        "loss": mean_loss,
        "perplexity": math.exp(mean_loss),
        "supervised_tokens": total_supervised,
        "physical_input_positions": total_input_positions,
        "forward_batches": forward_batches,
        "pack_sha256": observed_sha,
    }


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    external_root = Path(args.external_root)
    config, summary = _load_contracts(
        Path(args.config),
        Path(args.summary),
    )
    runtime = config["evaluation_runtime"]
    if runtime != {
        "device": "cpu",
        "precision": "fp32",
        "threads": 12,
        "batch_size": 4,
    }:
        raise RuntimeError("validation runtime contract changed")
    torch.set_num_threads(12)

    execution = load_pretraining_execution_contract(
        ROOT / "configs" / "pretraining-execution-l004.json",
        repo_root=ROOT,
        external_root=external_root,
    )
    model, model_meta = _load_model(
        args.model,
        config=config,
        execution=execution,
        external_root=external_root,
    )
    model.eval()

    packing = config["packing"]
    if (
        packing.get("max_physical_sequence_length") != 1024
        or packing.get("eos_id") != 2
        or packing.get("ignore_index") != -100
        or packing.get("segment_attention_reset") is not False
    ):
        raise RuntimeError("validation packing/loss contract changed")

    results: dict[str, dict] = {}
    combined_weighted = 0.0
    combined_tokens = 0
    for category in sorted(summary["packs"]):
        meta = summary["packs"][category]
        pack_path = resolve_external_storage_uri(
            meta["storage"],
            external_root=external_root,
        )
        result = _evaluate_pack(
            model,
            pack_path,
            meta=meta,
            batch_size=4,
            max_length=1024,
            eos_id=2,
            ignore_index=-100,
        )
        results[category] = result
        combined_weighted += result["loss"] * result["supervised_tokens"]
        combined_tokens += result["supervised_tokens"]
        print(
            f"PASS model={args.model} category={category} "
            f"loss={result['loss']:.9f} "
            f"tokens={result['supervised_tokens']}",
            flush=True,
        )

    if combined_tokens != 300_000:
        raise RuntimeError("combined validation token count changed")
    combined_loss = combined_weighted / combined_tokens
    output = {
        "schema_version": 1,
        "stage": "L004",
        "evaluation_id": config["evaluation_id"],
        "evaluation_only": True,
        "partition": "validation",
        "final_holdout_touched": False,
        **model_meta,
        "runtime": runtime,
        "categories": results,
        "combined": {
            "loss": combined_loss,
            "perplexity": math.exp(combined_loss),
            "supervised_tokens": combined_tokens,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "pretraining_authorized": False,
        "public_release_eligibility": "not_qualified",
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        f"combined_loss={combined_loss:.9f} "
        f"combined_tokens={combined_tokens}"
    )
    print(f"validation_{args.model}=PASS")


if __name__ == "__main__":
    main()
