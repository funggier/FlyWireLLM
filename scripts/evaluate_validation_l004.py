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
from flywire_llm.pretraining_continuation import (
    load_continuation_authorization,
    load_continuation_state,
)
from flywire_llm.pretraining_execution import (
    load_pretraining_execution_contract,
)
from flywire_llm.pretraining_expansion import (
    load_expansion_authorization,
    load_expansion_state,
)
from flywire_llm.pretraining_scaling import (
    load_scaling_authorization,
    load_scaling_state,
)
from flywire_llm.pretraining_rolling import (
    load_rolling_authorization,
    load_rolling_state,
)
from flywire_llm.pretraining_rolling_expansion import (
    load_rolling_expansion_authorization,
    load_rolling_expansion_state,
)
from flywire_llm.pretraining_rolling_scaling import (
    load_rolling_scaling_authorization,
    load_rolling_scaling_state,
)
from flywire_llm.pretraining_warmup_guard import (
    load_warmup_guard_authorization,
    load_warmup_guard_state,
)
from flywire_llm.pretraining_warmup_followup import (
    load_warmup_followup_authorization,
    load_warmup_followup_state,
)
from flywire_llm.pretraining_warmup_penultimate import (
    load_warmup_penultimate_authorization,
    load_warmup_penultimate_state,
)
from flywire_llm.pretraining_warmup_boundary import (
    load_warmup_boundary_authorization,
    load_warmup_boundary_state,
)
from flywire_llm.pretraining_post_warmup import (
    load_post_warmup_authorization,
    load_post_warmup_state,
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
        choices=("step0", "step4", "step8", "step16", "step32", "step40", "step56", "step88", "step104", "step120", "step136", "step153", "step169"),
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

    if model_name == "step4":
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
            raise RuntimeError(
                "loaded checkpoint supervised-token count changed"
            )
        return loaded.model, {
            "model": "step4",
            "optimizer_step": 4,
            "supervised_tokens_seen": 262_144,
            "checkpoint_sha256": observed_sha,
        }

    if model_name == "step8":
        authorization = load_continuation_authorization(
            ROOT / "configs" / "pretraining-tranche-l004-v2.json",
            repo_root=ROOT,
            external_root=external_root,
        )
        state = load_continuation_state(
            authorization.run_root / "state.json",
            authorization=authorization,
            verify_files=True,
        )
        if state.optimizer_step != 8:
            raise RuntimeError("continuation state is not optimizer step 8")
        if state.supervised_tokens_seen != 524_288:
            raise RuntimeError("step8 supervised-token progress changed")
        checkpoint = authorization.run_root / state.checkpoint_file
        observed_sha = sha256_file(checkpoint)
        if observed_sha != state.checkpoint_sha256:
            raise RuntimeError("step8 checkpoint SHA-256 mismatch")
        loaded = load_pretraining_checkpoint(
            checkpoint,
            expected_lineage=lineage_from_contract(execution),
        )
        if loaded.progress.optimizer_step != 8:
            raise RuntimeError("loaded checkpoint is not optimizer step 8")
        if loaded.progress.supervised_tokens_seen != 524_288:
            raise RuntimeError("loaded step8 supervised-token count changed")
        return loaded.model, {
            "model": "step8",
            "optimizer_step": 8,
            "supervised_tokens_seen": 524_288,
            "checkpoint_sha256": observed_sha,
            "continuation_authorization_sha256": (
                authorization.authorization_sha256
            ),
        }

    if model_name == "step16":
        authorization = load_expansion_authorization(
            ROOT / "configs" / "pretraining-tranche-l004-v3.json",
            repo_root=ROOT,
            external_root=external_root,
        )
        state = load_expansion_state(
            authorization.run_root / "state.json",
            authorization=authorization,
            verify_files=True,
        )
        if state.optimizer_step != 16:
            raise RuntimeError("expansion state is not optimizer step 16")
        if state.supervised_tokens_seen != 1_048_576:
            raise RuntimeError("step16 supervised-token progress changed")
        checkpoint = authorization.run_root / state.checkpoint_file
        observed_sha = sha256_file(checkpoint)
        if observed_sha != state.checkpoint_sha256:
            raise RuntimeError("step16 checkpoint SHA-256 mismatch")
        loaded = load_pretraining_checkpoint(
            checkpoint,
            expected_lineage=lineage_from_contract(execution),
        )
        if loaded.progress.optimizer_step != 16:
            raise RuntimeError("loaded checkpoint is not optimizer step 16")
        if loaded.progress.supervised_tokens_seen != 1_048_576:
            raise RuntimeError("loaded step16 supervised-token count changed")
        return loaded.model, {
            "model": "step16",
            "optimizer_step": 16,
            "supervised_tokens_seen": 1_048_576,
            "checkpoint_sha256": observed_sha,
            "expansion_authorization_sha256": (
                authorization.authorization_sha256
            ),
        }

    if model_name == "step32":
        authorization = load_scaling_authorization(
            ROOT / "configs" / "pretraining-tranche-l004-v4.json",
            repo_root=ROOT,
            external_root=external_root,
        )
        state = load_scaling_state(
            authorization.run_root / "state.json",
            authorization=authorization,
            verify_files=True,
        )
        if state.optimizer_step != 32:
            raise RuntimeError("scaling state is not optimizer step 32")
        if state.supervised_tokens_seen != 2_097_152:
            raise RuntimeError("step32 supervised-token progress changed")
        checkpoint = authorization.run_root / state.checkpoint_file
        observed_sha = sha256_file(checkpoint)
        if observed_sha != state.checkpoint_sha256:
            raise RuntimeError("step32 checkpoint SHA-256 mismatch")
        loaded = load_pretraining_checkpoint(
            checkpoint,
            expected_lineage=lineage_from_contract(execution),
        )
        if loaded.progress.optimizer_step != 32:
            raise RuntimeError("loaded checkpoint is not optimizer step 32")
        if loaded.progress.supervised_tokens_seen != 2_097_152:
            raise RuntimeError("loaded step32 supervised-token count changed")
        return loaded.model, {
            "model": "step32",
            "optimizer_step": 32,
            "supervised_tokens_seen": 2_097_152,
            "checkpoint_sha256": observed_sha,
            "scaling_authorization_sha256": (
                authorization.authorization_sha256
            ),
        }

    if model_name == "step40":
        authorization = load_rolling_authorization(
            ROOT / "configs" / "pretraining-tranche-l004-v5.json",
            repo_root=ROOT,
            external_root=external_root,
        )
        state = load_rolling_state(
            authorization.run_root / "state.json",
            authorization=authorization,
            verify_files=True,
        )
        if state.optimizer_step != 40:
            raise RuntimeError("rolling state is not optimizer step 40")
        if state.supervised_tokens_seen != 2_621_440:
            raise RuntimeError("step40 supervised-token progress changed")
        checkpoint = authorization.run_root / state.checkpoint_file
        observed_sha = sha256_file(checkpoint)
        if observed_sha != state.checkpoint_sha256:
            raise RuntimeError("step40 checkpoint SHA-256 mismatch")
        loaded = load_pretraining_checkpoint(
            checkpoint,
            expected_lineage=lineage_from_contract(execution),
        )
        if loaded.progress.optimizer_step != 40:
            raise RuntimeError("loaded checkpoint is not optimizer step 40")
        if loaded.progress.supervised_tokens_seen != 2_621_440:
            raise RuntimeError("loaded step40 supervised-token count changed")
        return loaded.model, {
            "model": "step40",
            "optimizer_step": 40,
            "supervised_tokens_seen": 2_621_440,
            "checkpoint_sha256": observed_sha,
            "rolling_authorization_sha256": (
                authorization.authorization_sha256
            ),
        }

    if model_name == "step56":
        authorization = load_rolling_expansion_authorization(
            ROOT / "configs" / "pretraining-tranche-l004-v6.json",
            repo_root=ROOT,
            external_root=external_root,
        )
        state = load_rolling_expansion_state(
            authorization.run_root / "state.json",
            authorization=authorization,
            verify_files=True,
        )
        if state.optimizer_step != 56:
            raise RuntimeError("rolling expansion state is not optimizer step 56")
        if state.supervised_tokens_seen != 3_670_016:
            raise RuntimeError("step56 supervised-token progress changed")
        checkpoint = authorization.run_root / state.checkpoint_file
        observed_sha = sha256_file(checkpoint)
        if observed_sha != state.checkpoint_sha256:
            raise RuntimeError("step56 checkpoint SHA-256 mismatch")
        loaded = load_pretraining_checkpoint(
            checkpoint,
            expected_lineage=lineage_from_contract(execution),
        )
        if loaded.progress.optimizer_step != 56:
            raise RuntimeError("loaded checkpoint is not optimizer step 56")
        if loaded.progress.supervised_tokens_seen != 3_670_016:
            raise RuntimeError("loaded step56 supervised-token count changed")
        return loaded.model, {
            "model": "step56",
            "optimizer_step": 56,
            "supervised_tokens_seen": 3_670_016,
            "checkpoint_sha256": observed_sha,
            "rolling_expansion_authorization_sha256": (
                authorization.authorization_sha256
            ),
        }

    if model_name == "step88":
        authorization = load_rolling_scaling_authorization(
            ROOT / "configs" / "pretraining-tranche-l004-v7.json",
            repo_root=ROOT,
            external_root=external_root,
        )
        state = load_rolling_scaling_state(
            authorization.run_root / "state.json",
            authorization=authorization,
            verify_files=True,
        )
        if state.optimizer_step != 88:
            raise RuntimeError("rolling scaling state is not optimizer step 88")
        if state.supervised_tokens_seen != 5_767_168:
            raise RuntimeError("step88 supervised-token progress changed")
        checkpoint = authorization.run_root / state.checkpoint_file
        observed_sha = sha256_file(checkpoint)
        if observed_sha != state.checkpoint_sha256:
            raise RuntimeError("step88 checkpoint SHA-256 mismatch")
        loaded = load_pretraining_checkpoint(
            checkpoint,
            expected_lineage=lineage_from_contract(execution),
        )
        if loaded.progress.optimizer_step != 88:
            raise RuntimeError("loaded checkpoint is not optimizer step 88")
        if loaded.progress.supervised_tokens_seen != 5_767_168:
            raise RuntimeError("loaded step88 supervised-token count changed")
        return loaded.model, {
            "model": "step88",
            "optimizer_step": 88,
            "supervised_tokens_seen": 5_767_168,
            "checkpoint_sha256": observed_sha,
            "rolling_scaling_authorization_sha256": (
                authorization.authorization_sha256
            ),
        }

    if model_name == "step104":
        authorization = load_warmup_guard_authorization(
            ROOT / "configs" / "pretraining-tranche-l004-v8.json",
            repo_root=ROOT,
            external_root=external_root,
        )
        state = load_warmup_guard_state(
            authorization.run_root / "state.json",
            authorization=authorization,
            verify_files=True,
        )
        if state.optimizer_step != 104:
            raise RuntimeError("warmup guard state is not optimizer step 104")
        if state.supervised_tokens_seen != 6_815_744:
            raise RuntimeError("step104 supervised-token progress changed")
        checkpoint = authorization.run_root / state.checkpoint_file
        observed_sha = sha256_file(checkpoint)
        if observed_sha != state.checkpoint_sha256:
            raise RuntimeError("step104 checkpoint SHA-256 mismatch")
        loaded = load_pretraining_checkpoint(
            checkpoint,
            expected_lineage=lineage_from_contract(execution),
        )
        if loaded.progress.optimizer_step != 104:
            raise RuntimeError("loaded checkpoint is not optimizer step 104")
        if loaded.progress.supervised_tokens_seen != 6_815_744:
            raise RuntimeError("loaded step104 supervised-token count changed")
        return loaded.model, {
            "model": "step104",
            "optimizer_step": 104,
            "supervised_tokens_seen": 6_815_744,
            "checkpoint_sha256": observed_sha,
            "warmup_guard_authorization_sha256": (
                authorization.authorization_sha256
            ),
        }

    if model_name == "step120":
        authorization = load_warmup_followup_authorization(
            ROOT / "configs" / "pretraining-tranche-l004-v9.json",
            repo_root=ROOT,
            external_root=external_root,
        )
        state = load_warmup_followup_state(
            authorization.run_root / "state.json",
            authorization=authorization,
            verify_files=True,
        )
        if state.optimizer_step != 120:
            raise RuntimeError("warmup follow-up state is not optimizer step 120")
        if state.supervised_tokens_seen != 7_864_320:
            raise RuntimeError("step120 supervised-token progress changed")
        checkpoint = authorization.run_root / state.checkpoint_file
        observed_sha = sha256_file(checkpoint)
        if observed_sha != state.checkpoint_sha256:
            raise RuntimeError("step120 checkpoint SHA-256 mismatch")
        loaded = load_pretraining_checkpoint(
            checkpoint,
            expected_lineage=lineage_from_contract(execution),
        )
        if loaded.progress.optimizer_step != 120:
            raise RuntimeError("loaded checkpoint is not optimizer step 120")
        if loaded.progress.supervised_tokens_seen != 7_864_320:
            raise RuntimeError("loaded step120 supervised-token count changed")
        return loaded.model, {
            "model": "step120",
            "optimizer_step": 120,
            "supervised_tokens_seen": 7_864_320,
            "checkpoint_sha256": observed_sha,
            "warmup_followup_authorization_sha256": (
                authorization.authorization_sha256
            ),
        }

    if model_name == "step136":
        authorization = load_warmup_penultimate_authorization(
            ROOT / "configs" / "pretraining-tranche-l004-v10.json",
            repo_root=ROOT,
            external_root=external_root,
        )
        state = load_warmup_penultimate_state(
            authorization.run_root / "state.json",
            authorization=authorization,
            verify_files=True,
        )
        if state.optimizer_step != 136:
            raise RuntimeError("warmup penultimate state is not optimizer step 136")
        if state.supervised_tokens_seen != 8_912_896:
            raise RuntimeError("step136 supervised-token progress changed")
        checkpoint = authorization.run_root / state.checkpoint_file
        observed_sha = sha256_file(checkpoint)
        if observed_sha != state.checkpoint_sha256:
            raise RuntimeError("step136 checkpoint SHA-256 mismatch")
        loaded = load_pretraining_checkpoint(
            checkpoint,
            expected_lineage=lineage_from_contract(execution),
        )
        if loaded.progress.optimizer_step != 136:
            raise RuntimeError("loaded checkpoint is not optimizer step 136")
        if loaded.progress.supervised_tokens_seen != 8_912_896:
            raise RuntimeError("loaded step136 supervised-token count changed")
        return loaded.model, {
            "model": "step136",
            "optimizer_step": 136,
            "supervised_tokens_seen": 8_912_896,
            "checkpoint_sha256": observed_sha,
            "warmup_penultimate_authorization_sha256": (
                authorization.authorization_sha256
            ),
        }

    if model_name == "step153":
        authorization = load_warmup_boundary_authorization(
            ROOT / "configs" / "pretraining-tranche-l004-v11.json",
            repo_root=ROOT,
            external_root=external_root,
        )
        state = load_warmup_boundary_state(
            authorization.run_root / "state.json",
            authorization=authorization,
            verify_files=True,
        )
        if state.optimizer_step != 153:
            raise RuntimeError("warmup boundary state is not optimizer step 153")
        if state.supervised_tokens_seen != 10_027_008:
            raise RuntimeError("step153 supervised-token progress changed")
        checkpoint = authorization.run_root / state.checkpoint_file
        observed_sha = sha256_file(checkpoint)
        if observed_sha != state.checkpoint_sha256:
            raise RuntimeError("step153 checkpoint SHA-256 mismatch")
        loaded = load_pretraining_checkpoint(
            checkpoint,
            expected_lineage=lineage_from_contract(execution),
        )
        if loaded.progress.optimizer_step != 153:
            raise RuntimeError("loaded checkpoint is not optimizer step 153")
        if loaded.progress.supervised_tokens_seen != 10_027_008:
            raise RuntimeError("loaded step153 supervised-token count changed")
        return loaded.model, {
            "model": "step153",
            "optimizer_step": 153,
            "supervised_tokens_seen": 10_027_008,
            "checkpoint_sha256": observed_sha,
            "warmup_boundary_authorization_sha256": (
                authorization.authorization_sha256
            ),
        }

    authorization = load_post_warmup_authorization(
        ROOT / "configs" / "pretraining-tranche-l004-v12.json",
        repo_root=ROOT,
        external_root=external_root,
    )
    state = load_post_warmup_state(
        authorization.run_root / "state.json",
        authorization=authorization,
        verify_files=True,
    )
    if state.optimizer_step != 169:
        raise RuntimeError("post-warmup state is not optimizer step 169")
    if state.supervised_tokens_seen != 11_075_584:
        raise RuntimeError("step169 supervised-token progress changed")
    checkpoint = authorization.run_root / state.checkpoint_file
    observed_sha = sha256_file(checkpoint)
    if observed_sha != state.checkpoint_sha256:
        raise RuntimeError("step169 checkpoint SHA-256 mismatch")
    loaded = load_pretraining_checkpoint(
        checkpoint,
        expected_lineage=lineage_from_contract(execution),
    )
    if loaded.progress.optimizer_step != 169:
        raise RuntimeError("loaded checkpoint is not optimizer step 169")
    if loaded.progress.supervised_tokens_seen != 11_075_584:
        raise RuntimeError("loaded step169 supervised-token count changed")
    return loaded.model, {
        "model": "step169",
        "optimizer_step": 169,
        "supervised_tokens_seen": 11_075_584,
        "checkpoint_sha256": observed_sha,
        "post_warmup_authorization_sha256": (
            authorization.authorization_sha256
        ),
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
