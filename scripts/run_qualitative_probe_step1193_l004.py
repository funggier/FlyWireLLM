from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import torch

from flywire_llm.pretraining_checkpoint import (
    lineage_from_contract,
    load_pretraining_checkpoint,
)
from flywire_llm.pretraining_execution import (
    load_pretraining_execution_contract,
    verify_base50m_tokenizer,
)
from flywire_llm.pretraining_post_warmup_extended import (
    load_post_warmup_extended_authorization,
    load_post_warmup_extended_state,
)
from flywire_llm.pretraining_post_warmup_sustained import (
    load_post_warmup_sustained_authorization,
    load_post_warmup_sustained_state,
)


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--external-root", required=True)
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "qualitative-probe-l004-v2.json"),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("step681", "step1193"),
        default=["step681", "step1193"],
    )
    parser.add_argument(
        "--output-json",
        default=str(
            ROOT
            / "results"
            / "l004"
            / "qualitative-probe-step681-step1193-v2.json"
        ),
    )
    return parser.parse_args()


def load_probe_config(path: str | Path) -> dict:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("qualitative probe config must be a JSON object")
    expected = {
        "schema_version": 1,
        "stage": "L004",
        "probe_id": "base50m-qualitative-generation-v2",
        "observational_only": True,
        "evaluation_only": True,
        "synthetic_prompts_only": True,
        "source_partitions_read": [],
        "final_holdout_touched": False,
        "models": ["step681", "step1193"],
        "runtime": {
            "device": "cpu",
            "precision": "fp32",
            "threads": 12,
        },
        "decoding": {
            "method": "greedy",
            "max_new_tokens": 32,
            "stop_on_eos": True,
            "add_bos": False,
        },
        "public_release_eligibility": "not_qualified",
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"qualitative probe contract changed: {key}")

    prompts = payload.get("prompts")
    if not isinstance(prompts, list) or len(prompts) != 12:
        raise RuntimeError("qualitative probe must contain exactly 12 prompts")
    ids: set[str] = set()
    categories: dict[str, int] = {
        "general_thai": 0,
        "general_english": 0,
        "technical_scientific_code": 0,
    }
    for row in prompts:
        if not isinstance(row, dict):
            raise RuntimeError("qualitative prompt must be an object")
        if set(row) != {"id", "category", "kind", "text"}:
            raise RuntimeError("qualitative prompt fields changed")
        prompt_id = row["id"]
        if not isinstance(prompt_id, str) or not prompt_id or prompt_id in ids:
            raise RuntimeError("qualitative prompt id is invalid or duplicated")
        ids.add(prompt_id)
        category = row["category"]
        if category not in categories:
            raise RuntimeError("qualitative prompt category changed")
        categories[category] += 1
        if not isinstance(row["kind"], str) or not row["kind"]:
            raise RuntimeError("qualitative prompt kind is invalid")
        if not isinstance(row["text"], str) or not row["text"]:
            raise RuntimeError("qualitative prompt text is invalid")
    if categories != {
        "general_thai": 4,
        "general_english": 4,
        "technical_scientific_code": 4,
    }:
        raise RuntimeError("qualitative prompt category balance changed")
    return payload


def _load_model(model_name: str, *, execution, external_root: Path):
    if model_name == "step681":
        authorization = load_post_warmup_extended_authorization(
            ROOT / "configs" / "pretraining-tranche-l004-v17.json",
            repo_root=ROOT,
            external_root=external_root,
        )
        state = load_post_warmup_extended_state(
            authorization.run_root / "state.json",
            authorization=authorization,
            verify_files=True,
        )
        expected_step = 681
        expected_tokens = 44_630_016
        auth_field = "post_warmup_extended_authorization_sha256"
    elif model_name == "step1193":
        authorization = load_post_warmup_sustained_authorization(
            ROOT / "configs" / "pretraining-tranche-l004-v18.json",
            repo_root=ROOT,
            external_root=external_root,
        )
        state = load_post_warmup_sustained_state(
            authorization.run_root / "state.json",
            authorization=authorization,
            verify_files=True,
        )
        expected_step = 1193
        expected_tokens = 78_184_448
        auth_field = "post_warmup_sustained_authorization_sha256"
    else:
        raise RuntimeError(f"unsupported qualitative probe model: {model_name}")

    if state.optimizer_step != expected_step:
        raise RuntimeError(f"{model_name} optimizer step changed")
    if state.supervised_tokens_seen != expected_tokens:
        raise RuntimeError(f"{model_name} supervised-token progress changed")
    checkpoint = authorization.run_root / state.checkpoint_file
    observed_sha = sha256_file(checkpoint)
    if observed_sha != state.checkpoint_sha256:
        raise RuntimeError(f"{model_name} checkpoint SHA-256 mismatch")
    loaded = load_pretraining_checkpoint(
        checkpoint,
        expected_lineage=lineage_from_contract(execution),
    )
    if loaded.progress.optimizer_step != expected_step:
        raise RuntimeError(f"loaded checkpoint is not {model_name}")
    if loaded.progress.supervised_tokens_seen != expected_tokens:
        raise RuntimeError(f"loaded {model_name} supervised-token count changed")
    return loaded.model, {
        "model": model_name,
        "optimizer_step": expected_step,
        "supervised_tokens_seen": expected_tokens,
        "checkpoint_sha256": observed_sha,
        auth_field: authorization.authorization_sha256,
    }


def greedy_generate_ids(
    model,
    prompt_ids: list[int],
    *,
    eos_id: int,
    max_new_tokens: int,
) -> tuple[list[int], bool]:
    if not prompt_ids:
        raise RuntimeError("qualitative prompt tokenization is empty")
    if max_new_tokens <= 0:
        raise RuntimeError("max_new_tokens must be positive")
    context = [int(value) for value in prompt_ids]
    generated: list[int] = []
    eos_reached = False
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            window = context[-int(model.config.max_seq_len) :]
            input_ids = torch.tensor([window], dtype=torch.long)
            output = model(input_ids)
            next_id = int(torch.argmax(output.logits[0, -1]).item())
            generated.append(next_id)
            context.append(next_id)
            if next_id == eos_id:
                eos_reached = True
                break
    return generated, eos_reached


def main() -> None:
    args = parse_args()
    external_root = Path(args.external_root)
    config_path = Path(args.config)
    config = load_probe_config(config_path)
    requested_models = list(args.models)
    if len(set(requested_models)) != len(requested_models):
        raise RuntimeError("qualitative probe models must not be duplicated")

    torch.set_num_threads(int(config["runtime"]["threads"]))
    execution = load_pretraining_execution_contract(
        ROOT / "configs" / "pretraining-execution-l004.json",
        repo_root=ROOT,
        external_root=external_root,
    )
    tokenizer = verify_base50m_tokenizer(
        execution.tokenizer_path,
        expected_sha256=execution.tokenizer_sha256,
        expected_vocab_size=execution.tokenizer_vocab_size,
    )

    decoding = config["decoding"]
    model_results: dict[str, dict] = {}
    for model_name in requested_models:
        model, model_meta = _load_model(
            model_name,
            execution=execution,
            external_root=external_root,
        )
        model.eval()
        rows: list[dict] = []
        for prompt in config["prompts"]:
            prompt_text = prompt["text"]
            prompt_ids = tokenizer.encode(
                prompt_text,
                add_bos=bool(decoding["add_bos"]),
                add_eos=False,
            )
            if tokenizer.decode(prompt_ids) != prompt_text:
                raise RuntimeError(
                    f"prompt roundtrip changed for {prompt['id']}"
                )
            generated_ids, eos_reached = greedy_generate_ids(
                model,
                prompt_ids,
                eos_id=tokenizer.eos_id,
                max_new_tokens=int(decoding["max_new_tokens"]),
            )
            full_ids = prompt_ids + generated_ids
            full_text = tokenizer.decode(full_ids)
            if not full_text.startswith(prompt_text):
                raise RuntimeError(
                    f"generated decode lost prompt prefix for {prompt['id']}"
                )
            completion = full_text[len(prompt_text) :]
            row = {
                **prompt,
                "prompt_token_count": len(prompt_ids),
                "generated_token_ids": generated_ids,
                "generated_token_count": len(generated_ids),
                "eos_reached": eos_reached,
                "completion": completion,
                "full_text": full_text,
            }
            rows.append(row)
            print(
                f"PASS model={model_name} prompt={prompt['id']} "
                f"generated_tokens={len(generated_ids)} "
                f"eos_reached={str(eos_reached).lower()}",
                flush=True,
            )
        model_results[model_name] = {
            **model_meta,
            "prompts": rows,
        }
        del model
        gc.collect()

    output = {
        "schema_version": 1,
        "stage": "L004",
        "probe_id": config["probe_id"],
        "observational_only": True,
        "evaluation_only": True,
        "synthetic_prompts_only": True,
        "source_partitions_read": [],
        "final_holdout_touched": False,
        "config_sha256": sha256_file(config_path),
        "tokenizer_sha256": execution.tokenizer_sha256,
        "tokenizer_vocab_size": execution.tokenizer_vocab_size,
        "runtime": config["runtime"],
        "decoding": config["decoding"],
        "models": model_results,
        "public_release_eligibility": "not_qualified",
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print("final_holdout_touched=false")
    print("qualitative_probe_observational_only=true")
    print("qualitative_probe=PASS")


if __name__ == "__main__":
    main()
