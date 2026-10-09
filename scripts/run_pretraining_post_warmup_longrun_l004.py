from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import torch

from flywire_llm.pretraining_checkpoint import (
    lineage_from_contract,
    load_pretraining_checkpoint,
    restore_pretraining_rng,
    save_pretraining_checkpoint,
)
from flywire_llm.pretraining_data import (
    PackedPretrainingData,
    cursor_from_state_dict,
    cursor_state_dict,
    load_pretraining_data_contract,
)
from flywire_llm.pretraining_execution import (
    load_pretraining_execution_contract,
)
from flywire_llm.pretraining_post_warmup_longrun import (
    PretrainingPostWarmupLongrunError,
    atomic_write_json,
    build_post_warmup_longrun_state_payload,
    checkpoint_filename,
    load_post_warmup_longrun_authorization,
    load_post_warmup_longrun_state,
    metric_filename,
    prune_prior_checkpoint,
    sha256_file,
)
from flywire_llm.pretraining_training import (
    build_adamw,
    build_primary_schedule,
    gradient_clip_norm,
    learning_rate_for_update,
    load_training_configuration,
    pretraining_update,
)


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Execute the authorized post-warmup Base-50M pretraining "
            "eighth post-warmup cosine-decay longrun tranche from optimizer step 1193 through step 2217."
        )
    )
    parser.add_argument("--external-root", required=True)
    parser.add_argument(
        "--authorization",
        default=str(
            ROOT / "configs" / "pretraining-tranche-l004-v19.json"
        ),
    )
    parser.add_argument("--stop-after-updates", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=ROOT,
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


def _validate_git_state(authorization) -> str:
    if _git("status", "--porcelain"):
        raise PretrainingPostWarmupLongrunError(
            "production rolling requires a clean Git worktree"
        )
    head = _git("rev-parse", "HEAD")
    for name, commit in (
        ("runner_commit", authorization.runner_commit),
        ("evidence_commit", authorization.evidence_commit),
    ):
        completed = subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, head],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if completed.returncode != 0:
            raise PretrainingPostWarmupLongrunError(
                f"{name} is not an ancestor of current HEAD"
            )
    return head


def _load_local_runtime() -> dict:
    payload = json.loads(
        (
            ROOT / "configs" / "pretraining-local-runtime-l004.json"
        ).read_text(encoding="utf-8")
    )
    expected = {
        "schema_version": 1,
        "stage": "L004",
        "runtime_id": "base50m-local-cpu-v1",
        "device": "cpu",
        "precision": "fp32",
        "threads": 12,
        "micro_batch_size": 1,
        "max_physical_sequence_length": 1024,
        "global_supervised_tokens_per_update": 65_536,
        "optimizer_steps": 7_630,
        "full_updates": 7_629,
        "final_update_supervised_tokens": 25_856,
        "bf16_status": "rejected_for_current_cpu_runtime",
        "production_optimizer_steps_recorded": 0,
        "public_release_eligibility": "not_qualified",
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise PretrainingPostWarmupLongrunError(
                f"local runtime {key} changed"
            )
    return payload


def _validate_free_space(run_root: Path) -> None:
    probe = run_root
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    if shutil.disk_usage(probe).free < 8 * 1024**3:
        raise PretrainingPostWarmupLongrunError(
            "fewer than 8 GiB free on post-warmup run volume"
        )


def _source_runtime(authorization, execution, data_contract, training):
    loaded = load_pretraining_checkpoint(
        authorization.source.checkpoint,
        expected_lineage=lineage_from_contract(execution),
    )
    expected = authorization.source
    if loaded.progress.optimizer_step != expected.optimizer_step:
        raise PretrainingPostWarmupLongrunError(
            "source checkpoint optimizer step changed"
        )
    if (
        loaded.progress.supervised_tokens_seen
        != expected.supervised_tokens_seen
    ):
        raise PretrainingPostWarmupLongrunError(
            "source checkpoint supervised progress changed"
        )
    if (
        loaded.progress.physical_input_positions_seen
        != expected.physical_input_positions_seen
    ):
        raise PretrainingPostWarmupLongrunError(
            "source checkpoint physical progress changed"
        )
    if loaded.progress.microbatches_seen != expected.microbatches_seen:
        raise PretrainingPostWarmupLongrunError(
            "source checkpoint microbatch progress changed"
        )
    if loaded.data_state != expected.data_state:
        raise PretrainingPostWarmupLongrunError(
            "source checkpoint data state changed"
        )
    optimizer = build_adamw(loaded.model, training)
    optimizer.load_state_dict(loaded.optimizer_state)
    data_generator = torch.Generator()
    restore_pretraining_rng(loaded, data_generator=data_generator)
    cursor = cursor_from_state_dict(
        loaded.data_state,
        contract=data_contract,
    )
    return (
        loaded.model,
        optimizer,
        data_generator,
        loaded.progress,
        cursor,
    )


def _resume_runtime(authorization, execution, data_contract, training):
    state_path = authorization.run_root / "state.json"
    if not state_path.is_file():
        raise PretrainingPostWarmupLongrunError(
            "resume requested but post-warmup state.json is missing"
        )
    state = load_post_warmup_longrun_state(
        state_path,
        authorization=authorization,
        verify_files=True,
    )
    checkpoint = authorization.run_root / state.checkpoint_file
    loaded = load_pretraining_checkpoint(
        checkpoint,
        expected_lineage=lineage_from_contract(execution),
    )
    if loaded.progress.optimizer_step != state.optimizer_step:
        raise PretrainingPostWarmupLongrunError(
            "checkpoint/state optimizer step mismatch"
        )
    if loaded.progress.supervised_tokens_seen != state.supervised_tokens_seen:
        raise PretrainingPostWarmupLongrunError(
            "checkpoint/state supervised progress mismatch"
        )
    if (
        loaded.progress.physical_input_positions_seen
        != state.physical_input_positions_seen
    ):
        raise PretrainingPostWarmupLongrunError(
            "checkpoint/state physical progress mismatch"
        )
    if loaded.progress.microbatches_seen != state.microbatches_seen:
        raise PretrainingPostWarmupLongrunError(
            "checkpoint/state microbatch progress mismatch"
        )
    optimizer = build_adamw(loaded.model, training)
    optimizer.load_state_dict(loaded.optimizer_state)
    data_generator = torch.Generator()
    restore_pretraining_rng(loaded, data_generator=data_generator)
    cursor = cursor_from_state_dict(
        loaded.data_state,
        contract=data_contract,
    )
    return (
        loaded.model,
        optimizer,
        data_generator,
        loaded.progress,
        cursor,
    )


def _assert_fresh_run_root(run_root: Path) -> None:
    if run_root.exists() and any(run_root.iterdir()):
        raise PretrainingPostWarmupLongrunError(
            "fresh post-warmup run root is not empty; use --resume"
        )


def main() -> None:
    args = parse_args()
    authorization = load_post_warmup_longrun_authorization(
        args.authorization,
        repo_root=ROOT,
        external_root=args.external_root,
    )
    git_head = _validate_git_state(authorization)
    execution = load_pretraining_execution_contract(
        ROOT / "configs" / "pretraining-execution-l004.json",
        repo_root=ROOT,
        external_root=args.external_root,
    )
    data_contract = load_pretraining_data_contract(
        ROOT / "configs" / "pretraining-data-runtime-l004.json",
        repo_root=ROOT,
        external_root=args.external_root,
        verify_external_hashes=True,
    )
    local = _load_local_runtime()
    training = load_training_configuration(
        ROOT / "configs" / "training-base-50m-v1.json"
    )
    schedule = build_primary_schedule(training)
    if execution.checkpoint_lane != authorization.checkpoint_lane:
        raise PretrainingPostWarmupLongrunError(
            "authorization/execution checkpoint lane mismatch"
        )
    if (
        data_contract.global_supervised_tokens_per_update
        != schedule.global_tokens_per_update
    ):
        raise PretrainingPostWarmupLongrunError(
            "data/schedule global token contract mismatch"
        )
    _validate_free_space(authorization.run_root)

    if args.validate_only:
        print(f"authorization_id={authorization.authorization_id}")
        print(f"authorization_sha256={authorization.authorization_sha256}")
        print(f"source_step={authorization.start_optimizer_step}")
        print(f"end_step={authorization.end_optimizer_step}")
        print(
            "target_additional_supervised_tokens="
            f"{authorization.target_additional_supervised_tokens}"
        )
        print(
            "source_checkpoint_sha256="
            f"{authorization.source.checkpoint_sha256}"
        )
        print(f"run_root={authorization.run_root_storage}")
        print("post_warmup_validation=PASS")
        return

    if args.resume:
        (
            model,
            optimizer,
            data_generator,
            progress,
            cursor,
        ) = _resume_runtime(
            authorization,
            execution,
            data_contract,
            training,
        )
        previous_checkpoint_path: Path | None = (
            authorization.run_root / checkpoint_filename(progress.optimizer_step)
        )
    else:
        _assert_fresh_run_root(authorization.run_root)
        (
            model,
            optimizer,
            data_generator,
            progress,
            cursor,
        ) = _source_runtime(
            authorization,
            execution,
            data_contract,
            training,
        )
        previous_checkpoint_path = None

    remaining = authorization.end_optimizer_step - progress.optimizer_step
    if remaining <= 0:
        raise PretrainingPostWarmupLongrunError(
            "post-warmup tranche is already complete"
        )
    updates = (
        remaining
        if args.stop_after_updates is None
        else args.stop_after_updates
    )
    if (
        not isinstance(updates, int)
        or updates <= 0
        or updates > remaining
    ):
        raise PretrainingPostWarmupLongrunError(
            "requested post-warmup updates exceed authorization"
        )

    authorization.run_root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(local["threads"]))
    lineage = lineage_from_contract(execution)

    with PackedPretrainingData(data_contract) as data:
        for _ in range(updates):
            next_step = progress.optimizer_step + 1
            packed = list(
                data.iter_update_microbatches(
                    cursor,
                    batch_size=int(local["micro_batch_size"]),
                    max_input_length=int(
                        local["max_physical_sequence_length"]
                    ),
                    supervised_budget=65_536,
                )
            )
            if not packed:
                raise PretrainingPostWarmupLongrunError(
                    "packed runtime produced no post-warmup microbatches"
                )
            next_cursor = packed[-1].cursor
            supervised = sum(
                item.supervised_positions for item in packed
            )
            physical = sum(int(item.input_ids.numel()) for item in packed)
            if supervised != 65_536:
                raise PretrainingPostWarmupLongrunError(
                    "post-warmup update lacks 65536 supervised tokens"
                )
            lr = learning_rate_for_update(schedule, next_step)
            started = time.perf_counter()
            result = pretraining_update(
                model,
                optimizer,
                [(item.input_ids, item.targets) for item in packed],
                learning_rate=lr,
                clip_norm=gradient_clip_norm(training),
            )
            elapsed = time.perf_counter() - started
            if result.supervised_tokens != supervised:
                raise PretrainingPostWarmupLongrunError(
                    "optimizer/data supervised-token mismatch"
                )

            progress = type(progress)(
                optimizer_step=next_step,
                supervised_tokens_seen=(
                    progress.supervised_tokens_seen + supervised
                ),
                microbatches_seen=(
                    progress.microbatches_seen + len(packed)
                ),
                physical_input_positions_seen=(
                    progress.physical_input_positions_seen + physical
                ),
            )
            cursor = next_cursor
            if (
                cursor.supervised_tokens_consumed
                != progress.supervised_tokens_seen
            ):
                raise PretrainingPostWarmupLongrunError(
                    "post-warmup cursor/training progress mismatch"
                )

            checkpoint_relative = Path(checkpoint_filename(next_step))
            checkpoint_path = authorization.run_root / checkpoint_relative
            save_pretraining_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                lineage=lineage,
                progress=progress,
                primary_token_budget=execution.primary_token_budget,
                primary_seed=execution.primary_seed,
                data_generator=data_generator,
                data_state=cursor_state_dict(cursor),
            )
            checkpoint_sha = sha256_file(checkpoint_path)

            metric_relative = Path(metric_filename(next_step))
            metric_path = authorization.run_root / metric_relative
            metric_payload = {
                "schema_version": 1,
                "stage": "L004",
                "run_id": authorization.authorization_id,
                "authorization_sha256": authorization.authorization_sha256,
                "execution_git_head": git_head,
                "checkpoint_lane": authorization.checkpoint_lane,
                "rolling_checkpoint_retention": 1,
                "prune_prior_checkpoint_after_state_commit": True,
                "optimizer_step": next_step,
                "supervised_tokens_this_update": supervised,
                "physical_input_positions_this_update": physical,
                "microbatches_this_update": len(packed),
                "cumulative_supervised_tokens": (
                    progress.supervised_tokens_seen
                ),
                "cumulative_physical_input_positions": (
                    progress.physical_input_positions_seen
                ),
                "cumulative_microbatches": progress.microbatches_seen,
                "data_state": cursor_state_dict(cursor),
                "learning_rate": result.learning_rate,
                "mean_loss": result.mean_loss,
                "gradient_norm": result.gradient_norm,
                "update_seconds": elapsed,
                "supervised_tokens_per_second": supervised / elapsed,
                "checkpoint": {
                    "file": checkpoint_relative.as_posix(),
                    "bytes": checkpoint_path.stat().st_size,
                    "sha256": checkpoint_sha,
                },
                "pretraining_complete": False,
                "public_release_eligibility": "not_qualified",
            }
            atomic_write_json(metric_path, metric_payload)
            metric_sha = sha256_file(metric_path)

            state_payload = build_post_warmup_longrun_state_payload(
                authorization,
                optimizer_step=progress.optimizer_step,
                supervised_tokens_seen=progress.supervised_tokens_seen,
                physical_input_positions_seen=(
                    progress.physical_input_positions_seen
                ),
                microbatches_seen=progress.microbatches_seen,
                data_state=cursor_state_dict(cursor),
                checkpoint_path=checkpoint_path,
                checkpoint_sha256=checkpoint_sha,
                metric_path=metric_path,
                metric_sha256=metric_sha,
            )
            atomic_write_json(
                authorization.run_root / "state.json",
                state_payload,
            )
            pruned_checkpoint = None
            if previous_checkpoint_path is not None:
                if prune_prior_checkpoint(
                    authorization.run_root,
                    previous_checkpoint_path,
                    checkpoint_path,
                ):
                    pruned_checkpoint = previous_checkpoint_path.name
            previous_checkpoint_path = checkpoint_path
            print(
                "post_warmup_update_complete "
                f"step={next_step} supervised={supervised} "
                f"cumulative={progress.supervised_tokens_seen} "
                f"loss={result.mean_loss:.6f} "
                f"lr={result.learning_rate:.10f} "
                f"seconds={elapsed:.3f} "
                f"checkpoint_sha256={checkpoint_sha} "
                f"pruned={pruned_checkpoint or 'none'}",
                flush=True,
            )

    print(
        f"final_optimizer_step={progress.optimizer_step} "
        f"supervised_tokens_seen={progress.supervised_tokens_seen}"
    )
    print("post_warmup_longrun_tranche_execution=PASS")


if __name__ == "__main__":
    main()
