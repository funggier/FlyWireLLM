from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import torch

from flywire_llm.model import BlankCausalLM
from flywire_llm.pretraining_checkpoint import (
    PretrainingProgress,
    lineage_from_contract,
    load_pretraining_checkpoint,
    restore_pretraining_rng,
    save_pretraining_checkpoint,
)
from flywire_llm.pretraining_data import (
    PackedDataCursor,
    PackedPretrainingData,
    cursor_from_state_dict,
    cursor_state_dict,
    load_pretraining_data_contract,
)
from flywire_llm.pretraining_execution import (
    load_pretraining_execution_contract,
)
from flywire_llm.pretraining_training import (
    build_adamw,
    build_primary_schedule,
    gradient_clip_norm,
    learning_rate_for_update,
    load_training_configuration,
    pretraining_update,
)
from flywire_llm.pretraining_tranche import (
    PretrainingTrancheError,
    atomic_write_json,
    build_state_payload,
    checkpoint_filename,
    load_bounded_tranche_authorization,
    load_tranche_state,
    metric_filename,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Execute the separately authorized first bounded Base-50M "
            "production pretraining tranche."
        )
    )
    parser.add_argument("--external-root", required=True)
    parser.add_argument(
        "--authorization",
        default=str(
            ROOT
            / "configs"
            / "pretraining-tranche-l004-v1.json"
        ),
    )
    parser.add_argument(
        "--stop-after-updates",
        type=int,
        default=None,
        help=(
            "Bound this invocation to N additional optimizer updates. "
            "The authorization still caps the tranche at four updates."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume only from the authorized state.json checkpoint.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate authorization/contracts without creating run output.",
    )
    return parser.parse_args()


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=ROOT,
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


def _validate_git_state(authorization) -> str:
    dirty = _git("status", "--porcelain")
    if dirty:
        raise PretrainingTrancheError(
            "production tranche requires a clean Git worktree"
        )
    head = _git("rev-parse", "HEAD")
    for field, commit in (
        ("foundation_commit", authorization.foundation_commit),
        ("runner_commit", authorization.runner_commit),
    ):
        completed = subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, head],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if completed.returncode != 0:
            raise PretrainingTrancheError(
                f"{field} is not an ancestor of current HEAD"
            )
    return head


def _load_local_runtime() -> dict:
    payload = json.loads(
        (
            ROOT
            / "configs"
            / "pretraining-local-runtime-l004.json"
        ).read_text(encoding="utf-8")
    )
    fixed = {
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
    for key, expected in fixed.items():
        if payload.get(key) != expected:
            raise PretrainingTrancheError(
                f"local runtime {key} changed"
            )
    return payload


def _fresh_runtime(execution, training):
    torch.set_num_threads(12)
    torch.manual_seed(execution.primary_seed)
    model = BlankCausalLM(execution.model_config)
    if model.parameter_count != 50_213_376:
        raise PretrainingTrancheError(
            "Base-50M parameter count changed"
        )
    optimizer = build_adamw(model, training)
    data_generator = torch.Generator().manual_seed(
        execution.primary_seed
    )
    progress = PretrainingProgress(
        optimizer_step=0,
        supervised_tokens_seen=0,
        microbatches_seen=0,
        physical_input_positions_seen=0,
    )
    return (
        model,
        optimizer,
        data_generator,
        progress,
        PackedDataCursor(),
    )


def _resume_runtime(
    *,
    authorization,
    execution,
    data_contract,
    training,
):
    state_path = authorization.run_root / "state.json"
    if not state_path.is_file():
        raise PretrainingTrancheError(
            "resume requested but authorized state.json is missing"
        )
    state = load_tranche_state(
        state_path,
        authorization=authorization,
        verify_files=True,
    )
    checkpoint_path = (
        authorization.run_root / state.checkpoint_file
    )
    lineage = lineage_from_contract(execution)
    loaded = load_pretraining_checkpoint(
        checkpoint_path,
        expected_lineage=lineage,
    )
    if loaded.model.config != execution.model_config:
        raise PretrainingTrancheError(
            "checkpoint model config does not match execution contract"
        )
    if loaded.metadata.get("primary_seed") != execution.primary_seed:
        raise PretrainingTrancheError(
            "checkpoint primary seed changed"
        )
    if (
        loaded.metadata.get("primary_token_budget")
        != execution.primary_token_budget
    ):
        raise PretrainingTrancheError(
            "checkpoint primary token budget changed"
        )
    if loaded.progress.optimizer_step != state.optimizer_step:
        raise PretrainingTrancheError(
            "checkpoint/state optimizer step mismatch"
        )
    if (
        loaded.progress.supervised_tokens_seen
        != state.supervised_tokens_seen
    ):
        raise PretrainingTrancheError(
            "checkpoint/state supervised progress mismatch"
        )
    if (
        loaded.progress.physical_input_positions_seen
        != state.physical_input_positions_seen
    ):
        raise PretrainingTrancheError(
            "checkpoint/state physical progress mismatch"
        )
    if loaded.progress.microbatches_seen != state.microbatches_seen:
        raise PretrainingTrancheError(
            "checkpoint/state microbatch progress mismatch"
        )

    optimizer = build_adamw(loaded.model, training)
    optimizer.load_state_dict(loaded.optimizer_state)
    data_generator = torch.Generator()
    restore_pretraining_rng(
        loaded,
        data_generator=data_generator,
    )
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
    if run_root.exists():
        entries = list(run_root.iterdir())
        if entries:
            raise PretrainingTrancheError(
                "fresh run root is not empty; use --resume only for "
                "an authorized existing tranche"
            )


def _validate_free_space(run_root: Path) -> None:
    probe = run_root
    while not probe.exists():
        if probe.parent == probe:
            break
        probe = probe.parent
    usage = shutil.disk_usage(probe)
    # Four AdamW checkpoints for Base-50M are expected to require a few GB.
    # Keep a conservative floor so step 1 cannot begin on a nearly full disk.
    if usage.free < 8 * 1024**3:
        raise PretrainingTrancheError(
            "fewer than 8 GiB free on the authorized run volume"
        )


def main() -> None:
    args = parse_args()
    authorization = load_bounded_tranche_authorization(
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
    if execution.checkpoint_lane != authorization.checkpoint_lane:
        raise PretrainingTrancheError(
            "authorization/execution checkpoint lane mismatch"
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
    if (
        data_contract.global_supervised_tokens_per_update
        != schedule.global_tokens_per_update
        or local["global_supervised_tokens_per_update"]
        != schedule.global_tokens_per_update
    ):
        raise PretrainingTrancheError(
            "global token/update contract mismatch"
        )

    _validate_free_space(authorization.run_root)
    if args.validate_only:
        print(f"authorization_id={authorization.authorization_id}")
        print(f"authorization_sha256={authorization.authorization_sha256}")
        print(f"foundation_commit={authorization.foundation_commit}")
        print(f"runner_commit={authorization.runner_commit}")
        print(f"current_git_head={git_head}")
        print(
            f"max_optimizer_updates={authorization.max_optimizer_updates}"
        )
        print(
            "target_supervised_tokens="
            f"{authorization.target_supervised_tokens}"
        )
        print(f"run_root={authorization.run_root_storage}")
        print("bounded_tranche_validation=PASS")
        return

    if args.resume:
        (
            model,
            optimizer,
            data_generator,
            progress,
            cursor,
        ) = _resume_runtime(
            authorization=authorization,
            execution=execution,
            data_contract=data_contract,
            training=training,
        )
    else:
        _assert_fresh_run_root(authorization.run_root)
        (
            model,
            optimizer,
            data_generator,
            progress,
            cursor,
        ) = _fresh_runtime(execution, training)

    remaining_updates = (
        authorization.max_optimizer_updates
        - progress.optimizer_step
    )
    if remaining_updates <= 0:
        raise PretrainingTrancheError(
            "bounded tranche is already complete"
        )
    updates_this_invocation = (
        remaining_updates
        if args.stop_after_updates is None
        else args.stop_after_updates
    )
    if (
        not isinstance(updates_this_invocation, int)
        or updates_this_invocation <= 0
        or updates_this_invocation > remaining_updates
    ):
        raise PretrainingTrancheError(
            "requested updates exceed the remaining bounded authorization"
        )

    authorization.run_root.mkdir(parents=True, exist_ok=True)
    lineage = lineage_from_contract(execution)
    torch.set_num_threads(int(local["threads"]))

    with PackedPretrainingData(data_contract) as data:
        for _ in range(updates_this_invocation):
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
                raise PretrainingTrancheError(
                    "packed runtime produced no microbatches"
                )
            next_cursor = packed[-1].cursor
            supervised = sum(
                item.supervised_positions for item in packed
            )
            physical = sum(
                int(item.input_ids.numel()) for item in packed
            )
            if supervised != 65_536:
                raise PretrainingTrancheError(
                    "production update did not contain exactly 65536 "
                    "supervised tokens"
                )
            lr = learning_rate_for_update(schedule, next_step)
            started = time.perf_counter()
            result = pretraining_update(
                model,
                optimizer,
                [
                    (item.input_ids, item.targets)
                    for item in packed
                ],
                learning_rate=lr,
                clip_norm=gradient_clip_norm(training),
            )
            elapsed = time.perf_counter() - started
            if result.supervised_tokens != supervised:
                raise PretrainingTrancheError(
                    "optimizer/data supervised-token mismatch"
                )

            progress = PretrainingProgress(
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
                raise PretrainingTrancheError(
                    "data cursor/training progress mismatch after update"
                )

            checkpoint_relative = Path(
                checkpoint_filename(next_step)
            )
            checkpoint_path = (
                authorization.run_root / checkpoint_relative
            )
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
                "authorization_sha256": (
                    authorization.authorization_sha256
                ),
                "foundation_commit": authorization.foundation_commit,
                "runner_commit": authorization.runner_commit,
                "execution_git_head": git_head,
                "checkpoint_lane": authorization.checkpoint_lane,
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

            state_payload = build_state_payload(
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
            print(
                "production_update_complete "
                f"step={next_step} "
                f"supervised={supervised} "
                f"cumulative={progress.supervised_tokens_seen} "
                f"loss={result.mean_loss:.6f} "
                f"lr={result.learning_rate:.10f} "
                f"seconds={elapsed:.3f} "
                f"checkpoint_sha256={checkpoint_sha}",
                flush=True,
            )

    final_state = load_tranche_state(
        authorization.run_root / "state.json",
        authorization=authorization,
        verify_files=True,
    )
    print(f"optimizer_step={final_state.optimizer_step}")
    print(
        "supervised_tokens_seen="
        f"{final_state.supervised_tokens_seen}"
    )
    print(
        "tranche_complete="
        f"{str(final_state.optimizer_step == authorization.max_optimizer_updates).lower()}"
    )
    print("bounded_tranche_execution=PASS")


if __name__ == "__main__":
    main()
