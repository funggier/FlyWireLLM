from __future__ import annotations

import argparse
import json
from pathlib import Path

from flywire_llm.pretraining_checkpoint import (
    lineage_from_contract,
    load_pretraining_checkpoint,
)
from flywire_llm.pretraining_data import (
    cursor_from_state_dict,
    load_pretraining_data_contract,
)
from flywire_llm.pretraining_execution import (
    load_pretraining_execution_contract,
)
from flywire_llm.pretraining_post_warmup import (
    load_post_warmup_authorization,
    load_post_warmup_state,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--external-root", required=True)
    parser.add_argument(
        "--authorization",
        default=str(
            ROOT / "configs" / "pretraining-tranche-l004-v12.json"
        ),
    )
    parser.add_argument("--output-json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    authorization = load_post_warmup_authorization(
        args.authorization,
        repo_root=ROOT,
        external_root=args.external_root,
    )
    state_path = authorization.run_root / "state.json"
    state = load_post_warmup_state(
        state_path,
        authorization=authorization,
        verify_files=True,
    )
    if state.optimizer_step != 169:
        raise RuntimeError("post warmup did not stop at optimizer step 169")
    if state.supervised_tokens_seen != 11_075_584:
        raise RuntimeError("post warmup supervised-token total changed")

    checkpoint_dir = authorization.run_root / "checkpoints"
    retained = sorted(
        path.name for path in checkpoint_dir.glob("*.pt") if path.is_file()
    )
    if retained != ["step-000169.pt"]:
        raise RuntimeError(
            "rolling checkpoint retention changed "
            f"expected=['step-000169.pt'] observed={retained}"
        )

    step_rows: list[dict[str, object]] = []
    losses: list[float] = []
    pruned_count = 0
    for step in range(154, 170):
        metric = (
            authorization.run_root
            / "metrics"
            / f"step-{step:06d}.json"
        )
        if not metric.is_file():
            raise RuntimeError(f"missing rolling metric for step {step}")
        payload = json.loads(metric.read_text(encoding="utf-8"))
        if payload.get("optimizer_step") != step:
            raise RuntimeError(f"metric step changed for step {step}")
        if payload.get("run_id") != authorization.authorization_id:
            raise RuntimeError(f"run id changed for step {step}")
        if payload.get("authorization_sha256") != (
            authorization.authorization_sha256
        ):
            raise RuntimeError(
                f"authorization hash changed for step {step}"
            )
        if payload.get("rolling_checkpoint_retention") != 1:
            raise RuntimeError(
                f"rolling retention changed for step {step}"
            )
        if payload.get(
            "prune_prior_checkpoint_after_state_commit"
        ) is not True:
            raise RuntimeError(
                f"prune ordering policy changed for step {step}"
            )
        if payload.get("supervised_tokens_this_update") != 65_536:
            raise RuntimeError(
                f"supervised update size changed for step {step}"
            )
        expected_cumulative = step * 65_536
        if payload.get("cumulative_supervised_tokens") != expected_cumulative:
            raise RuntimeError(
                f"cumulative token progress changed for step {step}"
            )

        checkpoint_meta = payload.get("checkpoint")
        if not isinstance(checkpoint_meta, dict):
            raise RuntimeError(
                f"checkpoint metadata missing for step {step}"
            )
        checkpoint_file = checkpoint_meta.get("file")
        checkpoint_sha = checkpoint_meta.get("sha256")
        checkpoint_bytes = checkpoint_meta.get("bytes")
        expected_file = f"checkpoints/step-{step:06d}.pt"
        if checkpoint_file != expected_file:
            raise RuntimeError(
                f"checkpoint filename changed for step {step}"
            )
        if not isinstance(checkpoint_sha, str) or len(checkpoint_sha) != 64:
            raise RuntimeError(
                f"checkpoint hash invalid for step {step}"
            )
        if checkpoint_bytes != 602_706_571:
            raise RuntimeError(
                f"checkpoint byte size changed for step {step}"
            )
        checkpoint = authorization.run_root / checkpoint_file
        if step < 169:
            if checkpoint.exists():
                raise RuntimeError(
                    f"historical checkpoint was not pruned for step {step}"
                )
            pruned_count += 1
        else:
            if not checkpoint.is_file():
                raise RuntimeError("final rolling checkpoint is missing")
            if checkpoint.stat().st_size != checkpoint_bytes:
                raise RuntimeError(
                    "final rolling checkpoint byte size mismatch"
                )
            if sha256_file(checkpoint) != checkpoint_sha:
                raise RuntimeError(
                    "final rolling checkpoint SHA-256 mismatch"
                )

        metric_sha = sha256_file(metric)
        loss = float(payload["mean_loss"])
        losses.append(loss)
        step_rows.append(
            {
                "optimizer_step": step,
                "supervised_tokens_this_update": int(
                    payload["supervised_tokens_this_update"]
                ),
                "cumulative_supervised_tokens": int(
                    payload["cumulative_supervised_tokens"]
                ),
                "physical_input_positions_this_update": int(
                    payload["physical_input_positions_this_update"]
                ),
                "microbatches_this_update": int(
                    payload["microbatches_this_update"]
                ),
                "learning_rate": float(payload["learning_rate"]),
                "mean_loss": loss,
                "gradient_norm": float(payload["gradient_norm"]),
                "update_seconds": float(payload["update_seconds"]),
                "checkpoint_bytes": int(checkpoint_bytes),
                "checkpoint_sha256": str(checkpoint_sha),
                "checkpoint_retained": step == 169,
                "metric_sha256": metric_sha,
            }
        )
        print(
            f"PASS step={step} retained={str(step == 169).lower()} "
            f"checkpoint_sha256={checkpoint_sha} "
            f"metric_sha256={metric_sha} loss={loss:.6f}",
            flush=True,
        )

    if pruned_count != 15:
        raise RuntimeError("rolling historical prune count changed")

    execution = load_pretraining_execution_contract(
        ROOT / "configs" / "pretraining-execution-l004.json",
        repo_root=ROOT,
        external_root=args.external_root,
    )
    data = load_pretraining_data_contract(
        ROOT / "configs" / "pretraining-data-runtime-l004.json",
        repo_root=ROOT,
        external_root=args.external_root,
        verify_external_hashes=False,
    )
    final_checkpoint = authorization.run_root / state.checkpoint_file
    loaded = load_pretraining_checkpoint(
        final_checkpoint,
        expected_lineage=lineage_from_contract(execution),
    )
    cursor = cursor_from_state_dict(
        loaded.data_state,
        contract=data,
    )
    if loaded.progress.optimizer_step != 169:
        raise RuntimeError("final rolling checkpoint step changed")
    if loaded.progress.supervised_tokens_seen != 11_075_584:
        raise RuntimeError(
            "final rolling checkpoint token progress changed"
        )
    if cursor.supervised_tokens_consumed != 11_075_584:
        raise RuntimeError(
            "final rolling checkpoint data cursor changed"
        )
    if loaded.metadata.get("pretraining_started") is not True:
        raise RuntimeError("final rolling checkpoint not started")
    if loaded.metadata.get("pretraining_complete") is not False:
        raise RuntimeError(
            "rolling expansion must not claim full pretraining completion"
        )

    state_payload = json.loads(state_path.read_text(encoding="utf-8"))
    if state_payload.get("tranche_complete") is not True:
        raise RuntimeError("rolling tranche completion flag changed")
    if state_payload.get("rolling_checkpoint_retention") != 1:
        raise RuntimeError("rolling state retention changed")
    if state_payload.get(
        "prune_prior_checkpoint_after_state_commit"
    ) is not True:
        raise RuntimeError("rolling state prune policy changed")
    if state_payload.get("public_release_eligibility") != "not_qualified":
        raise RuntimeError("public release eligibility changed")

    summary = {
        "schema_version": 1,
        "stage": "L004",
        "qualification_id": "base50m-post-warmup-1-v1",
        "authorization_sha256": authorization.authorization_sha256,
        "runner_commit": authorization.runner_commit,
        "evidence_commit": authorization.evidence_commit,
        "checkpoint_lane": authorization.checkpoint_lane,
        "source_optimizer_step": 153,
        "source_checkpoint_sha256": (
            authorization.source.checkpoint_sha256
        ),
        "production_optimizer_steps": 169,
        "additional_optimizer_steps": 16,
        "supervised_tokens_seen": 11_075_584,
        "additional_supervised_tokens": 1_048_576,
        "fraction_of_500m": 11_075_584 / 500_000_000,
        "pretraining_started": True,
        "pretraining_complete": False,
        "tranche_complete": True,
        "rolling_checkpoint_retention": 1,
        "retained_checkpoint_count": 1,
        "historical_checkpoints_pruned": 15,
        "public_release_eligibility": "not_qualified",
        "state_sha256": sha256_file(state_path),
        "final_checkpoint": {
            "file": state.checkpoint_file,
            "bytes": state.checkpoint_bytes,
            "sha256": state.checkpoint_sha256,
        },
        "final_data_state": state.data_state,
        "loss_sequence": losses,
        "loss_change_step154_to_step169": losses[-1] - losses[0],
        "steps": step_rows,
    }
    rendered = json.dumps(summary, indent=2) + "\n"
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8", newline="\n")
    print(f"state_sha256={summary['state_sha256']}")
    print("rolling_retained_checkpoint_count=1")
    print("rolling_historical_checkpoints_pruned=15")
    print("final_checkpoint_load=PASS")
    print("post_warmup_tranche_verification=PASS")


if __name__ == "__main__":
    main()
