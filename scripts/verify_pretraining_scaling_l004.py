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
from flywire_llm.pretraining_scaling import (
    load_scaling_authorization,
    load_scaling_state,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--external-root", required=True)
    parser.add_argument(
        "--authorization",
        default=str(
            ROOT / "configs" / "pretraining-tranche-l004-v4.json"
        ),
    )
    parser.add_argument("--output-json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    authorization = load_scaling_authorization(
        args.authorization,
        repo_root=ROOT,
        external_root=args.external_root,
    )
    state_path = authorization.run_root / "state.json"
    state = load_scaling_state(
        state_path,
        authorization=authorization,
        verify_files=True,
    )
    if state.optimizer_step != 32:
        raise RuntimeError("fourth tranche did not stop at optimizer step 32")
    if state.supervised_tokens_seen != 2_097_152:
        raise RuntimeError("fourth tranche supervised-token total changed")

    step_rows: list[dict[str, object]] = []
    losses: list[float] = []
    for step in range(17, 33):
        checkpoint = (
            authorization.run_root
            / "checkpoints"
            / f"step-{step:06d}.pt"
        )
        metric = (
            authorization.run_root
            / "metrics"
            / f"step-{step:06d}.json"
        )
        if not checkpoint.is_file() or not metric.is_file():
            raise RuntimeError(
                f"missing fourth-tranche artifacts for step {step}"
            )
        payload = json.loads(metric.read_text(encoding="utf-8"))
        if payload.get("optimizer_step") != step:
            raise RuntimeError(f"metric step changed for step {step}")
        expected_cumulative = step * 65_536
        if payload.get("cumulative_supervised_tokens") != expected_cumulative:
            raise RuntimeError(
                f"cumulative token progress changed for step {step}"
            )
        if payload.get("supervised_tokens_this_update") != 65_536:
            raise RuntimeError(
                f"supervised update size changed for step {step}"
            )
        checkpoint_sha = sha256_file(checkpoint)
        metric_sha = sha256_file(metric)
        checkpoint_meta = payload.get("checkpoint")
        if not isinstance(checkpoint_meta, dict):
            raise RuntimeError(
                f"checkpoint metadata missing for step {step}"
            )
        if checkpoint_meta.get("sha256") != checkpoint_sha:
            raise RuntimeError(
                f"checkpoint SHA-256 mismatch for step {step}"
            )
        if checkpoint_meta.get("bytes") != checkpoint.stat().st_size:
            raise RuntimeError(
                f"checkpoint byte size mismatch for step {step}"
            )
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
                "supervised_tokens_per_second": float(
                    payload["supervised_tokens_per_second"]
                ),
                "checkpoint_bytes": checkpoint.stat().st_size,
                "checkpoint_sha256": checkpoint_sha,
                "metric_sha256": metric_sha,
            }
        )
        print(
            f"PASS step={step} "
            f"checkpoint_bytes={checkpoint.stat().st_size} "
            f"checkpoint_sha256={checkpoint_sha} "
            f"metric_sha256={metric_sha} loss={loss:.6f}",
            flush=True,
        )

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
    if loaded.progress.optimizer_step != 32:
        raise RuntimeError("final scaling checkpoint step changed")
    if loaded.progress.supervised_tokens_seen != 2_097_152:
        raise RuntimeError(
            "final scaling checkpoint token progress changed"
        )
    if cursor.supervised_tokens_consumed != 2_097_152:
        raise RuntimeError(
            "final scaling checkpoint data cursor changed"
        )
    if loaded.metadata.get("pretraining_started") is not True:
        raise RuntimeError("final scaling checkpoint not started")
    if loaded.metadata.get("pretraining_complete") is not False:
        raise RuntimeError(
            "fourth tranche must not claim full pretraining completion"
        )

    state_payload = json.loads(state_path.read_text(encoding="utf-8"))
    if state_payload.get("tranche_complete") is not True:
        raise RuntimeError("fourth tranche completion flag changed")
    if state_payload.get("public_release_eligibility") != "not_qualified":
        raise RuntimeError("public release eligibility changed")

    summary = {
        "schema_version": 1,
        "stage": "L004",
        "qualification_id": "base50m-fourth-tranche-v1",
        "authorization_sha256": authorization.authorization_sha256,
        "runner_commit": authorization.runner_commit,
        "evidence_commit": authorization.evidence_commit,
        "checkpoint_lane": authorization.checkpoint_lane,
        "source_optimizer_step": 16,
        "source_checkpoint_sha256": (
            authorization.source.checkpoint_sha256
        ),
        "production_optimizer_steps": 32,
        "additional_optimizer_steps": 16,
        "supervised_tokens_seen": 2_097_152,
        "additional_supervised_tokens": 1_048_576,
        "fraction_of_500m": 2_097_152 / 500_000_000,
        "pretraining_started": True,
        "pretraining_complete": False,
        "tranche_complete": True,
        "public_release_eligibility": "not_qualified",
        "state_sha256": sha256_file(state_path),
        "final_checkpoint": {
            "file": state.checkpoint_file,
            "bytes": state.checkpoint_bytes,
            "sha256": state.checkpoint_sha256,
        },
        "final_data_state": state.data_state,
        "loss_sequence": losses,
        "loss_change_step17_to_step32": losses[-1] - losses[0],
        "steps": step_rows,
    }
    rendered = json.dumps(summary, indent=2) + "\n"
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            rendered,
            encoding="utf-8",
            newline="\n",
        )
    print(f"state_sha256={summary['state_sha256']}")
    print("final_checkpoint_load=PASS")
    print("fourth_bounded_tranche_verification=PASS")


if __name__ == "__main__":
    main()
