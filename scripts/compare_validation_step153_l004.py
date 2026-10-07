from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUTH_SHA = "b8482919369a8cc402f8addaa7c1a122d59a1fd28ce52e9dfe9b78fb78eb1f3b"
CHECKPOINT_SHA = "f00d36e288094aab5bc0bec52a62dd134e92505ee6a87f3797e8e70d05cc3700"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--authorization",
        default=str(
            ROOT / "configs" / "pretraining-tranche-l004-v11.json"
        ),
    )
    parser.add_argument(
        "--step136",
        default=str(
            ROOT / "results" / "l004" / "validation-step136-v1.json"
        ),
    )
    parser.add_argument(
        "--step153",
        default=str(
            ROOT / "results" / "l004" / "validation-step153-v1.json"
        ),
    )
    parser.add_argument(
        "--output",
        default=str(
            ROOT / "results" / "l004"
            / "validation-comparison-step136-step153-v1.json"
        ),
    )
    return parser.parse_args()


def _load(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path}: expected JSON object")
    return payload


def main() -> None:
    args = parse_args()
    authorization = _load(args.authorization)
    step136 = _load(args.step136)
    step153 = _load(args.step153)
    gate = authorization.get("post_tranche_validation_gate")
    expected_gate = {
        "evaluation_id": "base50m-validation-v1",
        "baseline_optimizer_step": 136,
        "candidate_optimizer_step": 153,
        "same_validation_pack_required": True,
        "final_holdout_must_remain_untouched": True,
        "combined_loss_must_not_increase_vs_step136": True,
        "max_category_relative_loss_increase_vs_step136": 0.005,
        "require_all_losses_finite": True,
    }
    if gate != expected_gate:
        raise RuntimeError("predeclared warmup-boundary validation gate changed")

    for payload, model, step, tokens in (
        (step136, "step136", 136, 8_912_896),
        (step153, "step153", 153, 10_027_008),
    ):
        if payload.get("schema_version") != 1:
            raise RuntimeError(f"{model}: validation schema changed")
        if payload.get("stage") != "L004":
            raise RuntimeError(f"{model}: validation stage changed")
        if payload.get("evaluation_id") != gate["evaluation_id"]:
            raise RuntimeError(f"{model}: evaluation id changed")
        if payload.get("partition") != "validation":
            raise RuntimeError(f"{model}: validation partition changed")
        if payload.get("final_holdout_touched") is not False:
            raise RuntimeError(f"{model}: final holdout was touched")
        if payload.get("model") != model:
            raise RuntimeError(f"{model}: model label changed")
        if payload.get("optimizer_step") != step:
            raise RuntimeError(f"{model}: optimizer step changed")
        if payload.get("supervised_tokens_seen") != tokens:
            raise RuntimeError(f"{model}: training-token progress changed")
        if payload["combined"].get("supervised_tokens") != 300_000:
            raise RuntimeError(f"{model}: validation token total changed")

    if step153.get("warmup_boundary_authorization_sha256") != AUTH_SHA:
        raise RuntimeError(
            "step153 warmup-boundary authorization hash changed"
        )
    if step153.get("checkpoint_sha256") != CHECKPOINT_SHA:
        raise RuntimeError("step153 checkpoint hash changed")

    categories = sorted(step136["categories"])
    if categories != sorted(step153["categories"]):
        raise RuntimeError("step136/step153 validation category sets differ")

    max_relative = float(
        gate["max_category_relative_loss_increase_vs_step136"]
    )
    comparisons: dict[str, dict[str, object]] = {}
    category_gate_pass = True
    same_pack = True
    for category in categories:
        before = step136["categories"][category]
        after = step153["categories"][category]
        if before.get("pack_sha256") != after.get("pack_sha256"):
            same_pack = False
        before_loss = float(before["loss"])
        after_loss = float(after["loss"])
        finite = math.isfinite(before_loss) and math.isfinite(after_loss)
        absolute = after_loss - before_loss
        relative = absolute / before_loss
        passed = finite and relative <= max_relative
        category_gate_pass = category_gate_pass and passed
        comparisons[category] = {
            "step136_loss": before_loss,
            "step153_loss": after_loss,
            "absolute_change": absolute,
            "relative_change": relative,
            "pack_sha256": after["pack_sha256"],
            "gate_pass": passed,
        }
        print(
            f"{category} step136={before_loss:.9f} "
            f"step153={after_loss:.9f} "
            f"relative={relative:.6%} "
            f"gate_pass={str(passed).lower()}"
        )

    if gate["same_validation_pack_required"] and not same_pack:
        raise RuntimeError(
            "step136/step153 did not use identical validation packs"
        )

    before_combined = float(step136["combined"]["loss"])
    after_combined = float(step153["combined"]["loss"])
    combined_finite = (
        math.isfinite(before_combined) and math.isfinite(after_combined)
    )
    combined_pass = combined_finite and (
        not gate["combined_loss_must_not_increase_vs_step136"]
        or after_combined <= before_combined
    )
    passed = (
        same_pack
        and category_gate_pass
        and combined_pass
        and step153.get("final_holdout_touched") is False
    )
    output = {
        "schema_version": 1,
        "stage": "L004",
        "evaluation_id": gate["evaluation_id"],
        "comparison": "step136_vs_step153",
        "partition": "validation",
        "same_validation_pack": same_pack,
        "final_holdout_touched": False,
        "authorization_sha256": AUTH_SHA,
        "candidate_checkpoint_sha256": step153["checkpoint_sha256"],
        "production_optimizer_steps": 153,
        "supervised_training_tokens_seen": 10_027_008,
        "validation_supervised_tokens_per_model": 300_000,
        "categories": comparisons,
        "combined": {
            "step136_loss": before_combined,
            "step153_loss": after_combined,
            "absolute_change": after_combined - before_combined,
            "relative_change": (
                (after_combined - before_combined) / before_combined
            ),
            "gate_pass": combined_pass,
        },
        "gate_contract": gate,
        "gate_pass": passed,
        "post_warmup_policy_may_be_considered": passed,
        "pretraining_complete": False,
        "optimizer_warmup_complete": True,
        "public_release_eligibility": "not_qualified",
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        f"combined step136={before_combined:.9f} "
        f"step153={after_combined:.9f} "
        f"relative={output['combined']['relative_change']:.6%} "
        f"gate_pass={str(combined_pass).lower()}"
    )
    print(f"same_validation_pack={str(same_pack).lower()}")
    print("holdout_touched=false")
    print(f"post_tranche_validation_gate_pass={str(passed).lower()}")
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
