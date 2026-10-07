from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUTH_SHA = "8ef449c63c6be4e2129a0604413fbfbb13f6d706a2c046d779079415981f0259"
CHECKPOINT_SHA = "136f554d7321be08ec266ac44296ae6870aadc7edb8d468812173ab240b8a463"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--authorization",
        default=str(
            ROOT / "configs" / "pretraining-tranche-l004-v4.json"
        ),
    )
    parser.add_argument(
        "--step16",
        default=str(
            ROOT / "results" / "l004" / "validation-step16-v1.json"
        ),
    )
    parser.add_argument(
        "--step32",
        default=str(
            ROOT / "results" / "l004" / "validation-step32-v1.json"
        ),
    )
    parser.add_argument(
        "--output",
        default=str(
            ROOT / "results" / "l004"
            / "validation-comparison-step16-step32-v1.json"
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
    step16 = _load(args.step16)
    step32 = _load(args.step32)
    gate = authorization.get("post_tranche_validation_gate")
    expected_gate = {
        "evaluation_id": "base50m-validation-v1",
        "baseline_optimizer_step": 16,
        "candidate_optimizer_step": 32,
        "same_validation_pack_required": True,
        "final_holdout_must_remain_untouched": True,
        "combined_loss_must_not_increase_vs_step16": True,
        "max_category_relative_loss_increase_vs_step16": 0.005,
        "require_all_losses_finite": True,
    }
    if gate != expected_gate:
        raise RuntimeError("predeclared fourth-tranche validation gate changed")

    for payload, model, step, tokens in (
        (step16, "step16", 16, 1_048_576),
        (step32, "step32", 32, 2_097_152),
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

    if step32.get("scaling_authorization_sha256") != AUTH_SHA:
        raise RuntimeError("step32 scaling authorization hash changed")
    if step32.get("checkpoint_sha256") != CHECKPOINT_SHA:
        raise RuntimeError("step32 checkpoint hash changed")

    categories = sorted(step16["categories"])
    if categories != sorted(step32["categories"]):
        raise RuntimeError("step16/step32 validation category sets differ")

    max_relative = float(
        gate["max_category_relative_loss_increase_vs_step16"]
    )
    comparisons: dict[str, dict[str, object]] = {}
    category_gate_pass = True
    same_pack = True
    for category in categories:
        before = step16["categories"][category]
        after = step32["categories"][category]
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
            "step16_loss": before_loss,
            "step32_loss": after_loss,
            "absolute_change": absolute,
            "relative_change": relative,
            "pack_sha256": after["pack_sha256"],
            "gate_pass": passed,
        }
        print(
            f"{category} step16={before_loss:.9f} "
            f"step32={after_loss:.9f} "
            f"relative={relative:.6%} "
            f"gate_pass={str(passed).lower()}"
        )

    if gate["same_validation_pack_required"] and not same_pack:
        raise RuntimeError(
            "step16/step32 did not use identical validation packs"
        )

    before_combined = float(step16["combined"]["loss"])
    after_combined = float(step32["combined"]["loss"])
    combined_finite = (
        math.isfinite(before_combined) and math.isfinite(after_combined)
    )
    combined_pass = combined_finite and (
        not gate["combined_loss_must_not_increase_vs_step16"]
        or after_combined <= before_combined
    )
    passed = (
        same_pack
        and category_gate_pass
        and combined_pass
        and step32.get("final_holdout_touched") is False
    )
    output = {
        "schema_version": 1,
        "stage": "L004",
        "evaluation_id": gate["evaluation_id"],
        "comparison": "step16_vs_step32",
        "partition": "validation",
        "same_validation_pack": same_pack,
        "final_holdout_touched": False,
        "authorization_sha256": AUTH_SHA,
        "candidate_checkpoint_sha256": step32["checkpoint_sha256"],
        "production_optimizer_steps": 32,
        "supervised_training_tokens_seen": 2_097_152,
        "validation_supervised_tokens_per_model": 300_000,
        "categories": comparisons,
        "combined": {
            "step16_loss": before_combined,
            "step32_loss": after_combined,
            "absolute_change": after_combined - before_combined,
            "relative_change": (
                (after_combined - before_combined) / before_combined
            ),
            "gate_pass": combined_pass,
        },
        "gate_contract": gate,
        "gate_pass": passed,
        "larger_tranche_may_be_considered": passed,
        "pretraining_complete": False,
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
        f"combined step16={before_combined:.9f} "
        f"step32={after_combined:.9f} "
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
