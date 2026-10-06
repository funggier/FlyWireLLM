from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--authorization",
        default=str(
            ROOT / "configs" / "pretraining-tranche-l004-v2.json"
        ),
    )
    parser.add_argument(
        "--step0",
        default=str(ROOT / "results" / "l004" / "validation-step0-v1.json"),
    )
    parser.add_argument(
        "--step4",
        default=str(ROOT / "results" / "l004" / "validation-step4-v1.json"),
    )
    parser.add_argument(
        "--step8",
        default=str(ROOT / "results" / "l004" / "validation-step8-v1.json"),
    )
    parser.add_argument(
        "--output",
        default=str(
            ROOT
            / "results"
            / "l004"
            / "validation-comparison-step4-step8-v1.json"
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
    step0 = _load(args.step0)
    step4 = _load(args.step4)
    step8 = _load(args.step8)

    gate = authorization.get("post_tranche_validation_gate")
    expected_gate = {
        "evaluation_id": "base50m-validation-v1",
        "baseline_optimizer_step": 4,
        "candidate_optimizer_step": 8,
        "same_validation_pack_required": True,
        "final_holdout_must_remain_untouched": True,
        "combined_loss_must_not_increase_vs_step4": True,
        "max_category_relative_loss_increase_vs_step4": 0.005,
        "require_all_losses_finite": True,
    }
    if gate != expected_gate:
        raise RuntimeError("predeclared post-tranche validation gate changed")

    for payload, model, step in (
        (step0, "step0", 0),
        (step4, "step4", 4),
        (step8, "step8", 8),
    ):
        if payload.get("schema_version") != 1:
            raise RuntimeError(f"{model}: validation schema changed")
        if payload.get("stage") != "L004":
            raise RuntimeError(f"{model}: validation stage changed")
        if payload.get("evaluation_id") != gate["evaluation_id"]:
            raise RuntimeError(f"{model}: validation evaluation id changed")
        if payload.get("partition") != "validation":
            raise RuntimeError(f"{model}: validation partition changed")
        if payload.get("final_holdout_touched") is not False:
            raise RuntimeError(f"{model}: final holdout was touched")
        if payload.get("model") != model:
            raise RuntimeError(f"{model}: model label changed")
        if payload.get("optimizer_step") != step:
            raise RuntimeError(f"{model}: optimizer step changed")
        if payload["combined"].get("supervised_tokens") != 300_000:
            raise RuntimeError(f"{model}: validation token total changed")

    if step8.get("continuation_authorization_sha256") != (
        "458cf90c0d0f38f407bc01d3a5e4af70706497610a3610347ce0cd75bcfff30d"
    ):
        raise RuntimeError("step8 continuation authorization hash changed")
    if step8.get("checkpoint_sha256") != (
        "2552f503438bf635f0e54a4fc1fc4bdcef3da57ad734c9ea7b0eebd379500e21"
    ):
        raise RuntimeError("step8 checkpoint hash changed")

    categories = sorted(step4["categories"])
    if categories != sorted(step8["categories"]):
        raise RuntimeError("step4/step8 validation category sets differ")
    if categories != sorted(step0["categories"]):
        raise RuntimeError("step0 validation category set differs")

    max_relative = float(
        gate["max_category_relative_loss_increase_vs_step4"]
    )
    comparisons: dict[str, dict[str, object]] = {}
    category_gate_pass = True
    same_pack = True
    for category in categories:
        before = step4["categories"][category]
        after = step8["categories"][category]
        origin = step0["categories"][category]
        if before.get("pack_sha256") != after.get("pack_sha256"):
            same_pack = False
        if origin.get("pack_sha256") != after.get("pack_sha256"):
            same_pack = False
        before_loss = float(before["loss"])
        after_loss = float(after["loss"])
        origin_loss = float(origin["loss"])
        finite = all(
            math.isfinite(value)
            for value in (origin_loss, before_loss, after_loss)
        )
        absolute = after_loss - before_loss
        relative = absolute / before_loss
        passed = finite and relative <= max_relative
        category_gate_pass = category_gate_pass and passed
        comparisons[category] = {
            "step0_loss": origin_loss,
            "step4_loss": before_loss,
            "step8_loss": after_loss,
            "step4_to_step8_absolute_change": absolute,
            "step4_to_step8_relative_change": relative,
            "step0_to_step8_absolute_change": after_loss - origin_loss,
            "step0_to_step8_relative_change": (
                (after_loss - origin_loss) / origin_loss
            ),
            "pack_sha256": after["pack_sha256"],
            "gate_pass": passed,
        }
        print(
            f"{category} step4={before_loss:.9f} "
            f"step8={after_loss:.9f} "
            f"relative={relative:.6%} "
            f"gate_pass={str(passed).lower()}"
        )

    if gate["same_validation_pack_required"] and not same_pack:
        raise RuntimeError("step4/step8 did not use identical validation packs")

    step4_combined = float(step4["combined"]["loss"])
    step8_combined = float(step8["combined"]["loss"])
    step0_combined = float(step0["combined"]["loss"])
    combined_finite = all(
        math.isfinite(value)
        for value in (step0_combined, step4_combined, step8_combined)
    )
    combined_pass = combined_finite and (
        not gate["combined_loss_must_not_increase_vs_step4"]
        or step8_combined <= step4_combined
    )
    passed = (
        same_pack
        and category_gate_pass
        and combined_pass
        and step8.get("final_holdout_touched") is False
    )
    output = {
        "schema_version": 1,
        "stage": "L004",
        "evaluation_id": gate["evaluation_id"],
        "comparison": "step4_vs_step8",
        "partition": "validation",
        "same_validation_pack": same_pack,
        "final_holdout_touched": False,
        "authorization_sha256": (
            "458cf90c0d0f38f407bc01d3a5e4af70706497610a3610347ce0cd75bcfff30d"
        ),
        "candidate_checkpoint_sha256": step8["checkpoint_sha256"],
        "production_optimizer_steps": 8,
        "supervised_training_tokens_seen": 524_288,
        "validation_supervised_tokens_per_model": 300_000,
        "categories": comparisons,
        "combined": {
            "step0_loss": step0_combined,
            "step4_loss": step4_combined,
            "step8_loss": step8_combined,
            "step4_to_step8_absolute_change": (
                step8_combined - step4_combined
            ),
            "step4_to_step8_relative_change": (
                (step8_combined - step4_combined) / step4_combined
            ),
            "step0_to_step8_absolute_change": (
                step8_combined - step0_combined
            ),
            "step0_to_step8_relative_change": (
                (step8_combined - step0_combined) / step0_combined
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
        f"combined step4={step4_combined:.9f} "
        f"step8={step8_combined:.9f} "
        f"relative={output['combined']['step4_to_step8_relative_change']:.6%} "
        f"gate_pass={str(combined_pass).lower()}"
    )
    print(f"same_validation_pack={str(same_pack).lower()}")
    print("holdout_touched=false")
    print(f"post_tranche_validation_gate_pass={str(passed).lower()}")
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
