from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "validation-l004-v1.json"),
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
        "--output",
        default=str(
            ROOT / "results" / "l004" / "validation-comparison-v1.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    step0 = json.loads(Path(args.step0).read_text(encoding="utf-8"))
    step4 = json.loads(Path(args.step4).read_text(encoding="utf-8"))

    if config.get("evaluation_id") != "base50m-validation-v1":
        raise RuntimeError("validation evaluation id changed")
    for payload, model, step in (
        (step0, "step0", 0),
        (step4, "step4", 4),
    ):
        if payload.get("evaluation_id") != config["evaluation_id"]:
            raise RuntimeError(f"{model}: evaluation id mismatch")
        if payload.get("model") != model:
            raise RuntimeError(f"{model}: model label mismatch")
        if payload.get("optimizer_step") != step:
            raise RuntimeError(f"{model}: optimizer step mismatch")
        if payload.get("partition") != "validation":
            raise RuntimeError(f"{model}: partition changed")
        if payload.get("final_holdout_touched") is not False:
            raise RuntimeError(f"{model}: holdout was touched")
        if payload["combined"]["supervised_tokens"] != 300_000:
            raise RuntimeError(f"{model}: validation token total changed")

    categories = sorted(step0["categories"])
    if categories != sorted(step4["categories"]):
        raise RuntimeError("validation category sets differ")
    gate = config["gate"]
    max_relative = float(gate["max_category_relative_loss_increase"])
    comparisons: dict[str, dict] = {}
    category_gate_pass = True
    for category in categories:
        before = float(step0["categories"][category]["loss"])
        after = float(step4["categories"][category]["loss"])
        if not math.isfinite(before) or not math.isfinite(after):
            category_gate_pass = False
        absolute = after - before
        relative = absolute / before
        passed = relative <= max_relative
        category_gate_pass = category_gate_pass and passed
        comparisons[category] = {
            "step0_loss": before,
            "step4_loss": after,
            "absolute_change": absolute,
            "relative_change": relative,
            "gate_pass": passed,
        }
        print(
            f"{category} step0={before:.9f} step4={after:.9f} "
            f"delta={absolute:.9f} relative={relative:.6%} "
            f"gate_pass={str(passed).lower()}"
        )

    before_combined = float(step0["combined"]["loss"])
    after_combined = float(step4["combined"]["loss"])
    combined_delta = after_combined - before_combined
    combined_relative = combined_delta / before_combined
    finite = math.isfinite(before_combined) and math.isfinite(after_combined)
    combined_pass = finite and (
        not gate["combined_loss_must_not_increase"]
        or after_combined <= before_combined
    )
    passed = category_gate_pass and combined_pass
    output = {
        "schema_version": 1,
        "stage": "L004",
        "evaluation_id": config["evaluation_id"],
        "partition": "validation",
        "final_holdout_touched": False,
        "comparison": "step0_vs_step4",
        "production_optimizer_steps": 4,
        "supervised_training_tokens_seen": 262_144,
        "validation_supervised_tokens_per_model": 300_000,
        "categories": comparisons,
        "combined": {
            "step0_loss": before_combined,
            "step4_loss": after_combined,
            "absolute_change": combined_delta,
            "relative_change": combined_relative,
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
        f"combined step0={before_combined:.9f} "
        f"step4={after_combined:.9f} "
        f"delta={combined_delta:.9f} "
        f"relative={combined_relative:.6%} "
        f"gate_pass={str(combined_pass).lower()}"
    )
    print(f"validation_gate_pass={str(passed).lower()}")
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
