from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUTH_SHA = "d8e1ffe91ab4af0aed58fc5635674e9b87923b2cc96347344283706743d94846"
CHECKPOINT_SHA = "ccce01d29686a76edfe52fbaa531cf198a82bfd8aa17ded34625b8c6f14b979e"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--authorization",
        default=str(
            ROOT / "configs" / "pretraining-tranche-l004-v12.json"
        ),
    )
    parser.add_argument(
        "--step153",
        default=str(
            ROOT / "results" / "l004" / "validation-step153-v1.json"
        ),
    )
    parser.add_argument(
        "--step169",
        default=str(
            ROOT / "results" / "l004" / "validation-step169-v1.json"
        ),
    )
    parser.add_argument(
        "--output",
        default=str(
            ROOT / "results" / "l004"
            / "validation-comparison-step153-step169-v1.json"
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
    step153 = _load(args.step153)
    step169 = _load(args.step169)
    gate = authorization.get("post_tranche_validation_gate")
    expected_gate = {
        "evaluation_id": "base50m-validation-v1",
        "baseline_optimizer_step": 153,
        "candidate_optimizer_step": 169,
        "same_validation_pack_required": True,
        "final_holdout_must_remain_untouched": True,
        "combined_loss_must_not_increase_vs_step153": True,
        "max_category_relative_loss_increase_vs_step153": 0.005,
        "require_all_losses_finite": True,
    }
    if gate != expected_gate:
        raise RuntimeError("predeclared post-warmup validation gate changed")

    for payload, model, step, tokens in (
        (step153, "step153", 153, 10_027_008),
        (step169, "step169", 169, 11_075_584),
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

    if step169.get("post_warmup_authorization_sha256") != AUTH_SHA:
        raise RuntimeError("step169 post-warmup authorization hash changed")
    if step169.get("checkpoint_sha256") != CHECKPOINT_SHA:
        raise RuntimeError("step169 checkpoint hash changed")

    categories = sorted(step153["categories"])
    if categories != sorted(step169["categories"]):
        raise RuntimeError("step153/step169 validation category sets differ")

    max_relative = float(
        gate["max_category_relative_loss_increase_vs_step153"]
    )
    comparisons: dict[str, dict[str, object]] = {}
    category_gate_pass = True
    same_pack = True
    for category in categories:
        before = step153["categories"][category]
        after = step169["categories"][category]
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
            "step153_loss": before_loss,
            "step169_loss": after_loss,
            "absolute_change": absolute,
            "relative_change": relative,
            "pack_sha256": after["pack_sha256"],
            "gate_pass": passed,
        }
        print(
            f"{category} step153={before_loss:.9f} "
            f"step169={after_loss:.9f} "
            f"relative={relative:.6%} "
            f"gate_pass={str(passed).lower()}"
        )

    if gate["same_validation_pack_required"] and not same_pack:
        raise RuntimeError(
            "step153/step169 did not use identical validation packs"
        )

    before_combined = float(step153["combined"]["loss"])
    after_combined = float(step169["combined"]["loss"])
    combined_finite = (
        math.isfinite(before_combined) and math.isfinite(after_combined)
    )
    combined_pass = combined_finite and (
        not gate["combined_loss_must_not_increase_vs_step153"]
        or after_combined <= before_combined
    )
    passed = (
        same_pack
        and category_gate_pass
        and combined_pass
        and step169.get("final_holdout_touched") is False
    )
    output = {
        "schema_version": 1,
        "stage": "L004",
        "evaluation_id": gate["evaluation_id"],
        "comparison": "step153_vs_step169",
        "partition": "validation",
        "same_validation_pack": same_pack,
        "final_holdout_touched": False,
        "authorization_sha256": AUTH_SHA,
        "candidate_checkpoint_sha256": step169["checkpoint_sha256"],
        "production_optimizer_steps": 169,
        "supervised_training_tokens_seen": 11_075_584,
        "validation_supervised_tokens_per_model": 300_000,
        "categories": comparisons,
        "combined": {
            "step153_loss": before_combined,
            "step169_loss": after_combined,
            "absolute_change": after_combined - before_combined,
            "relative_change": (
                (after_combined - before_combined) / before_combined
            ),
            "gate_pass": combined_pass,
        },
        "gate_contract": gate,
        "gate_pass": passed,
        "larger_post_warmup_tranche_may_be_considered": passed,
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
        f"combined step153={before_combined:.9f} "
        f"step169={after_combined:.9f} "
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
