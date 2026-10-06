from __future__ import annotations

import argparse
import json
from pathlib import Path

from flywire_llm.pretraining_continuation import (
    load_continuation_authorization,
    load_continuation_state,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--external-root", required=True)
    parser.add_argument(
        "--pack-summary",
        default=str(ROOT / "results" / "l004" / "validation-pack-v1.json"),
    )
    parser.add_argument(
        "--step8",
        default=str(ROOT / "results" / "l004" / "validation-step8-v1.json"),
    )
    parser.add_argument(
        "--comparison",
        default=str(
            ROOT
            / "results"
            / "l004"
            / "validation-comparison-step4-step8-v1.json"
        ),
    )
    parser.add_argument(
        "--second-tranche",
        default=str(
            ROOT / "results" / "l004" / "second-bounded-tranche-v1.json"
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
    summary = _load(args.pack_summary)
    step8 = _load(args.step8)
    comparison = _load(args.comparison)
    tranche = _load(args.second_tranche)
    authorization = load_continuation_authorization(
        ROOT / "configs" / "pretraining-tranche-l004-v2.json",
        repo_root=ROOT,
        external_root=args.external_root,
    )
    state = load_continuation_state(
        authorization.run_root / "state.json",
        authorization=authorization,
        verify_files=True,
    )

    if state.optimizer_step != 8 or state.supervised_tokens_seen != 524_288:
        raise RuntimeError("step8 continuation state changed")
    if tranche.get("production_optimizer_steps") != 8:
        raise RuntimeError("second-tranche summary step changed")
    if tranche.get("supervised_tokens_seen") != 524_288:
        raise RuntimeError("second-tranche summary token progress changed")
    if tranche.get("authorization_sha256") != (
        authorization.authorization_sha256
    ):
        raise RuntimeError("second-tranche authorization hash changed")
    if tranche.get("state_sha256") != sha256_file(
        authorization.run_root / "state.json"
    ):
        raise RuntimeError("second-tranche state hash changed")
    final = tranche.get("final_checkpoint")
    if not isinstance(final, dict):
        raise RuntimeError("second-tranche final checkpoint metadata missing")
    if final.get("sha256") != state.checkpoint_sha256:
        raise RuntimeError("second-tranche final checkpoint hash changed")
    if final.get("bytes") != state.checkpoint_bytes:
        raise RuntimeError("second-tranche final checkpoint size changed")

    if step8.get("schema_version") != 1 or step8.get("stage") != "L004":
        raise RuntimeError("step8 validation schema/stage changed")
    if step8.get("evaluation_id") != "base50m-validation-v1":
        raise RuntimeError("step8 validation evaluation id changed")
    if step8.get("model") != "step8":
        raise RuntimeError("step8 validation model label changed")
    if step8.get("optimizer_step") != 8:
        raise RuntimeError("step8 validation optimizer step changed")
    if step8.get("supervised_tokens_seen") != 524_288:
        raise RuntimeError("step8 validation training progress changed")
    if step8.get("checkpoint_sha256") != state.checkpoint_sha256:
        raise RuntimeError("step8 validation checkpoint hash changed")
    if step8.get("continuation_authorization_sha256") != (
        authorization.authorization_sha256
    ):
        raise RuntimeError("step8 validation authorization hash changed")
    if step8.get("partition") != "validation":
        raise RuntimeError("step8 validation partition changed")
    if step8.get("final_holdout_touched") is not False:
        raise RuntimeError("step8 validation touched final holdout")
    if step8["combined"].get("supervised_tokens") != 300_000:
        raise RuntimeError("step8 validation token total changed")

    packs = summary.get("packs")
    categories = step8.get("categories")
    if not isinstance(packs, dict) or not isinstance(categories, dict):
        raise RuntimeError("validation pack/category metadata missing")
    if set(packs) != set(categories):
        raise RuntimeError("step8 validation category set changed")
    for category in sorted(packs):
        pack_meta = packs[category]
        observed = categories[category]
        if observed.get("supervised_tokens") != 100_000:
            raise RuntimeError(f"{category}: validation tokens changed")
        if observed.get("pack_sha256") != pack_meta.get("sha256"):
            raise RuntimeError(f"{category}: validation pack hash changed")
        loss = float(observed["loss"])
        if not (0.0 < loss < 100.0):
            raise RuntimeError(f"{category}: invalid validation loss")
        print(
            f"PASS step8_category={category} "
            f"loss={loss:.9f} pack_sha256={observed['pack_sha256']}"
        )

    raw_auth = _load(ROOT / "configs" / "pretraining-tranche-l004-v2.json")
    expected_gate = raw_auth.get("post_tranche_validation_gate")
    if comparison.get("gate_contract") != expected_gate:
        raise RuntimeError("post-tranche validation gate contract changed")
    if comparison.get("same_validation_pack") is not True:
        raise RuntimeError("step4/step8 validation packs differ")
    if comparison.get("final_holdout_touched") is not False:
        raise RuntimeError("post-tranche comparison touched holdout")
    if comparison.get("gate_pass") is not True:
        raise RuntimeError("post-tranche validation gate failed")
    if comparison.get("larger_tranche_may_be_considered") is not True:
        raise RuntimeError("post-tranche expansion decision changed")
    if comparison.get("candidate_checkpoint_sha256") != (
        state.checkpoint_sha256
    ):
        raise RuntimeError("comparison candidate checkpoint hash changed")

    print(
        "combined_loss "
        f"step4={comparison['combined']['step4_loss']:.9f} "
        f"step8={comparison['combined']['step8_loss']:.9f} "
        f"relative={comparison['combined']['step4_to_step8_relative_change']:.6%}"
    )
    print(f"step8_checkpoint_sha256={state.checkpoint_sha256}")
    print("same_validation_pack=true")
    print("holdout_touched=false")
    print("post_tranche_validation_verification=PASS")


if __name__ == "__main__":
    main()
