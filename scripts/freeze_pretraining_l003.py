from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from flywire_llm.pretraining_freeze import load_pretraining_freeze


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_freeze(report_path: Path) -> dict:
    report_path = report_path.resolve()
    root = ROOT.resolve()
    try:
        report_relative = report_path.relative_to(root).as_posix()
    except ValueError as exc:
        raise RuntimeError(
            "global manifest report must be inside repository"
        ) from exc

    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema_version") != 1 or report.get("stage") != "L003":
        raise RuntimeError("global manifest report schema/stage changed")
    if report.get("global_cross_source_dedup_applied") is not True:
        raise RuntimeError("global cross-source dedup is not complete")
    if report.get("pretraining_authorized") is not False:
        raise RuntimeError(
            "global manifest report must not self-authorize pretraining"
        )

    mixture = report.get("mixture", {}).get("research_only", {})
    if mixture.get("ready") is not True:
        raise RuntimeError("research_only mixture is not ready")
    for gap in ("thai_gap", "english_gap", "technical_gap", "total_gap"):
        if mixture.get(gap) != 0:
            raise RuntimeError(f"research_only mixture has non-zero {gap}")

    registry = report.get("input_registry")
    decisions = report.get("decisions")
    tokenizer = report.get("tokenizer_model")
    if not all(isinstance(x, dict) for x in (registry, decisions, tokenizer)):
        raise RuntimeError("global report is missing frozen pins")
    if decisions.get("contains_raw_text") is not False:
        raise RuntimeError("decision manifest must be metadata-only")
    if tokenizer.get("candidate_id") != "base50m-unigram-32000-v1":
        raise RuntimeError("tokenizer candidate changed")

    return {
        "schema_version": 1,
        "stage": "L003",
        "checkpoint_lane": "research_only",
        "optimizer_steps_completed": 0,
        "checkpoint_lane_immutable_after_first_optimizer_step": True,
        "source_manifest": {
            "path": registry["path"],
            "sha256": registry["sha256"],
            "frozen": True,
        },
        "global_manifest_report": {
            "path": report_relative,
            "sha256": sha256_file(report_path),
            "frozen": True,
        },
        "evaluation_manifest": {
            "validation_partition": "validation",
            "final_holdout_partition": "holdout",
            "pretraining_fit_partitions": ["train"],
            "tokenizer_fit_must_exclude_holdout": True,
            "model_fit_must_exclude_holdout": True,
            "decisions_sha256": decisions["sha256"],
            "frozen": True,
        },
        "tokenizer": {
            "candidate_id": tokenizer["candidate_id"],
            "model_sha256": tokenizer["sha256"],
        },
        "flywiremodel_automatic_export_allowed": False,
        "public_release_eligibility": "not_qualified",
        "pretraining_authorized": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        default=str(ROOT / "results" / "l003" / "global-manifest-v2.json"),
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "configs" / "pretraining-freeze-l003.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_path = Path(args.report)
    output = Path(args.output)
    payload = build_freeze(report_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    freeze = load_pretraining_freeze(output, repo_root=ROOT)
    print(f"checkpoint_lane={freeze.checkpoint_lane}")
    print(f"optimizer_steps_completed={freeze.optimizer_steps_completed}")
    print(f"source_manifest_sha256={freeze.source_manifest_sha256}")
    print(
        "global_manifest_report_sha256="
        f"{freeze.global_manifest_report_sha256}"
    )
    print(f"decisions_sha256={freeze.decisions_sha256}")
    print(f"tokenizer_sha256={freeze.tokenizer_sha256}")
    print("pretraining_authorized=true")
    print("public_release_eligibility=not_qualified")


if __name__ == "__main__":
    main()
