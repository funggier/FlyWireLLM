from __future__ import annotations

import argparse
import json
from pathlib import Path

from flywire_llm.pretraining_freeze import load_pretraining_freeze


ROOT = Path(__file__).resolve().parents[1]

_EXPECTED_V5 = {
    "technical-fineweb2-thai-v5-1pct.json": {
        "records_sampled": 3383,
        "technical_tokens": 53094,
        "technical_token_fraction": 0.01792033762896254,
    },
    "technical-fineweb-en-014-v5-1pct.json": {
        "records_sampled": 1666,
        "technical_tokens": 248949,
        "technical_token_fraction": 0.19793674267722547,
    },
}


def _load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path}: expected JSON object")
    return payload


def audit_v5() -> None:
    for name, expected in _EXPECTED_V5.items():
        path = ROOT / "results" / "l003" / name
        report = _load_json(path)
        if report.get("classifier") != "technical-heuristic-v5":
            raise RuntimeError(f"{name}: classifier changed")
        if report.get("records_sampled") != expected["records_sampled"]:
            raise RuntimeError(f"{name}: sampled record count changed")
        technical_tokens = report.get("tokens_by_category", {}).get(
            "technical_scientific_code"
        )
        if technical_tokens != expected["technical_tokens"]:
            raise RuntimeError(f"{name}: technical token count changed")
        if (
            report.get("technical_token_fraction")
            != expected["technical_token_fraction"]
        ):
            raise RuntimeError(f"{name}: technical token fraction changed")
        if "corpus_path" in report:
            raise RuntimeError(f"{name}: local corpus path leaked")
        storage = report.get("corpus_storage")
        if (
            not isinstance(storage, str)
            or not storage.startswith("external://FlyWireLLM-data/")
        ):
            raise RuntimeError(f"{name}: external corpus storage missing")
        if report.get("local_path_recorded") is not False:
            raise RuntimeError(f"{name}: local_path_recorded changed")
        if report.get("training_authorized") is not False:
            raise RuntimeError(f"{name}: calibration self-authorized training")
        if report.get("calibration_only") is not True:
            raise RuntimeError(f"{name}: calibration_only changed")
        print(
            f"PASS technical_v5={name} "
            f"technical_tokens={technical_tokens}"
        )


def audit_v1_failure() -> None:
    path = ROOT / "results" / "l003" / "global-manifest-v1.json"
    report = _load_json(path)
    if report.get("global_cross_source_dedup_applied") is not True:
        raise RuntimeError("v1 five-corpus global dedup was not complete")
    records = report.get("records", {})
    expected_records = {
        "total": 2_609_182,
        "accepted": 2_608_512,
        "excluded": 670,
        "exact_duplicates": 3,
        "near_duplicates": 667,
    }
    if records != expected_records:
        raise RuntimeError("v1 five-corpus record accounting changed")
    research = report.get("mixture", {}).get("research_only", {})
    if research.get("ready") is not False:
        raise RuntimeError(
            "v1 five-corpus evidence must remain a failed mixture attempt"
        )
    expected_gaps = {
        "total_gap": 26_974_429,
        "thai_gap": 0,
        "english_gap": 81_481_741,
        "technical_gap": 62_326_681,
    }
    for key, value in expected_gaps.items():
        if research.get(key) != value:
            raise RuntimeError(f"v1 five-corpus {key} changed")
    if report.get("pretraining_authorized") is not False:
        raise RuntimeError("v1 five-corpus must not authorize pretraining")
    print(
        "PASS five_corpus_v1=FAILED_AS_EXPECTED "
        "english_gap=81481741 technical_gap=62326681"
    )


def audit_final_freeze(freeze_path: Path) -> None:
    freeze = load_pretraining_freeze(freeze_path, repo_root=ROOT)
    if freeze.checkpoint_lane != "research_only":
        raise RuntimeError("L003 final checkpoint lane must be research_only")
    if freeze.optimizer_steps_completed != 0:
        raise RuntimeError("L003 freeze was not declared before training")
    if freeze.pretraining_authorized is not True:
        raise RuntimeError("L003 final freeze did not authorize pretraining")
    print(
        "PASS final_freeze checkpoint_lane=research_only "
        "optimizer_steps_completed=0 pretraining_authorized=true"
    )
    print(f"source_manifest_sha256={freeze.source_manifest_sha256}")
    print(
        "global_manifest_report_sha256="
        f"{freeze.global_manifest_report_sha256}"
    )
    print(f"decisions_sha256={freeze.decisions_sha256}")
    print(f"tokenizer_sha256={freeze.tokenizer_sha256}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--freeze",
        default=str(ROOT / "configs" / "pretraining-freeze-l003.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit_v5()
    audit_v1_failure()
    audit_final_freeze(Path(args.freeze))
    print(
        "decision=L003 GREEN / TECHNICAL HEURISTIC V5 FROZEN / "
        "GLOBAL EXACT+NEAR DEDUP QUALIFIED / 500M RESEARCH-ONLY MIXTURE "
        "QUALIFIED / PRETRAINING AUTHORIZED FOR RESEARCH-ONLY LINEAGE / "
        "PUBLIC RELEASE NOT QUALIFIED"
    )


if __name__ == "__main__":
    main()
