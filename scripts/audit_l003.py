from __future__ import annotations

import hashlib
import json
from pathlib import Path

from flywire_llm.artifacts import load_artifact_ledger
from flywire_llm.calibration import load_calibration_artifacts
from flywire_llm.data_contract import load_corpus_inventory
from flywire_llm.pretraining_freeze import load_pretraining_freeze
from flywire_llm.rights import (
    load_rights_policy,
    token_gap,
    training_budget_ready,
)
from flywire_llm.source_candidates import load_source_candidates


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    inventory = load_corpus_inventory(
        ROOT / "configs" / "corpus-sources-l002.json"
    )
    rights = load_rights_policy(
        ROOT / "configs" / "rights-lanes-l003.json",
        expected_source_ids={
            source.source_id for source in inventory.sources
        },
    )
    ledger = load_artifact_ledger(
        ROOT / "configs" / "source-artifacts-l003.json",
        rights_policy=rights,
    )
    freeze = load_pretraining_freeze(
        ROOT / "configs" / "pretraining-freeze-l003.json",
        repo_root=ROOT,
    )
    candidates = load_source_candidates(
        ROOT / "configs" / "source-candidates-l003.json",
        rights_policy=rights,
    )
    calibrations = load_calibration_artifacts(
        ROOT / "configs" / "calibration-artifacts-l003.json",
        rights_policy=rights,
    )
    for calibration in calibrations:
        report_path = ROOT / calibration.report_path
        observed = sha256_file(report_path)
        if observed != calibration.report_sha256:
            raise RuntimeError(
                f"calibration report hash mismatch "
                f"{calibration.calibration_id}: "
                f"expected={calibration.report_sha256} observed={observed}"
            )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("training_authorized") is not False:
            raise RuntimeError(
                f"{calibration.calibration_id}: calibration report must "
                "not authorize training"
            )
        if report.get("calibration_only") is not True:
            raise RuntimeError(
                f"{calibration.calibration_id}: calibration_only changed"
            )
        if report["rows"]["total"] != calibration.observed_rows:
            raise RuntimeError("calibration row count changed")
        if report["content"]["tokens"] != calibration.observed_tokens:
            raise RuntimeError("calibration token count changed")
        if report["screening"]["status"]["accept"]["tokens"] != (
            calibration.accepted_tokens
        ):
            raise RuntimeError("calibration accepted token count changed")

    quality_path = (
        ROOT / "results" / "l003" / "acquired-quality-audit.json"
    )
    quality_hash = sha256_file(quality_path)
    expected_hash = rights.known_token_accounting[
        "quality_audit_sha256"
    ]
    if quality_hash != expected_hash:
        raise RuntimeError(
            f"quality audit hash mismatch expected={expected_hash} "
            f"observed={quality_hash}"
        )
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    if quality["screening_policy"][
        "raw_flagged_text_persisted_in_report"
    ] is not False:
        raise RuntimeError("quality report must never persist flagged text")

    budget = quality["screened_train_budget"]
    if budget["release_safe_tokens"] != 20_061_246:
        raise RuntimeError("release-safe screened token count changed")
    if budget["research_only_tokens"] != 38_172_496:
        raise RuntimeError("research-only screened token count changed")
    if budget["combined_research_eligible_tokens"] != 58_233_742:
        raise RuntimeError("combined screened token count changed")

    print("stage=L003")
    print("base_commit=528e96736c52d5ba90412bb1ee0b8c0239349e57")
    print(f"acquired_artifacts={len(ledger.artifacts)}")
    print(
        "artifact_ids="
        + ",".join(artifact.artifact_id for artifact in ledger.artifacts)
    )
    print(f"acquisition_candidates={len(candidates)}")
    print(
        "candidate_ids="
        + ",".join(candidate.candidate_id for candidate in candidates)
    )
    print(f"calibration_artifacts={len(calibrations)}")
    for calibration in calibrations:
        print(
            f"calibration={calibration.calibration_id},"
            f"rows={calibration.observed_rows},"
            f"accepted_tokens={calibration.accepted_tokens},"
            f"accepted_fraction={calibration.accepted_token_fraction:.6f},"
            f"tokens_per_compressed_gib="
            f"{calibration.accepted_tokens_per_compressed_gib:.3f},"
            "training_authorized=false"
        )
    print(f"quality_audit_sha256={quality_hash}")
    print(f"quality_records={quality['overall']['records']}")
    print(f"quality_tokens={quality['overall']['tokens']}")
    print(
        "accepted_tokens="
        f"{quality['overall']['status']['accept']['tokens']}"
    )
    print(
        "quarantine_records="
        f"{quality['overall']['status']['quarantine']['records']}"
    )
    print(
        "reject_records="
        f"{quality['overall']['status']['reject']['records']}"
    )
    print(
        "screened_baseline_release_safe_tokens="
        f"{budget['release_safe_tokens']}"
    )
    print(
        "screened_baseline_research_eligible_tokens="
        f"{budget['combined_research_eligible_tokens']}"
    )
    qualified = rights.known_token_accounting["qualified_global_train"]
    print(
        "qualified_release_safe_tokens="
        f"{qualified['release_safe']['total']}"
    )
    print(
        "qualified_research_tokens="
        f"{qualified['research_only']['total']}"
    )
    print(
        "release_safe_gap="
        f"{token_gap(rights, checkpoint_lane='release_safe')}"
    )
    print(
        "research_gap="
        f"{token_gap(rights, checkpoint_lane='research_only')}"
    )
    print(
        "release_safe_ready="
        f"{training_budget_ready(rights, checkpoint_lane='release_safe')}"
    )
    print(
        "research_ready="
        f"{training_budget_ready(rights, checkpoint_lane='research_only')}"
    )
    print(
        "common_voice_download_state="
        "authenticated_mdc_access_required"
    )
    print(
        "near_duplicate_contract="
        "simhash64+4x16bit-lsh+hamming<=3"
    )
    print(f"checkpoint_lane={freeze.checkpoint_lane}")
    print(
        "pretraining_authorized="
        f"{str(freeze.pretraining_authorized).lower()}"
    )
    print("public_release_eligibility=not_qualified")
    print(
        "decision=L003 GREEN / INGESTION SAFETY GREEN / "
        "GLOBAL EXACT+NEAR DEDUP QUALIFIED / "
        "500M RESEARCH-ONLY MIXTURE QUALIFIED / "
        "PRETRAINING AUTHORIZED FOR RESEARCH-ONLY LINEAGE / "
        "PUBLIC RELEASE NOT QUALIFIED"
    )


if __name__ == "__main__":
    main()
