from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from flywire_llm.artifacts import load_artifact_ledger
from flywire_llm.data_contract import load_corpus_inventory
from flywire_llm.global_manifest import (
    evaluate_global_manifest_mixture,
    materialize_global_manifest,
)
from flywire_llm.manifest_inputs import load_global_manifest_inputs
from flywire_llm.mixing import load_mixture_contract
from flywire_llm.rights import load_rights_policy
from flywire_llm.sentencepiece_tokenizer import SentencePieceTokenizer


ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_ID = "base50m-unigram-32000-v1"
EXPECTED_TOKENIZER_SHA256 = (
    "998bc75f058e554d4f50b6cbc77c52898e3446e516c3b25a126b99b113513818"
)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_tokenizer_model(
    path: str | Path,
    *,
    expected_sha256: str = EXPECTED_TOKENIZER_SHA256,
) -> str:
    path = Path(path)
    if not path.is_file():
        raise RuntimeError("tokenizer model file is missing")
    observed = _sha256_file(path)
    if observed != expected_sha256:
        raise RuntimeError(
            "tokenizer model SHA-256 mismatch "
            f"expected={expected_sha256} observed={observed}"
        )
    return observed


def finalize_report(
    report: dict,
    *,
    registry_path: str | Path,
    registry_display_path: str,
    output_storage: str,
    tokenizer_sha256: str,
    mixture: dict,
) -> dict:
    prefix = "external://FlyWireLLM-data/"
    if (
        not isinstance(output_storage, str)
        or not output_storage.startswith(prefix)
        or output_storage == prefix
    ):
        raise ValueError(
            "output_storage must use external://FlyWireLLM-data/"
        )
    if not isinstance(registry_display_path, str) or not registry_display_path:
        raise ValueError("registry_display_path must be non-empty")

    final = dict(report)
    decisions = dict(final["decisions"])
    decisions["storage"] = (
        output_storage.rstrip("/") + "/" + decisions["path"]
    )
    final["decisions"] = decisions
    final["input_registry"] = {
        "path": registry_display_path,
        "sha256": _sha256_file(registry_path),
    }
    final["tokenizer_model"] = {
        "candidate_id": TOKENIZER_ID,
        "sha256": tokenizer_sha256,
    }
    final["global_cross_source_dedup_applied"] = True
    final["mixture"] = mixture
    final["summary_storage"] = output_storage.rstrip("/") + "/summary.json"
    # Mixture readiness is necessary but not sufficient. Evaluation/source
    # freezes and checkpoint lineage authorization are separate L003 gates.
    final["pretraining_authorized"] = False
    return final


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize the L003 five-corpus global exact/near-dedup "
            "manifest and evaluate the frozen 500M mixture contract."
        )
    )
    parser.add_argument(
        "--registry",
        default=str(ROOT / "configs" / "global-manifest-inputs-l003.json"),
        help="Frozen global-manifest input registry to materialize.",
    )
    parser.add_argument(
        "--external-root",
        required=True,
        help="Filesystem path corresponding to external://FlyWireLLM-data/",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Frozen Base-50M 32K SentencePiece model.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="External directory for metadata-only decisions/index/summary.",
    )
    parser.add_argument(
        "--output-storage",
        required=True,
        help="external://FlyWireLLM-data/ URI corresponding to --output-dir.",
    )
    parser.add_argument(
        "--report",
        default=str(ROOT / "results" / "l003" / "global-manifest-v1.json"),
        help="Tracked metadata-only report path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer_sha256 = verify_tokenizer_model(args.model)

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
    registry_path = Path(args.registry).resolve()
    try:
        registry_display_path = registry_path.relative_to(
            ROOT.resolve()
        ).as_posix()
    except ValueError as exc:
        raise RuntimeError(
            "registry must be inside the FlyWireLLM repository"
        ) from exc
    plan = load_global_manifest_inputs(
        registry_path,
        repo_root=ROOT,
        rights_policy=rights,
        artifact_ledger=ledger,
    )
    if plan.tokenizer != TOKENIZER_ID:
        raise RuntimeError("global input registry tokenizer changed")

    tokenizer = SentencePieceTokenizer(args.model)
    raw_report = materialize_global_manifest(
        plan,
        external_root=args.external_root,
        output_dir=args.output_dir,
        token_counter=lambda text: len(tokenizer.encode(text)),
    )
    contract = load_mixture_contract(ROOT / "configs" / "mixture-l003.json")
    mixture = evaluate_global_manifest_mixture(raw_report, contract)
    final = finalize_report(
        raw_report,
        registry_path=registry_path,
        registry_display_path=registry_display_path,
        output_storage=args.output_storage,
        tokenizer_sha256=tokenizer_sha256,
        mixture=mixture,
    )

    summary_path = Path(args.output_dir) / "summary.json"
    tracked_report = Path(args.report)
    tracked_report.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(final, ensure_ascii=False, indent=2) + "\n"
    summary_path.write_text(
        encoded,
        encoding="utf-8",
        newline="\n",
    )
    tracked_report.write_text(
        encoded,
        encoding="utf-8",
        newline="\n",
    )

    print(f"records_total={final['records']['total']}")
    print(f"records_accepted={final['records']['accepted']}")
    print(f"records_excluded={final['records']['excluded']}")
    print(
        "exact_duplicates="
        f"{final['records']['exact_duplicates']}"
    )
    print(
        "near_duplicates="
        f"{final['records']['near_duplicates']}"
    )
    for lane in ("release_safe", "research_only"):
        lane_mix = final["mixture"][lane]
        print(
            f"mixture_{lane}_ready="
            f"{str(lane_mix['ready']).lower()}"
        )
        print(
            f"mixture_{lane}_total_gap="
            f"{lane_mix['total_gap']}"
        )
        print(
            f"mixture_{lane}_thai_gap="
            f"{lane_mix['thai_gap']}"
        )
        print(
            f"mixture_{lane}_english_gap="
            f"{lane_mix['english_gap']}"
        )
        print(
            f"mixture_{lane}_technical_gap="
            f"{lane_mix['technical_gap']}"
        )
    print("pretraining_authorized=false")


if __name__ == "__main__":
    main()
