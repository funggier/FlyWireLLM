from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from flywire_llm.artifacts import (
    load_artifact_ledger,
    resolve_external_storage_uri,
)
from flywire_llm.data_contract import load_corpus_inventory
from flywire_llm.manifest_inputs import (
    load_global_manifest_inputs,
    verify_global_manifest_input_file,
)
from flywire_llm.rights import load_rights_policy


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(
    path: str | Path,
    *,
    expected_bytes: int | None = None,
    expected_sha256: str,
    label: str,
) -> None:
    path = Path(path)
    if not path.is_file():
        raise RuntimeError(f"{label}: file is missing")
    if expected_bytes is not None and path.stat().st_size != expected_bytes:
        raise RuntimeError(
            f"{label}: byte size mismatch expected={expected_bytes} "
            f"observed={path.stat().st_size}"
        )
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise RuntimeError(
            f"{label}: SHA-256 mismatch expected={expected_sha256} "
            f"observed={observed}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify an L003 global manifest against every frozen external "
            "accepted corpus, decision manifest, summary and tokenizer."
        )
    )
    parser.add_argument("--report", required=True)
    parser.add_argument("--external-root", required=True)
    parser.add_argument("--model", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_path = Path(args.report).resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema_version") != 1 or report.get("stage") != "L003":
        raise RuntimeError("global manifest report schema/stage changed")
    if report.get("global_cross_source_dedup_applied") is not True:
        raise RuntimeError("global cross-source dedup is not complete")
    if report.get("pretraining_authorized") is not False:
        raise RuntimeError(
            "global manifest report must not self-authorize pretraining"
        )

    tokenizer = report.get("tokenizer_model")
    if not isinstance(tokenizer, dict):
        raise RuntimeError("global report missing tokenizer_model")
    if tokenizer.get("candidate_id") != "base50m-unigram-32000-v1":
        raise RuntimeError("global report tokenizer candidate changed")
    verify_file(
        args.model,
        expected_sha256=tokenizer["sha256"],
        label="tokenizer_model",
    )
    print(
        f"PASS tokenizer_model sha256={tokenizer['sha256']}",
        flush=True,
    )

    registry = report.get("input_registry")
    if not isinstance(registry, dict):
        raise RuntimeError("global report missing input_registry")
    registry_path = (ROOT / registry["path"]).resolve()
    try:
        registry_path.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise RuntimeError("input registry escapes repository") from exc
    verify_file(
        registry_path,
        expected_sha256=registry["sha256"],
        label="input_registry",
    )

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
    plan = load_global_manifest_inputs(
        registry_path,
        repo_root=ROOT,
        rights_policy=rights,
        artifact_ledger=ledger,
    )
    for item in plan.inputs:
        path = verify_global_manifest_input_file(
            item,
            external_root=args.external_root,
        )
        print(
            f"PASS input={item.input_id} bytes={item.accepted_bytes} "
            f"sha256={item.accepted_sha256} path={path}",
            flush=True,
        )

    decisions = report.get("decisions")
    if not isinstance(decisions, dict):
        raise RuntimeError("global report missing decisions")
    if decisions.get("contains_raw_text") is not False:
        raise RuntimeError("decision manifest must be metadata-only")
    decisions_path = resolve_external_storage_uri(
        decisions["storage"],
        external_root=args.external_root,
    )
    verify_file(
        decisions_path,
        expected_bytes=decisions["bytes"],
        expected_sha256=decisions["sha256"],
        label="global_decisions",
    )
    print(
        f"PASS decisions bytes={decisions['bytes']} "
        f"sha256={decisions['sha256']}",
        flush=True,
    )

    summary_path = resolve_external_storage_uri(
        report["summary_storage"],
        external_root=args.external_root,
    )
    tracked_sha = sha256_file(report_path)
    verify_file(
        summary_path,
        expected_bytes=report_path.stat().st_size,
        expected_sha256=tracked_sha,
        label="external_summary",
    )
    print(
        f"PASS external_summary sha256={tracked_sha}",
        flush=True,
    )
    print(f"verified_inputs={len(plan.inputs)}", flush=True)
    print("global_external_qualification=PASS", flush=True)


if __name__ == "__main__":
    main()
