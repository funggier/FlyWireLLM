from __future__ import annotations

import argparse
from pathlib import Path

from flywire_llm.pretraining_checkpoint import lineage_from_contract
from flywire_llm.pretraining_data import load_pretraining_data_contract
from flywire_llm.pretraining_execution import (
    load_pretraining_execution_contract,
)
from flywire_llm.pretraining_training import (
    build_primary_schedule,
    load_training_configuration,
)


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--external-root", required=True)
    parser.add_argument(
        "--contract",
        default=str(ROOT / "configs" / "pretraining-execution-l004.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contract = load_pretraining_execution_contract(
        args.contract,
        repo_root=ROOT,
        external_root=args.external_root,
    )
    data = load_pretraining_data_contract(
        ROOT / "configs" / "pretraining-data-runtime-l004.json",
        repo_root=ROOT,
        external_root=args.external_root,
        verify_external_hashes=True,
    )
    training = load_training_configuration(
        ROOT / "configs" / "training-base-50m-v1.json"
    )
    schedule = build_primary_schedule(training)
    lineage = lineage_from_contract(contract)

    print("stage=L004")
    print(f"profile={contract.profile}")
    print(f"checkpoint_lane={contract.checkpoint_lane}")
    print(
        "model_parameters="
        f"{contract.model_config.exact_parameter_count}"
    )
    print(f"tokenizer_vocab={contract.tokenizer_vocab_size}")
    print(f"tokenizer_sha256={contract.tokenizer_sha256}")
    print(f"data_runtime_sha256={contract.data_runtime_sha256}")
    print(f"token_stream_sha256={data.token_stream_sha256}")
    print(f"block_order_sha256={data.block_order_sha256}")
    print(f"total_supervised_tokens={data.total_supervised_tokens}")
    print(
        "source_manifest_sha256="
        f"{lineage.source_manifest_sha256}"
    )
    print(
        "global_manifest_report_sha256="
        f"{lineage.global_manifest_report_sha256}"
    )
    print(f"decisions_sha256={lineage.decisions_sha256}")
    print(f"primary_token_budget={schedule.target_tokens}")
    print(
        "global_tokens_per_update="
        f"{schedule.global_tokens_per_update}"
    )
    print(f"optimizer_steps={schedule.optimizer_steps}")
    print(f"warmup_steps={schedule.warmup_steps}")
    print("pretraining_runtime_contract=PASS")


if __name__ == "__main__":
    main()
