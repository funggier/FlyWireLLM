from __future__ import annotations

import json
from pathlib import Path

from flywire_llm.config import BlankLLMConfig
from flywire_llm.data_contract import (
    eligible_source_ids,
    load_corpus_inventory,
)
from flywire_llm.planning import build_token_budget_plan


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    config = BlankLLMConfig.from_dict(
        json.loads(
            (ROOT / "configs" / "base-50m-v1.json").read_text(
                encoding="utf-8"
            )
        )
    )
    training = json.loads(
        (ROOT / "configs" / "training-base-50m-v1.json").read_text(
            encoding="utf-8"
        )
    )
    inventory = load_corpus_inventory(
        ROOT / "configs" / "corpus-sources-l002.json"
    )
    pilot = json.loads(
        (
            ROOT
            / "results"
            / "l002"
            / "tatoeba-unigram-8192-pilot.json"
        ).read_text(encoding="utf-8")
    )

    assert config.exact_parameter_count == 50_213_376
    assert pilot["benchmark"]["roundtrip_failures"] == 0
    assert pilot["benchmark"]["unk_tokens"] == 0
    assert pilot["tokenizer"]["final_base50m_tokenizer"] is False

    print("stage=L002")
    print("baseline_release=v0.1.0")
    print("profile=base-50m-v1")
    print(f"parameters={config.exact_parameter_count}")
    print(f"vocab_target={config.vocab_size}")
    print(f"context_initial={config.max_seq_len}")
    print(f"primary_token_budget={inventory.target_tokens}")
    print(
        "approved_sources="
        + ",".join(eligible_source_ids(inventory))
    )
    print(
        "conditional_sources="
        + ",".join(
            source.source_id
            for source in inventory.sources
            if source.license_audit_status == "conditional"
        )
    )
    print(
        "blocked_sources="
        + ",".join(
            source.source_id
            for source in inventory.sources
            if source.license_audit_status == "blocked"
        )
    )
    print(
        "pilot_vocab="
        f"{pilot['tokenizer']['actual_vocab_size']}"
    )
    print(
        "pilot_compression_vs_byte="
        f"{pilot['benchmark']['compression_vs_byte_overall']:.6f}"
    )
    print(
        "pilot_thai_compression_vs_byte="
        f"{pilot['benchmark']['compression_vs_byte_thai']:.6f}"
    )
    print(
        "pilot_english_compression_vs_byte="
        f"{pilot['benchmark']['compression_vs_byte_english']:.6f}"
    )
    print(
        "pilot_roundtrip_failures="
        f"{pilot['benchmark']['roundtrip_failures']}"
    )
    print(f"pilot_unk_tokens={pilot['benchmark']['unk_tokens']}")

    schedule = training["schedule"]
    for target in training["token_budgets"]:
        plan = build_token_budget_plan(
            target,
            global_tokens_per_update=training[
                "global_tokens_per_update"
            ],
            warmup_fraction=schedule["warmup_fraction"],
            peak_learning_rate=schedule["peak_learning_rate"],
            min_learning_rate=schedule["min_learning_rate"],
        )
        print(
            f"budget_{target}_steps={plan.optimizer_steps},"
            f"warmup={plan.warmup_steps},"
            f"scheduled={plan.scheduled_tokens},"
            f"overshoot={plan.overshoot_tokens}"
        )

    candidate_path = (
        ROOT / "results" / "l002" / "base50m-unigram-32000.json"
    )
    if candidate_path.exists():
        candidate = json.loads(
            candidate_path.read_text(encoding="utf-8")
        )
        print(
            "candidate_32k_roundtrip_failures="
            f"{candidate['benchmark']['overall']['roundtrip_failures']}"
        )
        print(
            "candidate_32k_unk_tokens="
            f"{candidate['benchmark']['overall']['unk_tokens']}"
        )
        print(
            "candidate_32k_compression_vs_byte="
            f"{candidate['benchmark']['overall']['compression_vs_byte']:.6f}"
        )
    else:
        print("candidate_32k_status=pending")


if __name__ == "__main__":
    main()
