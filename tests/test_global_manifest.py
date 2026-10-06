from __future__ import annotations

import hashlib
import json

import pytest

from flywire_llm.data_contract import dedup_fingerprint
from flywire_llm.global_manifest import (
    CompactGlobalDeduplicator,
    GlobalCandidate,
    GlobalDeduplicator,
    GlobalManifestError,
    category_for_text,
    parse_global_candidate,
    priority_key,
)
from flywire_llm.manifest_inputs import (
    GlobalManifestInput,
    GlobalManifestInputPlan,
)
from flywire_llm.mixing import MixtureContract
from flywire_llm.quality import simhash64
import flywire_llm.global_manifest as global_manifest


def _payload(
    *,
    record_id: str,
    text: str,
    partition: str = "train",
    lane: str = "research_only",
    source_id: str = "fineweb2-thai",
    artifact_id: str = "artifact-a",
    language: str = "th",
) -> dict:
    return {
        "record_id": record_id,
        "text": text,
        "language": language,
        "source_id": source_id,
        "artifact_id": artifact_id,
        "source_group_id": dedup_fingerprint(text),
        "partition": partition,
        "rights_lane": lane,
        "content_sha256": hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest(),
        "dedup_fingerprint": dedup_fingerprint(text),
        "simhash64": f"{simhash64(text):016x}",
    }


def test_global_candidate_recomputes_all_content_fingerprints():
    payload = _payload(record_id="r1", text="ข้อมูลตัวอย่าง")
    candidate = parse_global_candidate(payload)
    assert candidate.canonical_id.endswith("::r1")

    bad = dict(payload)
    bad["simhash64"] = "0" * 16
    with pytest.raises(GlobalManifestError, match="simhash64 does not match"):
        parse_global_candidate(bad)

    bad = dict(payload)
    bad["dedup_fingerprint"] = "0" * 64
    with pytest.raises(
        GlobalManifestError,
        match="dedup_fingerprint does not match",
    ):
        parse_global_candidate(bad)


def test_priority_preserves_holdout_then_validation_then_train():
    text = "same text"
    holdout = parse_global_candidate(
        _payload(
            record_id="h",
            text=text,
            partition="holdout",
            lane="release_safe",
            source_id="tatoeba-sentences",
        )
    )
    validation = parse_global_candidate(
        _payload(
            record_id="v",
            text=text,
            partition="validation",
            lane="release_safe",
            source_id="tatoeba-sentences",
        )
    )
    train = parse_global_candidate(
        _payload(
            record_id="t",
            text=text,
            partition="train",
            lane="release_safe",
            source_id="tatoeba-sentences",
        )
    )
    assert priority_key(holdout) < priority_key(validation) < priority_key(train)


def test_release_safe_precedes_research_only_within_partition():
    safe = parse_global_candidate(
        _payload(
            record_id="safe",
            text="safe",
            lane="release_safe",
            source_id="tatoeba-sentences",
        )
    )
    research = parse_global_candidate(
        _payload(
            record_id="research",
            text="research",
            lane="research_only",
            source_id="fineweb-english",
            language="en",
        )
    )
    assert priority_key(safe) < priority_key(research)


def test_exact_duplicate_is_excluded_and_points_to_canonical(tmp_path):
    text = "identical source text"
    first = parse_global_candidate(
        _payload(
            record_id="first",
            text=text,
            partition="holdout",
            lane="release_safe",
            source_id="tatoeba-sentences",
            language="en",
        )
    )
    second = parse_global_candidate(
        _payload(
            record_id="second",
            text=text,
            partition="train",
            lane="research_only",
            source_id="fineweb-english",
            artifact_id="artifact-b",
            language="en",
        )
    )
    with GlobalDeduplicator(tmp_path) as dedup:
        a = dedup.consider(
            first,
            token_count=4,
            primary_category="general_english",
        )
        b = dedup.consider(
            second,
            token_count=4,
            primary_category="general_english",
        )
        assert a.status == "accept"
        assert b.status == "exclude"
        assert b.reason == "cross_source_exact_duplicate"
        assert b.duplicate_of == first.canonical_id
        assert dedup.accepted_record_count == 1


def test_noncanonical_priority_order_fails_closed(tmp_path):
    train = parse_global_candidate(
        _payload(
            record_id="train",
            text="train text",
            partition="train",
            lane="release_safe",
            source_id="tatoeba-sentences",
            language="en",
        )
    )
    holdout = parse_global_candidate(
        _payload(
            record_id="holdout",
            text="holdout text",
            partition="holdout",
            lane="release_safe",
            source_id="tatoeba-sentences",
            language="en",
        )
    )
    with GlobalDeduplicator(tmp_path) as dedup:
        dedup.consider(
            train,
            token_count=2,
            primary_category="general_english",
        )
        with pytest.raises(
            GlobalManifestError,
            match="canonical priority order",
        ):
            dedup.consider(
                holdout,
                token_count=2,
                primary_category="general_english",
            )


def test_category_routing_uses_current_technical_classifier():
    assert category_for_text(
        text=(
            "แบบจำลองโครงข่ายประสาทใช้ชุดข้อมูลและพารามิเตอร์ "
            "พร้อมเกรเดียนต์"
        ),
        language="th",
    ) == "technical_scientific_code"
    assert category_for_text(
        text="ครอบครัวเดินเล่นในสวนและรับประทานอาหารเย็น",
        language="th",
    ) == "general_thai"
    assert category_for_text(
        text="ordinary family travel story",
        language="en",
    ) == "general_english"


def _write_manifest_fixture(path, rows):
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        for row in rows
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8", newline="\n")
    return len(payload.encode("utf-8")), hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()


def _input_for_file(
    *,
    input_id,
    source_id,
    artifact_id,
    lane,
    language,
    storage,
    byte_count,
    sha256,
):
    return GlobalManifestInput(
        input_id=input_id,
        source_id=source_id,
        artifact_id=artifact_id,
        rights_lane=lane,
        language=language,
        summary_path="fixture-summary.json",
        summary_sha256="0" * 64,
        accepted_storage=storage,
        accepted_bytes=byte_count,
        accepted_sha256=sha256,
    )


def test_materialize_global_manifest_prioritizes_partition_and_emits_metadata_only(
    tmp_path,
):
    external = tmp_path / "external"
    research_path = external / "fixture" / "research.jsonl"
    safe_path = external / "fixture" / "safe.jsonl"
    duplicate_text = "shared canonical sentence"
    research_rows = [
        _payload(
            record_id="research-holdout",
            text=duplicate_text,
            partition="holdout",
            lane="research_only",
            source_id="source-research",
            artifact_id="artifact-research",
            language="en",
        ),
    ]
    safe_rows = [
        _payload(
            record_id="safe-train-duplicate",
            text=duplicate_text,
            partition="train",
            lane="release_safe",
            source_id="source-safe",
            artifact_id="artifact-safe",
            language="en",
        ),
        _payload(
            record_id="safe-train-unique",
            text="unique general sentence",
            partition="train",
            lane="release_safe",
            source_id="source-safe",
            artifact_id="artifact-safe",
            language="en",
        ),
    ]
    for row in research_rows + safe_rows:
        row.pop("rights_lane")

    research_bytes, research_sha = _write_manifest_fixture(
        research_path,
        research_rows,
    )
    safe_bytes, safe_sha = _write_manifest_fixture(safe_path, safe_rows)
    plan = GlobalManifestInputPlan(
        tokenizer="base50m-unigram-32000-v1",
        technical_classifier="technical-heuristic-v5",
        near_duplicate_hamming_distance=3,
        inputs=(
            _input_for_file(
                input_id="safe",
                source_id="source-safe",
                artifact_id="artifact-safe",
                lane="release_safe",
                language="en",
                storage="external://FlyWireLLM-data/fixture/safe.jsonl",
                byte_count=safe_bytes,
                sha256=safe_sha,
            ),
            _input_for_file(
                input_id="research",
                source_id="source-research",
                artifact_id="artifact-research",
                lane="research_only",
                language="en",
                storage="external://FlyWireLLM-data/fixture/research.jsonl",
                byte_count=research_bytes,
                sha256=research_sha,
            ),
        ),
        pretraining_authorized=False,
    )

    report = global_manifest.materialize_global_manifest(
        plan,
        external_root=external,
        output_dir=tmp_path / "global",
        token_counter=lambda text: len(text.split()),
    )

    assert report["records"] == {
        "total": 3,
        "accepted": 2,
        "excluded": 1,
        "exact_duplicates": 1,
        "near_duplicates": 0,
    }
    assert report["train_capacity"]["release_safe"]["general_english"] == 3
    assert report["train_capacity"]["research_only"]["general_english"] == 3

    decisions = [
        json.loads(line)
        for line in (tmp_path / "global" / "decisions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert decisions[0]["record_id"] == "research-holdout"
    assert decisions[1]["record_id"] == "safe-train-duplicate"
    assert decisions[1]["duplicate_of"].endswith("::research-holdout")
    assert all("text" not in decision for decision in decisions)


def test_compact_global_deduplicator_matches_exact_and_near_semantics():
    first = GlobalCandidate(
        record_id="first",
        text="first",
        language="en",
        source_id="a",
        artifact_id="a1",
        source_group_id="g1",
        partition="train",
        rights_lane="release_safe",
        content_sha256="1" * 64,
        dedup_fingerprint="1" * 64,
        simhash64_hex="123456789abcdef0",
    )
    exact = GlobalCandidate(
        record_id="exact",
        text="exact",
        language="en",
        source_id="b",
        artifact_id="b1",
        source_group_id="g2",
        partition="train",
        rights_lane="research_only",
        content_sha256="2" * 64,
        dedup_fingerprint="1" * 64,
        simhash64_hex="ffffffffffffffff",
    )
    near = GlobalCandidate(
        record_id="near",
        text="near",
        language="en",
        source_id="c",
        artifact_id="c1",
        source_group_id="g3",
        partition="train",
        rights_lane="research_only",
        content_sha256="3" * 64,
        dedup_fingerprint="3" * 64,
        simhash64_hex=f"{(0x123456789ABCDEF0 ^ 0b111):016x}",
    )
    with CompactGlobalDeduplicator() as dedup:
        accepted = dedup.consider(
            first,
            token_count=1,
            primary_category="general_english",
        )
        exact_decision = dedup.consider(
            exact,
            token_count=1,
            primary_category="general_english",
        )
        near_decision = dedup.consider(
            near,
            token_count=1,
            primary_category="general_english",
        )
        assert dedup.accepted_record_count == 1
    assert accepted.status == "accept"
    assert exact_decision.reason == "cross_source_exact_duplicate"
    assert exact_decision.duplicate_of == first.canonical_id
    assert near_decision.reason == "cross_source_near_duplicate"
    assert near_decision.duplicate_of == first.canonical_id
    assert near_decision.hamming_distance == 3


def test_global_deduplicator_excludes_three_bit_near_duplicate(tmp_path):
    first = GlobalCandidate(
        record_id="first",
        text="first",
        language="en",
        source_id="a",
        artifact_id="a1",
        source_group_id="g1",
        partition="train",
        rights_lane="release_safe",
        content_sha256="1" * 64,
        dedup_fingerprint="1" * 64,
        simhash64_hex="123456789abcdef0",
    )
    second = GlobalCandidate(
        record_id="second",
        text="second",
        language="en",
        source_id="b",
        artifact_id="b1",
        source_group_id="g2",
        partition="train",
        rights_lane="research_only",
        content_sha256="2" * 64,
        dedup_fingerprint="2" * 64,
        simhash64_hex=f"{(0x123456789ABCDEF0 ^ 0b111):016x}",
    )
    with GlobalDeduplicator(tmp_path) as dedup:
        assert dedup.consider(
            first,
            token_count=1,
            primary_category="general_english",
        ).status == "accept"
        decision = dedup.consider(
            second,
            token_count=1,
            primary_category="general_english",
        )
    assert decision.status == "exclude"
    assert decision.reason == "cross_source_near_duplicate"
    assert decision.duplicate_of == first.canonical_id
    assert decision.hamming_distance == 3


def test_global_mixture_evaluation_can_be_ready_without_authorizing_pretraining():
    report = {
        "train_capacity": {
            "release_safe": {
                "general_thai": 100,
                "general_english": 100,
                "technical_scientific_code": 0,
                "flywire_domain": 0,
            },
            "research_only": {
                "general_thai": 200_000_000,
                "general_english": 200_000_000,
                "technical_scientific_code": 100_000_000,
                "flywire_domain": 0,
            },
        }
    }
    contract = MixtureContract(
        primary_target_tokens=500_000_000,
        thai_minimum_tokens=200_000_000,
        english_minimum_tokens=200_000_000,
        technical_base_tokens=50_000_000,
        flywire_target_tokens=50_000_000,
        flywire_maximum_tokens=50_000_000,
    )
    result = global_manifest.evaluate_global_manifest_mixture(report, contract)
    assert result["release_safe"]["ready"] is False
    assert result["research_only"]["ready"] is True
    assert result["pretraining_authorized"] is False


def test_materialize_global_manifest_fails_before_processing_on_hash_mismatch(
    tmp_path,
):
    external = tmp_path / "external"
    path = external / "fixture" / "bad.jsonl"
    rows = [
        _payload(
            record_id="r1",
            text="one sentence",
            source_id="source-safe",
            artifact_id="artifact-safe",
            lane="release_safe",
            language="en",
        )
    ]
    rows[0].pop("rights_lane")
    byte_count, _ = _write_manifest_fixture(path, rows)
    item = _input_for_file(
        input_id="bad",
        source_id="source-safe",
        artifact_id="artifact-safe",
        lane="release_safe",
        language="en",
        storage="external://FlyWireLLM-data/fixture/bad.jsonl",
        byte_count=byte_count,
        sha256="0" * 64,
    )
    plan = GlobalManifestInputPlan(
        tokenizer="base50m-unigram-32000-v1",
        technical_classifier="technical-heuristic-v5",
        near_duplicate_hamming_distance=3,
        inputs=(item,),
        pretraining_authorized=False,
    )

    with pytest.raises(Exception, match="accepted SHA-256 mismatch"):
        global_manifest.materialize_global_manifest(
            plan,
            external_root=external,
            output_dir=tmp_path / "global",
            token_counter=lambda text: len(text.split()),
        )
