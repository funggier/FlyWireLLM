from __future__ import annotations

import json
from pathlib import Path

import scripts.calibrate_technical_mix as calibrate_technical_mix
from flywire_llm.technical import classify_technical_text


ROOT = Path(__file__).resolve().parents[1]


def test_plain_general_prose_stays_general():
    result = classify_technical_text(
        "The family walked through the park after lunch and enjoyed the weather."
    )
    assert result.category == "general"


def test_python_code_is_technical():
    result = classify_technical_text(
        "import torch\ndef train(model, dataset):\n"
        "    loss = model(dataset)\n    loss.backward()"
    )
    assert result.category == "technical_scientific_code"
    assert "code_keywords" in result.signals


def test_scientific_method_text_is_technical():
    result = classify_technical_text(
        "The neural model used a 3.2e-4 learning rate. "
        "The dataset contained molecular receptor measurements and "
        "statistical regression experiments with model parameters."
    )
    assert result.category == "technical_scientific_code"


def test_doi_and_units_contribute_explainable_signals():
    result = classify_technical_text(
        "doi:10.1038/s41467-019-09069-1. The sample measured 25 nm "
        "and the experiment used molecular protein receptor data."
    )
    assert result.category == "technical_scientific_code"
    assert "doi" in result.signals
    assert "units" in result.signals


def test_single_technical_word_does_not_reclassify_general_text():
    result = classify_technical_text(
        "The word model appeared once in an otherwise ordinary story "
        "about travel, food, music, friends, and a long summer holiday."
    )
    assert result.category == "general"


def test_classifier_is_deterministic():
    text = (
        "SELECT value FROM dataset; the database query uses an algorithm "
        "and optimization parameters."
    )
    assert classify_technical_text(text) == classify_technical_text(text)


def test_thai_scientific_text_is_technical():
    result = classify_technical_text(
        "แบบจำลองโครงข่ายประสาทใช้ชุดข้อมูลสำหรับการฝึก "
        "และปรับพารามิเตอร์ด้วยเกรเดียนต์จากการทดลอง"
    )
    assert result.category == "technical_scientific_code"
    assert "thai_strong_technical_terms" in result.signals


def test_thai_general_text_stays_general():
    result = classify_technical_text(
        "เมื่อเช้านี้ครอบครัวเดินเล่นในสวนแล้วแวะซื้ออาหารก่อนกลับบ้าน"
    )
    assert result.category == "general"


def test_repeated_generic_thai_terms_do_not_inflate_score():
    result = classify_technical_text(
        ("สถิติ ความน่าจะเป็น การทดลอง ฐานข้อมูล " * 20)
        + "เป็นบทความทั่วไปเกี่ยวกับการเดิมพัน"
    )
    assert result.category == "general"


def test_three_distinct_support_terms_are_technical():
    result = classify_technical_text(
        "บทความอธิบายโมเลกุล โปรตีน และตัวรับในเซลล์"
    )
    assert result.category == "technical_scientific_code"
    assert "thai_support_technical_terms" in result.signals


def test_thai_technical_lexicon_is_real_utf8_not_question_mark_placeholder():
    source = (ROOT / "src" / "flywire_llm" / "technical.py").read_text(
        encoding="utf-8"
    )
    assert '"อัลกอริทึม"' in source
    assert '"ประสาทวิทยา"' in source
    assert '"????????"' not in source


def test_single_strong_thai_term_does_not_route_gambling_page():
    result = classify_technical_text(
        "สล็อตเกมออนไลน์มีเมทริกซ์หกวงล้อและวิธีชนะหลายรูปแบบ"
    )
    assert result.category == "general"
    assert result.signals == ("thai_strong_technical_terms",)


def test_calibration_report_location_metadata_is_external_only():
    storage = (
        "external://FlyWireLLM-data/L003/FineWeb2/tha_Thai/"
        "derived/005_00002-full-v1/accepted.jsonl"
    )
    assert calibrate_technical_mix.report_location_metadata(storage) == {
        "corpus_storage": storage,
        "local_path_recorded": False,
    }


def test_v3_v4_negative_history_reports_are_path_sanitized():
    names = (
        "technical-fineweb2-thai-v3-1pct.json",
        "technical-fineweb2-thai-v4-1pct.json",
        "technical-fineweb-en-014-v3-1pct.json",
        "technical-fineweb-en-014-v4-1pct.json",
    )
    for name in names:
        report = json.loads(
            (ROOT / "results" / "l003" / name).read_text(encoding="utf-8")
        )
        assert "corpus_path" not in report
        assert report["corpus_storage"].startswith(
            "external://FlyWireLLM-data/"
        )
        assert report["local_path_recorded"] is False
        assert report["training_authorized"] is False
        assert report["calibration_only"] is True


def test_v5_calibration_reports_are_frozen_and_path_sanitized():
    thai = json.loads(
        (ROOT / "results" / "l003" / "technical-fineweb2-thai-v5-1pct.json")
        .read_text(encoding="utf-8")
    )
    english = json.loads(
        (ROOT / "results" / "l003" / "technical-fineweb-en-014-v5-1pct.json")
        .read_text(encoding="utf-8")
    )
    assert thai["classifier"] == "technical-heuristic-v5"
    assert thai["records_sampled"] == 3383
    assert thai["tokens_by_category"]["technical_scientific_code"] == 53094
    assert thai["technical_token_fraction"] == 0.01792033762896254
    assert english["classifier"] == "technical-heuristic-v5"
    assert english["records_sampled"] == 1666
    assert english["tokens_by_category"]["technical_scientific_code"] == 248949
    assert english["technical_token_fraction"] == 0.19793674267722547
    for report in (thai, english):
        assert "corpus_path" not in report
        assert report["corpus_storage"].startswith("external://FlyWireLLM-data/")
        assert report["local_path_recorded"] is False
        assert report["training_authorized"] is False
        assert report["calibration_only"] is True
