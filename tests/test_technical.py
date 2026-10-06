from __future__ import annotations

from flywire_llm.technical import classify_technical_text


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
        "?????????????????????????????????????????????????????? "
        "???????????????????????????????"
    )
    assert result.category == "technical_scientific_code"
    assert "thai_technical_terms" in result.signals
