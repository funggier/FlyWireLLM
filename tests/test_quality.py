from __future__ import annotations

from flywire_llm.quality import (
    ScreeningDisposition,
    hamming_distance64,
    likely_near_duplicate,
    screen_text,
    simhash64,
    simhash_candidate_keys,
    simhash_lsh_bands,
)


def _codes(result):
    return {finding.code for finding in result.findings}


def test_clean_thai_english_and_identifiers_are_accepted():
    result = screen_text(
        "FlyWireLLM เก็บ root_id=720575940630024566 และ learning_rate=6e-4"
    )
    assert result.disposition == ScreeningDisposition.ACCEPT
    assert result.findings == ()


def test_email_and_ip_are_review_not_automatic_reject():
    result = screen_text(
        "Contact alice@example.org from 192.168.1.10 for the test."
    )
    assert result.disposition == ScreeningDisposition.REVIEW
    assert {"possible_email", "possible_ipv4"} <= _codes(result)


def test_luhn_payment_card_candidate_is_rejected():
    result = screen_text("card 4111 1111 1111 1111")
    assert result.disposition == ScreeningDisposition.REJECT
    assert "possible_payment_card" in _codes(result)


def test_secret_like_material_is_rejected():
    result = screen_text(
        "api_key=abcdefghijklmnopqrstuvwxyz012345"
    )
    assert result.disposition == ScreeningDisposition.REJECT
    assert "possible_secret" in _codes(result)


def test_nul_and_bad_unicode_replacement_are_rejected():
    nul = screen_text("abc\x00def")
    assert nul.disposition == ScreeningDisposition.REJECT
    assert "nul_byte" in _codes(nul)

    replacement = screen_text("abc" + "\ufffd" * 3)
    assert replacement.disposition == ScreeningDisposition.REJECT
    assert "excess_unicode_replacement" in _codes(replacement)


def test_long_repeated_character_run_is_reviewed():
    result = screen_text("normal text " + "x" * 30)
    assert result.disposition == ScreeningDisposition.REVIEW
    assert "suspicious_repeated_character_runs" in _codes(result)


def test_simhash_is_deterministic_and_exact_match_is_near_duplicate():
    text = "the quick brown fox jumps over the lazy dog"
    left = simhash64(text)
    right = simhash64(text)
    assert left == right
    assert hamming_distance64(left, right) == 0
    assert likely_near_duplicate(text, text)


def test_simhash_validates_unsigned_64bit_values():
    try:
        hamming_distance64(-1, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("negative fingerprint must be rejected")

def test_four_band_lsh_captures_any_three_bit_simhash_difference():
    base = 0x123456789ABCDEF0
    changed = base ^ (1 << 0) ^ (1 << 16) ^ (1 << 32)
    assert hamming_distance64(base, changed) == 3
    base_bands = set(simhash_lsh_bands(base, bands=4))
    changed_bands = set(simhash_lsh_bands(changed, bands=4))
    assert base_bands & changed_bands


def test_simhash_candidate_keys_are_stable_and_band_scoped():
    text = "FlyWireLLM near duplicate candidate generation"
    keys = simhash_candidate_keys(text)
    assert keys == simhash_candidate_keys(text)
    assert len(keys) == 4
    assert [key.split(":", 1)[0] for key in keys] == [
        "0",
        "1",
        "2",
        "3",
    ]
