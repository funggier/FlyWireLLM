from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class ScreeningDisposition(str, Enum):
    ACCEPT = "accept"
    REVIEW = "review"
    REJECT = "reject"


@dataclass(frozen=True)
class ScreeningFinding:
    code: str
    severity: str
    count: int


@dataclass(frozen=True)
class TextScreeningResult:
    disposition: ScreeningDisposition
    characters: int
    utf8_bytes: int
    line_count: int
    control_ratio: float
    replacement_ratio: float
    repeated_run_ratio: float
    findings: tuple[ScreeningFinding, ...]


_EMAIL_RE = re.compile(
    r"(?<![\w.+-])[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}(?![\w.-])",
    re.IGNORECASE,
)
_IPV4_RE = re.compile(
    r"(?<!\d)(?:"
    r"(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}"
    r"(?:25[0-5]|2[0-4]\d|1?\d?\d)(?!\d)"
)
_PHONE_RE = re.compile(
    r"(?<!\\w)(?:"
    r"\\+\\d[\\d(). -]{7,}\\d"
    r"|\\d{2,4}[- ]\\d[\\d -]{5,}\\d"
    r")(?!\\w)"
)
_CREDIT_CARD_CANDIDATE_RE = re.compile(
    r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)"
)
_SECRET_RE = re.compile(
    r"(?i)(?:"
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
    r"|(?:api[_-]?key|access[_-]?token|secret[_-]?key)"
    r"\s*[:=]\s*['\"]?[A-Za-z0-9_\-/.+=]{16,}"
    r")"
)


def _luhn_valid(digits: str) -> bool:
    if not digits.isdigit() or not 13 <= len(digits) <= 19:
        return False
    total = 0
    parity = len(digits) % 2
    for index, char in enumerate(digits):
        value = int(char)
        if index % 2 == parity:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def _count_credit_card_candidates(text: str) -> int:
    count = 0
    for match in _CREDIT_CARD_CANDIDATE_RE.finditer(text):
        digits = re.sub(r"\D", "", match.group(0))
        if _luhn_valid(digits):
            count += 1
    return count


def _control_ratio(text: str) -> float:
    if not text:
        return 0.0
    controls = 0
    for char in text:
        if char in "\n\r\t":
            continue
        if unicodedata.category(char) in {"Cc", "Cf"}:
            controls += 1
    return controls / len(text)


def _replacement_ratio(text: str) -> float:
    if not text:
        return 0.0
    return text.count("\ufffd") / len(text)


def _repeated_run_ratio(text: str) -> float:
    """Fraction of characters participating in suspicious 12+ char runs."""
    if not text:
        return 0.0
    repeated = 0
    start = 0
    while start < len(text):
        end = start + 1
        while end < len(text) and text[end] == text[start]:
            end += 1
        run = end - start
        if run >= 12 and not text[start].isspace():
            repeated += run
        start = end
    return repeated / len(text)


def screen_text(
    text: str,
    *,
    min_characters: int = 1,
    max_characters: int = 2_000_000,
    reject_control_ratio: float = 0.01,
    reject_replacement_ratio: float = 0.005,
    review_repeated_run_ratio: float = 0.10,
) -> TextScreeningResult:
    if not isinstance(text, str):
        raise TypeError("text must be str")
    if min_characters < 1 or max_characters < min_characters:
        raise ValueError("invalid character bounds")

    findings: list[ScreeningFinding] = []
    characters = len(text)
    control_ratio = _control_ratio(text)
    replacement_ratio = _replacement_ratio(text)
    repeated_run_ratio = _repeated_run_ratio(text)

    def add(code: str, severity: str, count: int = 1) -> None:
        if count:
            findings.append(
                ScreeningFinding(
                    code=code,
                    severity=severity,
                    count=count,
                )
            )

    if characters < min_characters:
        add("too_short", "reject")
    if characters > max_characters:
        add("too_long", "reject")
    if not text.strip():
        add("blank_or_whitespace", "reject")
    if "\x00" in text:
        add("nul_byte", "reject", text.count("\x00"))
    if control_ratio > reject_control_ratio:
        add("excess_control_characters", "reject")
    if replacement_ratio > reject_replacement_ratio:
        add("excess_unicode_replacement", "reject")
    if repeated_run_ratio > review_repeated_run_ratio:
        add("suspicious_repeated_character_runs", "review")

    add("possible_email", "review", len(_EMAIL_RE.findall(text)))
    add("possible_ipv4", "review", len(_IPV4_RE.findall(text)))
    add("possible_phone", "review", len(_PHONE_RE.findall(text)))
    add(
        "possible_payment_card",
        "reject",
        _count_credit_card_candidates(text),
    )
    add("possible_secret", "reject", len(_SECRET_RE.findall(text)))

    if any(item.severity == "reject" for item in findings):
        disposition = ScreeningDisposition.REJECT
    elif any(item.severity == "review" for item in findings):
        disposition = ScreeningDisposition.REVIEW
    else:
        disposition = ScreeningDisposition.ACCEPT

    return TextScreeningResult(
        disposition=disposition,
        characters=characters,
        utf8_bytes=len(text.encode("utf-8")),
        line_count=text.count("\n") + 1,
        control_ratio=control_ratio,
        replacement_ratio=replacement_ratio,
        repeated_run_ratio=repeated_run_ratio,
        findings=tuple(findings),
    )


def normalized_word_shingles(
    text: str,
    *,
    width: int = 5,
) -> tuple[str, ...]:
    if width < 1:
        raise ValueError("width must be positive")
    normalized = " ".join(
        unicodedata.normalize("NFC", text).casefold().split()
    )
    words = normalized.split()
    if not words:
        return ()
    if len(words) <= width:
        return (" ".join(words),)
    return tuple(
        " ".join(words[index : index + width])
        for index in range(len(words) - width + 1)
    )


def simhash64(text: str, *, shingle_width: int = 5) -> int:
    import hashlib

    shingles = normalized_word_shingles(text, width=shingle_width)
    if not shingles:
        return 0
    weights = [0] * 64
    for shingle in shingles:
        digest = hashlib.blake2b(
            shingle.encode("utf-8"),
            digest_size=8,
            person=b"FWLLM003",
        ).digest()
        value = int.from_bytes(digest, "big")
        for bit in range(64):
            weights[bit] += 1 if value & (1 << bit) else -1
    fingerprint = 0
    for bit, weight in enumerate(weights):
        if weight >= 0:
            fingerprint |= 1 << bit
    return fingerprint


def hamming_distance64(left: int, right: int) -> int:
    if not 0 <= left < 2**64 or not 0 <= right < 2**64:
        raise ValueError("fingerprints must be unsigned 64-bit integers")
    return (left ^ right).bit_count()


def likely_near_duplicate(
    left: str,
    right: str,
    *,
    max_hamming_distance: int = 3,
    shingle_width: int = 5,
) -> bool:
    if not 0 <= max_hamming_distance <= 64:
        raise ValueError("invalid Hamming distance threshold")
    return (
        hamming_distance64(
            simhash64(left, shingle_width=shingle_width),
            simhash64(right, shingle_width=shingle_width),
        )
        <= max_hamming_distance
    )

def simhash_lsh_bands(
    fingerprint: int,
    *,
    bands: int = 4,
) -> tuple[tuple[int, int], ...]:
    """Return equal-width LSH bands for a 64-bit SimHash.

    With four 16-bit bands, fingerprints at Hamming distance <= 3 are
    guaranteed to share at least one exact band by the pigeonhole principle.
    """
    if not 0 <= fingerprint < 2**64:
        raise ValueError("fingerprint must be an unsigned 64-bit integer")
    if bands < 1 or 64 % bands != 0:
        raise ValueError("bands must be a positive divisor of 64")
    width = 64 // bands
    mask = (1 << width) - 1
    return tuple(
        (band, (fingerprint >> (band * width)) & mask)
        for band in range(bands)
    )


def simhash_candidate_keys(
    text: str,
    *,
    bands: int = 4,
    shingle_width: int = 5,
) -> tuple[str, ...]:
    fingerprint = simhash64(text, shingle_width=shingle_width)
    width = 64 // bands
    hex_width = (width + 3) // 4
    return tuple(
        f"{band}:{value:0{hex_width}x}"
        for band, value in simhash_lsh_bands(
            fingerprint,
            bands=bands,
        )
    )

