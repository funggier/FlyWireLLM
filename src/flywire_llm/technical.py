from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class TechnicalClassification:
    category: str
    score: int
    signals: tuple[str, ...]


_CODE_PATTERNS = (
    (
        "code_keywords",
        re.compile(
            r"(?im)^\s*(?:def|class|import|from|function|const|let|var|SELECT|"
            r"CREATE TABLE|public static|#include|package)\b"
        ),
    ),
    (
        "stack_trace",
        re.compile(
            r"(?i)(?:Traceback \(most recent call last\)|Exception in thread|"
            r"at [\w.$]+\([\w.]+:\d+\))"
        ),
    ),
    (
        "shell_prompt",
        re.compile(r"(?m)^\s*(?:\$|>>>|PS [^>]+>)\s+\S"),
    ),
)

_SCIENCE_PATTERNS = (
    (
        "doi",
        re.compile(r"(?i)\bdoi\s*:?\s*10\.\d{4,9}/\S+"),
    ),
    (
        "arxiv",
        re.compile(r"(?i)\barxiv\s*:?\s*\d{4}\.\d{4,5}\b"),
    ),
    (
        "scientific_notation",
        re.compile(r"(?<!\w)[+-]?\d+(?:\.\d+)?[eE][+-]?\d+"),
    ),
    (
        "equation",
        re.compile(r"(?:\$[^$]{3,}\$|\\frac\{|\\sum\b|\\int\b)"),
    ),
    (
        "units",
        re.compile(
            r"(?i)(?<!\w)\d+(?:\.\d+)?\s*(?:nm|um|mm|cm|km|ms|us|ns|"
            r"hz|khz|mhz|ghz|mv|ma|kg|mg|ug|mol|mmol|kpa|mpa)\b"
        ),
    ),
)

_THAI_TECH_TERMS = (
    "??????????",
    "?????????",
    "?????????????",
    "?????????????????????",
    "??????????????????",
    "????????????",
    "??????????????",
    "???????????????",
    "???????????",
    "??????",
    "????????",
    "????????",
    "?????",
    "????????",
    "?????????",
    "?????????????",
    "?????",
    "???????",
    "????",
    "????????",
    "??????????",
    "??????",
    "?????",
    "????????",
    "??????",
    "???????????",
    "????????",
    "??????????",
    "??????",
    "????????????",
    "????????",
    "?????????",
)

_TECH_TERMS = frozenset(
    """
    algorithm algorithms api architecture array arrays compiler database databases
    distributed encryption endpoint endpoints framework frameworks function functions
    kernel latency memory network neural neuron neurons optimization parameter parameters
    protocol protocols query queries runtime schema schemas server servers software
    tensor tensors theorem theorems training transformer transformers variable variables
    vector vectors inference gradient gradients matrix matrices probability statistical
    statistics regression dataset datasets benchmark benchmarks experiment experiments
    molecular molecule molecules protein proteins receptor receptors genome genomic
    chromosome chromosomes physics chemistry biology neuroscience circuit circuits
    synapse synapses electron electrons quantum equation equations calculus derivative
    derivatives integral integrals differential simulation simulations model models
    microscopy sequencing transcriptome transcriptomics
    """.split()
)


def _word_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z][A-Za-z0-9_+-]*", text.casefold())


def classify_technical_text(
    text: str,
    *,
    technical_threshold: int = 4,
) -> TechnicalClassification:
    """Deterministic, explainable mixture-routing heuristic."""
    if not isinstance(text, str):
        raise TypeError("text must be str")
    if technical_threshold < 1:
        raise ValueError("technical_threshold must be positive")

    signals: list[str] = []
    score = 0

    for name, pattern in _CODE_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            signals.append(name)
            score += min(4, len(matches) * 2)

    for name, pattern in _SCIENCE_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            signals.append(name)
            score += min(3, len(matches))

    words = _word_tokens(text)
    if words:
        hits = sum(word in _TECH_TERMS for word in words)
        density = hits / len(words)
        if hits >= 4:
            signals.append("technical_terms")
            score += min(5, hits // 2)
        if hits >= 3 and density >= 0.03:
            signals.append("technical_term_density")
            score += 2

    thai_hits = sum(text.count(term) for term in _THAI_TECH_TERMS)
    if thai_hits >= 3:
        signals.append("thai_technical_terms")
        score += min(6, thai_hits // 2 + 1)
    elif thai_hits == 2:
        signals.append("thai_technical_terms")
        score += 2
    elif thai_hits == 1:
        signals.append("thai_technical_term")
        score += 1

    symbol_count = sum(
        text.count(symbol)
        for symbol in ("{", "}", ";", "=>", "::")
    )
    if symbol_count >= 8:
        signals.append("code_symbol_density")
        score += min(4, symbol_count // 8)

    category = (
        "technical_scientific_code"
        if score >= technical_threshold
        else "general"
    )
    return TechnicalClassification(
        category=category,
        score=score,
        signals=tuple(sorted(set(signals))),
    )
