"""Strict scoring.

Every score must be reproducible from the raw model output, so the scorer is deliberately
conservative: it extracts the model's final answer, then compares it either numerically or as
a normalised short string. `SCORER_VERSION` takes part in the candidate-cache key, so
changing scoring invalidates the cache instead of silently changing results.
"""

from __future__ import annotations

import re

SCORER_VERSION = "strict-v1"

_PREFIXES = re.compile(
    r"^\s*(?:final\s+answer|answer|result|the\s+answer\s+is|值|答案)\s*[:：]?\s*",
    re.IGNORECASE,
)
_NUMBER = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
_WS = re.compile(r"\s+")

NUMERIC_FAMILIES = [
    "numerical_arithmetic",
    "unit_conversion",
    "logic_constraint",
    "code_boundary",
    "symbolic_manipulation",
]


def first_line(raw: str) -> str:
    for line in (raw or "").splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def strip_prefix(text: str) -> str:
    text = text.strip().strip("`*_ ")
    return _PREFIXES.sub("", text).strip()


def parse_number(text: str) -> float | None:
    """Parse a single number out of a short answer, tolerating thousands separators."""
    text = strip_prefix(text).strip()
    match = _NUMBER.search(text)
    if not match:
        return None
    token = match.group(0).replace(",", "")
    try:
        return float(token)
    except ValueError:
        return None


def normalize_text(text: str) -> str:
    text = strip_prefix(text).lower()
    text = text.strip("\"'`")
    text = re.sub(r"[.,;:!?]+$", "", text)
    return _WS.sub(" ", text).strip()


def looks_numeric(gold: str) -> bool:
    gold = gold.strip()
    if not gold:
        return False
    return bool(re.fullmatch(r"[-+]?\d[\d,]*(?:\.\d+)?", gold))


def score_answer(prediction: str, gold: str, family: str = "", tol: float = 1e-6) -> tuple[bool, float]:
    """Return (correct, score in {0.0, 1.0}).

    Numeric comparison is used whenever the gold answer is numeric; otherwise a normalised
    exact string match is used.
    """
    pred_line = strip_prefix(first_line(prediction))
    if not pred_line:
        return False, 0.0

    if looks_numeric(gold) or family in NUMERIC_FAMILIES:
        gold_num = parse_number(gold)
        pred_num = parse_number(pred_line)
        if gold_num is not None and pred_num is not None:
            ok = abs(gold_num - pred_num) <= max(tol, tol * abs(gold_num))
            return ok, 1.0 if ok else 0.0

    pred_norm = normalize_text(pred_line)
    gold_norm = normalize_text(gold)
    if not pred_norm or not gold_norm:
        return False, 0.0
    if pred_norm == gold_norm:
        return True, 1.0

    # Single-token answers (symbols, YES/NO, words) may be wrapped in extra words; longer
    # gold strings must match exactly after normalisation.
    if len(gold_norm.split()) > 1:
        return False, 0.0
    tokens = re.findall(r"[a-z0-9\u4e00-\u9fff]+", pred_norm)
    if gold_norm in tokens:
        return True, 1.0
    return False, 0.0

