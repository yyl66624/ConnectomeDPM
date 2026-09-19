"""Deduplication and near-duplicate auditing.

DEVELOPMENT.md section 4: "同一模板生成的改写不能跨 train / test". Splitting by
`template_family` is the primary defence; the helpers here are the *audit* that proves it
held, plus an exact/near-duplicate check on the raw prompt text.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable

from .schemas import Sample

_WS = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^a-z0-9\u4e00-\u9fff ]+")


def normalize_text(text: str) -> str:
    """Aggressive normalisation used for duplicate detection only (never for scoring)."""
    text = text.lower()
    text = _NON_ALNUM.sub(" ", text)
    return _WS.sub(" ", text).strip()


def shingles(text: str, n: int = 3) -> frozenset[str]:
    tokens = normalize_text(text).split()
    if len(tokens) < n:
        return frozenset([" ".join(tokens)]) if tokens else frozenset()
    return frozenset(" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def exact_duplicate_groups(samples: Iterable[Sample]) -> dict[str, list[str]]:
    """sample_ids sharing an identical normalised prompt, for groups larger than one."""
    buckets: dict[str, list[str]] = defaultdict(list)
    for s in samples:
        buckets[normalize_text(s.prompt)].append(s.sample_id)
    return {k: v for k, v in buckets.items() if len(v) > 1}


def near_duplicate_pairs(
    samples: Iterable[Sample],
    threshold: float = 0.85,
    max_block: int = 512,
) -> list[tuple[str, str, float]]:
    """Jaccard near-duplicate pairs, computed with an inverted index (no O(n^2) scan)."""
    samples = list(samples)
    shingle_sets = [shingles(s.prompt) for s in samples]
    index: dict[str, list[int]] = defaultdict(list)
    for i, sh in enumerate(shingle_sets):
        for token in sh:
            index[token].append(i)

    pairs: set[tuple[int, int]] = set()
    for token, ids in index.items():
        if len(ids) > max_block:
            continue
        for a_pos in range(len(ids)):
            for b_pos in range(a_pos + 1, len(ids)):
                i, j = ids[a_pos], ids[b_pos]
                pairs.add((i, j))

    out: list[tuple[str, str, float]] = []
    for i, j in sorted(pairs):
        score = jaccard(shingle_sets[i], shingle_sets[j])
        if score >= threshold:
            out.append((samples[i].sample_id, samples[j].sample_id, round(score, 4)))
    return out


def template_split_leakage(splits: dict[str, list[Sample]]) -> dict[str, list[str]]:
    """Return template families that appear in more than one split (must be empty)."""
    owner: dict[str, set[str]] = defaultdict(set)
    for split_name, rows in splits.items():
        for s in rows:
            owner[s.template_family].add(split_name)
    return {tpl: sorted(names) for tpl, names in owner.items() if len(names) > 1}


def prompt_split_leakage(splits: dict[str, list[Sample]], threshold: float = 0.85) -> list[dict]:
    """Cross-split exact and near-duplicate prompt overlaps."""
    names = sorted(splits)
    findings: list[dict] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            exact_a = {normalize_text(s.prompt) for s in splits[a]}
            exact_b = {normalize_text(s.prompt) for s in splits[b]}
            shared = exact_a & exact_b
            if shared:
                findings.append({"kind": "exact", "splits": [a, b], "count": len(shared)})
            pairs = near_duplicate_pairs(list(splits[a]) + list(splits[b]), threshold)
            a_ids = {s.sample_id for s in splits[a]}
            cross = [p for p in pairs if (p[0] in a_ids) != (p[1] in a_ids)]
            if cross:
                findings.append({"kind": "near", "splits": [a, b], "count": len(cross),
                                 "examples": cross[:5]})
    return findings

