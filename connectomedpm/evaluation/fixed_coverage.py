"""Fixed-coverage evaluation (DEVELOPMENT.md section 13.1).

Every method ranks requests by *its own* router score only, and the same fraction is then
executed against the shared candidate cache. Test answers never participate in ranking.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..data.schemas import CandidateRecord
from ..router.inference import BASE_ACTION
from .metrics import compute_metrics, resolve_decisions


def top_k_mask(scores: np.ndarray, coverage: float) -> np.ndarray:
    """Boolean mask of the highest-scoring `coverage` fraction (ties broken stably)."""
    scores = np.asarray(scores, dtype=np.float64)
    n = scores.shape[0]
    if n == 0 or coverage <= 0:
        return np.zeros(n, dtype=bool)
    k = int(round(coverage * n))
    k = max(min(k, n), 0)
    if k == 0:
        return np.zeros(n, dtype=bool)
    order = np.argsort(-scores, kind="stable")
    mask = np.zeros(n, dtype=bool)
    mask[order[:k]] = True
    return mask


def evaluate_fixed_coverage(
    records: Sequence[CandidateRecord],
    scores: np.ndarray,
    block_ids: Sequence[str],
    *,
    coverage: float = 0.10,
) -> dict:
    """Evaluate one method at one coverage level."""
    scores = np.asarray(scores, dtype=np.float64)
    best_idx = scores.argmax(axis=1) if scores.size else np.zeros(0, dtype=int)
    best_score = scores[np.arange(scores.shape[0]), best_idx] if scores.size else np.zeros(0)
    mask = top_k_mask(best_score, coverage)

    chosen = [block_ids[i] if keep else BASE_ACTION
              for i, keep in zip(best_idx, mask)]
    decisions = resolve_decisions(records, chosen, intervened=mask)
    metrics = compute_metrics(decisions)
    metrics["requested_coverage"] = float(coverage)
    return metrics


def coverage_sweep(
    records: Sequence[CandidateRecord],
    scores: np.ndarray,
    block_ids: Sequence[str],
    coverages: Sequence[float] = (0.05, 0.10, 0.20),
) -> dict[str, dict]:
    return {
        f"{c:.2f}": evaluate_fixed_coverage(records, scores, block_ids, coverage=c)
        for c in coverages
    }
