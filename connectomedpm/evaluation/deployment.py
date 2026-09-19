"""Deployment evaluation at the frozen threshold (DEVELOPMENT.md section 13.2)."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..data.schemas import CandidateRecord
from ..router.inference import BASE_ACTION
from .metrics import compute_metrics, resolve_decisions


def evaluate_deployment(
    records: Sequence[CandidateRecord],
    scores: np.ndarray,
    block_ids: Sequence[str],
    *,
    threshold: float,
) -> dict:
    scores = np.asarray(scores, dtype=np.float64)
    best_idx = scores.argmax(axis=1) if scores.size else np.zeros(0, dtype=int)
    best_score = scores[np.arange(scores.shape[0]), best_idx] if scores.size else np.zeros(0)
    mask = best_score >= threshold
    chosen = [block_ids[i] if keep else BASE_ACTION for i, keep in zip(best_idx, mask)]
    metrics = compute_metrics(resolve_decisions(records, chosen, intervened=mask))
    metrics["threshold"] = float(threshold)
    metrics["actual_coverage"] = metrics["coverage"]
    return metrics

