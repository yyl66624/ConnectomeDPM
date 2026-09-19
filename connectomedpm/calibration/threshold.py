"""Learn the deployment threshold on `D_cal` only.

The threshold is frozen before the test set is touched. Two protocols are supported:

* `max_net_repair`  — pick the threshold with the best calibration net repair
                      (ties broken towards lower coverage, i.e. the more conservative rule).
* `max_coverage_at_risk` — the largest coverage whose selective break risk stays under
                      `max_break_risk`.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..data.schemas import CandidateRecord
from ..evaluation.metrics import compute_metrics, resolve_decisions
from ..router.inference import BASE_ACTION


def _evaluate_threshold(
    records: Sequence[CandidateRecord],
    scores: np.ndarray,
    block_ids: Sequence[str],
    threshold: float,
) -> dict:
    best_idx = scores.argmax(axis=1) if scores.size else np.zeros(0, dtype=int)
    best_score = scores[np.arange(scores.shape[0]), best_idx] if scores.size else np.zeros(0)
    mask = best_score >= threshold
    chosen = [block_ids[i] if keep else BASE_ACTION for i, keep in zip(best_idx, mask)]
    metrics = compute_metrics(resolve_decisions(records, chosen, intervened=mask))
    metrics["threshold"] = float(threshold)
    return metrics


def learn_threshold(
    records: Sequence[CandidateRecord],
    scores: np.ndarray,
    block_ids: Sequence[str],
    *,
    protocol: str = "max_net_repair",
    candidates: int = 101,
    max_break_risk: float = 0.02,
) -> dict:
    """Return the chosen threshold plus the full calibration curve."""
    scores = np.asarray(scores, dtype=np.float64)
    best_idx = scores.argmax(axis=1) if scores.size else np.zeros(0, dtype=int)
    best_score = scores[np.arange(scores.shape[0]), best_idx] if scores.size else np.zeros(0)
    if best_score.size == 0:
        return {"threshold": 0.0, "curve": [], "protocol": protocol}

    lo, hi = float(best_score.min()), float(best_score.max())
    grid = np.linspace(lo - 1e-6, hi + 1e-6, int(candidates))
    curve = [_evaluate_threshold(records, scores, block_ids, float(t)) for t in grid]

    if protocol == "max_coverage_at_risk":
        feasible = [row for row in curve
                    if row.get("selective_break_risk", 0.0) <= max_break_risk
                    and row.get("breaks", 0) <= row.get("fixes", 0)]
        pool = feasible or curve
        best = max(pool, key=lambda row: (row["coverage"], row["net_repair"]))
    else:
        best = max(curve, key=lambda row: (row["net_repair"], -row["coverage"]))

    return {
        "threshold": float(best["threshold"]),
        "protocol": protocol,
        "selection_metrics": best,
        "curve": curve,
    }

