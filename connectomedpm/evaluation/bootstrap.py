"""Clustered bootstrap (DEVELOPMENT.md section 19).

Prompts produced from the same template are *not* independent. Resampling is therefore done
over template families, not over samples.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from ..utils.seed import numpy_rng


def clustered_bootstrap(
    clusters: Sequence[str],
    statistic: Callable[[np.ndarray], float],
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """Percentile CI for `statistic` under template-family resampling.

    Args:
        clusters: cluster label per sample.
        statistic: function mapping an index array to a scalar.
    """
    clusters = np.asarray(clusters)
    unique = np.unique(clusters)
    index_by_cluster = {c: np.flatnonzero(clusters == c) for c in unique}
    rng = numpy_rng(seed)

    values = np.empty(int(n_boot), dtype=np.float64)
    for b in range(int(n_boot)):
        picked = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([index_by_cluster[c] for c in picked])
        values[b] = statistic(idx)

    lo = float(np.quantile(values, alpha / 2))
    hi = float(np.quantile(values, 1 - alpha / 2))
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "ci_low": lo,
        "ci_high": hi,
        "alpha": float(alpha),
        "n_boot": int(n_boot),
        "n_clusters": int(len(unique)),
    }


def paired_clustered_bootstrap(
    clusters: Sequence[str],
    statistic_a: Callable[[np.ndarray], float],
    statistic_b: Callable[[np.ndarray], float],
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """CI for the *paired* difference A - B, resampling clusters once per replicate."""
    clusters = np.asarray(clusters)
    unique = np.unique(clusters)
    index_by_cluster = {c: np.flatnonzero(clusters == c) for c in unique}
    rng = numpy_rng(seed)

    diffs = np.empty(int(n_boot), dtype=np.float64)
    for b in range(int(n_boot)):
        picked = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([index_by_cluster[c] for c in picked])
        diffs[b] = statistic_a(idx) - statistic_b(idx)

    lo = float(np.quantile(diffs, alpha / 2))
    hi = float(np.quantile(diffs, 1 - alpha / 2))
    return {
        "mean": float(diffs.mean()),
        "std": float(diffs.std()),
        "ci_low": lo,
        "ci_high": hi,
        "alpha": float(alpha),
        "n_boot": int(n_boot),
        "n_clusters": int(len(unique)),
        "p_positive": float((diffs > 0).mean()),
    }
