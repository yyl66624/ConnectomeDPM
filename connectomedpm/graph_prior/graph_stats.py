"""Descriptive graph statistics, computed identically for every topology."""

from __future__ import annotations

import numpy as np


def graph_stats(
    A: np.ndarray,
    self_loops: np.ndarray | None = None,
    node_sizes: np.ndarray | None = None,
    modules: list[str] | None = None,
) -> dict:
    A = np.asarray(A, dtype=np.float64)
    k = A.shape[0]
    support = A > 0
    n_edges = int(support.sum())
    possible = k * (k - 1)

    in_deg = support.sum(axis=0)
    out_deg = support.sum(axis=1)
    in_str = A.sum(axis=0)
    out_str = A.sum(axis=1)

    stats = {
        "n_nodes": int(k),
        "n_edges": n_edges,
        "density": float(n_edges / possible) if possible else 0.0,
        "self_loop_weight_total": float(np.sum(self_loops)) if self_loops is not None else None,
        "in_degree": _describe(in_deg),
        "out_degree": _describe(out_deg),
        "in_strength": _describe(in_str),
        "out_strength": _describe(out_str),
        "weight_total": float(A.sum()),
        "reciprocity": float(
            (support & support.T).sum() / max(int(support.sum()), 1)
        ),
        "gini_out_strength": _gini(out_str),
        "gini_in_strength": _gini(in_str),
    }
    if node_sizes is not None:
        stats["node_size"] = _describe(np.asarray(node_sizes, dtype=np.float64))
    if modules:
        stats["modules"] = _module_stats(A, modules)
    return stats


def _describe(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(x.mean()),
        "std": float(x.std()),
        "min": float(x.min()),
        "max": float(x.max()),
        "median": float(np.median(x)),
    }


def _gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = x.size
    if n == 0 or x.sum() <= 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2 * (index * x).sum()) / (n * x.sum()) - (n + 1) / n)


def _module_stats(A: np.ndarray, modules: list[str]) -> dict:
    labels = sorted(set(modules))
    idx = {label: i for i, label in enumerate(labels)}
    assign = np.array([idx[m] for m in modules])
    m = len(labels)
    block_counts = np.zeros((m, m), dtype=np.int64)
    support = A > 0
    for a in range(m):
        for b in range(m):
            block_counts[a, b] = int(support[np.ix_(assign == a, assign == b)].sum())
    sizes = np.array([int((assign == a).sum()) for a in range(m)])
    within_density = [
        float(block_counts[a, a] / max(sizes[a] * (sizes[a] - 1), 1)) for a in range(m)
    ]
    return {
        "n_modules": m,
        "module_sizes": sizes.tolist(),
        "block_counts": block_counts.tolist(),
        "within_module_density": within_density,
    }
