"""Per-graph node features.

DEVELOPMENT.md section 12: every graph recomputes its *own* node features. MaleCNS features
are never copied onto a random graph.
"""

from __future__ import annotations

import numpy as np

FEATURE_NAMES: tuple[str, ...] = (
    "log_self_loop",
    "log_in_strength",
    "log_out_strength",
    "log_in_degree",
    "log_out_degree",
    "log_node_size",
    "log_reciprocity",
)


def node_features(
    A: np.ndarray,
    self_loops: np.ndarray,
    node_sizes: np.ndarray | None = None,
    standardize: bool = False,
) -> np.ndarray:
    """Build the (K, 7) node feature matrix.

    `standardize=True` z-scores each column **within this graph**, which keeps real and null
    graphs on the same footing (section 12) while making the features scale-free.
    """
    A = np.asarray(A, dtype=np.float64)
    k = A.shape[0]
    log = lambda x: np.log1p(np.maximum(np.asarray(x, dtype=np.float64), 0.0))  # noqa: E731

    in_strength = A.sum(axis=0)
    out_strength = A.sum(axis=1)
    in_degree = (A > 0).sum(axis=0)
    out_degree = (A > 0).sum(axis=1)
    if node_sizes is None:
        node_sizes = np.ones(k, dtype=np.float64)
    reciprocal = np.minimum(A, A.T).sum(axis=1)

    feats = np.stack(
        [
            log(self_loops),
            log(in_strength),
            log(out_strength),
            log(in_degree),
            log(out_degree),
            log(node_sizes),
            log(reciprocal),
        ],
        axis=1,
    )
    if standardize:
        mu = feats.mean(axis=0, keepdims=True)
        sd = feats.std(axis=0, keepdims=True)
        feats = (feats - mu) / np.where(sd > 1e-8, sd, 1.0)
    return feats

