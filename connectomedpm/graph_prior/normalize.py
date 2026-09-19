"""Edge transform and fixed transition operators (DEVELOPMENT.md section 8 steps 5-6).

Conventions used everywhere downstream:

``A[i, j]``            weight of the directed edge i -> j.
``P_in[i, j]``         column-stochastic: ``(P_in s)[i]`` is the mass *arriving* at i.
``P_out[i, j]``        column-stochastic: ``(P_out s)[i]`` is the mass leaving i.

Both operators are column-stochastic so that they act on a distribution over nodes without
changing its total mass. The diagonal is zeroed; original self-loops are preserved separately
as a node feature. A node with no incoming (or no outgoing) mass keeps its mass in place.
"""

from __future__ import annotations

import numpy as np

EDGE_TRANSFORMS = ("log1p", "raw", "binary")


def edge_transform(A: np.ndarray, mode: str = "log1p") -> np.ndarray:
    A = np.asarray(A, dtype=np.float64)
    if mode == "log1p":
        return np.log1p(np.maximum(A, 0.0))
    if mode == "raw":
        return np.maximum(A, 0.0).copy()
    if mode == "binary":
        return (A > 0).astype(np.float64)
    raise ValueError(f"unknown edge transform {mode!r}; expected one of {EDGE_TRANSFORMS}")


def split_self_loops(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (A_without_diagonal, self_loop_vector)."""
    A = np.asarray(A, dtype=np.float64).copy()
    self_loops = np.diag(A).copy()
    np.fill_diagonal(A, 0.0)
    return A, self_loops


def transition_operators(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build (P_in, P_out) from a zero-diagonal adjacency matrix."""
    A = np.asarray(A, dtype=np.float64)
    k = A.shape[0]
    if A.shape != (k, k):
        raise ValueError("adjacency must be square")

    # P_in: normalise each *column* (incoming mass into j comes from column j).
    col = A.sum(axis=0, keepdims=True)
    P_in = np.divide(A, col, out=np.zeros_like(A), where=col > 0)
    empty_in = (col[0] == 0)
    P_in[:, empty_in] = 0.0
    P_in[empty_in, empty_in] = 1.0

    # P_out: normalise each *row*, then transpose so columns are sources.
    row = A.sum(axis=1, keepdims=True)
    R = np.divide(A, row, out=np.zeros_like(A), where=row > 0)
    P_out = R.T.copy()
    empty_out = (row[:, 0] == 0)
    P_out[:, empty_out] = 0.0
    P_out[empty_out, empty_out] = 1.0

    return P_in, P_out


def check_column_stochastic(P: np.ndarray, tol: float = 1e-6) -> bool:
    P = np.asarray(P, dtype=np.float64)
    if P.shape[0] != P.shape[1]:
        return False
    if np.any(P < -tol):
        return False
    return bool(np.allclose(P.sum(axis=0), 1.0, atol=tol))


def propagate(P: np.ndarray, s: np.ndarray, steps: int = 2) -> list[np.ndarray]:
    """Return [P s, P^2 s, ..., P^steps s] for a batch of node distributions s."""
    out: list[np.ndarray] = []
    current = s
    for _ in range(steps):
        current = P @ current
        out.append(current)
    return out

