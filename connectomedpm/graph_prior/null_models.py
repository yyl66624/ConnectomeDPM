"""Matched null topologies (DEVELOPMENT.md section 11).

N0 No-Neighbor   : zero adjacency. With our operator convention this makes P_in = P_out = I,
                   so z = [s, s, s, s, s]: the router can still be trained, it just gets no
                   graph information. This is the *primary* check that edges matter.
N1 Uniform       : same edge count, endpoints uniform. Weak baseline only.
N2 Degree        : preserves in-degree and out-degree sequences (directed configuration
                   model), weights re-assigned from the real weight multiset.
N3 Module        : additionally preserves module sizes, module-block edge counts and hence
                   within/between-module density.

Every null keeps the real graph's node count and the real graph's weight multiset, so the
comparison isolates *wiring*, exactly as section 12 demands.
"""

from __future__ import annotations

import numpy as np

NULL_KINDS = ("no_neighbor", "uniform_random", "degree_preserving", "module_preserving")


def _weight_multiset(A: np.ndarray) -> np.ndarray:
    vals = A[A > 0]
    return np.sort(vals.astype(np.float64))


def _assign_weights(A_new: np.ndarray, A_ref: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Put the reference weight multiset onto the new edge support, in a random order."""
    support = np.argwhere(A_new > 0)
    weights = _weight_multiset(A_ref)
    if len(support) != len(weights):
        raise ValueError(f"support size {len(support)} != weight multiset {len(weights)}")
    perm = rng.permutation(len(weights))
    out = np.zeros_like(A_new, dtype=np.float64)
    for (i, j), w in zip(support, weights[perm]):
        out[i, j] = w
    return out


def _random_simple_support(k: int, n_edges: int, rng: np.random.Generator) -> set[tuple[int, int]]:
    """Uniformly sample `n_edges` distinct off-diagonal pairs."""
    possible = k * (k - 1)
    if n_edges > possible:
        raise ValueError(
            f"cannot place {n_edges} distinct directed edges on {k} nodes "
            f"(only {possible} off-diagonal pairs exist)"
        )
    pairs = np.array([(i, j) for i in range(k) for j in range(k) if i != j], dtype=int)
    chosen = rng.permutation(len(pairs))[:n_edges]
    return {tuple(int(x) for x in pairs[c]) for c in chosen}


def _swap_randomize(
    edges: set[tuple[int, int]],
    rng: np.random.Generator,
    n_swaps: int,
    module_of: dict[int, str] | None = None,
) -> set[tuple[int, int]]:
    """Directed edge-swap randomisation.

    Swapping ``(a -> b), (c -> d)`` into ``(a -> d), (c -> b)`` preserves every in-degree and
    every out-degree exactly while keeping the graph simple. When `module_of` is given the
    swap is only accepted between edges with the same (source-module, target-module) pair,
    which additionally preserves every module-block edge count.
    """
    edge_list = list(edges)
    present = set(edges)
    if len(edge_list) < 2:
        return present

    for _ in range(int(n_swaps)):
        i = int(rng.integers(0, len(edge_list)))
        j = int(rng.integers(0, len(edge_list)))
        if i == j:
            continue
        a, b = edge_list[i]
        c, d = edge_list[j]
        if module_of is not None:
            if module_of[a] != module_of[c] or module_of[b] != module_of[d]:
                continue
        n1, n2 = (a, d), (c, b)
        if n1[0] == n1[1] or n2[0] == n2[1]:
            continue
        if n1 in present or n2 in present:
            continue
        present.discard((a, b))
        present.discard((c, d))
        present.add(n1)
        present.add(n2)
        edge_list[i], edge_list[j] = n1, n2
    return present


def _support_to_matrix(edges: set[tuple[int, int]], k: int) -> np.ndarray:
    A = np.zeros((k, k), dtype=np.float64)
    for i, j in edges:
        A[i, j] = 1.0
    return A


def build_null(
    kind: str,
    A: np.ndarray,
    *,
    seed: int = 0,
    modules: list[str] | None = None,
) -> np.ndarray:
    """Return a null adjacency with the same shape as `A`."""
    if kind not in NULL_KINDS:
        raise ValueError(f"unknown null kind {kind!r}; expected one of {NULL_KINDS}")
    A = np.asarray(A, dtype=np.float64)
    k = A.shape[0]
    rng = np.random.default_rng(int(seed))

    if kind == "no_neighbor":
        return np.zeros_like(A)

    support = A > 0
    ref_support = support.copy()
    n_edges = int(ref_support.sum())
    if n_edges == 0:
        return np.zeros_like(A)

    if kind == "uniform_random":
        # Section 11 N1: "weak baseline, must not be the main biological control".
        # We draw a plain random support with the same edge count, then reuse the real
        # weight multiset. Degrees are deliberately *not* preserved.
        support = _random_simple_support(k, n_edges, rng)
        return _assign_weights(_support_to_matrix(support, k), A, rng)

    if kind == "degree_preserving":
        swapped = _swap_randomize({tuple(map(int, ij)) for ij in np.argwhere(ref_support)},
                                  rng, n_swaps=10 * n_edges)
        assert len(swapped) == n_edges, "edge-swap randomisation changed the edge count"
        return _assign_weights(_support_to_matrix(swapped, k), A, rng)

    if kind == "module_preserving":
        if modules is None:
            raise ValueError("module_preserving requires `modules`")
        module_of = {i: str(m) for i, m in enumerate(modules)}
        swapped = _swap_randomize({tuple(map(int, ij)) for ij in np.argwhere(ref_support)},
                                  rng, n_swaps=10 * n_edges, module_of=module_of)
        assert len(swapped) == n_edges, "module-preserving randomisation changed the edge count"
        return _assign_weights(_support_to_matrix(swapped, k), A, rng)

    raise AssertionError("unreachable")


# ---------------------------------------------------------------------------
# Constraint validation (used by tests and by the null-graph report)
# ---------------------------------------------------------------------------

def validate_null(kind: str, A: np.ndarray, null_A: np.ndarray, modules: list[str] | None = None) -> dict:
    """Return a dict of measured deviations; an empty `violations` list means success."""
    A = np.asarray(A, dtype=np.float64)
    null_A = np.asarray(null_A, dtype=np.float64)
    ref, got = A > 0, null_A > 0
    violations: list[str] = []
    checks: dict = {}

    checks["n_nodes"] = {"ref": int(A.shape[0]), "null": int(null_A.shape[0])}
    if A.shape != null_A.shape:
        violations.append("shape mismatch")

    if kind == "no_neighbor":
        checks["n_edges"] = {"ref": int(ref.sum()), "null": int(got.sum())}
        if got.any():
            violations.append("no_neighbor null has edges")
        return {"violations": violations, "checks": checks}

    checks["n_edges"] = {"ref": int(ref.sum()), "null": int(got.sum())}
    if int(ref.sum()) != int(got.sum()):
        violations.append("edge count differs")

    if kind in ("degree_preserving", "module_preserving"):
        checks["in_degree"] = {"ref": ref.sum(axis=0).tolist(), "null": got.sum(axis=0).tolist()}
        checks["out_degree"] = {"ref": ref.sum(axis=1).tolist(), "null": got.sum(axis=1).tolist()}
        if not np.array_equal(ref.sum(axis=0), got.sum(axis=0)):
            violations.append("in-degree sequence differs")
        if not np.array_equal(ref.sum(axis=1), got.sum(axis=1)):
            violations.append("out-degree sequence differs")

    if kind == "module_preserving":
        if modules is None:
            violations.append("modules required for validation")
        else:
            labels = sorted(set(modules))
            idx = {label: i for i, label in enumerate(labels)}
            assign = np.array([idx[m] for m in modules])
            m = len(labels)
            ref_blocks = np.zeros((m, m), dtype=int)
            new_blocks = np.zeros((m, m), dtype=int)
            for a in range(m):
                for b in range(m):
                    ref_blocks[a, b] = int(ref[np.ix_(assign == a, assign == b)].sum())
                    new_blocks[a, b] = int(got[np.ix_(assign == a, assign == b)].sum())
            checks["block_counts"] = {"ref": ref_blocks.tolist(), "null": new_blocks.tolist()}
            if not np.array_equal(ref_blocks, new_blocks):
                violations.append("module block counts differ")

    ref_w = np.sort(A[A > 0])
    null_w = np.sort(null_A[null_A > 0])
    checks["weight_multiset_equal"] = bool(
        len(ref_w) == len(null_w) and np.allclose(ref_w, null_w)
    )
    if not checks["weight_multiset_equal"]:
        violations.append("weight multiset differs")

    return {"violations": violations, "checks": checks}
