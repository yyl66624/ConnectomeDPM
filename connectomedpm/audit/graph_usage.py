"""Graph-usage audits (DEVELOPMENT.md section 25).

If routing does not change when the graph changes, the project must not claim that topology
is doing any work. These audits make that failure mode explicit.
"""

from __future__ import annotations

import copy

import numpy as np
import torch

from ..router.inference import score_blocks


def _identity_like(P: np.ndarray) -> np.ndarray:
    return np.eye(P.shape[0], dtype=np.float64)


def no_neighbor_counterfactual(model, features: np.ndarray, block_ids, *, lam: float = 1.0):
    """Audit A: replace P_in/P_out with identity, keeping every learned weight.

    This answers "how much would the router change if the edges were removed?" for an
    already-trained model, without retraining.
    """
    twin = copy.deepcopy(model)
    k = int(twin.operator.P_in.shape[0])
    with torch.no_grad():
        twin.operator.P_in.copy_(torch.eye(k, dtype=twin.operator.P_in.dtype,
                                           device=twin.operator.P_in.device))
        twin.operator.P_out.copy_(torch.eye(k, dtype=twin.operator.P_out.dtype,
                                            device=twin.operator.P_out.device))
    return score_blocks(twin, features, block_ids, lam=lam)


def permute_graph(graph: dict, perm: np.ndarray) -> dict:
    """Return a copy of `graph` with nodes relabelled by `perm` (new node i == old perm[i])."""
    perm = np.asarray(perm, dtype=int)
    out = dict(graph)

    def _permute(mat, axes):
        m = np.asarray(mat)
        return m[np.ix_(perm, perm)] if axes == 2 else m[perm]

    out["adjacency"] = _permute(graph["adjacency"], 2)
    out["transition_in"] = _permute(graph["transition_in"], 2)
    out["transition_out"] = _permute(graph["transition_out"], 2)
    out["node_features"] = _permute(graph["node_features"], 1)
    out["self_loops"] = _permute(graph["self_loops"], 1)
    out["node_sizes"] = _permute(graph["node_sizes"], 1)
    if graph.get("node_modules") is not None:
        out["node_modules"] = [graph["node_modules"][i] for i in perm]
    return out


def permuted_twin(model, perm: np.ndarray):
    """Audit C: a router that is identical except that its nodes were relabelled.

    The projector's output rows, the operators, and the action head's input columns are all
    permuted consistently. A correct implementation must therefore produce identical scores;
    any deviation is an indexing bug, not a discovery.
    """
    twin = copy.deepcopy(model)
    perm = np.asarray(perm, dtype=int)
    k = int(twin.operator.P_in.shape[0])
    steps = int(twin.operator.steps)
    with torch.no_grad():
        for name in ("P_in", "P_out"):
            buf = getattr(twin.operator, name)
            buf.copy_(buf[np.ix_(perm, perm)])
        if twin.operator.node_features.shape[1] > 0:
            twin.operator.node_features.copy_(twin.operator.node_features[perm])
        w = twin.projector.net.weight
        b = twin.projector.net.bias
        w.copy_(w[perm])
        if b is not None:
            b.copy_(b[perm])

        # z is laid out as [s | Pin^1 | Pout^1 | ... | Pin^steps | Pout^steps], so the head's
        # input columns must be permuted inside every block.
        head_net = twin.head.net
        first = head_net[0] if isinstance(head_net, torch.nn.Sequential) else head_net
        index: list[int] = []
        for block in range(1 + 2 * steps):
            index.extend(block * k + perm)
        first.weight.copy_(first.weight[:, torch.as_tensor(index, dtype=torch.long)])
    return twin


def graph_sensitivity(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    block_ids,
    *,
    threshold: float = 0.0,
) -> dict:
    """Audit D: how much does routing move when the topology moves?"""
    a, b = np.asarray(scores_a), np.asarray(scores_b)
    if a.shape != b.shape:
        raise ValueError(f"score shapes differ: {a.shape} vs {b.shape}")
    choice_a = a.argmax(axis=1)
    choice_b = b.argmax(axis=1)
    intervene_a = a.max(axis=1) >= threshold
    intervene_b = b.max(axis=1) >= threshold

    flat_a, flat_b = a.reshape(-1), b.reshape(-1)
    corr = float(np.corrcoef(flat_a, flat_b)[0, 1]) if flat_a.size > 1 else float("nan")
    return {
        "action_change_rate": float((choice_a != choice_b).mean()),
        "coverage_change": float(intervene_b.mean() - intervene_a.mean()),
        "score_correlation": corr,
        "mean_abs_score_delta": float(np.abs(flat_a - flat_b).mean()),
        "max_abs_score_delta": float(np.abs(flat_a - flat_b).max()),
        "n_samples": int(a.shape[0]),
        "n_blocks": len(block_ids),
    }
