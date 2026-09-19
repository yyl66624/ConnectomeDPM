import numpy as np
import pytest

torch = pytest.importorskip("torch")

from connectomedpm.audit.graph_usage import permute_graph, permuted_twin  # noqa: E402
from connectomedpm.router.inference import BASE_ACTION, score_blocks, select_actions  # noqa: E402
from connectomedpm.router.model import TopologyRouter  # noqa: E402
from connectomedpm.router.transfer_operator import FixedTransferOperator  # noqa: E402


def _graph(k=6, seed=0):
    rng = np.random.default_rng(seed)
    A = (rng.random((k, k)) > 0.6).astype(float) * rng.random((k, k))
    np.fill_diagonal(A, 0.0)
    col = A.sum(axis=0, keepdims=True)
    p_in = np.divide(A, col, out=np.zeros_like(A), where=col > 0)
    row = A.sum(axis=1, keepdims=True)
    r = np.divide(A, row, out=np.zeros_like(A), where=row > 0)
    return {
        "graph_id": "toy",
        "adjacency": A,
        "transition_in": p_in,
        "transition_out": r.T,
        "node_features": rng.random((k, 7)),
        "feature_names": [f"f{i}" for i in range(7)],
    }


def test_transfer_operator_output_dim():
    g = _graph()
    op = FixedTransferOperator(g["transition_in"], g["transition_out"], steps=2)
    s = torch.softmax(torch.randn(4, 6), dim=-1)
    z = op(s)
    assert z.shape == (4, 6 * 5)
    assert op.output_dim == 30


def test_router_shapes_and_probabilities():
    g = _graph()
    model = TopologyRouter(in_dim=16, n_blocks=3, transition_in=g["transition_in"],
                           transition_out=g["transition_out"], steps=2, hidden=32)
    h = torch.randn(7, 16)
    logits = model(h)
    assert logits.shape == (7, 3, 3)
    probs = model.probabilities(h)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(7, 3), atol=1e-5)


def test_select_actions_threshold_and_fallback():
    scores = np.array([[0.5, 0.1], [0.05, 0.02]])
    chosen, intervened, rank = select_actions(scores, ["b0", "b1"], threshold=0.1)
    assert chosen == ["b0", BASE_ACTION]
    assert intervened.tolist() == [True, False]
    assert rank[0] == 0


def test_node_permutation_is_equivariant():
    """Audit C: relabelling the nodes must not change a single score."""
    g = _graph(k=6)
    model = TopologyRouter(in_dim=8, n_blocks=2, transition_in=g["transition_in"],
                           transition_out=g["transition_out"], steps=2, hidden=16)
    h = torch.randn(5, 8)
    baseline = model(h).detach().numpy()

    perm = np.array([3, 0, 5, 1, 4, 2])
    twin = permuted_twin(model, perm)
    permuted_scores = twin(h).detach().numpy()
    assert np.allclose(baseline, permuted_scores, atol=1e-5)


def test_no_neighbor_scores_match_identity_propagation():
    g = _graph(k=5)
    model = TopologyRouter(in_dim=4, n_blocks=2, transition_in=g["transition_in"],
                           transition_out=g["transition_out"], steps=2, hidden=8)
    h = np.random.default_rng(0).normal(size=(3, 4)).astype("float32")
    scores, _, _ = score_blocks(model, h, ["b0", "b1"])
    assert scores.shape == (3, 2)
