import numpy as np
import pandas as pd
import pytest

from connectomedpm.graph_prior.build import build_null_topologies, build_topology
from connectomedpm.graph_prior.normalize import check_column_stochastic


def _fixture(n_communities: int = 40, per_community: int = 6, seed: int = 0):
    """A tiny synthetic connectome with the same column contract as MaleCNS."""
    rng = np.random.default_rng(seed)
    rows = []
    body = 1000
    communities = {}
    for c in range(n_communities):
        community = f"supertype_{c:03d}"
        module = f"superclass_{c % 4}"
        for _ in range(per_community):
            rows.append({"bodyId": body, "superclass": f"cb_{module}",
                         "supertype": community, "statusLabel": "Traced"})
            communities[body] = community
            body += 1
    annotations = pd.DataFrame(rows)

    edges = []
    for _ in range(4000):
        a = int(rng.choice(annotations["bodyId"]))
        b = int(rng.choice(annotations["bodyId"]))
        if a != b:
            edges.append({"pre": a, "post": b, "weight": float(int(rng.integers(1, 30)))})
    return pd.DataFrame(edges), annotations


def test_build_topology_shapes_and_direction():
    edges, annotations = _fixture()
    graph = build_topology(edges, annotations, graph_id="toy", k=8)
    assert graph["adjacency"].shape == (8, 8)
    assert graph["transition_in"].shape == (8, 8)
    assert np.allclose(np.diag(graph["adjacency"]), 0.0), "diagonal must be zeroed"
    assert check_column_stochastic(graph["transition_in"])
    assert check_column_stochastic(graph["transition_out"])
    assert np.all(graph["node_sizes"] > 0)


def test_build_topology_is_deterministic():
    edges, annotations = _fixture()
    a = build_topology(edges, annotations, graph_id="toy", k=8)
    b = build_topology(edges, annotations, graph_id="toy", k=8)
    assert np.array_equal(a["adjacency"], b["adjacency"])
    assert a["node_names"] == b["node_names"]
    assert a["coarse_mapping"] == b["coarse_mapping"]


def test_coarse_graining_covers_every_community_once():
    edges, annotations = _fixture()
    graph = build_topology(edges, annotations, graph_id="toy", k=8)
    mapping = graph["coarse_mapping"]
    assert len({row["community"] for row in mapping}) == len(mapping)
    assert all(0 <= row["routing_node"] < 8 for row in mapping)


def test_node_features_are_per_graph():
    """Section 12: null graphs must not inherit the real graph's node features."""
    edges, annotations = _fixture()
    real = build_topology(edges, annotations, graph_id="toy", k=8)
    nulls = build_null_topologies(real, kinds=("degree_preserving",), seed=0)
    assert not np.allclose(real["node_features"], nulls[0]["node_features"])


def test_k_larger_than_communities_is_rejected():
    edges, annotations = _fixture(n_communities=4)
    with pytest.raises(ValueError):
        build_topology(edges, annotations, graph_id="toy", k=32)

