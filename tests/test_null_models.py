import numpy as np

from connectomedpm.graph_prior.build import build_null_topologies, build_topology
from connectomedpm.graph_prior.null_models import NULL_KINDS, build_null, validate_null
from tests.test_graph_pipeline import _fixture


def _real(k=8):
    edges, annotations = _fixture()
    return build_topology(edges, annotations, graph_id="toy", k=k)


def test_all_null_kinds_build_with_right_shape():
    real = _real()
    for kind in NULL_KINDS:
        null = build_null(kind, real["adjacency_raw"], seed=0, modules=real["node_modules"])
        assert null.shape == real["adjacency_raw"].shape
        assert np.allclose(np.diag(null), 0.0)


def test_no_neighbor_has_no_edges():
    real = _real()
    null = build_null("no_neighbor", real["adjacency_raw"], seed=0)
    assert not (null > 0).any()


def test_degree_preserving_matches_degrees_and_weights():
    real = _real()
    null = build_null("degree_preserving", real["adjacency_raw"], seed=0)
    report = validate_null("degree_preserving", real["adjacency_raw"], null)
    assert report["violations"] == [], report["violations"]


def test_module_preserving_matches_block_counts():
    real = _real()
    null = build_null("module_preserving", real["adjacency_raw"], seed=0,
                      modules=real["node_modules"])
    report = validate_null("module_preserving", real["adjacency_raw"], null,
                           modules=real["node_modules"])
    assert report["violations"] == [], report["violations"]


def test_uniform_random_keeps_edge_count_and_weights():
    real = _real()
    null = build_null("uniform_random", real["adjacency_raw"], seed=0)
    assert int((null > 0).sum()) == int((real["adjacency_raw"] > 0).sum())
    assert np.allclose(np.sort(null[null > 0]), np.sort(real["adjacency_raw"][real["adjacency_raw"] > 0]))


def test_nulls_are_seed_reproducible():
    real = _real()
    a = build_null("degree_preserving", real["adjacency_raw"], seed=7)
    b = build_null("degree_preserving", real["adjacency_raw"], seed=7)
    c = build_null("degree_preserving", real["adjacency_raw"], seed=8)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_build_null_topologies_returns_every_kind():
    real = _real()
    graphs = build_null_topologies(real, seed=0, instances=2)
    assert len(graphs) == len(NULL_KINDS) * 2
    ids = [g["graph_id"] for g in graphs]
    assert len(ids) == len(set(ids))
