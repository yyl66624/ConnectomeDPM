"""The MaleCNS -> K-node topology pipeline (DEVELOPMENT.md section 8) and its nulls.

Pipeline, in order and with no language-side signal anywhere in it:

    1. main node set  = neurons labelled `cb_*` (central brain), minus non-neuronal statuses
    2. edges          = both endpoints inside the main node set
    3. communities    = the release's cell-type hierarchy (`supertype` by default)
    4. modules        = the coarser level of the same hierarchy (`superclass`)
    5. coarse-grain   = deterministic size-balanced merge of C communities into K nodes
    6. operators      = log1p edge transform, zero diagonal, column-stochastic P_in / P_out
    7. features       = self-loop weight, strengths, degrees, node size, reciprocity
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils.hash import stable_hash
from .coarse_grain import coarse_grain
from .features import FEATURE_NAMES, node_features
from .filter import (
    aggregate_communities,
    assign_communities,
    filter_edges_within,
    select_nodes_by_label_prefix,
)
from .graph_stats import graph_stats
from .malecns_loader import find_column
from .normalize import check_column_stochastic, edge_transform, split_self_loops, transition_operators

DEFAULTS = {
    "region_column": "superclass",
    "region_prefix": "cb_",
    "status_column": "statusLabel",
    "excluded_status": ("Glia", "Out of scope"),
    "community_column": "supertype",
    "module_column": "superclass",
    "min_community_size": 1,
    "edge_transform": "log1p",
    "steps": 2,
}


def build_topology(
    edges: pd.DataFrame,
    annotations: pd.DataFrame,
    *,
    graph_id: str,
    k: int = 32,
    region_column: str | None = None,
    region_prefix: str | None = None,
    status_column: str | None = None,
    excluded_status: tuple = (),
    community_column: str | None = None,
    module_column: str | None = None,
    min_community_size: int | None = None,
    transform: str | None = None,
    steps: int | None = None,
    provenance: dict | None = None,
) -> dict:
    """Build one K-node topology. Returns a dict with arrays, stats and the full mapping."""
    cfg = dict(DEFAULTS)
    for key, value in {
        "region_column": region_column,
        "region_prefix": region_prefix,
        "status_column": status_column,
        "community_column": community_column,
        "module_column": module_column,
        "min_community_size": min_community_size,
        "edge_transform": transform,
        "steps": steps,
    }.items():
        if value is not None:
            cfg[key] = value
    if excluded_status:
        cfg["excluded_status"] = tuple(excluded_status)

    # --- steps 1-2: main node set and induced subgraph --------------------------------
    main_nodes = select_nodes_by_label_prefix(
        annotations,
        column=cfg["region_column"],
        prefix=cfg["region_prefix"],
        status_column=cfg["status_column"],
        excluded_status=cfg["excluded_status"],
    )
    node_ids = main_nodes["bodyId"].astype("int64").tolist()
    induced = filter_edges_within(edges, node_ids)

    # --- steps 3-4: community and module labels ---------------------------------------
    labelled, community_column_used = assign_communities(
        main_nodes,
        community_keys=(cfg["community_column"],),
        min_community_size=int(cfg["min_community_size"]),
    )
    module_col = find_column(labelled, (cfg["module_column"], cfg["community_column"]))
    labelled["module"] = (
        labelled[module_col].astype(str) if module_col is not None else "module0"
    )
    node_to_community = dict(zip(labelled["bodyId"].astype("int64"), labelled["community"]))
    node_to_module = dict(zip(labelled["bodyId"].astype("int64"), labelled["module"]))

    community_weights, community_names, community_sizes = aggregate_communities(
        induced, node_to_community
    )
    community_modules = {}
    for name in community_names:
        tally: dict[str, int] = {}
        for body_id, community in node_to_community.items():
            if community == name:
                label = node_to_module.get(body_id, "module0")
                tally[label] = tally.get(label, 0) + 1
        community_modules[name] = max(sorted(tally), key=lambda key: tally[key]) if tally else "module0"

    # --- step 5: deterministic coarse-graining ----------------------------------------
    cg = coarse_grain(
        community_weights,
        community_names,
        community_sizes,
        k=int(k),
        community_modules=community_modules,
    )

    # --- step 6: edge transform + fixed operators -------------------------------------
    # Self-loops are removed *before* the transform so that the diagonal is zero everywhere
    # downstream, and they are kept only as a node feature (DEVELOPMENT.md section 8 step 5).
    raw_no_diagonal, raw_self_loops = split_self_loops(cg.community_matrix)
    transformed = edge_transform(cg.community_matrix, cfg["edge_transform"])
    adjacency, self_loops = split_self_loops(transformed)
    p_in, p_out = transition_operators(adjacency)
    assert check_column_stochastic(p_in), "P_in is not column-stochastic"
    assert check_column_stochastic(p_out), "P_out is not column-stochastic"

    # --- step 7: node features (recomputed per graph, never copied) -------------------
    feats = node_features(adjacency, self_loops, cg.node_sizes, standardize=True)
    stats = graph_stats(adjacency, self_loops, cg.node_sizes, modules=cg.node_modules)
    stats["source"] = {
        "n_annotated_bodies": int(len(annotations)),
        "n_main_nodes": int(len(node_ids)),
        "n_edges_induced": int(len(induced)),
        "n_communities": int(len(community_names)),
        "community_column": str(community_column_used),
        "module_column": str(module_col),
        "region_column": cfg["region_column"],
        "region_prefix": cfg["region_prefix"],
        "filters": main_nodes.attrs.get("filters", {}),
        "edge_transform": cfg["edge_transform"],
    }
    stats["module_balance"] = {
        label: int(sum(1 for m in cg.node_modules if m == label))
        for label in sorted(set(cg.node_modules))
    }

    return {
        "graph_id": graph_id,
        "graph_type": "malecns",
        "node_names": cg.node_names,
        "adjacency": adjacency,
        "transition_in": p_in,
        "transition_out": p_out,
        "node_features": feats,
        "feature_names": list(FEATURE_NAMES),
        "self_loops": self_loops,
        "node_sizes": cg.node_sizes,
        "node_modules": cg.node_modules,
        "stats": stats,
        "coarse_mapping": cg.mapping,
        "provenance": provenance or {},
        "adjacency_raw": raw_no_diagonal,
        "self_loops_raw": raw_self_loops,
        "community_names": community_names,
        "community_sizes": community_sizes,
        "community_weights": community_weights,
        "steps": int(cfg["steps"]),
        "config": cfg,
    }


def topology_hash(real: dict) -> str:
    """Identity of the wiring, used in the run manifest and as part of the cache key."""
    return stable_hash(
        {
            "graph_id": real["graph_id"],
            "adjacency": np.round(np.asarray(real["adjacency"], dtype=np.float64), 6).tolist(),
            "node_names": list(real["node_names"]),
        },
        length=32,
    )


def build_null_topologies(
    real: dict,
    kinds: tuple[str, ...] = ("no_neighbor", "uniform_random", "degree_preserving",
                              "module_preserving"),
    *,
    seed: int = 0,
    instances: int = 1,
) -> list[dict]:
    """Build every matched null for one real topology.

    Each null recomputes its own operators, features and statistics from its own adjacency
    (DEVELOPMENT.md section 12): the real node features are never copied across.
    """
    from .null_models import NULL_KINDS, build_null

    out: list[dict] = []
    for kind in kinds:
        if kind not in NULL_KINDS:
            raise ValueError(f"unknown null kind {kind!r}")
        for instance in range(int(instances)):
            null_seed = int(seed) + 1000 * (NULL_KINDS.index(kind) + 1) + instance
            null_adjacency_raw = build_null(
                kind,
                real["adjacency_raw"],
                seed=null_seed,
                modules=real.get("node_modules"),
            )
            adjacency, self_loops = split_self_loops(
                edge_transform(null_adjacency_raw, real["config"]["edge_transform"])
            )
            p_in, p_out = transition_operators(adjacency)
            node_sizes = np.asarray(real["node_sizes"])
            feats = node_features(adjacency, self_loops, node_sizes, standardize=True)
            stats = graph_stats(adjacency, self_loops, node_sizes,
                                modules=real.get("node_modules"))
            stats["null_kind"] = kind
            stats["null_seed"] = null_seed
            out.append(
                {
                    "graph_id": f"{real['graph_id']}__{kind}" + (
                        f"_i{instance}" if instances > 1 else ""
                    ),
                    "graph_type": kind,
                    "node_names": list(real["node_names"]),
                    "adjacency": adjacency,
                    "transition_in": p_in,
                    "transition_out": p_out,
                    "node_features": feats,
                    "feature_names": list(FEATURE_NAMES),
                    "self_loops": self_loops,
                    "node_sizes": node_sizes,
                    "node_modules": list(real.get("node_modules") or []),
                    "stats": stats,
                    "coarse_mapping": None,
                    "provenance": {"derived_from": real["graph_id"],
                                   "null_kind": kind, "seed": null_seed},
                    "adjacency_raw": null_adjacency_raw,
                    "steps": int(real["steps"]),
                    "config": dict(real["config"]),
                }
            )
    return out
