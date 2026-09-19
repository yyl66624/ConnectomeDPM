"""Main-node selection and community aggregation (DEVELOPMENT.md section 8, steps 2-3).

This module is deliberately blind to the language side of the project: it takes only the
connectome tables, never task performance, never router scores, never `D_test*`.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .malecns_loader import find_column

# Annotation columns that describe *where* a neuron lives / which population it belongs to.
REGION_CANDIDATES = (
    "region", "centralBrain", "central_brain", "primaryRoi", "roi",
    "somaNeuromere", "neuropil", "compartment", "bodyPart",
)
COMMUNITY_CANDIDATES = (
    "superClass", "super_class", "superclass",
    "class", "cell_class", "cellClass",
    "cellType", "cell_type", "celltype",
)


def select_main_nodes(
    annotations: pd.DataFrame,
    *,
    region_key: str | None = None,
    region_values: Iterable[str] | None = None,
    min_confidence: float | None = None,
    require_status: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Return the subset of annotated neurons that form the main node set.

    Args:
        region_key: column holding the region/compartment label. If None, auto-detected.
        region_values: accepted values for that column. If None, all values are accepted
            (documented in the returned `attrs`).
        min_confidence: if a confidence column exists, keep rows at or above it.
        require_status: accepted values of the `status` column (e.g. proofread labels).
    """
    df = annotations.copy()
    kept_filters: dict = {}

    if region_key is not None and region_values is not None:
        col = find_column(df, (region_key,))
        if col is None:
            raise KeyError(f"region column {region_key!r} not found")
        values = {str(v).lower() for v in region_values}
        mask = df[col].astype(str).str.lower().isin(values)
        df = df[mask]
        kept_filters["region"] = {"column": str(col), "values": sorted(values)}

    if min_confidence is not None:
        col = find_column(df, ("confidence", "conf", "minconf", "score"))
        if col is not None:
            df = df[df[col].astype(float) >= float(min_confidence)]
            kept_filters["min_confidence"] = {"column": str(col), "value": float(min_confidence)}

    if require_status is not None:
        col = find_column(df, ("status", "review_status", "proofread"))
        if col is not None:
            values = {str(v).lower() for v in require_status}
            df = df[df[col].astype(str).str.lower().isin(values)]
            kept_filters["status"] = {"column": str(col), "values": sorted(values)}

    df.attrs["filters"] = kept_filters
    df.attrs["n_input"] = int(len(annotations))
    return df


def assign_communities(
    main_nodes: pd.DataFrame,
    *,
    community_keys: Iterable[str] = COMMUNITY_CANDIDATES,
    min_community_size: int = 1,
) -> tuple[pd.DataFrame, str]:
    """Attach a `community` label to each main node using the first usable annotation column.

    Returns the frame (with `community` column) and the name of the column that was used.
    Neurons with a missing label are grouped under `unknown`, so no neuron is silently lost.
    """
    df = main_nodes.copy()
    col = None
    for cand in community_keys:
        found = find_column(df, (cand,))
        if found is not None and df[found].notna().any():
            col = found
            break
    if col is None:
        raise KeyError(
            "no community column found; available: " + ", ".join(str(c) for c in df.columns)
        )

    community = df[col].astype(str).fillna("unknown")
    community = community.where(df[col].notna(), "unknown")
    df["community"] = community

    sizes = df["community"].value_counts()
    small = set(sizes[sizes < int(min_community_size)].index)
    if small:
        # Deterministic: fold very small classes together rather than dropping them.
        df["community"] = np.where(df["community"].isin(small), "other_small", df["community"])
    return df, str(col)


def filter_edges_within(edges: pd.DataFrame, node_ids: Iterable[int]) -> pd.DataFrame:
    """Keep only edges whose *both* endpoints are in the main node set (step 2)."""
    node_set = pd.Index(sorted(set(int(x) for x in node_ids)))
    mask = edges["pre"].isin(node_set) & edges["post"].isin(node_set)
    return edges.loc[mask].reset_index(drop=True)


def select_nodes_by_label_prefix(
    annotations: pd.DataFrame,
    *,
    column: str,
    prefix: str,
    status_column: str | None = None,
    excluded_status: Iterable[str] = (),
) -> pd.DataFrame:
    """Main node set = neurons whose label in `column` starts with `prefix`.

    For MaleCNS v1.0 the central brain is encoded in `superclass` with the `cb_` prefix
    (`cb_intrinsic`, `cb_sensory`, `cb_motor`, `cb_endocrine`, ...). Non-neuronal and
    out-of-scope segments are removed through `status_column`.
    """
    col = find_column(annotations, (column,))
    if col is None:
        raise KeyError(f"column {column!r} not found in annotations")
    df = annotations.copy()
    mask = df[col].astype(str).str.lower().str.startswith(prefix.lower())
    df = df[mask]
    filters = {"region": {"column": str(col), "prefix": prefix},
               "n_after_region": int(len(df))}

    if status_column is not None and excluded_status:
        status_col = find_column(df, (status_column,))
        if status_col is not None:
            excluded = {str(s).lower() for s in excluded_status}
            before = len(df)
            df = df[~df[status_col].astype(str).str.lower().isin(excluded)]
            filters["status"] = {"column": str(status_col),
                                 "excluded": sorted(excluded),
                                 "n_removed": int(before - len(df))}

    df = df[df["bodyId"].notna()].copy()
    df.attrs["filters"] = filters
    df.attrs["n_input"] = int(len(annotations))
    return df


def aggregate_communities(
    edges: pd.DataFrame,
    node_to_community: dict[int, str],
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Aggregate directed edge weights between communities (step 3).

    Returns:
        W: (C, C) float matrix of total weight from community i to community j.
        communities: ordered community names (sorted for determinism).
        sizes: (C,) neuron counts per community.
    """
    pre = edges["pre"].map(node_to_community)
    post = edges["post"].map(node_to_community)
    valid = pre.notna() & post.notna()
    pre = pre[valid].astype(str)
    post = post[valid].astype(str)
    weights = edges.loc[valid, "weight"].to_numpy(dtype=np.float64)

    communities = sorted(set(pre.unique()) | set(post.unique()))
    index = {name: i for i, name in enumerate(communities)}
    c = len(communities)

    W = np.zeros((c, c), dtype=np.float64)
    np.add.at(W, (pre.map(index).to_numpy(), post.map(index).to_numpy()), weights)

    size_map = pd.Series(1, index=list(node_to_community.values()))
    counts = size_map.groupby(level=0).sum()
    sizes = np.array([int(counts.get(name, 0)) for name in communities], dtype=np.int64)
    return W, communities, sizes
