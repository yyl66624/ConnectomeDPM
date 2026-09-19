"""Graph artefact bundle I/O.

DEVELOPMENT.md section 8 step 6 defines the on-disk format. The numpy payload lives in a
single `.npz`; everything else is JSON so it stays diffable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..utils.io import ensure_dir, read_json, write_json


def save_graph(
    out_dir: str | Path,
    *,
    graph_id: str,
    graph_type: str,
    node_names: list[str],
    adjacency: np.ndarray,
    transition_in: np.ndarray,
    transition_out: np.ndarray,
    node_features: np.ndarray,
    feature_names: list[str],
    self_loops: np.ndarray,
    node_sizes: np.ndarray,
    stats: dict,
    manifest: dict,
    mapping: list[dict] | None = None,
    node_modules: list[str] | None = None,
    steps: int | None = None,
    config: dict | None = None,
) -> Path:
    out_dir = ensure_dir(out_dir)
    np.savez_compressed(
        out_dir / "graph.npz",
        adjacency=adjacency,
        transition_in=transition_in,
        transition_out=transition_out,
        node_features=node_features,
        self_loops=self_loops,
        node_sizes=node_sizes,
    )
    write_json(
        out_dir / "graph_meta.json",
        {
            "graph_id": graph_id,
            "graph_type": graph_type,
            "node_names": node_names,
            "feature_names": feature_names,
            "node_modules": node_modules,
            "steps": steps,
            "config": config or {},
            "shapes": {
                "adjacency": list(np.asarray(adjacency).shape),
                "transition_in": list(np.asarray(transition_in).shape),
                "transition_out": list(np.asarray(transition_out).shape),
                "node_features": list(np.asarray(node_features).shape),
            },
        },
    )
    write_json(out_dir / "graph_stats.json", stats)
    write_json(out_dir / "manifest.json", manifest)
    if mapping is not None:
        write_json(out_dir / "coarse_mapping.json", mapping)
    return out_dir


def load_graph(path: str | Path) -> dict:
    """Load a graph bundle. `path` may be the directory or the `.npz` file."""
    path = Path(path)
    if path.is_file():
        npz_path, base = path, path.parent
    else:
        npz_path, base = path / "graph.npz", path
    data = np.load(npz_path)
    meta = read_json(base / "graph_meta.json")
    stats = read_json(base / "graph_stats.json")
    manifest = read_json(base / "manifest.json")
    mapping_path = base / "coarse_mapping.json"
    mapping = read_json(mapping_path) if mapping_path.exists() else None
    return {
        "graph_id": meta["graph_id"],
        "graph_type": meta["graph_type"],
        "node_names": meta["node_names"],
        "feature_names": meta["feature_names"],
        "node_modules": meta.get("node_modules"),
        "steps": meta.get("steps"),
        "config": meta.get("config") or {},
        "adjacency": data["adjacency"],
        "transition_in": data["transition_in"],
        "transition_out": data["transition_out"],
        "node_features": data["node_features"],
        "self_loops": data["self_loops"],
        "node_sizes": data["node_sizes"],
        "stats": stats,
        "manifest": manifest,
        "coarse_mapping": mapping,
    }


def list_graphs(root: str | Path) -> list[str]:
    root = Path(root)
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if (p / "graph.npz").exists())
