#!/usr/bin/env python3
"""Generate a tiny synthetic connectome in the MaleCNS column contract.

Used to smoke-test the whole pipeline (graph -> nulls -> blocks -> cache -> router -> report)
without waiting for the 1 GB release download. It is *not* a substitute for the real data:
graphs built from it are labelled `synthetic` and must never be reported as MaleCNS results.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow.feather as feather
import pandas as pd


def make(
    out_dir: Path,
    *,
    n_communities: int = 24,
    per_community: int = 8,
    n_edges: int = 3000,
    ood_prefix: str = "ol_",
    seed: int = 0,
) -> tuple[Path, Path]:
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    body = 10000
    for c in range(n_communities):
        # 3/4 of the communities are central brain (cb_*), the rest visual (ol_*).
        region = "ol" if (c % 4 == 3) else "cb"
        for _ in range(per_community):
            rows.append(
                {
                    "bodyId": body,
                    "superclass": f"{region}_intrinsic",
                    "supertype": f"{region}_supertype_{c:03d}",
                    "class": f"{region}_class_{c % 5}",
                    "statusLabel": "Traced",
                }
            )
            body += 1
    annotations = pd.DataFrame(rows)

    ids = annotations["bodyId"].to_numpy()
    community = annotations["supertype"].to_numpy()
    # Prefer within-community edges so the coarse-grained graph has real block structure.
    edges = []
    for _ in range(n_edges):
        if rng.random() < 0.7:
            c = int(rng.integers(0, n_communities))
            pool = ids[community == f"{'ol' if (c % 4 == 3) else 'cb'}_supertype_{c:03d}"]
            a, b = int(rng.choice(pool)), int(rng.choice(pool))
        else:
            a, b = int(rng.choice(ids)), int(rng.choice(ids))
        if a != b:
            edges.append({"bodyId_pre": a, "bodyId_post": b, "weight": float(int(rng.integers(1, 40)))})
    edge_df = pd.DataFrame(edges)

    ann_path = out_dir / "body-annotations-male-cns-synthetic.feather"
    w_path = out_dir / "connectome-weights-male-cns-synthetic.feather"
    feather.write_feather(annotations, ann_path)
    feather.write_feather(edge_df, w_path)
    print(f"annotations: {len(annotations)} rows -> {ann_path}")
    print(f"edges: {len(edge_df)} rows -> {w_path}")
    return ann_path, w_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-communities", type=int, default=24)
    parser.add_argument("--per-community", type=int, default=8)
    parser.add_argument("--n-edges", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    make(
        Path(args.out_dir),
        n_communities=args.n_communities,
        per_community=args.per_community,
        n_edges=args.n_edges,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
