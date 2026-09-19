"""Deterministic community -> K routing node coarse-graining (DEVELOPMENT.md section 8 step 4).

The rule is fixed and reproducible: communities are visited largest-first (ties broken by
name), and each one is placed into the routing node that currently holds the fewest neurons.
No language-side signal participates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CoarseGrainResult:
    mapping: list[dict]          # one row per original community
    node_names: list[str]
    node_sizes: np.ndarray       # neurons per routing node
    community_matrix: np.ndarray  # (K, K) weights aggregated over communities
    community_to_node: dict      # community name -> routing node index
    node_modules: list[str]      # majority module per routing node


def coarse_grain(
    community_weights: np.ndarray,
    community_names: list[str],
    community_sizes: np.ndarray,
    *,
    k: int = 32,
    community_modules: dict | None = None,
) -> CoarseGrainResult:
    """Merge `C` communities into `k` routing nodes.

    Args:
        community_weights: (C, C) directed weight matrix between communities.
        community_names: length-C names, used only for deterministic ordering.
        community_sizes: length-C neuron counts.
        k: number of routing nodes.
        community_modules: optional community -> module label map, used by the
            module-preserving null model. Each routing node inherits the module of the
            largest community inside it.
    """
    c = len(community_names)
    if community_weights.shape != (c, c):
        raise ValueError(f"community_weights must be ({c}, {c}), got {community_weights.shape}")
    if k > c:
        raise ValueError(f"k={k} exceeds the number of communities ({c}); nothing to merge")

    node_sizes = np.zeros(k, dtype=np.int64)
    community_to_node: dict[str, int] = {}
    node_members: list[list[int]] = [[] for _ in range(k)]

    order = sorted(range(c), key=lambda i: (-int(community_sizes[i]), str(community_names[i])))
    for idx in order:
        # Deterministic: smallest current size, then lowest node index.
        target = min(range(k), key=lambda n: (int(node_sizes[n]), n))
        community_to_node[str(community_names[idx])] = target
        node_sizes[target] += int(community_sizes[idx])
        node_members[target].append(idx)

    node_matrix = np.zeros((k, k), dtype=np.float64)
    for i in range(c):
        ni = community_to_node[str(community_names[i])]
        for j in range(c):
            if community_weights[i, j] == 0.0:
                continue
            nj = community_to_node[str(community_names[j])]
            node_matrix[ni, nj] += community_weights[i, j]

    node_names = [f"R{n:02d}" for n in range(k)]

    node_modules: list[str] = []
    for n in range(k):
        if not community_modules or not node_members[n]:
            node_modules.append("module0")
            continue
        tally: dict[str, int] = {}
        for idx in node_members[n]:
            label = str(community_modules.get(str(community_names[idx]), "module0"))
            tally[label] = tally.get(label, 0) + int(community_sizes[idx])
        node_modules.append(max(sorted(tally), key=lambda key: tally[key]))

    mapping = [
        {
            "community": str(community_names[i]),
            "community_size": int(community_sizes[i]),
            "routing_node": community_to_node[str(community_names[i])],
            "routing_node_name": node_names[community_to_node[str(community_names[i])]],
        }
        for i in range(c)
    ]

    return CoarseGrainResult(
        mapping=mapping,
        node_names=node_names,
        node_sizes=node_sizes,
        community_matrix=node_matrix,
        community_to_node=community_to_node,
        node_modules=node_modules,
    )

