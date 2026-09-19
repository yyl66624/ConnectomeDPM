"""The fixed graph propagation.

This module is the entire reason the architecture is "graph-forced": the operators are
registered as buffers and are never trained, so the router cannot learn its way around the
topology. If swapping the graph does not move the routing, that is a real (and reportable)
result rather than a training artefact.

    z(x) = [ s, P_in s, P_out s, P_in^2 s, P_out^2 s ]        (DEVELOPMENT.md section 9)
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn


class FixedTransferOperator(nn.Module):
    def __init__(
        self,
        transition_in: np.ndarray,
        transition_out: np.ndarray,
        *,
        steps: int = 2,
        node_features: np.ndarray | None = None,
        feature_names: list[str] | None = None,
        use_node_features: bool = False,
    ):
        super().__init__()
        p_in = torch.as_tensor(np.asarray(transition_in, dtype=np.float32))
        p_out = torch.as_tensor(np.asarray(transition_out, dtype=np.float32))
        if p_in.shape != p_out.shape or p_in.ndim != 2 or p_in.shape[0] != p_in.shape[1]:
            raise ValueError(
                f"transition operators must be square and equal, got "
                f"{tuple(p_in.shape)} and {tuple(p_out.shape)}"
            )
        self.register_buffer("P_in", p_in, persistent=True)
        self.register_buffer("P_out", p_out, persistent=True)
        self.steps = int(steps)
        self.use_node_features = bool(use_node_features)

        if node_features is not None:
            self.register_buffer(
                "node_features", torch.as_tensor(np.asarray(node_features, dtype=np.float32))
            )
        else:
            self.register_buffer("node_features", torch.zeros(p_in.shape[0], 0))
        self.feature_names = list(feature_names or [])

        if self.use_node_features and self.node_features.shape[1] > 0:
            self.node_bias = nn.Parameter(torch.zeros(self.node_features.shape[1]))
        else:
            self.register_parameter("node_bias", None)

    @property
    def n_nodes(self) -> int:
        return int(self.P_in.shape[0])

    @property
    def output_dim(self) -> int:
        """Dimension of z: the raw s plus one block per propagation stage."""
        return self.n_nodes * (1 + 2 * self.steps)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """`s` is (B, K); returns z of shape (B, K * (1 + 2*steps))."""
        if s.shape[-1] != self.n_nodes:
            raise ValueError(f"expected {self.n_nodes} nodes, got {s.shape[-1]}")
        # P_in / P_out are column-stochastic, i.e. (P s)[i] = sum_j P[i, j] s[j] for a node
        # distribution s. For a batch of distributions we therefore multiply on the right by
        # the transpose: (s @ P.T)[b, i] = sum_j s[b, j] * P[i, j].
        parts = [s]
        cur_in, cur_out = s, s
        for _ in range(self.steps):
            cur_in = cur_in @ self.P_in.T
            cur_out = cur_out @ self.P_out.T
            parts.extend([cur_in, cur_out])
        return torch.cat(parts, dim=-1)

    def node_bias_vector(self) -> torch.Tensor | None:
        """Optional additive bias on the projected logits, derived from node features."""
        if not self.use_node_features or self.node_bias is None:
            return None
        scale = math.sqrt(max(int(self.node_features.shape[1]), 1))
        return (self.node_features @ self.node_bias).unsqueeze(0) / scale


def stage_names(n_nodes: int, steps: int) -> list[str]:
    names = [f"s[{i}]" for i in range(n_nodes)]
    for depth in range(1, steps + 1):
        names += [f"Pin^{depth}[{i}]" for i in range(n_nodes)]
        names += [f"Pout^{depth}[{i}]" for i in range(n_nodes)]
    return names
