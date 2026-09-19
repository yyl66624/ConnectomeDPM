"""Request features -> distribution over K graph nodes."""

from __future__ import annotations

import torch
import torch.nn as nn


class InputProjector(nn.Module):
    """s(x) = softmax(W_in h_x), optionally through a small MLP bottleneck.

    The bottleneck exists only to keep the parameter count identical across topologies
    (DEVELOPMENT.md section 12 requires an identical router for every graph). It never sees
    the topology.
    """

    def __init__(self, in_dim: int, n_nodes: int, hidden: int | None = None,
                 dropout: float = 0.0, temperature: float = 1.0):
        super().__init__()
        self.temperature = float(temperature)
        if hidden and hidden > 0:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, n_nodes),
            )
        else:
            self.net = nn.Linear(in_dim, n_nodes)

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h) / self.temperature

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.logits(h), dim=-1)

