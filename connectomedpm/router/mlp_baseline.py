"""Capacity-matched MLP baseline with no graph at all (DEVELOPMENT.md section 13.1).

This is the control that answers "would a plain function of the request features do just as
well?". It receives exactly the same input (h_x) and the same training data as the graph
routers; only the topology is absent.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .action_head import ActionHead


class MLPRouter(nn.Module):
    def __init__(self, *, in_dim: int, n_blocks: int, hidden: int = 256,
                 dropout: float = 0.0):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = ActionHead(hidden, n_blocks, hidden=None)

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(h))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.logits(h)

    def probabilities(self, h: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.logits(h), dim=-1)
