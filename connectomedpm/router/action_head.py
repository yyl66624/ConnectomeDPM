"""Per-block three-way action head: P(FIX), P(BREAK), P(UNCHANGED)."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..supervision.outcome_labeler import BREAK, FIX, UNCHANGED

CLASS_ORDER = (FIX, BREAK, UNCHANGED)


class ActionHead(nn.Module):
    def __init__(self, in_dim: int, n_blocks: int, hidden: int | None = None,
                 dropout: float = 0.0):
        super().__init__()
        self.n_blocks = int(n_blocks)
        self.n_classes = len(CLASS_ORDER)
        if hidden and hidden > 0:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, self.n_blocks * self.n_classes),
            )
        else:
            self.net = nn.Linear(in_dim, self.n_blocks * self.n_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Return logits shaped (B, n_blocks, 3)."""
        return self.net(z).view(z.shape[0], self.n_blocks, self.n_classes)

    def probabilities(self, z: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward(z), dim=-1)

