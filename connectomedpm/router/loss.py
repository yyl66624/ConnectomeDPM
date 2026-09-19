"""Router loss (DEVELOPMENT.md section 10).

BREAK is rare but expensive, so the three classes carry weights with
`w_break > w_fix >= w_unchanged`. The weights are chosen on D_router_dev only.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..supervision.outcome_labeler import BREAK, FIX, UNCHANGED
from .action_head import CLASS_ORDER

DEFAULT_CLASS_WEIGHTS = {FIX: 1.0, BREAK: 2.0, UNCHANGED: 1.0}


def class_weights_tensor(weights: dict | None = None, device="cpu") -> torch.Tensor:
    weights = weights or DEFAULT_CLASS_WEIGHTS
    ordered = [float(weights.get(name, 1.0)) for name in CLASS_ORDER]
    return torch.tensor(ordered, dtype=torch.float32, device=device)


class WeightedActionLoss(nn.Module):
    """Cross-entropy over (sample, block) pairs with class weights."""

    def __init__(self, weights: dict | None = None):
        super().__init__()
        self.register_buffer("class_weights", class_weights_tensor(weights), persistent=False)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """logits: (B, n_blocks, 3); targets: (B, n_blocks) class indices."""
        b, n_blocks, n_classes = logits.shape
        flat = logits.reshape(b * n_blocks, n_classes)
        flat_targets = targets.reshape(b * n_blocks)
        weight = self.class_weights.to(flat.device)
        return nn.functional.cross_entropy(flat, flat_targets, weight=weight)

