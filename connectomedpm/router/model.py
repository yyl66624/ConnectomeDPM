"""TopologyRouter: input projection + fixed graph propagation + action head.

Everything except the two small heads is fixed by the graph. Two routers are therefore
identical if and only if their topologies are identical, which is what the null comparison
needs (DEVELOPMENT.md section 12).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .action_head import ActionHead
from .input_projector import InputProjector
from .transfer_operator import FixedTransferOperator


class TopologyRouter(nn.Module):
    def __init__(
        self,
        *,
        in_dim: int,
        n_blocks: int,
        transition_in: np.ndarray,
        transition_out: np.ndarray,
        steps: int = 2,
        hidden: int | None = 128,
        dropout: float = 0.0,
        node_features: np.ndarray | None = None,
        feature_names: list[str] | None = None,
        use_node_features: bool = False,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.operator = FixedTransferOperator(
            transition_in,
            transition_out,
            steps=steps,
            node_features=node_features,
            feature_names=feature_names,
            use_node_features=use_node_features,
        )
        self.projector = InputProjector(
            in_dim, self.operator.n_nodes, hidden=None, temperature=temperature
        )
        self.head = ActionHead(
            self.operator.output_dim, n_blocks, hidden=hidden, dropout=dropout
        )

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        logits = self.projector.logits(h)
        bias = self.operator.node_bias_vector()
        if bias is not None:
            logits = logits + bias
        s = torch.softmax(logits, dim=-1)
        z = self.operator(s)
        return self.head(z)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.logits(h)

    def node_distribution(self, h: torch.Tensor) -> torch.Tensor:
        return self.projector(h)

    def probabilities(self, h: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.logits(h), dim=-1)

