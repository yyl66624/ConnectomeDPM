"""A single low-rank parameter-correction block.

    DeltaW = scale * A @ B        A: (in_features, rank), B: (rank, out_features)
    y      = W x + DeltaW x

This is the LoRA parameterisation, but the *use* is different from LoRA: blocks are trained
on individual, verifiable error families and are never merged. At inference at most one block
is active (DEVELOPMENT.md section 6).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn


@dataclass
class BlockConfig:
    block_id: str
    error_family: str
    layer_id: int
    module_name: str
    in_features: int
    out_features: int
    rank: int = 8
    scale: float = 1.0
    base_model: str = ""
    base_model_revision: str = ""
    training_seed: int = 0
    train_split_hash: str = ""
    validation_split_hash: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class ParameterBlock(nn.Module):
    """Low-rank correction block. `B` is zero-initialised so a fresh block is a no-op."""

    def __init__(self, config: BlockConfig):
        super().__init__()
        self.config = config
        self.block_id = config.block_id
        self.error_family = config.error_family
        self.layer_id = int(config.layer_id)
        self.module_name = config.module_name
        self.rank = int(config.rank)
        self.scale = float(config.scale)

        self.A = nn.Parameter(torch.empty(config.in_features, self.rank))
        self.B = nn.Parameter(torch.zeros(self.rank, config.out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    @property
    def in_features(self) -> int:
        return self.A.shape[0]

    @property
    def out_features(self) -> int:
        return self.B.shape[1]

    def delta_weight(self) -> torch.Tensor:
        """Materialised (out_features, in_features) update; used by audits, not by forward."""
        return self.scale * (self.B.transpose(0, 1) @ self.A.transpose(0, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != self.A.dtype:
            x = x.to(self.A.dtype)
        return self.scale * ((x @ self.A) @ self.B)

    def extra_repr(self) -> str:
        return (f"block_id={self.block_id}, family={self.error_family}, "
                f"layer={self.layer_id}/{self.module_name}, rank={self.rank}, scale={self.scale}")


def make_block(
    *,
    block_id: str,
    error_family: str,
    layer_id: int,
    module_name: str,
    in_features: int,
    out_features: int,
    rank: int = 8,
    scale: float = 1.0,
    base_model: str = "",
    base_model_revision: str = "",
    training_seed: int = 0,
) -> ParameterBlock:
    return ParameterBlock(
        BlockConfig(
            block_id=block_id,
            error_family=error_family,
            layer_id=int(layer_id),
            module_name=module_name,
            in_features=int(in_features),
            out_features=int(out_features),
            rank=int(rank),
            scale=float(scale),
            base_model=base_model,
            base_model_revision=base_model_revision,
            training_seed=int(training_seed),
        )
    )

