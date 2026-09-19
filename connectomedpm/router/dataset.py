"""Turn request features + candidate outcomes into router training tensors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from ..data.schemas import CandidateRecord
from ..supervision.outcome_labeler import label_index


@dataclass
class RouterTensors:
    h: torch.Tensor          # (N, d) request features, standardised
    y: torch.Tensor          # (N, n_blocks) class indices
    base_correct: torch.Tensor  # (N,) bool
    sample_ids: list[str]
    block_ids: list[str]


@dataclass
class FeatureScaler:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, features: np.ndarray) -> "FeatureScaler":
        features = np.asarray(features, dtype=np.float64)
        mean = features.mean(axis=0)
        std = features.std(axis=0)
        std = np.where(std > 1e-6, std, 1.0)
        return cls(mean=mean, std=std)

    def transform(self, features: np.ndarray) -> np.ndarray:
        return ((np.asarray(features, dtype=np.float64) - self.mean) / self.std).astype(np.float32)

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, payload: dict) -> "FeatureScaler":
        return cls(mean=np.asarray(payload["mean"], dtype=np.float64),
                   std=np.asarray(payload["std"], dtype=np.float64))


def build_tensors(
    features: np.ndarray,
    records: Sequence[CandidateRecord],
    block_ids: Sequence[str],
    scaler: FeatureScaler | None = None,
) -> tuple[RouterTensors, FeatureScaler]:
    """Build (h, y) tensors. `scaler` is fitted on this split when not supplied."""
    if len(features) != len(records):
        raise ValueError(f"features {len(features)} != records {len(records)}")
    if scaler is None:
        scaler = FeatureScaler.fit(features)
    h = scaler.transform(features)

    y = np.zeros((len(records), len(block_ids)), dtype=np.int64)
    base = np.zeros(len(records), dtype=bool)
    for i, record in enumerate(records):
        base[i] = record.base_correct
        for j, block_id in enumerate(block_ids):
            y[i, j] = label_index(record.label(block_id))

    return (
        RouterTensors(
            h=torch.from_numpy(h),
            y=torch.from_numpy(y),
            base_correct=torch.from_numpy(base),
            sample_ids=[r.sample_id for r in records],
            block_ids=list(block_ids),
        ),
        scaler,
    )


def class_balance(y: torch.Tensor, n_classes: int = 3) -> dict[int, int]:
    counts = torch.bincount(y.reshape(-1), minlength=n_classes)
    return {i: int(c) for i, c in enumerate(counts)}

