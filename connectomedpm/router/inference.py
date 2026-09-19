"""Routing at inference time (DEVELOPMENT.md section 9, Action Head + Fallback).

    S_b = P(FIX)_b - lambda * P(BREAK)_b - gamma * C_b

`gamma = 0` in the first version: cost-based selection is deliberately out of scope until
the quality signal itself is understood.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from .action_head import CLASS_ORDER
from .model import TopologyRouter

FIX_INDEX = CLASS_ORDER.index("FIX")
BREAK_INDEX = CLASS_ORDER.index("BREAK")
BASE_ACTION = "base"


@dataclass
class RoutingResult:
    scores: np.ndarray          # (N, n_blocks) S_b
    p_fix: np.ndarray           # (N, n_blocks)
    p_break: np.ndarray         # (N, n_blocks)
    chosen: list[str]           # block id or "base"
    intervened: np.ndarray      # (N,) bool
    coverage_rank: np.ndarray   # (N,) rank by descending best score


@torch.no_grad()
def score_blocks(
    model: TopologyRouter,
    features: np.ndarray,
    block_ids: Sequence[str],
    *,
    lam: float = 1.0,
    gamma: float = 0.0,
    costs: np.ndarray | None = None,
    batch_size: int = 4096,
    device: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (S, P_fix, P_break) each shaped (N, n_blocks)."""
    dev = torch.device(device) if device else next(model.parameters()).device
    model.eval()
    probs_all: list[np.ndarray] = []
    for start in range(0, len(features), batch_size):
        batch = torch.as_tensor(features[start:start + batch_size], dtype=torch.float32, device=dev)
        probs_all.append(torch.softmax(model(batch), dim=-1).float().cpu().numpy())
    probs = np.concatenate(probs_all, axis=0) if probs_all else np.zeros((0, len(block_ids), 3))

    p_fix = probs[:, :, FIX_INDEX]
    p_break = probs[:, :, BREAK_INDEX]
    scores = p_fix - lam * p_break
    if gamma and costs is not None:
        scores = scores - gamma * np.asarray(costs)[None, :]
    return scores, p_fix, p_break


def select_actions(
    scores: np.ndarray,
    block_ids: Sequence[str],
    *,
    threshold: float = 0.0,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Pick the best block per request, or fall back to `base` below `threshold`.

    Returns (chosen, intervened, coverage_rank) where coverage_rank is the descending rank of
    each request's best score (0 = most confident intervention).
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size == 0:
        return [], np.zeros(0, dtype=bool), np.zeros(0, dtype=np.int64)
    best_idx = scores.argmax(axis=1)
    best_score = scores[np.arange(scores.shape[0]), best_idx]
    intervened = best_score >= threshold
    chosen = [block_ids[i] if intervene else BASE_ACTION
              for i, intervene in zip(best_idx, intervened)]
    order = np.argsort(-best_score, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    return chosen, intervened, rank


def route(
    model: TopologyRouter,
    features: np.ndarray,
    block_ids: Sequence[str],
    *,
    threshold: float = 0.0,
    lam: float = 1.0,
    gamma: float = 0.0,
    costs: np.ndarray | None = None,
) -> RoutingResult:
    scores, p_fix, p_break = score_blocks(
        model, features, block_ids, lam=lam, gamma=gamma, costs=costs
    )
    chosen, intervened, coverage_rank = select_actions(scores, block_ids, threshold=threshold)
    return RoutingResult(
        scores=scores, p_fix=p_fix, p_break=p_break,
        chosen=chosen, intervened=intervened, coverage_rank=coverage_rank,
    )
