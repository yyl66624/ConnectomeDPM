"""Request features h_x.

The router's input projector consumes a single vector per request. We use the mean of the
last-hidden-state over the prompt tokens, computed with the base model *without* any block
active, so the feature depends on the request only.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from ..utils.hash import stable_hash
from .loader import Backbone, build_prompt


@torch.no_grad()
def extract_features(
    backbone: Backbone,
    prompts: Sequence[str],
    *,
    batch_size: int = 32,
    pooling: str = "mean",
) -> np.ndarray:
    """Return an (N, hidden_size) float32 array of request features."""
    model, tokenizer = backbone.model, backbone.tokenizer
    texts = [build_prompt(tokenizer, p, backbone.use_chat_template) for p in prompts]
    chunks: list[np.ndarray] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, add_special_tokens=False)
        enc = {k: v.to(backbone.device) for k, v in enc.items()}
        out = model(**enc, output_hidden_states=True)
        hidden = out.hidden_states[-1]                      # (B, T, d)
        mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        if pooling == "mean":
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        elif pooling == "last":
            lengths = mask.sum(dim=1).long() - 1
            pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), lengths]
        else:
            raise ValueError(f"unknown pooling {pooling!r}")
        chunks.append(pooled.float().cpu().numpy())

    return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 0), dtype=np.float32)


def feature_hash(features: np.ndarray, sample_ids: Sequence[str]) -> str:
    """Cheap identity for a feature matrix, stored in the routing trace and cache key."""
    rounded = np.round(np.asarray(features, dtype=np.float32), 5)
    return stable_hash(
        {"ids": list(sample_ids), "shape": list(rounded.shape),
         "sum": float(rounded.sum()), "abs": float(np.abs(rounded).sum())},
        length=32,
    )
