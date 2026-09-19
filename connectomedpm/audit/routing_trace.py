"""Sample-level routing traces (DEVELOPMENT.md section 23).

A run never publishes only averages. Every request records what the router saw, what it
chose, and what actually happened under the shared candidate cache.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..data.schemas import CandidateRecord
from ..router.inference import BASE_ACTION


def build_trace(
    records: Sequence[CandidateRecord],
    *,
    graph_id: str,
    block_ids: Sequence[str],
    scores: np.ndarray,
    p_fix: np.ndarray,
    p_break: np.ndarray,
    chosen: Sequence[str],
    intervened: Sequence[bool],
    coverage_rank: Sequence[int],
    threshold: float,
    request_feature_hash: str = "",
    latencies_ms: Sequence[float] | None = None,
) -> list[dict]:
    rows: list[dict] = []
    for i, record in enumerate(records):
        selected = chosen[i]
        if selected == BASE_ACTION:
            actual = "UNCHANGED"
        else:
            actual = record.label(selected)
        rows.append(
            {
                "sample_id": record.sample_id,
                "split": record.split,
                "task_family": record.task_family,
                "template_family": record.template_family,
                "base_correct": bool(record.base_correct),
                "request_feature_hash": request_feature_hash,
                "graph_id": graph_id,
                "router_score_per_block": {
                    b: float(scores[i, j]) for j, b in enumerate(block_ids)
                },
                "predicted_fix_per_block": {
                    b: float(p_fix[i, j]) for j, b in enumerate(block_ids)
                },
                "predicted_break_per_block": {
                    b: float(p_break[i, j]) for j, b in enumerate(block_ids)
                },
                "selected_action": selected,
                "fallback_or_intervene": "intervene" if intervened[i] else "fallback",
                "selected_block": None if selected == BASE_ACTION else selected,
                "actual_outcome": actual,
                "coverage_rank": int(coverage_rank[i]),
                "threshold": float(threshold),
                "latency_ms": None if latencies_ms is None else float(latencies_ms[i]),
            }
        )
    return rows


def trace_summary(trace: Sequence[dict]) -> dict:
    """Compact summary used in reports (the full trace stays on disk)."""
    n = len(trace)
    if n == 0:
        return {"n": 0}
    intervened = sum(1 for r in trace if r["fallback_or_intervene"] == "intervene")
    fixes = sum(1 for r in trace if r["actual_outcome"] == "FIX"
                and r["fallback_or_intervene"] == "intervene")
    breaks = sum(1 for r in trace if r["actual_outcome"] == "BREAK"
                 and r["fallback_or_intervene"] == "intervene")
    return {
        "n": n,
        "intervened": intervened,
        "coverage": intervened / n,
        "fixes": fixes,
        "breaks": breaks,
        "net_repair": fixes - breaks,
    }

