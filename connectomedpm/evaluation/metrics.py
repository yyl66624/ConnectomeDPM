"""Core metrics (DEVELOPMENT.md section 18).

Every metric keeps its numerator and denominator, because a bare average is not evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..data.schemas import CandidateRecord
from ..router.inference import BASE_ACTION


@dataclass
class Decisions:
    """Per-sample decisions resolved against the shared candidate cache."""

    sample_id: list[str]
    template_family: list[str]
    task_family: list[str]
    base_correct: np.ndarray
    intervened: np.ndarray
    chosen_block: list[str]
    chosen_correct: np.ndarray

    def __len__(self) -> int:
        return len(self.sample_id)


def resolve_decisions(
    records: Sequence[CandidateRecord],
    chosen: Sequence[str],
    intervened: Sequence[bool] | None = None,
) -> Decisions:
    """Turn (cache, selected action) into a decision table."""
    if len(records) != len(chosen):
        raise ValueError(f"records {len(records)} != chosen {len(chosen)}")
    base_correct = np.array([r.base_correct for r in records], dtype=bool)
    chosen_correct = np.zeros(len(records), dtype=bool)
    if intervened is None:
        intervened_arr = np.array([c != BASE_ACTION for c in chosen], dtype=bool)
    else:
        intervened_arr = np.asarray(intervened, dtype=bool)

    for i, (record, action) in enumerate(zip(records, chosen)):
        if action == BASE_ACTION:
            chosen_correct[i] = record.base_correct
        else:
            if action not in record.actions:
                raise KeyError(f"block {action!r} missing from cache for sample {record.sample_id}")
            chosen_correct[i] = bool(record.actions[action]["correct"])

    return Decisions(
        sample_id=[r.sample_id for r in records],
        template_family=[r.template_family for r in records],
        task_family=[r.task_family for r in records],
        base_correct=base_correct,
        intervened=intervened_arr,
        chosen_block=list(chosen),
        chosen_correct=chosen_correct,
    )


def compute_metrics(d: Decisions) -> dict:
    """All section-18 metrics with explicit numerators and denominators."""
    n = len(d)
    if n == 0:
        return {"n": 0}

    base_correct = d.base_correct
    intervened = d.intervened
    chosen_correct = d.chosen_correct

    base_correct_intervened = intervened & base_correct
    base_wrong = ~base_correct
    base_wrong_intervened = intervened & base_wrong

    fixes = int((base_wrong_intervened & chosen_correct).sum())
    breaks = int((base_correct_intervened & ~chosen_correct).sum())
    intervened_n = int(intervened.sum())

    final_correct = np.where(intervened, chosen_correct, base_correct)
    base_acc = float(base_correct.mean())
    final_acc = float(final_correct.mean())

    return {
        "n": int(n),
        "intervened": intervened_n,
        "coverage": intervened_n / n,
        "fixes": fixes,
        "breaks": breaks,
        "net_repair": fixes - breaks,
        "net_repair_per_1000": 1000.0 * (fixes - breaks) / n,
        "base_accuracy": base_acc,
        "final_accuracy": final_acc,
        "total_accuracy_delta": final_acc - base_acc,
        "fix_rate": fixes / max(int(base_wrong.sum()), 1),
        "fix_rate_denominator": int(base_wrong.sum()),
        "overall_break_rate": breaks / max(int(base_correct.sum()), 1),
        "overall_break_rate_denominator": int(base_correct.sum()),
        "selective_break_risk": breaks / max(int(base_correct_intervened.sum()), 1),
        "selective_break_risk_denominator": int(base_correct_intervened.sum()),
        "action_distribution": _action_distribution(d),
    }


def _action_distribution(d: Decisions) -> dict:
    out: dict[str, int] = {}
    for action in d.chosen_block:
        out[action] = out.get(action, 0) + 1
    return dict(sorted(out.items()))


def oracle_metrics(records: Sequence[CandidateRecord]) -> dict:
    """Upper bound: per sample, take the best block if it beats base."""
    block_ids = sorted({b for r in records for b in r.actions if b != BASE_ACTION})
    chosen = []
    for record in records:
        best = BASE_ACTION
        if not record.base_correct:
            for block_id in block_ids:
                if record.actions[block_id]["correct"]:
                    best = block_id
                    break
        chosen.append(best)
    return compute_metrics(resolve_decisions(records, chosen))


def metrics_by_family(d: Decisions) -> dict[str, dict]:
    """Per task-family breakdown, so a gain cannot hide inside one family."""
    out: dict[str, dict] = {}
    families = sorted(set(d.task_family))
    arr = np.array(d.task_family)
    for family in families:
        mask = arr == family
        sub = Decisions(
            sample_id=[s for s, keep in zip(d.sample_id, mask) if keep],
            template_family=[s for s, keep in zip(d.template_family, mask) if keep],
            task_family=[s for s, keep in zip(d.task_family, mask) if keep],
            base_correct=d.base_correct[mask],
            intervened=d.intervened[mask],
            chosen_block=[s for s, keep in zip(d.chosen_block, mask) if keep],
            chosen_correct=d.chosen_correct[mask],
        )
        out[family] = compute_metrics(sub)
    return out

