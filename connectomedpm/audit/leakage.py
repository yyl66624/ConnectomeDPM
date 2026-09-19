"""Leakage audits (DEVELOPMENT.md section 24).

These run before a formal experiment and fail loudly rather than silently inflating a result.
"""

from __future__ import annotations

from typing import Sequence

from ..data.dedup import exact_duplicate_groups, prompt_split_leakage, template_split_leakage
from ..data.schemas import Sample


def audit_splits(splits: dict[str, list[Sample]], *, near_duplicate_threshold: float = 0.9) -> dict:
    """Return a structured report; `ok` is False if any hard constraint is violated."""
    template_leaks = template_split_leakage(splits)
    prompt_leaks = prompt_split_leakage(splits, threshold=near_duplicate_threshold)
    dup_groups = exact_duplicate_groups([s for rows in splits.values() for s in rows])

    violations: list[str] = []
    if template_leaks:
        violations.append(f"template families span splits: {len(template_leaks)}")
    exact_prompt_leaks = [f for f in prompt_leaks if f["kind"] == "exact"]
    if exact_prompt_leaks:
        violations.append(f"exact prompt overlap across splits: {len(exact_prompt_leaks)} pairs")

    return {
        "ok": not violations,
        "violations": violations,
        "template_leaks": template_leaks,
        "prompt_leaks": prompt_leaks,
        "exact_duplicate_groups": len(dup_groups),
        "near_duplicate_threshold": near_duplicate_threshold,
    }


def audit_test_freeze(
    *,
    graph_built_before_test: bool,
    threshold_learned_on: str,
    router_selected_on: str,
    blocks_selected_on: str,
) -> dict:
    """Check the explicit prohibitions in DEVELOPMENT.md section 24."""
    violations: list[str] = []
    if not graph_built_before_test:
        violations.append("graph construction used test data")
    for name, value in (("threshold", threshold_learned_on),
                        ("router", router_selected_on),
                        ("blocks", blocks_selected_on)):
        if value.startswith("D_test"):
            violations.append(f"{name} was chosen on {value}")
    return {"ok": not violations, "violations": violations}
