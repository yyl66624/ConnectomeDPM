"""Canonical data types. `to_dict` output is the on-disk contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Sample:
    """One request.

    `template_family` is the *grouping* unit for splits and for clustered bootstrap: two
    samples generated from the same template are never treated as independent evidence, and
    are never allowed to straddle a train/test boundary.
    """

    sample_id: str
    prompt: str
    answer: str
    task_family: str
    template_family: str
    split: str = "unassigned"
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, row: dict) -> "Sample":
        return cls(
            sample_id=row["sample_id"],
            prompt=row["prompt"],
            answer=row["answer"],
            task_family=row["task_family"],
            template_family=row["template_family"],
            split=row.get("split", "unassigned"),
            metadata=row.get("metadata", {}) or {},
        )


@dataclass(frozen=True)
class ActionOutcome:
    """Outcome of one (sample, action) evaluation, where action is `base` or a block id."""

    sample_id: str
    action: str
    correct: bool
    score: float
    raw_output: str = ""
    outcome: str = ""
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class CandidateRecord:
    """One row of the candidate outcome cache (DEVELOPMENT.md section 7)."""

    sample_id: str
    split: str
    task_family: str
    template_family: str
    base_correct: bool
    actions: dict[str, dict]
    metadata: dict = field(default_factory=dict)

    def label(self, block_id: str) -> str:
        """Return FIX / BREAK / UNCHANGED for `block_id` relative to base."""
        row = self.actions[block_id]
        if not self.base_correct and row["correct"]:
            return "FIX"
        if self.base_correct and not row["correct"]:
            return "BREAK"
        return "UNCHANGED"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, row: dict) -> "CandidateRecord":
        return cls(
            sample_id=row["sample_id"],
            split=row["split"],
            task_family=row["task_family"],
            template_family=row["template_family"],
            base_correct=bool(row["base_correct"]),
            actions=row["actions"],
            metadata=row.get("metadata", {}) or {},
        )


@dataclass
class GraphArtifact:
    """Everything the router needs from a topology, plus the audit trail for it."""

    graph_id: str
    graph_type: str
    node_ids: list[str]
    adjacency: Any
    transition_in: Any
    transition_out: Any
    node_features: Any
    feature_names: list[str]
    stats: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    mapping: dict = field(default_factory=dict)
