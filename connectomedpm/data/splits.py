"""Frozen data partitions (DEVELOPMENT.md section 4).

The partitions are disjoint at the level of **template family**, so no rewrite of the same
template can straddle a train/test boundary. OOD is defined by *unseen task family*, not by a
random hold-out.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict

from ..utils.hash import stable_hash
from .schemas import Sample
from .task_registry import ALL_TASK_FAMILIES, OOD_TASK_FAMILIES

# Target share of template families per split, for the in-distribution families.
ID_SPLIT_RATIOS: dict[str, float] = {
    "D_block": 0.20,
    "D_router_train": 0.30,
    "D_router_dev": 0.10,
    "D_cal": 0.10,
    "D_test_ID": 0.30,
}

ALL_SPLITS: tuple[str, ...] = tuple(ID_SPLIT_RATIOS) + ("D_test_OOD",)


def _assign_families_to_splits(
    templates_by_family: dict[str, list[str]],
    ratios: dict[str, float],
    seed: int,
) -> dict[str, list[str]]:
    """Proportional allocation of template families to splits, with coverage guarantees.

    Allocation is per task family, so every split sees every family it can. Two constraints
    are enforced on top of the proportional quotas, because both the block bank and the
    router need training data:

    * every family contributes at least one template to `D_block`;
    * families with >= 2 templates also contribute at least one to `D_router_train`,
      with `D_test_ID` protected once a family has >= 5 templates.

    Without these guarantees a family with only three templates can be absorbed entirely by
    the evaluation splits and end up with no trainable block.
    """
    rng = random.Random(seed)
    assignment: dict[str, list[str]] = {name: [] for name in ratios}
    names = list(ratios)

    for family in sorted(templates_by_family):
        templates = sorted(templates_by_family[family])
        rng.shuffle(templates)
        n = len(templates)
        if n == 0:
            continue

        quota = {name: ratios[name] * n for name in names}
        take = {name: int(quota[name]) for name in names}

        if n >= 1:
            take["D_block"] = max(take["D_block"], 1)
        if n >= 2:
            take["D_router_train"] = max(take["D_router_train"], 1)
        if n >= 5:
            take["D_test_ID"] = max(take["D_test_ID"], 1)

        # Never hand out more templates than the family has.
        while sum(take.values()) > n:
            name = max(names, key=lambda k: (take[k] - quota[k], take[k]))
            if take[name] == 0:
                break
            take[name] -= 1

        # Distribute the remainder to whoever is furthest below its quota.
        remainder = n - sum(take.values())
        while remainder > 0:
            name = max(names, key=lambda k: quota[k] - take[k])
            take[name] += 1
            remainder -= 1

        cursor = 0
        for name in names:
            assignment[name].extend(templates[cursor:cursor + take[name]])
            cursor += take[name]
    return assignment


def build_splits(samples: list[Sample], seed: int = 0) -> dict[str, list[Sample]]:
    """Assign every sample to exactly one frozen split."""
    id_families = [f for f in ALL_TASK_FAMILIES if f not in OOD_TASK_FAMILIES]

    templates_by_family: dict[str, list[str]] = defaultdict(list)
    for s in samples:
        if s.task_family in id_families:
            if s.template_family not in templates_by_family[s.task_family]:
                templates_by_family[s.task_family].append(s.template_family)

    assignment = _assign_families_to_splits(templates_by_family, ID_SPLIT_RATIOS, seed)
    template_to_split: dict[str, str] = {}
    for split_name, templates in assignment.items():
        for tpl in templates:
            template_to_split[tpl] = split_name

    splits: dict[str, list[Sample]] = {name: [] for name in ALL_SPLITS}
    for s in samples:
        if s.task_family in OOD_TASK_FAMILIES:
            split_name = "D_test_OOD"
        else:
            split_name = template_to_split[s.template_family]
        splits[split_name].append(
            Sample(
                sample_id=s.sample_id, prompt=s.prompt, answer=s.answer,
                task_family=s.task_family, template_family=s.template_family,
                split=split_name, metadata=s.metadata,
            )
        )
    for rows in splits.values():
        rows.sort(key=lambda r: r.sample_id)
    return splits


def split_summary(splits: dict[str, list[Sample]]) -> dict:
    """Counts + hashes suitable for a manifest."""
    summary: dict = {"splits": {}, "split_seed_hash": None}
    for name, rows in splits.items():
        summary["splits"][name] = {
            "n_samples": len(rows),
            "n_template_families": len({r.template_family for r in rows}),
            "task_families": dict(sorted(Counter(r.task_family for r in rows).items())),
            "sample_ids_hash": stable_hash([r.sample_id for r in rows], length=32),
        }
    summary["split_seed_hash"] = stable_hash(
        {name: sorted({r.template_family for r in rows}) for name, rows in splits.items()},
        length=32,
    )
    return summary


def flatten(splits: dict[str, list[Sample]]) -> list[Sample]:
    return [s for rows in splits.values() for s in rows]
