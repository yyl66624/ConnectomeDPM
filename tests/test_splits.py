from connectomedpm.audit.leakage import audit_splits
from connectomedpm.data.splits import ALL_SPLITS, build_splits, split_summary
from connectomedpm.data.task_registry import OOD_TASK_FAMILIES, build_task_suite


def _splits():
    return build_splits(build_task_suite(seed=0, per_template=4), seed=0)


def test_all_splits_present_and_non_empty():
    splits = _splits()
    for name in ALL_SPLITS:
        assert splits[name], f"{name} is empty"


def test_no_template_family_spans_splits():
    splits = _splits()
    owner = {}
    for name, rows in splits.items():
        for row in rows:
            owner.setdefault(row.template_family, set()).add(name)
    assert all(len(v) == 1 for v in owner.values())


def test_ood_only_contains_ood_families():
    splits = _splits()
    assert {s.task_family for s in splits["D_test_OOD"]} == set(OOD_TASK_FAMILIES)
    for name, rows in splits.items():
        if name == "D_test_OOD":
            continue
        assert not ({s.task_family for s in rows} & set(OOD_TASK_FAMILIES))


def test_every_sample_assigned_exactly_once():
    splits = _splits()
    ids = [s.sample_id for rows in splits.values() for s in rows]
    assert len(ids) == len(set(ids))
    assert all(s.split != "unassigned" for rows in splits.values() for s in rows)


def test_leakage_audit_passes():
    audit = audit_splits(_splits())
    assert audit["ok"], audit["violations"]
    assert audit["template_leaks"] == {}


def test_split_summary_hashes_stable():
    assert (split_summary(_splits())["split_seed_hash"]
            == split_summary(_splits())["split_seed_hash"])

