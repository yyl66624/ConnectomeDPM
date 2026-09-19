from connectomedpm.data.task_registry import (
    ALL_TASK_FAMILIES, OOD_TASK_FAMILIES, build_task_suite,
)
from connectomedpm.supervision.scorer import score_answer


def test_suite_is_deterministic():
    a = build_task_suite(seed=0, per_template=3)
    b = build_task_suite(seed=0, per_template=3)
    assert [s.to_dict() for s in a] == [s.to_dict() for s in b]


def test_suite_changes_with_seed():
    a = build_task_suite(seed=0, per_template=3)
    b = build_task_suite(seed=1, per_template=3)
    assert [s.prompt for s in a] != [s.prompt for s in b]


def test_every_family_present_and_ood_disjoint():
    samples = build_task_suite(seed=0, per_template=2)
    families = {s.task_family for s in samples}
    assert families == set(ALL_TASK_FAMILIES)
    ood = {s.task_family for s in samples if s.task_family in OOD_TASK_FAMILIES}
    assert ood == set(OOD_TASK_FAMILIES)


def test_gold_answers_score_as_correct():
    """Every generated sample must be scorable: the gold answer marked correct against itself."""
    samples = build_task_suite(seed=0, per_template=2)
    for sample in samples:
        ok, score = score_answer(sample.answer, sample.answer, sample.task_family)
        assert ok and score == 1.0, (sample.sample_id, sample.answer)


def test_sample_ids_unique():
    samples = build_task_suite(seed=0, per_template=3)
    ids = [s.sample_id for s in samples]
    assert len(ids) == len(set(ids))

