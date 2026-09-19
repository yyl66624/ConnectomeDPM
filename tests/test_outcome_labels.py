from connectomedpm.supervision.outcome_labeler import (
    BREAK, FIX, UNCHANGED, label_from_index, label_index, label_outcome,
)


def test_label_semantics():
    assert label_outcome(False, True) == FIX
    assert label_outcome(True, False) == BREAK
    assert label_outcome(True, True) == UNCHANGED
    assert label_outcome(False, False) == UNCHANGED


def test_label_index_roundtrip():
    for label in (FIX, BREAK, UNCHANGED):
        assert label_from_index(label_index(label)) == label

