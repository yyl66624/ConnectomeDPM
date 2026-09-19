import numpy as np

from connectomedpm.data.schemas import CandidateRecord
from connectomedpm.evaluation.fixed_coverage import evaluate_fixed_coverage, top_k_mask
from connectomedpm.evaluation.metrics import compute_metrics, resolve_decisions
from connectomedpm.router.inference import BASE_ACTION


def _record(sid, base_correct, block_results):
    actions = {"base": {"correct": base_correct, "score": float(base_correct)}}
    for block, correct in block_results.items():
        actions[block] = {"correct": correct, "score": float(correct)}
    return CandidateRecord(
        sample_id=sid, split="D_test_ID", task_family="toy", template_family="t",
        base_correct=base_correct, actions=actions,
    )


def test_denominators_are_explicit():
    records = [
        _record("a", False, {"b0": True}),    # FIX
        _record("b", True, {"b0": False}),    # BREAK
        _record("c", True, {"b0": True}),     # UNCHANGED
        _record("d", False, {"b0": False}),   # UNCHANGED
    ]
    decisions = resolve_decisions(records, ["b0", "b0", BASE_ACTION, BASE_ACTION])
    metrics = compute_metrics(decisions)
    assert metrics["n"] == 4
    assert metrics["intervened"] == 2
    assert metrics["coverage"] == 0.5
    assert metrics["fixes"] == 1
    assert metrics["breaks"] == 1
    assert metrics["net_repair"] == 0
    assert metrics["fix_rate_denominator"] == 2          # base-wrong samples
    assert metrics["overall_break_rate_denominator"] == 2  # base-correct samples
    assert metrics["base_accuracy"] == 0.5
    assert metrics["final_accuracy"] == 0.5


def test_net_repair_per_1000_scales_with_n():
    records = [_record(f"s{i}", False, {"b0": True}) for i in range(10)]
    decisions = resolve_decisions(records, ["b0"] * 10)
    metrics = compute_metrics(decisions)
    assert metrics["net_repair"] == 10
    assert metrics["net_repair_per_1000"] == 1000.0


def test_fallback_uses_base_outcome():
    records = [_record("a", True, {"b0": False})]
    decisions = resolve_decisions(records, [BASE_ACTION])
    metrics = compute_metrics(decisions)
    assert metrics["breaks"] == 0
    assert metrics["final_accuracy"] == 1.0


def test_top_k_mask_selects_exact_count():
    scores = np.array([0.1, 0.9, 0.5, 0.3, 0.7])
    mask = top_k_mask(scores, 0.4)
    assert mask.sum() == 2
    assert mask[1] and mask[4]


def test_fixed_coverage_evaluation_uses_scores():
    records = [
        _record("a", False, {"b0": True}),
        _record("b", False, {"b0": False}),
    ]
    scores = np.array([[0.9], [0.1]])
    metrics = evaluate_fixed_coverage(records, scores, ["b0"], coverage=0.5)
    assert metrics["intervened"] == 1
    assert metrics["fixes"] == 1
    assert metrics["breaks"] == 0

