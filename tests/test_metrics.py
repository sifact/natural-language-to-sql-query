"""Tests for evaluated-task denominators and exact accuracy."""

from fractions import Fraction

from nl_sql.evaluation.metrics import calculate_metrics


def test_given_mixed_task_statuses_when_calculated_then_excludes_ineligible_tasks() -> (
    None
):
    metrics = calculate_metrics(
        ["passed", "failed", "passed", "incomplete", "unsupported", "skipped"]
    )

    assert metrics.evaluated_tasks == 3
    assert metrics.passed_tasks == 2
    assert metrics.failed_tasks == 1
    assert metrics.incomplete_tasks == 1
    assert metrics.unsupported_tasks == 1
    assert metrics.skipped_tasks == 1
    assert metrics.accuracy == Fraction(2, 3)


def test_given_no_comparable_results_when_calculated_then_accuracy_is_unavailable() -> (
    None
):
    metrics = calculate_metrics(["incomplete", "unsupported", "skipped"])

    assert metrics.evaluated_tasks == 0
    assert metrics.accuracy is None
