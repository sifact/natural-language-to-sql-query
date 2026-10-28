"""Aggregate task outcomes without scoring tasks that were not evaluated."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

TaskStatus = Literal["passed", "failed", "incomplete", "unsupported", "skipped"]


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    """Counts and exact accuracy over tasks that completed comparison."""

    evaluated_tasks: int
    passed_tasks: int
    failed_tasks: int
    incomplete_tasks: int
    unsupported_tasks: int
    skipped_tasks: int
    accuracy: Fraction | None


def calculate_metrics(statuses: Iterable[TaskStatus]) -> EvaluationMetrics:
    """Calculate accuracy over passed and failed tasks only."""
    counts = {
        "passed": 0,
        "failed": 0,
        "incomplete": 0,
        "unsupported": 0,
        "skipped": 0,
    }
    for status in statuses:
        if status not in counts:
            raise ValueError("Task status is outside the evaluation contract.")
        counts[status] += 1

    evaluated = counts["passed"] + counts["failed"]
    accuracy = Fraction(counts["passed"], evaluated) if evaluated else None
    return EvaluationMetrics(
        evaluated_tasks=evaluated,
        passed_tasks=counts["passed"],
        failed_tasks=counts["failed"],
        incomplete_tasks=counts["incomplete"],
        unsupported_tasks=counts["unsupported"],
        skipped_tasks=counts["skipped"],
        accuracy=accuracy,
    )
