"""Compare SQL result sets with value-based column and row matching."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from ..config import Settings
from ..database.provider import QueryResult
from .normalize import (
    NormalizedRow,
    NormalizedValue,
    UnsupportedValueError,
    normalize_value,
    values_match,
)

MatchStatus = Literal["passed", "failed", "unsupported"]


@dataclass(frozen=True, slots=True)
class MatchResult:
    """Outcome and safe reason code from one result comparison."""

    status: MatchStatus
    reason_code: str | None
    column_mapping: tuple[int, ...] | None = None


def match_results(
    gold: QueryResult,
    predicted: QueryResult,
    settings: Settings | None = None,
) -> MatchResult:
    """Match gold columns and rows by normalized values, allowing extra predicted columns."""
    runtime = settings or Settings()
    if gold.truncated or predicted.truncated:
        return MatchResult("unsupported", "truncated_result")

    gold_rows = _normalized_rows(gold)
    if isinstance(gold_rows, str):
        return MatchResult("unsupported", gold_rows)
    predicted_rows = _normalized_rows(predicted)
    if isinstance(predicted_rows, str):
        return MatchResult("unsupported", predicted_rows)

    if len(gold_rows) != len(predicted_rows):
        return MatchResult("failed", "row_count_mismatch")
    if len(gold.columns) > len(predicted.columns):
        return MatchResult("failed", "predicted_column_count_too_small")

    try:
        gold_columns = _columns(gold_rows, len(gold.columns))
        predicted_columns = _columns(predicted_rows, len(predicted.columns))
        column_candidates = _column_candidates(
            gold_columns,
            predicted_columns,
            absolute_tolerance=runtime.numeric_absolute_tolerance,
            relative_tolerance=runtime.numeric_relative_tolerance,
        )
    except UnsupportedValueError as error:
        return MatchResult("unsupported", str(error))

    if any(not candidates for candidates in column_candidates):
        return MatchResult("failed", "column_values_mismatch")

    assigned: dict[int, int] = {}
    used_columns: set[int] = set()
    result_mapping: tuple[int, ...] | None = None

    def search(remaining: set[int]) -> bool:
        nonlocal result_mapping
        if not remaining:
            result_mapping = tuple(
                assigned[index] for index in range(len(gold.columns))
            )
            return True

        gold_index = min(
            remaining,
            key=lambda index: sum(
                candidate not in used_columns for candidate in column_candidates[index]
            ),
        )
        next_remaining = remaining - {gold_index}
        for predicted_index in column_candidates[gold_index]:
            if predicted_index in used_columns:
                continue
            assigned[gold_index] = predicted_index
            used_columns.add(predicted_index)
            if _rows_can_match(
                gold_rows,
                predicted_rows,
                assigned,
                absolute_tolerance=runtime.numeric_absolute_tolerance,
                relative_tolerance=runtime.numeric_relative_tolerance,
            ) and search(next_remaining):
                return True
            used_columns.remove(predicted_index)
            del assigned[gold_index]
        return False

    if not search(set(range(len(gold.columns)))):
        return MatchResult("failed", "row_values_mismatch")
    return MatchResult("passed", None, result_mapping)


def _normalized_rows(
    result: QueryResult,
) -> list[NormalizedRow] | str:
    columns = result.columns
    if not isinstance(columns, list) or any(
        not isinstance(name, str) for name in columns
    ):
        return "invalid_column_labels"

    if result.value_rows is not None:
        if len(result.value_rows) != len(result.rows):
            return "inconsistent_result_rows"
        raw_rows: Sequence[object] = result.value_rows
    else:
        raw_rows = result.rows

    if len(set(columns)) != len(columns) and result.value_rows is None:
        return "duplicate_column_labels"

    normalized_rows: list[NormalizedRow] = []
    try:
        for row in raw_rows:
            if isinstance(row, Mapping):
                if any(column not in row for column in columns):
                    return "missing_row_value"
                values = [row[column] for column in columns]
            elif isinstance(row, Sequence) and not isinstance(row, (str, bytes)):
                values = list(row)
            else:
                return "invalid_result_row"
            if len(values) != len(columns):
                return "result_row_width_mismatch"
            normalized_rows.append(tuple(normalize_value(value) for value in values))
    except UnsupportedValueError as error:
        return str(error)
    return normalized_rows


def _columns(
    rows: list[NormalizedRow],
    column_count: int,
) -> list[list[NormalizedValue]]:
    return [[row[column_index] for row in rows] for column_index in range(column_count)]


def _column_candidates(
    gold_columns: list[list[NormalizedValue]],
    predicted_columns: list[list[NormalizedValue]],
    *,
    absolute_tolerance: Decimal,
    relative_tolerance: Decimal,
) -> list[list[int]]:
    candidates: list[list[int]] = []
    for gold_column in gold_columns:
        matching_columns: list[int] = []
        for predicted_index, predicted_column in enumerate(predicted_columns):
            if _value_multisets_match(
                gold_column,
                predicted_column,
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
            ):
                matching_columns.append(predicted_index)
        candidates.append(matching_columns)
    return candidates


def _value_multisets_match(
    gold_values: list[NormalizedValue],
    predicted_values: list[NormalizedValue],
    *,
    absolute_tolerance: Decimal,
    relative_tolerance: Decimal,
) -> bool:
    if len(gold_values) != len(predicted_values):
        return False
    adjacency = [
        [
            predicted_index
            for predicted_index, predicted_value in enumerate(predicted_values)
            if values_match(
                gold_value,
                predicted_value,
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
            )
        ]
        for gold_value in gold_values
    ]
    return _has_perfect_matching(adjacency, len(predicted_values))


def _rows_can_match(
    gold_rows: list[NormalizedRow],
    predicted_rows: list[NormalizedRow],
    assigned_columns: Mapping[int, int],
    *,
    absolute_tolerance: Decimal,
    relative_tolerance: Decimal,
) -> bool:
    adjacency: list[list[int]] = []
    for gold_row in gold_rows:
        matching_rows: list[int] = []
        for predicted_index, predicted_row in enumerate(predicted_rows):
            if all(
                values_match(
                    gold_row[gold_column],
                    predicted_row[predicted_column],
                    absolute_tolerance=absolute_tolerance,
                    relative_tolerance=relative_tolerance,
                )
                for gold_column, predicted_column in assigned_columns.items()
            ):
                matching_rows.append(predicted_index)
        adjacency.append(matching_rows)
    return _has_perfect_matching(adjacency, len(predicted_rows))


def _has_perfect_matching(
    adjacency: list[list[int]],
    right_count: int,
) -> bool:
    if len(adjacency) > right_count:
        return False

    matched_left: dict[int, int] = {}
    matched_right: dict[int, int] = {}
    for start_left in range(len(adjacency)):
        queue = deque([start_left])
        seen_left = {start_left}
        seen_right: set[int] = set()
        parent_right: dict[int, int] = {}
        free_right: int | None = None

        while queue and free_right is None:
            left = queue.popleft()
            for right in adjacency[left]:
                if right in seen_right:
                    continue
                seen_right.add(right)
                parent_right[right] = left
                previous_left = matched_right.get(right)
                if previous_left is None:
                    free_right = right
                    break
                if previous_left not in seen_left:
                    seen_left.add(previous_left)
                    queue.append(previous_left)

        if free_right is None:
            return False

        right = free_right
        while True:
            left = parent_right[right]
            previous_right = matched_left.get(left)
            matched_left[left] = right
            matched_right[right] = left
            if previous_right is None:
                break
            del matched_right[previous_right]
            right = previous_right

    return True
