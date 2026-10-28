"""Offline tests for SQL result normalization and comparison."""

from datetime import UTC, date, datetime, time
from decimal import Decimal

import pytest

from nl_sql.config import Settings
from nl_sql.database import QueryResult
from nl_sql.evaluation import (
    MatchResult,
    UnsupportedValueError,
    match_results,
    normalize_value,
    values_match,
)


def _result(
    columns: list[str],
    rows: list[tuple[object, ...]],
    *,
    truncated: bool = False,
) -> QueryResult:
    return QueryResult(
        columns=columns,
        rows=[
            {column: value for column, value in zip(columns, row, strict=True)}
            for row in rows
        ],
        truncated=truncated,
        value_rows=tuple(rows),
    )


def test_given_strings_when_normalized_then_trims_and_casefolds() -> None:
    assert normalize_value("  Straße  ") == normalize_value("STRASSE")


@pytest.mark.parametrize(
    ("left", "right", "absolute", "relative", "expected"),
    [
        ("1.10", "1.2", "0.1", "0", True),
        ("100", "109", "0", "0.1", True),
        ("100", "112", "0", "0.1", False),
        ("100", "111.5", "1", "0.1", False),
        ("1", "1.01", "0", "0", False),
    ],
)
def test_given_numbers_when_compared_then_obeys_decimal_tolerance_boundaries(
    left: str,
    right: str,
    absolute: str,
    relative: str,
    expected: bool,
) -> None:
    settings = Settings(
        numeric_absolute_tolerance=Decimal(absolute),
        numeric_relative_tolerance=Decimal(relative),
    )

    outcome = match_results(
        _result(["gold"], [(Decimal(left),)]),
        _result(["prediction"], [(Decimal(right),)]),
        settings,
    )

    assert outcome.status == ("passed" if expected else "failed")


@pytest.mark.parametrize(
    ("gold", "predicted"),
    [(1, Decimal(1)), (0.1, Decimal("0.1"))],
)
def test_given_mixed_numeric_types_when_compared_then_match_as_decimal(
    gold: object,
    predicted: object,
) -> None:
    outcome = match_results(_result(["g"], [(gold,)]), _result(["p"], [(predicted,)]))

    assert outcome.status == "passed"


@pytest.mark.parametrize(
    ("gold", "predicted", "expected"),
    [
        (None, None, "passed"),
        (None, "null", "failed"),
        (None, 0, "failed"),
    ],
)
def test_given_null_values_when_compared_then_matches_only_null(
    gold: object,
    predicted: object,
    expected: str,
) -> None:
    outcome = match_results(_result(["g"], [(gold,)]), _result(["p"], [(predicted,)]))

    assert outcome.status == expected


def test_given_native_date_time_values_when_normalized_then_uses_iso_representation() -> (
    None
):
    normalized_date = normalize_value(date(2024, 2, 3))
    normalized_time = normalize_value(time(10, 30))
    normalized_datetime = normalize_value(datetime(2024, 2, 3, 10, 30, tzinfo=UTC))

    assert (normalized_date.kind, normalized_date.value) == ("date", "2024-02-03")
    assert (normalized_time.kind, normalized_time.value) == ("time", "10:30:00")
    assert (normalized_datetime.kind, normalized_datetime.value) == (
        "datetime",
        "2024-02-03T10:30:00Z",
    )


@pytest.mark.parametrize(
    "text",
    [
        "2024-02-03",
        "2024-02-03T10:30:00Z",
        "10:30:00",
        '{"label":"Name"}',
        "[1,2]",
    ],
)
def test_given_date_or_json_looking_text_when_normalized_then_remains_string(
    text: str,
) -> None:
    normalized = normalize_value(f"  {text}  ")

    assert normalized.kind == "string"
    assert normalized.value == text.casefold()


@pytest.mark.parametrize(
    ("text", "typed_value"),
    [
        ("2024-02-03", date(2024, 2, 3)),
        (
            "2024-02-03T10:30:00Z",
            datetime(2024, 2, 3, 10, 30, tzinfo=UTC),
        ),
        ('{"a":1}', {"a": 1}),
        ("[1,2]", [1, 2]),
    ],
)
def test_given_date_or_json_looking_text_when_compared_with_typed_value_then_fails(
    text: str,
    typed_value: object,
) -> None:
    outcome = match_results(
        _result(["gold"], [(text,)]),
        _result(["prediction"], [(typed_value,)]),
    )

    assert outcome.status == "failed"


def test_given_json_objects_when_compared_then_ignores_key_order_recursively() -> None:
    gold = normalize_value({"z": [1, {"label": " Name "}], "a": True})
    predicted = normalize_value({"a": True, "z": [1, {"label": "name"}]})

    assert values_match(
        gold,
        predicted,
        absolute_tolerance=Decimal(0),
        relative_tolerance=Decimal(0),
    )


def test_given_unsupported_cell_when_compared_then_returns_unsupported() -> None:
    with pytest.raises(UnsupportedValueError, match="unsupported_value_type"):
        normalize_value({1, 2})

    outcome = match_results(_result(["g"], [(object(),)]), _result(["p"], [(1,)]))

    assert outcome == MatchResult("unsupported", "unsupported_value_type")


def test_given_reordered_rows_and_renamed_columns_when_compared_then_passes() -> None:
    outcome = match_results(
        _result(["gold_id", "gold_name"], [(1, "Alpha"), (2, "Beta")]),
        _result(["pred_name", "pred_id"], [("BETA", 2), ("alpha", 1)]),
    )

    assert outcome.status == "passed"
    assert outcome.column_mapping == (1, 0)


def test_given_extra_predicted_column_when_compared_then_maps_only_gold_columns() -> (
    None
):
    outcome = match_results(
        _result(["gold_id", "gold_name"], [(1, "Alpha"), (2, "Beta")]),
        _result(
            ["unrelated", "renamed_name", "renamed_id"],
            [("noise", "BETA", 2), ("noise", "alpha", 1)],
        ),
    )

    assert outcome.status == "passed"
    assert outcome.column_mapping == (2, 1)


def test_given_duplicate_rows_when_compared_then_matches_them_one_to_one() -> None:
    outcome = match_results(
        _result(["g"], [(1,), (1,), (2,)]),
        _result(["p"], [(2,), (1,), (1,)]),
    )

    assert outcome.status == "passed"


@pytest.mark.parametrize(
    ("gold_columns", "gold_rows", "predicted_columns", "predicted_rows", "reason"),
    [
        (
            ["g1", "g2"],
            [(1, 1), (1, 1), (2, 2), (2, 2)],
            ["p1", "p2"],
            [(1, 2), (1, 2), (2, 1), (2, 1)],
            "row_values_mismatch",
        ),
        (["g"], [(1,), (2,)], ["p"], [(1,), (2,), (3,)], "row_count_mismatch"),
        (["g"], [(1,), (2,), (3,)], ["p"], [(1,), (2,)], "row_count_mismatch"),
    ],
)
def test_given_unmatched_or_extra_rows_when_compared_then_fails_exact_matching(
    gold_columns: list[str],
    gold_rows: list[tuple[object, ...]],
    predicted_columns: list[str],
    predicted_rows: list[tuple[object, ...]],
    reason: str,
) -> None:
    outcome = match_results(
        _result(gold_columns, gold_rows),
        _result(predicted_columns, predicted_rows),
    )

    assert outcome.status == "failed"
    assert outcome.reason_code == reason


def test_given_truncated_result_when_compared_then_is_unsupported() -> None:
    outcome = match_results(
        _result(["g"], [(1,)]),
        _result(["p"], [(1,)], truncated=True),
    )

    assert outcome == MatchResult("unsupported", "truncated_result")


def test_given_ambiguous_column_values_when_compared_then_finds_consistent_mapping() -> (
    None
):
    outcome = match_results(
        _result(["g1", "g2"], [(1, 2), (2, 1)]),
        _result(["p1", "p2"], [(2, 1), (1, 2)]),
    )

    assert outcome.status == "passed"


def test_given_overlapping_numeric_rows_when_compared_then_uses_augmenting_match() -> (
    None
):
    settings = Settings(numeric_absolute_tolerance=Decimal("0.1"))

    outcome = match_results(
        _result(["g1", "g2"], [(Decimal("1.0"), 10), (Decimal("1.1"), 10)]),
        _result(["p1", "p2"], [(Decimal("1.05"), 10), (Decimal("0.95"), 10)]),
        settings,
    )

    assert outcome.status == "passed"


def test_given_individually_matching_columns_but_different_rows_when_compared_then_fails() -> (
    None
):
    outcome = match_results(
        _result(["g1", "g2", "g3"], [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]),
        _result(["p1", "p2", "p3"], [(0, 0, 0), (0, 0, 1), (1, 1, 0), (1, 1, 1)]),
    )

    assert outcome.status == "failed"
    assert outcome.reason_code == "row_values_mismatch"


def test_given_too_few_predicted_columns_when_compared_then_fails() -> None:
    outcome = match_results(
        _result(["g1", "g2"], [(1, 2)]),
        _result(["p1"], [(1,)]),
    )

    assert outcome.status == "failed"
    assert outcome.reason_code == "predicted_column_count_too_small"
