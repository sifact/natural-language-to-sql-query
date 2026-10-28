"""Normalize scalar and structured result values for deterministic comparison."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from decimal import Decimal, InvalidOperation
from typing import TypeAlias


class UnsupportedValueError(ValueError):
    """Raised when a result cell cannot be normalized without guessing."""


@dataclass(frozen=True, slots=True)
class NormalizedValue:
    """A typed normalized value that preserves SQL and JSON distinctions."""

    kind: str
    value: object


NormalizedRow: TypeAlias = tuple[NormalizedValue, ...]


def normalize_value(value: object) -> NormalizedValue:
    """Normalize strings, numbers, temporal values, JSON structures, and NULL."""
    if value is None:
        return NormalizedValue("null", None)
    if isinstance(value, bool):
        return NormalizedValue("boolean", value)
    if isinstance(value, Decimal):
        return _normalize_decimal(value)
    if isinstance(value, int):
        return NormalizedValue("number", Decimal(value))
    if isinstance(value, float):
        try:
            return _normalize_decimal(Decimal(str(value)))
        except InvalidOperation as error:
            raise UnsupportedValueError("non_finite_number") from error
    if isinstance(value, datetime):
        return NormalizedValue("datetime", _normalize_datetime(value))
    if isinstance(value, date):
        return NormalizedValue("date", value.isoformat())
    if isinstance(value, time):
        return NormalizedValue("time", value.isoformat())
    if isinstance(value, str):
        return _normalize_string(value)
    if isinstance(value, dict):
        return _normalize_object(value)
    if isinstance(value, (list, tuple)):
        return NormalizedValue("array", tuple(normalize_value(item) for item in value))
    raise UnsupportedValueError("unsupported_value_type")


def values_match(
    left: NormalizedValue,
    right: NormalizedValue,
    *,
    absolute_tolerance: Decimal,
    relative_tolerance: Decimal,
) -> bool:
    """Compare normalized values using Decimal tolerances and structural JSON rules."""
    if left.kind != right.kind:
        return False
    if left.kind == "number":
        left_number = left.value
        right_number = right.value
        if not isinstance(left_number, Decimal) or not isinstance(
            right_number, Decimal
        ):
            raise TypeError("Normalized numeric values must contain Decimal instances.")
        difference = abs(left_number - right_number)
        allowance = max(
            absolute_tolerance,
            relative_tolerance * max(abs(left_number), abs(right_number)),
        )
        return difference <= allowance
    if left.kind == "object":
        left_items = left.value
        right_items = right.value
        if not isinstance(left_items, tuple) or not isinstance(right_items, tuple):
            raise TypeError("Normalized objects must contain sorted item tuples.")
        if len(left_items) != len(right_items):
            return False
        return all(
            left_key == right_key
            and values_match(
                left_value,
                right_value,
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
            )
            for (left_key, left_value), (right_key, right_value) in zip(
                left_items,
                right_items,
                strict=True,
            )
        )
    if left.kind == "array":
        left_values = left.value
        right_values = right.value
        if not isinstance(left_values, tuple) or not isinstance(right_values, tuple):
            raise TypeError("Normalized arrays must contain tuples.")
        return len(left_values) == len(right_values) and all(
            values_match(
                left_item,
                right_item,
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
            )
            for left_item, right_item in zip(left_values, right_values, strict=True)
        )
    return left.value == right.value


def _normalize_decimal(value: Decimal) -> NormalizedValue:
    if not value.is_finite():
        raise UnsupportedValueError("non_finite_number")
    return NormalizedValue("number", value)


def _normalize_datetime(value: datetime) -> str:
    if value.tzinfo is not None and value.utcoffset() is not None:
        value = value.astimezone(UTC)
        return value.isoformat().replace("+00:00", "Z")
    return value.isoformat()


def _normalize_string(value: str) -> NormalizedValue:
    return NormalizedValue("string", value.strip().casefold())


def _normalize_object(value: dict[object, object]) -> NormalizedValue:
    if any(not isinstance(key, str) for key in value):
        raise UnsupportedValueError("non_string_json_key")
    return NormalizedValue(
        "object",
        tuple((key, normalize_value(value[key])) for key in sorted(value)),
    )
