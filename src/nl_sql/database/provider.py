"""Shared read-only database provider contracts and value serialization."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from datetime import time as datetime_time
from decimal import Decimal
from typing import Protocol
from uuid import UUID


class DatabaseToolError(Exception):
    """Raised when a database discovery or query operation cannot complete safely."""

    def __init__(self, message: str, *, category: str = "database_error") -> None:
        super().__init__(message)
        self.category = category


@dataclass(frozen=True)
class QueryResult:
    """Structured output from a bounded read-only SQL query."""

    columns: list[str]
    rows: list[dict[str, object]]
    truncated: bool
    value_rows: tuple[tuple[object, ...], ...] | None = None


def serialize_database_value(value: object) -> str:
    """Serialize common PostgreSQL scalar values for JSON tool and CLI output."""
    if isinstance(value, (date, datetime, datetime_time)):
        return value.isoformat()
    if isinstance(value, (Decimal, UUID, timedelta)):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


class DatabaseProvider(Protocol):
    """Read-only database discovery and query operations used by the agent."""

    def list_tables(self) -> list[str]: ...

    def inspect_schema(self) -> list[dict[str, object]]: ...

    def inspect_table(self, table_name: str) -> dict[str, object]: ...

    def query(self, sql: str) -> QueryResult: ...
