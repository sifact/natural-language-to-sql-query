"""Read-only SQLite discovery and query tools for the demo database."""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import quote

from ..config import Settings
from .provider import DatabaseToolError, QueryResult

QUERY_PROGRESS_INTERVAL = 1_000
ALLOWED_TABLES = frozenset({"customers", "products", "orders", "order_items"})
ALLOWED_FUNCTIONS = frozenset(
    {
        "avg",
        "coalesce",
        "count",
        "date",
        "datetime",
        "ifnull",
        "json_extract",
        "json_type",
        "json_valid",
        "length",
        "lower",
        "max",
        "min",
        "round",
        "strftime",
        "sum",
        "upper",
    }
)
ALLOWED_AUTHORIZER_ACTIONS = frozenset(
    {
        sqlite3.SQLITE_SELECT,
        sqlite3.SQLITE_READ,
        sqlite3.SQLITE_FUNCTION,
        sqlite3.SQLITE_RECURSIVE,
    }
)


class DatabaseTools:
    """Expose bounded schema discovery and read-only SQL over one SQLite database."""

    def __init__(
        self,
        database_path: Path,
        *,
        settings: Settings | None = None,
    ) -> None:
        """Create database tools for a local database path.

        Args:
            database_path: SQLite database path. Connections always use read-only mode.
        """
        self.database_path = database_path
        self.settings = settings or Settings()

    @contextmanager
    def _connect_read_only(self) -> Iterator[sqlite3.Connection]:
        uri = f"file:{quote(str(self.database_path.resolve()), safe='/')}?mode=ro"
        try:
            connection = sqlite3.connect(uri, uri=True)
        except sqlite3.Error as error:
            raise DatabaseToolError("Cannot open the demo database in read-only mode.") from error

        try:
            connection.row_factory = sqlite3.Row
            yield connection
        finally:
            connection.close()

    @staticmethod
    def _table_names(connection: sqlite3.Connection) -> list[str]:
        rows = connection.execute(
            """
            SELECT name
            FROM sqlite_schema
            WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """
        )
        return [row[0] for row in rows]

    @staticmethod
    def _quote_identifier(identifier: str) -> str:
        return '"' + identifier.replace('"', '""') + '"'

    def list_tables(self) -> list[str]:
        """List user tables in the demo database.

        Returns:
            Table names in stable alphabetical order.

        Raises:
            DatabaseToolError: If the database cannot be opened read-only.
        """
        with self._connect_read_only() as connection:
            return self._table_names(connection)

    def inspect_schema(self) -> list[dict[str, object]]:
        """Return columns and foreign-key relationships for all user tables.

        Returns:
            A list of table schemas.

        Raises:
            DatabaseToolError: If the database cannot be opened read-only.
        """
        with self._connect_read_only() as connection:
            return [self._inspect_table(connection, name) for name in self._table_names(connection)]

    def inspect_table(self, table_name: str) -> dict[str, object]:
        """Return columns and foreign-key relationships for one user table.

        Args:
            table_name: Name of a table in the database.

        Returns:
            The table name, columns, and relationships.

        Raises:
            DatabaseToolError: If the table is unknown or the database cannot be opened.
        """
        with self._connect_read_only() as connection:
            if table_name not in self._table_names(connection):
                raise DatabaseToolError(f"Unknown table: {table_name}")
            return self._inspect_table(connection, table_name)

    def _inspect_table(self, connection: sqlite3.Connection, table_name: str) -> dict[str, object]:
        identifier = self._quote_identifier(table_name)
        columns = [
            {
                "name": row[1],
                "type": row[2],
                "not_null": bool(row[3]),
                "primary_key": bool(row[5]),
            }
            for row in connection.execute(f"PRAGMA table_info({identifier})")
        ]
        relationships = [
            {
                "column": row[3],
                "referenced_table": row[2],
                "referenced_column": row[4],
            }
            for row in connection.execute(f"PRAGMA foreign_key_list({identifier})")
        ]
        return {
            "table_name": table_name,
            "columns": columns,
            "relationships": relationships,
        }

    def query(self, sql: str) -> QueryResult:
        """Execute one read-only SELECT with a two-second deadline and bounded results.

        Args:
            sql: A single SQL query accepted by the SQLite SELECT-only authorizer.

        Returns:
            Column names and at most 100 result rows, with truncation state.

        Raises:
            DatabaseToolError: If opening, authorizing, or executing the query fails.
        """
        with self._connect_read_only() as connection:
            timeout_seconds = self.settings.sqlite_query_timeout_seconds
            row_cap = self.settings.row_cap
            deadline = time.monotonic() + timeout_seconds
            deadline_exceeded = False

            def interrupt_after_deadline() -> int:
                nonlocal deadline_exceeded
                if time.monotonic() >= deadline:
                    deadline_exceeded = True
                    return 1
                return 0

            connection.set_authorizer(self._authorize)
            connection.set_progress_handler(interrupt_after_deadline, QUERY_PROGRESS_INTERVAL)

            try:
                cursor = connection.execute(sql)
                fetched_rows = cursor.fetchmany(row_cap + 1)
            except sqlite3.Error as error:
                if deadline_exceeded:
                    message = (
                        f"Query exceeded the {timeout_seconds:g}-second execution deadline."
                    )
                else:
                    message = f"Query rejected or failed: {error}"
                raise DatabaseToolError(message) from error

            truncated = len(fetched_rows) > row_cap
            rows = fetched_rows[:row_cap]
            return QueryResult(
                columns=[description[0] for description in cursor.description or ()],
                rows=[dict(row) for row in rows],
                truncated=truncated,
                value_rows=tuple(tuple(row) for row in rows),
            )

    @staticmethod
    def _authorize(
        action: int,
        argument_one: str | None,
        argument_two: str | None,
        database_name: str | None,
        _source: str | None,
    ) -> int:
        if action not in ALLOWED_AUTHORIZER_ACTIONS:
            return sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_READ:
            # CTE and count(*) reads may have no database name; base-table reads are checked separately.
            return (
                sqlite3.SQLITE_OK
                if database_name is None
                or (argument_one in ALLOWED_TABLES and database_name == "main")
                else sqlite3.SQLITE_DENY
            )
        if action == sqlite3.SQLITE_FUNCTION:
            function_name = (argument_two or argument_one or "").lower()
            return (
                sqlite3.SQLITE_OK
                if function_name in ALLOWED_FUNCTIONS
                else sqlite3.SQLITE_DENY
            )
        return sqlite3.SQLITE_OK
