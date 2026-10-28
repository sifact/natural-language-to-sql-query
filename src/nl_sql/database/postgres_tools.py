"""Read-only PostgreSQL discovery and query tools for local benchmark databases."""

from __future__ import annotations

import re
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

from ..config import Settings
from .provider import DatabaseToolError, QueryResult

if TYPE_CHECKING:
    from psycopg import Connection
    from sqlglot import exp as sqlglot_exp

MAX_SQL_LENGTH = 20_000

_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_$]{0,62}\Z")
_ALLOWED_FUNCTIONS = frozenset(
    {
        "abs",
        "avg",
        "ceil",
        "ceiling",
        "char_length",
        "cast",
        "coalesce",
        "concat",
        "count",
        "date_part",
        "date_trunc",
        "extract",
        "floor",
        "greatest",
        "least",
        "length",
        "lower",
        "max",
        "min",
        "nullif",
        "round",
        "substring",
        "sum",
        "timestamp_trunc",
        "trim",
        "upper",
        # sqlglot models CASE, boolean logic and JSON access as functions, so the
        # names below are needed by ordinary analytical SELECTs. They cover every
        # function in the Base-Lite gold SELECT queries. Server, file, network and
        # session functions (pg_*, dblink, lo_*, set_config, ...) stay excluded.
        "and",
        "or",
        "not",
        "if",
        "case",
        "exists",
        "json_extract",
        "json_extract_scalar",
        "jsonb_extract_path_text",
        "json_build_object",
        "jsonb_build_object",
        "jsonb_agg",
        "j_s_o_n_object_agg",
        "j_s_o_n_array_agg",
        "row_to_json",
        "array",
        "array_agg",
        "array_to_string",
        "array_remove",
        "string_to_array",
        "cardinality",
        "explode",
        "unnest",
        "concat_ws",
        "group_concat",
        "time_to_str",
        "current_date",
        "current_timestamp",
        "age",
        "power",
        "sqrt",
        "exp",
        "ln",
        "log",
        "mod",
        "sign",
        "stddev",
        "stddev_samp",
        "stddev_pop",
        "variance",
        "corr",
        "regr_slope",
        "percentile_cont",
        "percentile_disc",
        "median",
        "rank",
        "dense_rank",
        "percent_rank",
        "row_number",
        "ntile",
        "lead",
        "lag",
        "first_value",
        "last_value",
    }
)
_ROLE_CHECK_SQL = """
WITH RECURSIVE memberships(role_oid) AS (
    SELECT oid FROM pg_roles WHERE rolname = current_user
    UNION
    SELECT member.roleid
    FROM pg_auth_members AS member
    JOIN memberships AS inherited ON inherited.role_oid = member.member
)
SELECT role.rolsuper,
       role.rolcreaterole,
       role.rolcreatedb,
       role.rolreplication,
       role.rolbypassrls,
       EXISTS (
           SELECT 1
           FROM memberships
           JOIN pg_roles AS inherited ON inherited.oid = memberships.role_oid
           WHERE inherited.rolname IN (
               'pg_read_server_files',
               'pg_write_server_files',
               'pg_execute_server_program'
           )
       ) AS has_server_access_role,
       EXISTS (
           SELECT 1
           FROM memberships
           JOIN pg_roles AS inherited ON inherited.oid = memberships.role_oid
           WHERE inherited.rolsuper
              OR inherited.rolcreaterole
              OR inherited.rolcreatedb
              OR inherited.rolreplication
              OR inherited.rolbypassrls
       ) AS has_elevated_membership
FROM pg_roles AS role
WHERE role.rolname = current_user
"""
_CONNECT_PRIVILEGES_SQL = """
SELECT datname, has_database_privilege(current_user, oid, 'CONNECT')
FROM pg_database
WHERE datallowconn
"""


class PostgresDatabaseTools:
    """Expose bounded, read-only tools for a loopback-only run-owned database."""

    def __init__(
        self,
        dsn: str,
        *,
        allowed_schemas: tuple[str, ...] = ("public",),
        settings: Settings | None = None,
    ) -> None:
        """Create a provider for one local database whose name begins with ``nl_sql_``."""
        self.dsn = dsn
        self.allowed_schemas = tuple(sorted(set(allowed_schemas)))
        self.settings = settings or Settings()
        self.statement_timeout_seconds = self.settings.postgres_statement_timeout_seconds
        self.lock_timeout_seconds = self.settings.postgres_lock_timeout_seconds
        self.max_result_rows = self.settings.postgres_row_cap
        self._validate_configuration()

    def _validate_configuration(self) -> None:
        if not self.dsn.strip():
            raise DatabaseToolError(
                "A PostgreSQL connection string is required.",
                category="invalid_configuration",
            )
        if not self.allowed_schemas or any(
            _IDENTIFIER_PATTERN.fullmatch(schema) is None
            or schema.lower() == "information_schema"
            or schema.lower().startswith("pg_")
            for schema in self.allowed_schemas
        ):
            raise DatabaseToolError(
                "PostgreSQL schemas must be non-empty valid identifiers.",
                category="invalid_configuration",
            )
    def _validate_target(self) -> dict[str, str]:
        try:
            import psycopg
            from psycopg.conninfo import conninfo_to_dict
        except ImportError as error:
            raise DatabaseToolError(
                "PostgreSQL support is unavailable; install the benchmark extra.",
                category="driver_unavailable",
            ) from error

        try:
            options = conninfo_to_dict(self.dsn)
        except (psycopg.Error, ValueError) as error:
            raise DatabaseToolError(
                "The PostgreSQL connection configuration is invalid.",
                category="invalid_configuration",
            ) from error

        host = options.get("host", "")
        host_address = options.get("hostaddr", "")
        database = options.get("dbname", "")
        if (
            options.get("service")
            or (host and host not in _LOCAL_HOSTS and not host.startswith("/"))
            or (host_address and host_address not in _LOCAL_HOSTS)
            or not database.startswith("nl_sql_")
        ):
            raise DatabaseToolError(
                "PostgreSQL access is restricted to local nl_sql_ databases.",
                category="unsafe_target",
            )
        if host.startswith("/"):
            return {
                "host": host,
                "hostaddr": "",
                "dbname": database,
            }
        pinned_address = host_address or ("::1" if host == "::1" else "127.0.0.1")
        return {
            "host": host or "localhost",
            "hostaddr": pinned_address,
            "dbname": database,
        }

    @contextmanager
    def _connect_read_only(self) -> Iterator[Connection]:
        target_options = self._validate_target()
        try:
            import psycopg
        except ImportError as error:
            raise DatabaseToolError(
                "PostgreSQL support is unavailable; install the benchmark extra.",
                category="driver_unavailable",
            ) from error

        try:
            connection = psycopg.connect(self.dsn, connect_timeout=5, **target_options)
        except psycopg.Error as error:
            raise DatabaseToolError(
                "The local PostgreSQL database could not be opened.",
                category="database_unavailable",
            ) from error

        try:
            with connection.transaction():
                connection.execute("SET TRANSACTION READ ONLY")
                connection.execute(
                    "SELECT set_config('statement_timeout', %s, true), "
                    "set_config('lock_timeout', %s, true), "
                    "set_config('search_path', %s, true)",
                    (
                        f"{self.statement_timeout_seconds * 1_000}ms",
                        f"{self.lock_timeout_seconds * 1_000}ms",
                        ", ".join(f'"{schema}"' for schema in self.allowed_schemas),
                    ),
                )
                self._verify_role(connection)
                yield connection
        except DatabaseToolError:
            raise
        except psycopg.Error as error:
            category = (
                "query_timeout"
                if error.sqlstate in {"57014", "55P03"}
                else "database_error"
            )
            message = (
                "The PostgreSQL query exceeded its configured deadline."
                if category == "query_timeout"
                else "The PostgreSQL operation could not be completed safely."
            )
            raise DatabaseToolError(message, category=category) from error
        finally:
            connection.close()

    @staticmethod
    def _verify_role(connection: Connection) -> None:
        role = connection.execute(_ROLE_CHECK_SQL).fetchone()
        permitted_databases = connection.execute(_CONNECT_PRIVILEGES_SQL).fetchall()

        if role is None or any(role):
            raise DatabaseToolError(
                "The PostgreSQL query role has privileges outside the read-only policy.",
                category="unsafe_role",
            )
        connected_database = connection.info.dbname
        if (
            not permitted_databases
            or any(
                database != connected_database
                for database, can_connect in permitted_databases
                if can_connect
            )
            or not any(
                database == connected_database and can_connect
                for database, can_connect in permitted_databases
            )
        ):
            raise DatabaseToolError(
                "The PostgreSQL query role must have CONNECT access only to the active database.",
                category="unsafe_role",
            )

    def _table_records(self, connection: Connection) -> list[tuple[str, str]]:
        rows = connection.execute(
            """
            SELECT table_schema, table_name
            FROM information_schema.tables
            WHERE table_type = 'BASE TABLE'
              AND table_schema = ANY(%s)
            ORDER BY table_schema, table_name
            """,
            (list(self.allowed_schemas),),
        ).fetchall()
        return [(schema, name) for schema, name in rows]

    def _display_table_name(self, schema: str, table_name: str) -> str:
        if len(self.allowed_schemas) == 1 and schema == self.allowed_schemas[0]:
            return table_name
        return f"{schema}.{table_name}"

    def list_tables(self) -> list[str]:
        """List permitted base tables in stable order."""
        with self._connect_read_only() as connection:
            return [
                self._display_table_name(schema, table_name)
                for schema, table_name in self._table_records(connection)
            ]

    def inspect_schema(self) -> list[dict[str, object]]:
        """Return columns and foreign-key relationships for all permitted base tables."""
        with self._connect_read_only() as connection:
            return [
                self._inspect_table(connection, schema, table_name)
                for schema, table_name in self._table_records(connection)
            ]

    def inspect_table(self, table_name: str) -> dict[str, object]:
        """Return one permitted table's columns and foreign-key relationships."""
        schema, name = self._resolve_table_name(table_name)
        with self._connect_read_only() as connection:
            if (schema, name) not in self._table_records(connection):
                raise DatabaseToolError(
                    "The requested PostgreSQL table is unavailable.",
                    category="unknown_table",
                )
            return self._inspect_table(connection, schema, name)

    def _resolve_table_name(self, table_name: str) -> tuple[str, str]:
        parts = table_name.split(".")
        if len(parts) == 1:
            schema_candidates = self.allowed_schemas
            name = parts[0]
        elif len(parts) == 2:
            schema_candidates = (parts[0],)
            name = parts[1]
        else:
            schema_candidates = ()
            name = ""

        if (
            not name
            or _IDENTIFIER_PATTERN.fullmatch(name) is None
            or not schema_candidates
            or any(schema not in self.allowed_schemas for schema in schema_candidates)
        ):
            raise DatabaseToolError(
                "The requested PostgreSQL table identifier is invalid.",
                category="invalid_identifier",
            )

        with self._connect_read_only() as connection:
            matches = [
                (schema, table)
                for schema, table in self._table_records(connection)
                if schema in schema_candidates and table == name
            ]
        if len(matches) != 1:
            category = "unknown_table" if not matches else "ambiguous_table"
            message = (
                "The requested PostgreSQL table is unavailable."
                if not matches
                else "The PostgreSQL table name is ambiguous; use a schema-qualified name."
            )
            raise DatabaseToolError(message, category=category)
        return matches[0]

    def _inspect_table(
        self,
        connection: Connection,
        schema: str,
        table_name: str,
    ) -> dict[str, object]:
        columns = connection.execute(
            """
            SELECT column_info.column_name,
                   column_info.data_type,
                   column_info.is_nullable,
                   EXISTS (
                       SELECT 1
                       FROM information_schema.table_constraints AS constraint_info
                       JOIN information_schema.key_column_usage AS key_info
                         ON key_info.constraint_catalog = constraint_info.constraint_catalog
                        AND key_info.constraint_schema = constraint_info.constraint_schema
                        AND key_info.constraint_name = constraint_info.constraint_name
                        AND key_info.table_schema = constraint_info.table_schema
                        AND key_info.table_name = constraint_info.table_name
                       WHERE constraint_info.constraint_type = 'PRIMARY KEY'
                         AND constraint_info.table_schema = column_info.table_schema
                         AND constraint_info.table_name = column_info.table_name
                         AND key_info.column_name = column_info.column_name
                   ) AS is_primary_key
            FROM information_schema.columns AS column_info
            WHERE column_info.table_schema = %s
              AND column_info.table_name = %s
            ORDER BY column_info.ordinal_position
            """,
            (schema, table_name),
        ).fetchall()
        relationships = connection.execute(
            """
            SELECT key_info.column_name,
                   target_info.table_schema,
                   target_info.table_name,
                   target_info.column_name
            FROM information_schema.table_constraints AS constraint_info
            JOIN information_schema.key_column_usage AS key_info
              ON key_info.constraint_catalog = constraint_info.constraint_catalog
             AND key_info.constraint_schema = constraint_info.constraint_schema
             AND key_info.constraint_name = constraint_info.constraint_name
             AND key_info.table_schema = constraint_info.table_schema
             AND key_info.table_name = constraint_info.table_name
            JOIN information_schema.constraint_column_usage AS target_info
              ON target_info.constraint_catalog = constraint_info.constraint_catalog
             AND target_info.constraint_schema = constraint_info.constraint_schema
             AND target_info.constraint_name = constraint_info.constraint_name
            WHERE constraint_info.constraint_type = 'FOREIGN KEY'
              AND constraint_info.table_schema = %s
              AND constraint_info.table_name = %s
            ORDER BY key_info.ordinal_position
            """,
            (schema, table_name),
        ).fetchall()
        return {
            "table_name": self._display_table_name(schema, table_name),
            "columns": [
                {
                    "name": name,
                    "type": data_type,
                    "not_null": nullable == "NO",
                    "primary_key": primary_key,
                }
                for name, data_type, nullable, primary_key in columns
            ],
            "relationships": [
                {
                    "column": column,
                    "referenced_table": self._display_table_name(
                        target_schema, target_table
                    ),
                    "referenced_column": target_column,
                }
                for column, target_schema, target_table, target_column in relationships
            ],
        }

    def query(self, sql: str) -> QueryResult:
        """Execute one allowed SELECT, returning no more than the configured row limit."""
        query_ast = self._parse_query(sql)
        with self._connect_read_only() as connection:
            self._validate_query_tables(connection, query_ast)
            with connection.cursor(name=f"nl_sql_{uuid.uuid4().hex}") as cursor:
                cursor.execute(sql)
                fetched_rows = cursor.fetchmany(self.max_result_rows + 1)
                columns = [column.name for column in cursor.description or ()]

            truncated = len(fetched_rows) > self.max_result_rows
            rows = fetched_rows[: self.max_result_rows]
            return QueryResult(
                columns=columns,
                rows=[dict(zip(columns, row, strict=True)) for row in rows],
                truncated=truncated,
                value_rows=tuple(tuple(row) for row in rows),
            )

    def _parse_query(self, sql: str) -> sqlglot_exp.Expression:
        if not sql.strip() or len(sql) > MAX_SQL_LENGTH:
            raise DatabaseToolError(
                "The PostgreSQL query is empty or exceeds the size limit.",
                category="query_rejected",
            )
        try:
            import sqlglot
            from sqlglot import exp
        except ImportError as error:
            raise DatabaseToolError(
                "PostgreSQL SQL validation is unavailable; install the benchmark extra.",
                category="parser_unavailable",
            ) from error

        try:
            statements = [
                statement
                for statement in sqlglot.parse(sql, read="postgres")
                if statement is not None
            ]
        except sqlglot.errors.ParseError as error:
            raise DatabaseToolError(
                "The PostgreSQL query is not valid supported SQL.",
                category="query_rejected",
            ) from error

        if len(statements) != 1 or not isinstance(
            statements[0],
            (exp.Select, exp.Union, exp.Intersect, exp.Except),
        ):
            raise DatabaseToolError(
                "Exactly one PostgreSQL SELECT statement is permitted.",
                category="query_rejected",
            )

        query_ast = statements[0]
        if query_ast.find(exp.Into) is not None or query_ast.find(exp.Lock) is not None:
            raise DatabaseToolError(
                "PostgreSQL SELECT INTO and row-locking clauses are not permitted.",
                category="query_rejected",
            )

        for function in query_ast.find_all(exp.Func):
            function_name = (
                function.name.lower()
                if isinstance(function, exp.Anonymous)
                else function.sql_name().lower()
            )
            parent = function.parent
            if isinstance(parent, exp.Dot) and parent.expression is function:
                schema = parent.this.name.lower()
                if schema != "pg_catalog":
                    raise DatabaseToolError(
                        "Functions outside the approved PostgreSQL built-in set are not permitted.",
                        category="query_rejected",
                    )
            if function_name not in _ALLOWED_FUNCTIONS:
                raise DatabaseToolError(
                    "The PostgreSQL query uses a function outside the approved read-only set.",
                    category="query_rejected",
                )
        return query_ast

    def _validate_query_tables(
        self,
        connection: Connection,
        query_ast: sqlglot_exp.Expression,
    ) -> None:
        from sqlglot import exp
        from sqlglot.optimizer.scope import Scope, traverse_scope

        table_records = self._table_records(connection)
        name_schemas: dict[str, list[str]] = {}
        for schema, table_name in table_records:
            name_schemas.setdefault(table_name.lower(), []).append(schema)

        for scope in traverse_scope(query_ast):
            for table in scope.tables:
                if isinstance(scope.sources.get(table.alias_or_name), Scope):
                    continue
                if not isinstance(table.this, exp.Identifier):
                    raise DatabaseToolError(
                        "PostgreSQL table-valued functions are not permitted.",
                        category="query_rejected",
                    )
                name = table.name.lower()
                schema = table.db
                if table.catalog or (schema and schema not in self.allowed_schemas):
                    raise DatabaseToolError(
                        "The PostgreSQL query references a disallowed schema.",
                        category="query_rejected",
                    )
                matching_schemas = name_schemas.get(name, [])
                if schema and schema not in matching_schemas:
                    matching_schemas = []
                if not matching_schemas:
                    raise DatabaseToolError(
                        "The PostgreSQL query references a table outside the approved database schema.",
                        category="query_rejected",
                    )
                if not schema and len(matching_schemas) > 1:
                    raise DatabaseToolError(
                        "The PostgreSQL query must qualify tables with ambiguous names.",
                        category="query_rejected",
                    )
