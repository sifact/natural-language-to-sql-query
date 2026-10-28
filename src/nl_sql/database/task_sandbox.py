"""Create and destroy one isolated PostgreSQL clone for each evaluation task."""

from __future__ import annotations

import re
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Protocol

from ..config import Settings
from ..data.assets import BenchmarkTask, DatabaseInventory
from .postgres_tools import PostgresDatabaseTools
from .provider import DatabaseProvider

_RUN_ID = re.compile(r"[0-9a-f]{32}\Z")
_CLONE_ID = re.compile(r"nl_sql_[0-9a-f]{12}_[0-9a-f]{12}\Z")
_MAX_SETUP_SQL_BYTES = 100_000
_MAX_SETUP_STATEMENTS = 100


class TaskSandboxError(Exception):
    """Raised when an isolated PostgreSQL task database cannot be managed safely."""

    def __init__(self, message: str, *, reason_code: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


class TaskSandboxBackend(Protocol):
    """Backend operations used to manage a single run-owned database clone."""

    def validate_configuration(self) -> None: ...

    def create_clone(self, template_database: str, clone_name: str) -> None: ...

    def execute_sql(self, database_name: str, statements: Sequence[str]) -> None: ...

    def create_query_provider(
        self,
        database_name: str,
        inventory: DatabaseInventory,
    ) -> DatabaseProvider: ...

    def drop_clone(self, clone_name: str) -> None: ...


class PostgresTaskSandbox:
    """Run benchmark SQL inside an exact, uniquely named PostgreSQL clone."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        backend: TaskSandboxBackend | None = None,
    ) -> None:
        """Create a sandbox manager with an injectable offline-test backend."""
        self.settings = settings or Settings.from_env()
        self.backend = (
            backend
            if backend is not None
            else _PsycopgTaskSandboxBackend(self.settings)
        )
        self._owned_clones: set[str] = set()

    def validate_configuration(self) -> None:
        """Validate local connection configuration without opening a connection."""
        self.backend.validate_configuration()

    @contextmanager
    def open_task(
        self,
        task: BenchmarkTask,
        inventory: DatabaseInventory,
        run_id: str,
    ) -> Iterator[DatabaseProvider]:
        """Yield an agent provider while a run-owned clone is available."""
        if _RUN_ID.fullmatch(run_id) is None:
            raise TaskSandboxError(
                "The run identifier is invalid.",
                reason_code="invalid_run_id",
            )
        clone_name = f"nl_sql_{run_id[:12]}_{uuid.uuid4().hex[:12]}"
        if _CLONE_ID.fullmatch(clone_name) is None:
            raise TaskSandboxError(
                "A safe isolated database name could not be generated.",
                reason_code="invalid_clone_name",
            )
        if inventory.selected_database != task.selected_database:
            raise TaskSandboxError(
                "The selected task database does not match its inventory.",
                reason_code="database_inventory_mismatch",
            )
        self.backend.validate_configuration()
        setup_statements = _task_statements(task, "preprocess_sql")
        cleanup_statements = _task_statements(task, "clean_up_sqls")

        self.backend.create_clone(inventory.template_database, clone_name)
        self._owned_clones.add(clone_name)
        cleanup_error: TaskSandboxError | None = None
        try:
            self.backend.execute_sql(clone_name, setup_statements)
            yield self.backend.create_query_provider(clone_name, inventory)
        finally:
            try:
                self.backend.execute_sql(clone_name, cleanup_statements)
            except TaskSandboxError as error:
                cleanup_error = error

            try:
                if clone_name not in self._owned_clones:
                    raise TaskSandboxError(
                        "The run does not own the database selected for removal.",
                        reason_code="clone_ownership_mismatch",
                    )
                self.backend.drop_clone(clone_name)
                self._owned_clones.remove(clone_name)
            except TaskSandboxError as error:
                raise TaskSandboxError(
                    "The run-owned PostgreSQL clone could not be dropped.",
                    reason_code="clone_drop_failed",
                ) from error

            if cleanup_error is not None:
                raise TaskSandboxError(
                    "Task cleanup SQL failed; the run-owned clone was dropped.",
                    reason_code="cleanup_failed",
                ) from cleanup_error


def _task_statements(task: BenchmarkTask, field: str) -> tuple[str, ...]:
    statements = task.evaluator.record.get(field)
    if not isinstance(statements, list) or any(
        not isinstance(statement, str) or not statement.strip()
        for statement in statements
    ):
        raise TaskSandboxError(
            "A task setup or cleanup SQL list is invalid.",
            reason_code="invalid_setup_sql",
        )
    if len(statements) > _MAX_SETUP_STATEMENTS or any(
        len(statement.encode("utf-8")) > _MAX_SETUP_SQL_BYTES
        for statement in statements
    ):
        raise TaskSandboxError(
            "A task setup or cleanup SQL list exceeds the configured bound.",
            reason_code="setup_sql_limit_exceeded",
        )
    return tuple(statements)


class _PsycopgTaskSandboxBackend:
    """Local PostgreSQL implementation of the clone lifecycle contract."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def validate_configuration(self) -> None:
        setup_options = self._connection_options(self.settings.benchmark_setup_dsn)
        query_options = self._connection_options(self.settings.benchmark_query_dsn)
        setup_user = setup_options.get("user", "")
        query_user = query_options.get("user", "")
        if not setup_options.get("dbname"):
            raise TaskSandboxError(
                "The setup connection must name a maintenance database.",
                reason_code="database_configuration_missing",
            )
        if not setup_user or not query_user or setup_user == query_user:
            raise TaskSandboxError(
                "Separate named setup and query roles are required.",
                reason_code="database_roles_not_separated",
            )

    def create_clone(self, template_database: str, clone_name: str) -> None:
        if _CLONE_ID.fullmatch(clone_name) is None:
            raise TaskSandboxError(
                "The requested clone name is not run-owned.",
                reason_code="invalid_clone_name",
            )
        try:
            import psycopg
            from psycopg import sql
        except ImportError as error:
            raise TaskSandboxError(
                "PostgreSQL support is unavailable; install the benchmark extra.",
                reason_code="driver_unavailable",
            ) from error

        setup_options = self._connection_options(self.settings.benchmark_setup_dsn)
        query_options = self._connection_options(self.settings.benchmark_query_dsn)
        maintenance_database = setup_options.get("dbname", "")
        if not maintenance_database or maintenance_database == template_database:
            raise TaskSandboxError(
                "The setup role must connect through a separate maintenance database.",
                reason_code="database_configuration_missing",
            )
        setup_dsn = self._dsn_for_database(
            self.settings.benchmark_setup_dsn,
            maintenance_database,
        )
        created = False
        try:
            with psycopg.connect(setup_dsn, connect_timeout=5, autocommit=True) as conn:
                conn.execute(
                    sql.SQL("CREATE DATABASE {} TEMPLATE {}").format(
                        sql.Identifier(clone_name),
                        sql.Identifier(template_database),
                    )
                )
                created = True
                conn.execute(
                    sql.SQL("REVOKE CONNECT ON DATABASE {} FROM PUBLIC").format(
                        sql.Identifier(clone_name)
                    )
                )
                conn.execute(
                    sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(
                        sql.Identifier(clone_name),
                        sql.Identifier(query_options["user"]),
                    )
                )
        except (psycopg.Error, KeyError) as error:
            if created:
                try:
                    self.drop_clone(clone_name)
                except TaskSandboxError as drop_error:
                    raise drop_error from error
            raise TaskSandboxError(
                "The isolated PostgreSQL clone could not be prepared.",
                reason_code="database_setup_failed",
            ) from error

    def execute_sql(self, database_name: str, statements: Sequence[str]) -> None:
        if _CLONE_ID.fullmatch(database_name) is None:
            raise TaskSandboxError(
                "Setup SQL is restricted to a run-owned database clone.",
                reason_code="clone_ownership_mismatch",
            )
        if not statements:
            return
        try:
            import psycopg
        except ImportError as error:
            raise TaskSandboxError(
                "PostgreSQL support is unavailable; install the benchmark extra.",
                reason_code="driver_unavailable",
            ) from error
        dsn = self._dsn_for_database(
            self.settings.benchmark_setup_dsn,
            database_name,
        )
        try:
            with psycopg.connect(dsn, connect_timeout=5) as conn:
                conn.execute(
                    "SELECT set_config('statement_timeout', %s, false), "
                    "set_config('lock_timeout', %s, false)",
                    (
                        f"{self.settings.postgres_statement_timeout_seconds * 1_000}ms",
                        f"{self.settings.postgres_lock_timeout_seconds * 1_000}ms",
                    ),
                )
                for statement in statements:
                    conn.execute(statement, prepare=False)
        except psycopg.Error as error:
            raise TaskSandboxError(
                "Task setup or cleanup SQL could not be completed safely.",
                reason_code="task_sql_failed",
            ) from error

    def create_query_provider(
        self,
        database_name: str,
        inventory: DatabaseInventory,
    ) -> DatabaseProvider:
        if _CLONE_ID.fullmatch(database_name) is None:
            raise TaskSandboxError(
                "Queries are restricted to a run-owned database clone.",
                reason_code="clone_ownership_mismatch",
            )
        allowed_schemas = tuple(
            sorted({table.split(".", maxsplit=1)[0] for table in inventory.base_tables})
        ) or ("public",)
        dsn = self._dsn_for_database(
            self.settings.benchmark_query_dsn,
            database_name,
        )
        return PostgresDatabaseTools(
            dsn,
            allowed_schemas=allowed_schemas,
            settings=self.settings,
        )

    def drop_clone(self, clone_name: str) -> None:
        if _CLONE_ID.fullmatch(clone_name) is None:
            raise TaskSandboxError(
                "The requested database is not a run-owned clone.",
                reason_code="clone_ownership_mismatch",
            )
        try:
            import psycopg
            from psycopg import sql
        except ImportError as error:
            raise TaskSandboxError(
                "PostgreSQL support is unavailable; install the benchmark extra.",
                reason_code="driver_unavailable",
            ) from error
        options = self._connection_options(self.settings.benchmark_setup_dsn)
        dsn = self._dsn_for_database(
            self.settings.benchmark_setup_dsn,
            options.get("dbname", ""),
        )
        try:
            with psycopg.connect(dsn, connect_timeout=5, autocommit=True) as conn:
                conn.execute(
                    sql.SQL("DROP DATABASE {}").format(sql.Identifier(clone_name))
                )
        except psycopg.Error as error:
            raise TaskSandboxError(
                "The run-owned PostgreSQL clone could not be dropped.",
                reason_code="clone_drop_failed",
            ) from error

    def _connection_options(self, dsn: str) -> dict[str, str]:
        if not dsn.strip():
            raise TaskSandboxError(
                "Both local PostgreSQL role connections must be configured.",
                reason_code="database_configuration_missing",
            )
        try:
            import psycopg
            from psycopg.conninfo import conninfo_to_dict
        except ImportError as error:
            raise TaskSandboxError(
                "PostgreSQL support is unavailable; install the benchmark extra.",
                reason_code="driver_unavailable",
            ) from error
        try:
            options = conninfo_to_dict(dsn)
        except (psycopg.Error, TypeError, ValueError) as error:
            raise TaskSandboxError(
                "A local PostgreSQL connection configuration is invalid.",
                reason_code="invalid_database_configuration",
            ) from error
        host = options.get("host", "")
        host_address = options.get("hostaddr", "")
        if (
            options.get("service")
            or (
                host
                and host not in {"localhost", "127.0.0.1", "::1"}
                and not host.startswith("/")
            )
            or (host_address and host_address not in {"localhost", "127.0.0.1", "::1"})
        ):
            raise TaskSandboxError(
                "Benchmark database access is restricted to a local PostgreSQL server.",
                reason_code="unsafe_database_target",
            )
        return options

    def _dsn_for_database(self, dsn: str, database_name: str) -> str:
        try:
            from psycopg.conninfo import make_conninfo
        except ImportError as error:
            raise TaskSandboxError(
                "PostgreSQL support is unavailable; install the benchmark extra.",
                reason_code="driver_unavailable",
            ) from error
        options = self._connection_options(dsn)
        host = options.get("host", "")
        host_address = options.get("hostaddr", "")
        if host.startswith("/"):
            target = {
                "host": host,
                "hostaddr": "",
                "dbname": database_name,
            }
        else:
            target = {
                "host": host or "localhost",
                "hostaddr": host_address or ("::1" if host == "::1" else "127.0.0.1"),
                "dbname": database_name,
            }
        try:
            return make_conninfo(dsn, **target)
        except (TypeError, ValueError) as error:
            raise TaskSandboxError(
                "A local PostgreSQL connection configuration is invalid.",
                reason_code="invalid_database_configuration",
            ) from error
