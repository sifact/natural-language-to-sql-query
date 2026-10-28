"""Tests for clone ownership, setup/cleanup ordering, and exact-drop behavior."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from nl_sql.config import Settings
from nl_sql.data.assets import (
    BenchmarkTask,
    DatabaseInventory,
    EvaluatorTaskInput,
    ModelTaskInput,
)
from nl_sql.database import DatabaseProvider, QueryResult
from nl_sql.database.task_sandbox import (
    PostgresTaskSandbox,
    TaskSandboxError,
)


class FakeProvider:
    def list_tables(self) -> list[str]:
        return []

    def inspect_schema(self) -> list[dict[str, object]]:
        return []

    def inspect_table(self, table_name: str) -> dict[str, object]:
        return {"table_name": table_name}

    def query(self, sql: str) -> QueryResult:
        return QueryResult([], [], False, ())


class FakeBackend:
    def __init__(
        self,
        *,
        cleanup_failure: bool = False,
        drop_failure: bool = False,
        setup_failure: bool = False,
    ) -> None:
        self.cleanup_failure = cleanup_failure
        self.drop_failure = drop_failure
        self.setup_failure = setup_failure
        self.operations: list[tuple[str, str, tuple[str, ...] | None]] = []
        self.provider = FakeProvider()

    def validate_configuration(self) -> None:
        self.operations.append(("validate", "", None))

    def create_clone(self, template_database: str, clone_name: str) -> None:
        self.operations.append(("create", clone_name, (template_database,)))

    def execute_sql(self, database_name: str, statements: Sequence[str]) -> None:
        stage = (
            "cleanup" if statements and statements[0].startswith("DROP") else "setup"
        )
        self.operations.append((stage, database_name, tuple(statements)))
        if stage == "setup" and self.setup_failure:
            raise TaskSandboxError("setup failed", reason_code="task_sql_failed")
        if stage == "cleanup" and self.cleanup_failure:
            raise TaskSandboxError("cleanup failed", reason_code="task_sql_failed")

    def create_query_provider(
        self,
        database_name: str,
        inventory: DatabaseInventory,
    ) -> DatabaseProvider:
        self.operations.append(
            ("provider", database_name, (inventory.selected_database,))
        )
        return self.provider

    def drop_clone(self, clone_name: str) -> None:
        self.operations.append(("drop", clone_name, None))
        if self.drop_failure:
            raise TaskSandboxError("drop failed", reason_code="clone_drop_failed")


def _task() -> tuple[BenchmarkTask, DatabaseInventory]:
    task = BenchmarkTask(
        instance_id="task-1",
        selected_database="db-one",
        model_input=ModelTaskInput("Find an item."),
        evaluator=EvaluatorTaskInput(
            record={
                "preprocess_sql": ["CREATE TEMP TABLE task_setup (id int)"],
                "clean_up_sqls": ["DROP TABLE IF EXISTS task_setup"],
            }
        ),
    )
    inventory = DatabaseInventory(
        selected_database="db-one",
        template_database="nl_sql_template_db_one",
        asset_path="db-one",
        inventory_path="db-one/inventory.json",
        base_tables=("public.items",),
    )
    return task, inventory


def test_given_task_when_opened_then_creates_configures_and_drops_exact_clone() -> None:
    backend = FakeBackend()
    sandbox = PostgresTaskSandbox(backend=backend)
    task, inventory = _task()

    with sandbox.open_task(task, inventory, "a" * 32) as provider:
        assert provider is backend.provider

    clone_name = next(entry[1] for entry in backend.operations if entry[0] == "create")
    assert clone_name.startswith("nl_sql_aaaaaaaaaaaa_")
    assert [entry[0] for entry in backend.operations] == [
        "validate",
        "create",
        "setup",
        "provider",
        "cleanup",
        "drop",
    ]
    assert backend.operations[2][2] == ("CREATE TEMP TABLE task_setup (id int)",)
    assert backend.operations[4][2] == ("DROP TABLE IF EXISTS task_setup",)
    assert backend.operations[1][2] == ("nl_sql_template_db_one",)


def test_given_invalid_run_id_when_opened_then_does_not_create_database() -> None:
    backend = FakeBackend()
    sandbox = PostgresTaskSandbox(backend=backend)
    task, inventory = _task()

    with (
        pytest.raises(TaskSandboxError) as error,
        sandbox.open_task(task, inventory, "../not-a-run"),
    ):
        pass

    assert error.value.reason_code == "invalid_run_id"
    assert backend.operations == []


def test_given_cleanup_sql_failure_when_closed_then_drops_clone_and_aborts() -> None:
    backend = FakeBackend(cleanup_failure=True)
    sandbox = PostgresTaskSandbox(backend=backend)
    task, inventory = _task()

    with (
        pytest.raises(TaskSandboxError) as error,
        sandbox.open_task(task, inventory, "b" * 32),
    ):
        pass

    assert error.value.reason_code == "cleanup_failed"
    assert [entry[0] for entry in backend.operations][-1] == "drop"


def test_given_drop_failure_when_closed_then_reports_abort_without_retrying_other_names() -> (
    None
):
    backend = FakeBackend(drop_failure=True)
    sandbox = PostgresTaskSandbox(backend=backend)
    task, inventory = _task()

    with (
        pytest.raises(TaskSandboxError) as error,
        sandbox.open_task(task, inventory, "c" * 32),
    ):
        pass

    assert error.value.reason_code == "clone_drop_failed"
    clone_name = next(entry[1] for entry in backend.operations if entry[0] == "create")
    assert [entry for entry in backend.operations if entry[0] == "drop"] == [
        ("drop", clone_name, None)
    ]


def test_given_setup_failure_when_opened_then_still_cleans_and_drops_clone() -> None:
    backend = FakeBackend(setup_failure=True)
    sandbox = PostgresTaskSandbox(backend=backend)
    task, inventory = _task()

    with (
        pytest.raises(TaskSandboxError) as error,
        sandbox.open_task(task, inventory, "d" * 32),
    ):
        pass

    assert error.value.reason_code == "task_sql_failed"
    assert [entry[0] for entry in backend.operations] == [
        "validate",
        "create",
        "setup",
        "cleanup",
        "drop",
    ]


def test_given_remote_setup_target_when_validated_then_refuses_without_connecting() -> (
    None
):
    pytest.importorskip("psycopg")
    settings = Settings(
        benchmark_setup_dsn=(
            "host=database.example.test dbname=postgres user=setup_role"
        ),
        benchmark_query_dsn="host=localhost dbname=postgres user=query_role",
    )
    sandbox = PostgresTaskSandbox(settings)

    with pytest.raises(TaskSandboxError) as error:
        sandbox.validate_configuration()

    assert error.value.reason_code == "unsafe_database_target"


def test_given_shared_database_role_when_validated_then_rejects_role_reuse() -> None:
    pytest.importorskip("psycopg")
    settings = Settings(
        benchmark_setup_dsn="host=localhost dbname=postgres user=shared_role",
        benchmark_query_dsn="host=127.0.0.1 dbname=postgres user=shared_role",
    )
    sandbox = PostgresTaskSandbox(settings)

    with pytest.raises(TaskSandboxError) as error:
        sandbox.validate_configuration()

    assert error.value.reason_code == "database_roles_not_separated"
