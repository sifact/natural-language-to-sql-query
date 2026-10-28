"""Offline tests for authorized task execution and redacted evaluation reports."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

from nl_sql.agent.sql_agent import AgentResult
from nl_sql.agent.tracing import TaskTracer
from nl_sql.config import Settings
from nl_sql.data.assets import (
    AssetProvenance,
    BenchmarkTask,
    DatabaseInventory,
    EvaluatorTaskInput,
    ModelTaskInput,
    PreflightResult,
)
from nl_sql.data.selection import TaskSelection
from nl_sql.database import DatabaseProvider, DatabaseToolError, QueryResult
from nl_sql.database.task_sandbox import TaskSandboxError
from nl_sql.evaluation import (
    ReportError,
    TaskRunner,
    TaskRunnerError,
    TaskRunResult,
    build_report,
    serialize_report,
    validate_report_destination,
    write_report,
)

_CANONICAL_SQL = "SELECT gold_value FROM public.items"
_SUBMITTED_SQL = "SELECT submitted_value, extra FROM public.items"


class FakeDatabase:
    """Return authored value rows and record the SQL sent to the evaluator."""

    def __init__(
        self,
        *,
        submitted: QueryResult | None = None,
        canonical: QueryResult | None = None,
        error_sql: str | None = None,
    ) -> None:
        self.submitted = submitted or QueryResult(
            ["submitted_value", "extra"],
            [{"submitted_value": 1, "extra": "ignored"}],
            False,
            ((1, "ignored"),),
        )
        self.canonical = canonical or QueryResult(
            ["gold_value"],
            [{"gold_value": 1}],
            False,
            ((1,),),
        )
        self.error_sql = error_sql
        self.queries: list[str] = []

    def list_tables(self) -> list[str]:
        return ["items"]

    def inspect_schema(self) -> list[dict[str, object]]:
        return []

    def inspect_table(self, table_name: str) -> dict[str, object]:
        return {"table_name": table_name}

    def query(self, sql: str) -> QueryResult:
        self.queries.append(sql)
        if sql == self.error_sql:
            raise DatabaseToolError("Database failure.")
        if sql == _SUBMITTED_SQL:
            return self.submitted
        if sql == _CANONICAL_SQL:
            return self.canonical
        raise AssertionError("The runner executed SQL outside the test fixture.")


class FakeSandbox:
    """Provide one fake database and optionally fail during clone cleanup."""

    def __init__(
        self,
        database: FakeDatabase,
        *,
        cleanup_error: str | None = None,
    ) -> None:
        self.database = database
        self.cleanup_error = cleanup_error
        self.validated = False
        self.opened: list[str] = []

    def validate_configuration(self) -> None:
        self.validated = True

    @contextmanager
    def open_task(
        self,
        task: BenchmarkTask,
        inventory: DatabaseInventory,
        run_id: str,
    ) -> Iterator[DatabaseProvider]:
        assert inventory.selected_database == task.selected_database
        assert len(run_id) == 32
        self.opened.append(task.instance_id)
        try:
            yield self.database
        finally:
            if self.cleanup_error is not None:
                raise TaskSandboxError(
                    "The test clone could not be removed.",
                    reason_code=self.cleanup_error,
                )


class FakeAgent:
    def __init__(self, result: AgentResult) -> None:
        self.result = result

    def run(self, question: str) -> AgentResult:
        assert question == "How many rows?"
        return self.result


def _task(
    instance_id: str = "task-1",
    *,
    canonical_sql: list[str] | None = None,
    test_cases: list[object] | None = None,
) -> BenchmarkTask:
    return BenchmarkTask(
        instance_id=instance_id,
        selected_database="db-one",
        model_input=ModelTaskInput(query="How many rows?"),
        evaluator=EvaluatorTaskInput(
            record={
                "sol_sql": [_CANONICAL_SQL] if canonical_sql is None else canonical_sql,
                "test_cases": [] if test_cases is None else test_cases,
                "preprocess_sql": ["CREATE TEMP TABLE task_setup (id int)"],
                "clean_up_sqls": ["DROP TABLE IF EXISTS task_setup"],
            }
        ),
    )


def _preflight(
    *tasks: BenchmarkTask, run_blockers: tuple[str, ...] = ()
) -> PreflightResult:
    assets = tuple(
        AssetProvenance(
            kind=kind,
            path=f"{kind}.json",
            sha256="0" * 64,
            source_url="https://example.test/asset",
            retrieved_on="2026-01-01",
            terms_id="test-license",
            terms_url="https://example.test/terms",
            terms_verified_by_user=True,
            verified_on="2026-01-01",
            model_transmission_allowed=True,
        )
        for kind in ("public_tasks", "gold_tasks", "database", "database_inventory")
    )
    inventory = DatabaseInventory(
        selected_database="db-one",
        template_database="nl_sql_template_db_one",
        asset_path="db-one",
        inventory_path="db-one/inventory.json",
        base_tables=("public.items",),
    )
    return PreflightResult(
        dataset_revision="fixture-revision",
        assets=assets,
        databases={"db-one": inventory},
        tasks=tuple(tasks),
        observed_categories=("Query",),
        run_blockers=run_blockers,
    )


def _run(
    tmp_path: Path,
    tasks: tuple[BenchmarkTask, ...],
    *,
    settings: Settings | None = None,
    database: FakeDatabase | None = None,
    cleanup_error: str | None = None,
    result: AgentResult | None = None,
) -> tuple[TaskRunResult, FakeSandbox, FakeDatabase]:
    provider = database or FakeDatabase()
    sandbox = FakeSandbox(provider, cleanup_error=cleanup_error)
    agent_result = result or AgentResult(
        answer="There is one row.",
        sql=_SUBMITTED_SQL,
        rows=[{"wrong": "last executed result is not used"}],
        clarification=None,
        incomplete=False,
        trace=[],
        input_tokens=17,
        output_tokens=5,
    )
    runner = TaskRunner(
        settings or Settings(),
        sandbox=sandbox,
        agent_factory=lambda _input, _database, _tracer, _settings: FakeAgent(
            agent_result
        ),
        tracer_factory=lambda task_id: TaskTracer(
            task_id,
            trace_directory=tmp_path / "traces",
        ),
        template_verifier=lambda _preflight, _settings: None,
    )
    preflight = _preflight(*tasks)
    selection = TaskSelection(tasks, ())
    return (
        runner.run(
            preflight,
            selection,
            max_tasks=len(tasks),
            confirm_run=True,
            allow_model_data_transfer=True,
        ),
        sandbox,
        provider,
    )


@pytest.mark.parametrize(
    ("confirm_run", "allow_model_data_transfer", "reason"),
    [
        (False, True, "run_confirmation_required"),
        (True, False, "model_transfer_confirmation_required"),
    ],
)
def test_given_missing_run_opt_in_when_run_then_fails_before_sandbox(
    tmp_path: Path,
    confirm_run: bool,
    allow_model_data_transfer: bool,
    reason: str,
) -> None:
    task = _task()
    sandbox = FakeSandbox(FakeDatabase())
    runner = TaskRunner(
        Settings(),
        sandbox=sandbox,
        template_verifier=lambda _preflight, _settings: None,
    )

    with pytest.raises(TaskRunnerError) as error:
        runner.run(
            _preflight(task),
            TaskSelection((task,), ()),
            confirm_run=confirm_run,
            allow_model_data_transfer=allow_model_data_transfer,
        )

    assert error.value.reason_code == reason
    assert sandbox.validated is False


def test_given_selected_task_when_run_then_scores_submitted_sql_and_redacts_report(
    tmp_path: Path,
) -> None:
    task = _task()

    result, sandbox, database = _run(tmp_path, (task,))

    assert result.tasks[0].status == "passed"
    assert result.tasks[0].reason_code is None
    assert result.tasks[0].input_tokens == 17
    assert result.tasks[0].output_tokens == 5
    assert database.queries == [_SUBMITTED_SQL, _CANONICAL_SQL]
    assert sandbox.opened == ["task-1"]
    assert result.metrics is not None
    assert result.metrics.accuracy == Fraction(1)
    preflight = _preflight(task)
    selection = TaskSelection((task,), ())
    settings = Settings(
        model_url="https://model.example/v1",
        model_name="test-model",
        model_key="hidden-api-key",
        benchmark_setup_dsn="postgresql://setup:setup-secret@localhost/postgres",
        benchmark_query_dsn="postgresql://query:query-secret@localhost/postgres",
    )
    report = serialize_report(
        result,
        settings,
        preflight=preflight,
        selection=selection,
    )
    payload = build_report(
        result,
        settings,
        preflight=preflight,
        selection=selection,
    )
    assert report == serialize_report(
        result,
        settings,
        preflight=preflight,
        selection=selection,
    )
    assert "How many rows?" not in report
    assert _SUBMITTED_SQL not in report
    assert _CANONICAL_SQL not in report
    assert "There is one row." not in report
    assert "wrong" not in report
    assert "hidden-api-key" not in report
    assert "setup-secret" not in report
    assert "query-secret" not in report
    assert "model.example" not in report
    assert str(tmp_path) not in report
    assert payload["run_timestamp_utc"] == result.started_at_utc
    assert payload["selected_task_ids"] == ["task-1"]
    assert payload["selection_criteria"] == selection.criteria
    assert payload["status_counts"] == {
        "passed": 1,
        "failed": 0,
        "incomplete": 0,
        "unsupported": 0,
        "skipped": 0,
    }
    assert payload["score"] == {
        "numerator": 1,
        "denominator": 1,
        "accuracy": "1",
    }
    assert result.tasks[0].trace_path is not None
    assert Path(result.tasks[0].trace_path).name == result.tasks[0].trace_path
    assert payload["numeric_tolerances"] == {
        "absolute": "0",
        "relative": "0",
    }
    assert payload["input_file_hashes"] == [
        {"kind": "database", "sha256": "0" * 64},
        {"kind": "database_inventory", "sha256": "0" * 64},
        {"kind": "gold_tasks", "sha256": "0" * 64},
        {"kind": "public_tasks", "sha256": "0" * 64},
    ]


def _initialize_report_workspace(
    tmp_path: Path,
    *,
    ignore_rule: str = "data/benchmark/\n",
) -> Path:
    (tmp_path / ".gitignore").write_text(ignore_rule, encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(tmp_path), "init", "--quiet"],
        check=True,
        capture_output=True,
    )
    (tmp_path / "data" / "benchmark").mkdir(parents=True)
    return tmp_path


def test_given_report_when_written_then_uses_ignored_root_and_redacted_payload(
    tmp_path: Path,
) -> None:
    task = _task()
    result, _sandbox, _database = _run(tmp_path, (task,))
    preflight = _preflight(task)
    selection = TaskSelection((task,), ())
    workspace = _initialize_report_workspace(tmp_path)

    report_path = write_report(
        result,
        Settings(model_name="test-model", model_key="never-write-this"),
        preflight=preflight,
        selection=selection,
        workspace_root=workspace,
    )

    assert report_path == f"data/benchmark/reports/run-{result.run_id}.json"
    text = (workspace / report_path).read_text(encoding="utf-8")
    payload = json.loads(text)
    assert "never-write-this" not in text
    assert str(workspace) not in text
    assert payload["model"]["name"] == "test-model"


def test_given_unsafe_report_name_when_validated_then_rejects_it(
    tmp_path: Path,
) -> None:
    workspace = _initialize_report_workspace(tmp_path)

    with pytest.raises(ReportError):
        validate_report_destination(workspace, "../outside.json")
    with pytest.raises(ReportError):
        validate_report_destination(workspace, str(workspace / "outside.json"))


def test_given_symlinked_report_directory_when_validated_then_rejects_it(
    tmp_path: Path,
) -> None:
    workspace = _initialize_report_workspace(tmp_path)
    reports_directory = workspace / "data" / "benchmark" / "reports"
    outside_directory = workspace / "outside"
    outside_directory.mkdir()
    reports_directory.symlink_to(outside_directory, target_is_directory=True)

    with pytest.raises(ReportError):
        validate_report_destination(workspace, "safe.json")


def test_given_unignored_reports_root_when_validated_then_rejects_it(
    tmp_path: Path,
) -> None:
    workspace = _initialize_report_workspace(tmp_path, ignore_rule="")

    with pytest.raises(ReportError):
        validate_report_destination(workspace, "safe.json")


@pytest.mark.parametrize(
    ("task", "reason"),
    [
        (_task(canonical_sql=[]), "missing_gold_answer"),
        (
            _task(test_cases=[{"operator": "unknown"}]),
            "unknown_test_case_semantics",
        ),
    ],
)
def test_given_unsupported_gold_contract_when_run_then_does_not_open_database(
    tmp_path: Path,
    task: BenchmarkTask,
    reason: str,
) -> None:
    result, sandbox, _database = _run(tmp_path, (task,))

    assert result.tasks[0].status == "unsupported"
    assert result.tasks[0].reason_code == reason
    assert sandbox.validated is False
    assert sandbox.opened == []
    assert result.metrics is not None
    assert result.metrics.accuracy is None


def test_given_oversized_task_question_when_run_then_marks_task_unsupported(
    tmp_path: Path,
) -> None:
    task = replace(
        _task(),
        model_input=ModelTaskInput(query="q" * 11),
    )

    result, sandbox, database = _run(
        tmp_path,
        (task,),
        settings=Settings(max_tool_argument_length=10),
    )

    assert result.tasks[0].status == "unsupported"
    assert result.tasks[0].reason_code == "invalid_task_input"
    assert sandbox.opened == []
    assert database.queries == []


def test_given_oversized_submission_when_run_then_does_not_score_it(
    tmp_path: Path,
) -> None:
    result, _sandbox, database = _run(
        tmp_path,
        (_task(),),
        settings=Settings(max_tool_argument_length=100),
        result=AgentResult(
            answer="a" * 101,
            sql="SELECT 1",
            rows=[],
            clarification=None,
            incomplete=False,
            trace=[],
        ),
    )

    assert result.tasks[0].status == "incomplete"
    assert result.tasks[0].reason_code == "invalid_submission"
    assert database.queries == []


@pytest.mark.parametrize(
    ("database", "reason"),
    [
        (
            FakeDatabase(submitted=QueryResult(["x"], [{"x": 1}], True, ((1,),))),
            "submitted_result_truncated",
        ),
        (
            FakeDatabase(canonical=QueryResult(["x"], [{"x": 1}], True, ((1,),))),
            "canonical_result_truncated",
        ),
        (
            FakeDatabase(error_sql=_CANONICAL_SQL),
            "canonical_query_failed",
        ),
    ],
)
def test_given_uncomparable_result_when_run_then_marks_task_unsupported(
    tmp_path: Path,
    database: FakeDatabase,
    reason: str,
) -> None:
    result, _sandbox, _provider = _run(tmp_path, (_task(),), database=database)

    assert result.tasks[0].status == "unsupported"
    assert result.tasks[0].reason_code == reason
    assert result.metrics is not None
    assert result.metrics.accuracy is None


def test_given_clone_cleanup_failure_when_run_then_aborts_and_skips_remaining_tasks(
    tmp_path: Path,
) -> None:
    tasks = (_task("task-1"), _task("task-2"))
    result, sandbox, _database = _run(
        tmp_path,
        tasks,
        cleanup_error="clone_drop_failed",
    )

    assert [item.status for item in result.tasks] == ["incomplete", "skipped"]
    assert [item.reason_code for item in result.tasks] == [
        "clone_drop_failed",
        "run_aborted",
    ]
    assert result.aborted is True
    assert result.abort_reason == "clone_drop_failed"
    assert result.metrics is None
    assert sandbox.opened == ["task-1"]
    report = build_report(
        result,
        preflight=_preflight(*tasks),
        selection=TaskSelection(tasks, ()),
    )
    assert report["score"] is None
    assert report["status_counts"] == {
        "passed": 0,
        "failed": 0,
        "incomplete": 1,
        "unsupported": 0,
        "skipped": 1,
    }


def test_given_incomplete_asset_authorization_when_run_then_rejects_before_database(
    tmp_path: Path,
) -> None:
    task = _task()
    sandbox = FakeSandbox(FakeDatabase())
    runner = TaskRunner(
        Settings(),
        sandbox=sandbox,
        template_verifier=lambda _preflight, _settings: None,
    )

    with pytest.raises(TaskRunnerError) as error:
        runner.run(
            _preflight(task, run_blockers=("asset_terms_not_verified",)),
            TaskSelection((task,), ()),
            confirm_run=True,
            allow_model_data_transfer=True,
        )

    assert error.value.reason_code == "asset_authorization_blocked"
    assert sandbox.validated is False


def test_given_missing_database_asset_when_run_then_rejects_incomplete_assets(
    tmp_path: Path,
) -> None:
    task = _task()
    sandbox = FakeSandbox(FakeDatabase())
    runner = TaskRunner(
        Settings(),
        sandbox=sandbox,
        template_verifier=lambda _preflight, _settings: None,
    )
    preflight = _preflight(task)
    incomplete = replace(preflight, assets=preflight.assets[:-1])

    with pytest.raises(TaskRunnerError) as error:
        runner.run(
            incomplete,
            TaskSelection((task,), ()),
            confirm_run=True,
            allow_model_data_transfer=True,
        )

    assert error.value.reason_code == "incomplete_assets"
    assert sandbox.validated is False


def test_given_absolute_trace_path_when_reported_then_rejects_path(
    tmp_path: Path,
) -> None:
    result = _example_run_result(tmp_path)
    absolute_trace = replace(result.tasks[0], trace_path="/tmp/private-trace.jsonl")
    invalid_run = replace(result, tasks=(absolute_trace,))

    with pytest.raises(ReportError, match="relative file names"):
        task = _task()
        build_report(
            invalid_run,
            preflight=_preflight(task),
            selection=TaskSelection((task,), ()),
        )


def _example_run_result(tmp_path: Path) -> TaskRunResult:
    task = _task()
    result, _sandbox, _database = _run(tmp_path, (task,))
    return result
