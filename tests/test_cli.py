"""Tests for local command behavior without live model requests."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

from nl_sql import cli
from nl_sql.agent import ModelError, ModelResponse, ToolCall
from nl_sql.config import Settings
from nl_sql.data.assets import (
    BenchmarkTask,
    EvaluatorTaskInput,
    ModelTaskInput,
    PreflightResult,
)
from nl_sql.data.demo import initialize_database
from nl_sql.data.selection import TaskSelection
from nl_sql.database import DatabaseTools


class ScriptedModel:
    """Return prepared model turns for CLI integration tests."""

    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = responses

    def complete(
        self,
        _messages: list[dict[str, object]],
        _tools: list[dict[str, object]],
    ) -> ModelResponse:
        return self.responses.pop(0)


def tool_call(name: str, arguments: dict[str, object]) -> ModelResponse:
    return ModelResponse(
        content=None,
        tool_calls=[ToolCall(call_id="call-1", name=name, arguments=arguments)],
    )


def submit_answer_call(sql: str, answer: str) -> ModelResponse:
    return tool_call("submit_answer", {"sql": sql, "answer": answer})


@pytest.fixture()
def demo_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "demo.sqlite3"
    monkeypatch.setattr(cli, "DEFAULT_DATABASE_PATH", path)
    initialize_database(path)
    return path


def configured_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NL_SQL_BASE_URL", "https://model.example/v1")
    monkeypatch.setenv("NL_SQL_MODEL", "demo-model")
    monkeypatch.setenv("NL_SQL_API_KEY", "test-secret")


def test_given_init_command_when_database_exists_then_requires_force(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Arrange
    path = tmp_path / "demo.sqlite3"
    monkeypatch.setattr(cli, "DEFAULT_DATABASE_PATH", path)

    # Act
    first_result = cli.main(["init"])
    second_result = cli.main(["init"])
    output = capsys.readouterr()

    # Assert
    assert first_result == cli.EXIT_SUCCESS
    assert second_result == cli.EXIT_FAILURE
    assert "Use --force" in output.err
    assert path.exists()


def test_given_benchmark_run_without_opt_ins_when_parsed_then_refuses_to_start() -> (
    None
):
    # Act
    with pytest.raises(SystemExit) as error:
        cli.main(["benchmark", "run"])

    # Assert
    assert error.value.code == 2


def test_given_no_eligible_tasks_when_preflight_then_returns_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    preflight = PreflightResult(
        dataset_revision="fixture",
        assets=(),
        databases=MappingProxyType({}),
        tasks=(),
        observed_categories=("Query",),
        run_blockers=(),
    )
    monkeypatch.setattr(cli, "load_local_benchmark_assets", lambda: preflight)
    monkeypatch.setattr(
        cli,
        "select_article_shaped_tasks",
        lambda _result: TaskSelection((), ()),
    )

    exit_code = cli.main(["benchmark", "preflight"])
    output = capsys.readouterr()

    assert exit_code == cli.EXIT_FAILURE
    assert "no tasks meet" in output.err


def test_given_no_eligible_tasks_when_run_then_skips_database_verification(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    preflight = PreflightResult(
        dataset_revision="fixture",
        assets=(),
        databases=MappingProxyType({}),
        tasks=(),
        observed_categories=("Query",),
        run_blockers=(),
    )
    monkeypatch.setattr(cli, "load_local_benchmark_assets", lambda: preflight)
    monkeypatch.setattr(
        cli,
        "select_article_shaped_tasks",
        lambda _result: TaskSelection((), ()),
    )

    def reject_database_verification(_settings: Settings) -> None:
        pytest.fail("the task runner must not start for an empty selection")

    monkeypatch.setattr(cli, "TaskRunner", reject_database_verification)
    exit_code = cli.main(
        [
            "benchmark",
            "run",
            "--confirm-run",
            "--allow-model-data-transfer",
        ]
    )
    output = capsys.readouterr()

    assert exit_code == cli.EXIT_FAILURE
    assert "no tasks meet" in output.err


def test_given_authorized_benchmark_selection_when_run_then_writes_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    task = BenchmarkTask(
        instance_id="task-1",
        selected_database="db-one",
        model_input=ModelTaskInput(query="question"),
        evaluator=EvaluatorTaskInput(record={}),
    )
    preflight = PreflightResult(
        dataset_revision="fixture",
        assets=(),
        databases=MappingProxyType({}),
        tasks=(task,),
        observed_categories=("Query",),
        run_blockers=(),
    )
    selection = TaskSelection((task,), ())
    fake_run = SimpleNamespace(
        metrics=SimpleNamespace(
            passed_tasks=1,
            evaluated_tasks=1,
            accuracy=Fraction(1),
        ),
        aborted=False,
        tasks=(SimpleNamespace(status="passed"),),
    )

    class FakeRunner:
        def __init__(self, _settings: Settings) -> None:
            pass

        def run(
            self,
            received_preflight: PreflightResult,
            received_selection: TaskSelection,
            *,
            max_tasks: int,
            confirm_run: bool,
            allow_model_data_transfer: bool,
        ) -> SimpleNamespace:
            assert received_preflight is preflight
            assert received_selection is selection
            assert max_tasks == 1
            assert confirm_run is True
            assert allow_model_data_transfer is True
            return fake_run

    monkeypatch.setattr(cli, "load_local_benchmark_assets", lambda: preflight)
    monkeypatch.setattr(cli, "select_article_shaped_tasks", lambda _result: selection)
    monkeypatch.setattr(cli, "TaskRunner", FakeRunner)
    monkeypatch.setattr(cli, "validate_report_destination", lambda *_args: None)
    monkeypatch.setattr(
        cli,
        "write_report",
        lambda *_args, **_kwargs: "data/benchmark/reports/report.json",
    )

    exit_code = cli.main(
        [
            "benchmark",
            "run",
            "--confirm-run",
            "--allow-model-data-transfer",
            "--report",
            "report.json",
        ]
    )
    output = capsys.readouterr()

    assert exit_code == cli.EXIT_SUCCESS
    assert "Score: 1/1 (1)" in output.out
    assert "data/benchmark/reports/report.json" in output.out


def test_given_successful_question_when_asked_then_prints_answer_sql_and_rows(
    demo_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Arrange
    configured_environment(monkeypatch)
    scripted_model = ScriptedModel(
        [
            tool_call("query", {"sql": "SELECT COUNT(*) AS order_count FROM orders"}),
            submit_answer_call(
                "SELECT COUNT(*) AS order_count FROM orders",
                "There are four orders.",
            ),
        ]
    )
    monkeypatch.setattr(cli, "ChatCompletionClient", lambda _config: scripted_model)

    # Act
    exit_code = cli.main(["ask", "How many orders are there?"])
    output = capsys.readouterr()

    # Assert
    assert exit_code == cli.EXIT_SUCCESS
    assert "Answer: There are four orders." in output.out
    assert "SQL: SELECT COUNT(*) AS order_count FROM orders" in output.out
    assert 'Rows: [{"order_count": 4}]' in output.out
    assert DatabaseTools(demo_path).list_tables()


def test_given_missing_model_configuration_when_asked_then_returns_failure(
    demo_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Arrange
    for variable in ("NL_SQL_BASE_URL", "NL_SQL_MODEL", "NL_SQL_API_KEY"):
        monkeypatch.delenv(variable, raising=False)

    # Act
    exit_code = cli.main(["ask", "Count the orders"])
    output = capsys.readouterr()

    # Assert
    assert exit_code == cli.EXIT_FAILURE
    assert "NL_SQL_BASE_URL" in output.err
    assert "NL_SQL_API_KEY" in output.err


def test_given_clarification_response_when_asked_then_returns_incomplete_exit(
    demo_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    mocker,
) -> None:
    # Arrange
    configured_environment(monkeypatch)
    monkeypatch.setenv("NL_SQL_MAX_ROUNDS", "1")
    query = mocker.patch.object(DatabaseTools, "query")
    scripted_model = ScriptedModel(
        [
            ModelResponse(
                content='{"status":"clarification_required","question":"Which date range?"}',
                tool_calls=[],
            )
        ]
    )
    monkeypatch.setattr(cli, "ChatCompletionClient", lambda _config: scripted_model)

    # Act
    exit_code = cli.main(["ask", "Compare order totals"])
    output = capsys.readouterr()

    # Assert
    assert exit_code == cli.EXIT_INCOMPLETE
    assert "Clarification needed: Which date range?" in output.out
    query.assert_not_called()


def test_given_round_limit_when_asked_then_returns_incomplete_exit(
    demo_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Arrange
    configured_environment(monkeypatch)
    scripted_model = ScriptedModel(
        [tool_call("list_tables", {}) for _ in range(Settings().max_rounds)]
    )
    monkeypatch.setattr(cli, "ChatCompletionClient", lambda _config: scripted_model)

    # Act
    exit_code = cli.main(["ask", "Count the orders"])
    output = capsys.readouterr()

    # Assert
    assert exit_code == cli.EXIT_INCOMPLETE
    assert "round limit" in output.err


def test_given_model_failure_when_asked_then_returns_failure(
    demo_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Arrange
    configured_environment(monkeypatch)

    class FailingModel:
        def complete(
            self,
            _messages: list[dict[str, object]],
            _tools: list[dict[str, object]],
        ) -> ModelResponse:
            raise ModelError("Model endpoint returned HTTP 503.")

    monkeypatch.setattr(cli, "ChatCompletionClient", lambda _config: FailingModel())

    # Act
    exit_code = cli.main(["ask", "Count the orders"])
    output = capsys.readouterr()

    # Assert
    assert exit_code == cli.EXIT_FAILURE
    assert "HTTP 503" in output.err


def test_given_rejected_query_without_submission_when_asked_then_returns_incomplete(
    demo_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Arrange
    configured_environment(monkeypatch)
    monkeypatch.setenv("NL_SQL_MAX_ROUNDS", "2")
    scripted_model = ScriptedModel(
        [
            tool_call("query", {"sql": "DELETE FROM orders"}),
            ModelResponse(content="The query was rejected.", tool_calls=[]),
        ]
    )
    monkeypatch.setattr(cli, "ChatCompletionClient", lambda _config: scripted_model)

    # Act
    exit_code = cli.main(["ask", "Delete all orders"])
    output = capsys.readouterr()

    # Assert
    assert exit_code == cli.EXIT_INCOMPLETE
    assert "round limit reached before a final answer" in output.err


def test_given_valid_environment_when_asked_then_passes_validated_model_configuration(
    demo_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    configured_environment(monkeypatch)
    scripted_model = ScriptedModel([submit_answer_call("SELECT 1", "Done.")])
    captured_configs: list[Settings] = []

    def create_client(settings: Settings) -> ScriptedModel:
        captured_configs.append(settings)
        return scripted_model

    monkeypatch.setattr(cli, "ChatCompletionClient", create_client)

    # Act
    exit_code = cli.main(["ask", "Count the orders"])

    # Assert
    assert exit_code == cli.EXIT_SUCCESS
    assert captured_configs == [
        Settings(
            model_url="https://model.example/v1",
            model_name="demo-model",
            model_key="test-secret",
        )
    ]
