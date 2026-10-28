"""Tests for bounded tool orchestration without model calls or network access."""

import json
from decimal import Decimal

import pytest

from nl_sql.agent import (
    AgentError,
    ModelError,
    ModelResponse,
    SqlAgent,
    TaskTracer,
    ToolCall,
    TraceError,
)
from nl_sql.config import Settings
from nl_sql.data.demo import initialize_database
from nl_sql.data.knowledge import (
    ColumnMeaning,
    InMemoryKnowledgeCatalog,
    KnowledgeDefinition,
)
from nl_sql.database import DatabaseToolError, DatabaseTools, QueryResult


class ScriptedModel:
    """Return prepared model turns and retain the messages supplied to each turn."""

    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = responses
        self.calls: list[list[dict[str, object]]] = []
        self.tool_definitions: list[list[dict[str, object]]] = []

    def complete(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]],
    ) -> ModelResponse:
        self.calls.append(messages)
        self.tool_definitions.append(tools)
        return self.responses.pop(0)


@pytest.fixture()
def database(tmp_path) -> DatabaseTools:
    path = tmp_path / "demo.sqlite3"
    initialize_database(path)
    return DatabaseTools(path)


def tool_call(
    name: str, arguments: dict[str, object], call_id: str = "call-1"
) -> ModelResponse:
    return ModelResponse(
        content=None,
        tool_calls=[ToolCall(call_id=call_id, name=name, arguments=arguments)],
    )


def submit_answer_call(
    sql: str, answer: str, call_id: str = "submit-1"
) -> ModelResponse:
    return tool_call(
        "submit_answer",
        {"sql": sql, "answer": answer},
        call_id,
    )


def test_given_discovery_and_query_turns_when_run_then_returns_answer_sql_and_rows(
    database: DatabaseTools,
) -> None:
    # Arrange
    model = ScriptedModel(
        [
            tool_call("list_tables", {}),
            tool_call(
                "query", {"sql": "SELECT COUNT(*) AS order_count FROM orders"}, "call-2"
            ),
            submit_answer_call(
                "SELECT COUNT(*) AS order_count FROM orders",
                "There are four orders.",
            ),
        ]
    )
    agent = SqlAgent(model, database)

    # Act
    result = agent.run("How many orders are there?")

    # Assert
    assert result.answer == "There are four orders."
    assert result.sql == "SELECT COUNT(*) AS order_count FROM orders"
    assert result.rows == [{"order_count": 4}]


def test_given_plain_text_then_submission_when_run_then_completes_with_submitted_sql(
    database: DatabaseTools,
) -> None:
    model = ScriptedModel(
        [
            tool_call(
                "query",
                {"sql": "SELECT COUNT(*) AS order_count FROM orders"},
            ),
            ModelResponse(content="Premature unsubmitted answer.", tool_calls=[]),
            submit_answer_call("SELECT 1", "Submitted answer."),
        ]
    )
    agent = SqlAgent(model, database)

    result = agent.run("How many orders are there?")

    assert result.answer == "Submitted answer."
    assert result.sql == "SELECT 1"
    assert result.rows == []
    assert result.incomplete is False


def test_given_invalid_answer_submission_when_run_then_continues_until_valid_submission(
    database: DatabaseTools,
) -> None:
    model = ScriptedModel(
        [
            tool_call("submit_answer", {"sql": " ", "answer": "Missing SQL"}),
            submit_answer_call("SELECT 1", "Valid answer.", "submit-2"),
        ]
    )
    agent = SqlAgent(model, database)

    result = agent.run("Return a result.")

    assert result.answer == "Valid answer."
    assert [item.outcome for item in result.trace] == ["error", "success"]


def test_given_tool_calls_after_submission_when_run_then_does_not_execute_them(
    database: DatabaseTools,
    mocker,
) -> None:
    list_tables = mocker.patch.object(database, "list_tables")
    response = ModelResponse(
        content=None,
        tool_calls=[
            ToolCall(
                call_id="submit-1",
                name="submit_answer",
                arguments={"sql": "SELECT 1", "answer": "Submitted."},
            ),
            ToolCall(
                call_id="call-2",
                name="list_tables",
                arguments={},
            ),
        ],
    )
    agent = SqlAgent(ScriptedModel([response]), database)

    result = agent.run("Return a result.")

    assert result.answer == "Submitted."
    list_tables.assert_not_called()
    assert [item.outcome for item in result.trace] == ["success", "error"]


def test_given_plain_text_without_submission_when_round_limit_reached_then_is_incomplete(
    database: DatabaseTools,
) -> None:
    model = ScriptedModel(
        [ModelResponse(content="This is not a submitted answer.", tool_calls=[])]
    )
    agent = SqlAgent(model, database, settings=Settings(max_rounds=1))

    result = agent.run("Return a result.")

    assert result.incomplete is True
    assert result.answer is None
    assert result.sql is None


def test_given_token_usage_on_multiple_turns_when_run_then_sums_known_counts(
    database: DatabaseTools,
) -> None:
    model = ScriptedModel(
        [
            ModelResponse(
                content="I am checking the result.",
                tool_calls=[],
                input_tokens=5,
                output_tokens=2,
            ),
            ModelResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        call_id="submit-1",
                        name="submit_answer",
                        arguments={"sql": "SELECT 1", "answer": "One."},
                    )
                ],
                input_tokens=11,
                output_tokens=3,
            ),
        ]
    )

    result = SqlAgent(model, database).run("Return a result.")

    assert result.input_tokens == 16
    assert result.output_tokens == 5


def test_given_missing_token_usage_on_one_turn_when_run_then_marks_that_count_unavailable(
    database: DatabaseTools,
) -> None:
    model = ScriptedModel(
        [
            ModelResponse(
                content="I am checking the result.",
                tool_calls=[],
                input_tokens=5,
                output_tokens=2,
            ),
            ModelResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        call_id="submit-1",
                        name="submit_answer",
                        arguments={"sql": "SELECT 1", "answer": "One."},
                    )
                ],
                input_tokens=11,
            ),
        ]
    )

    result = SqlAgent(model, database).run("Return a result.")

    assert result.input_tokens == 16
    assert result.output_tokens is None


def test_given_task_tracer_when_agent_runs_then_records_messages_and_tool_results(
    database: DatabaseTools,
    tmp_path,
) -> None:
    sql = "SELECT COUNT(*) AS order_count FROM orders"
    tracer = TaskTracer("task-42", trace_directory=tmp_path)
    model = ScriptedModel(
        [
            tool_call("query", {"sql": sql}),
            submit_answer_call(sql, "There are four orders."),
        ]
    )
    agent = SqlAgent(model, database, tracer=tracer)

    result = agent.run("How many orders are there?")

    records = [
        json.loads(line)
        for line in tracer.path.read_text(encoding="utf-8").splitlines()
    ]
    tool_events = [
        record["data"] for record in records if record["event"] == "tool_call"
    ]
    assert result.sql == sql
    assert [record["sequence"] for record in records] == list(range(len(records)))
    assert [event["name"] for event in tool_events] == ["query", "submit_answer"]
    assert tool_events[0]["arguments"] == {"sql": sql}
    assert tool_events[0]["result"]["rows"] == [{"order_count": 4}]
    assert list(tmp_path.glob("*.jsonl")) == [tracer.path]


def test_given_trace_write_failure_when_submission_succeeds_then_surfaces_error(
    database: DatabaseTools,
    tmp_path,
    mocker,
) -> None:
    tracer = TaskTracer("task-43", trace_directory=tmp_path)
    mocker.patch.object(
        tracer,
        "record_tool_call",
        side_effect=TraceError("Trace write failed."),
    )
    agent = SqlAgent(
        ScriptedModel([submit_answer_call("SELECT 1", "Submitted.")]),
        database,
        tracer=tracer,
    )

    with pytest.raises(TraceError, match="Trace write failed"):
        agent.run("Return a result.")


@pytest.mark.parametrize(
    ("question", "sql", "answer", "expected_rows"),
    [
        (
            "How many orders are there?",
            "SELECT COUNT(*) AS order_count FROM orders",
            "There are four orders.",
            [{"order_count": 4}],
        ),
        (
            "What is the total value of shipped orders?",
            (
                "SELECT SUM(quantity * unit_price) AS total "
                "FROM order_items JOIN orders USING (order_id) WHERE status = 'shipped'"
            ),
            "The shipped-order total is 350.",
            [{"total": 350.0}],
        ),
        (
            "How many orders are expedited?",
            (
                "SELECT COUNT(*) AS expedited_count FROM orders "
                "WHERE json_extract(metadata, '$.expedited') = 1"
            ),
            "There are two expedited orders.",
            [{"expedited_count": 2}],
        ),
    ],
)
def test_given_fixed_demo_question_when_run_then_returns_expected_synthetic_result(
    database: DatabaseTools,
    question: str,
    sql: str,
    answer: str,
    expected_rows: list[dict[str, object]],
) -> None:
    # Arrange
    model = ScriptedModel(
        [
            tool_call("query", {"sql": sql}),
            submit_answer_call(sql, answer),
        ]
    )
    agent = SqlAgent(model, database)

    # Act
    result = agent.run(question)

    # Assert
    assert result.answer == answer
    assert result.sql == sql
    assert result.rows == expected_rows


def test_given_unallowlisted_tool_when_run_then_does_not_execute_it(
    database: DatabaseTools,
) -> None:
    # Arrange
    model = ScriptedModel(
        [
            tool_call("delete_everything", {}),
            submit_answer_call("SELECT 1", "I could not use that tool."),
        ]
    )
    agent = SqlAgent(model, database)

    # Act
    result = agent.run("Count the orders.")

    # Assert
    assert result.trace[0].outcome == "error"
    assert result.sql == "SELECT 1"


def test_given_invalid_tool_arguments_when_run_then_rejects_before_database_call(
    database: DatabaseTools,
    mocker,
) -> None:
    # Arrange
    list_tables = mocker.patch.object(database, "list_tables")
    model = ScriptedModel(
        [
            tool_call("list_tables", {"unexpected": "value"}),
            submit_answer_call("SELECT 1", "The tool request was invalid."),
        ]
    )
    agent = SqlAgent(model, database)

    # Act
    result = agent.run("Count the orders.")

    # Assert
    list_tables.assert_not_called()
    assert result.trace[0].outcome == "error"


def test_given_clarification_response_when_run_then_does_not_execute_query(
    database: DatabaseTools,
    mocker,
) -> None:
    # Arrange
    query = mocker.patch.object(database, "query")
    model = ScriptedModel(
        [
            ModelResponse(
                content='{"status":"clarification_required","question":"Which date range?"}',
                tool_calls=[],
            ),
            ModelResponse(
                content='{"status":"clarification_required","question":"Which date range?"}',
                tool_calls=[],
            ),
        ]
    )
    agent = SqlAgent(model, database, settings=Settings(max_rounds=2))

    # Act
    result = agent.run("Which orders should I compare?")

    # Assert
    query.assert_not_called()
    assert result.clarification == "Which date range?"
    assert result.incomplete is True


def test_given_repeating_tool_calls_when_run_then_returns_incomplete_after_fifteen_rounds(
    database: DatabaseTools,
) -> None:
    # Arrange
    model = ScriptedModel(
        [tool_call("list_tables", {}) for _ in range(Settings().max_rounds)]
    )
    agent = SqlAgent(model, database)

    # Act
    result = agent.run("Count the orders.")

    # Assert
    assert result.incomplete is True
    assert result.answer is None
    assert len(model.calls) == 15


def test_given_provider_failure_when_run_then_surfaces_model_error(
    database: DatabaseTools,
) -> None:
    # Arrange
    class FailingModel:
        def complete(
            self,
            _messages: list[dict[str, object]],
            _tools: list[dict[str, object]],
        ) -> ModelResponse:
            raise ModelError("Model endpoint returned HTTP 503.")

    agent = SqlAgent(FailingModel(), database)

    # Act
    with pytest.raises(ModelError, match="HTTP 503"):
        agent.run("Count the orders.")


def test_given_database_tool_failure_when_run_then_returns_error_to_model(
    database: DatabaseTools,
    mocker,
) -> None:
    # Arrange
    mocker.patch.object(
        database,
        "inspect_schema",
        side_effect=DatabaseToolError("Schema inspection failed."),
    )
    model = ScriptedModel(
        [
            tool_call("inspect_schema", {}),
            submit_answer_call("SELECT 1", "Schema inspection failed."),
        ]
    )
    agent = SqlAgent(model, database)

    # Act
    result = agent.run("What tables are available?")

    # Assert
    assert result.answer == "Schema inspection failed."
    assert result.trace[0].outcome == "error"
    assert "Schema inspection failed." in model.calls[1][-1]["content"]


def test_given_postgres_scalars_when_tool_result_is_serialized_then_preserves_agent_flow(
    database: DatabaseTools,
    mocker,
) -> None:
    # Arrange
    mocker.patch.object(
        database,
        "query",
        return_value=QueryResult(
            columns=["total"],
            rows=[{"total": Decimal("12.50")}],
            truncated=False,
        ),
    )
    model = ScriptedModel(
        [
            tool_call("query", {"sql": "SELECT SUM(amount) AS total FROM orders"}),
            submit_answer_call(
                "SELECT SUM(amount) AS total FROM orders",
                "The total is 12.50.",
            ),
        ]
    )
    agent = SqlAgent(model, database, domain_hints="")

    # Act
    result = agent.run("What is the total?")

    # Assert
    tool_payload = next(
        json.loads(message["content"])
        for message in model.calls[1]
        if message.get("role") == "tool"
    )
    assert tool_payload["rows"] == [{"total": "12.50"}]
    assert result.rows == [{"total": Decimal("12.50")}]
    assert "synthetic orders" not in model.calls[0][0]["content"]


def test_given_knowledge_catalog_when_agent_runs_then_uses_bounded_discovery_only(
    database: DatabaseTools,
) -> None:
    # Arrange
    hidden_required_id = "PRIVATE-KB-942"
    catalog = InMemoryKnowledgeCatalog(
        column_meanings=[
            ColumnMeaning("orders", "status", "Lifecycle state of an order.")
        ],
        knowledge_definitions=[
            KnowledgeDefinition(
                "Order status lifecycle",
                "The lifecycle state records whether an order is pending or shipped.",
            ),
            KnowledgeDefinition(
                "Invoice settlement",
                "Invoices are settled after payment reconciliation.",
            ),
        ],
    )
    model = ScriptedModel(
        [
            tool_call(
                "lookup_column_meaning",
                {"table_name": "orders", "column_name": "status"},
            ),
            tool_call(
                "search_knowledge",
                {"query": "What does order status mean?", "limit": 5},
                "call-2",
            ),
            submit_answer_call(
                "SELECT status FROM orders",
                "Status describes the order lifecycle.",
            ),
        ]
    )
    agent = SqlAgent(model, database, knowledge_catalog=catalog)

    # Act
    result = agent.run("What does order status mean?")

    # Assert
    conversation = json.dumps(model.calls)
    tool_names = {tool["function"]["name"] for tool in model.tool_definitions[0]}
    tool_payloads = [
        json.loads(message["content"])
        for message in model.calls[-1]
        if message.get("role") == "tool"
    ]
    assert result.answer == "Status describes the order lifecycle."
    assert hidden_required_id not in conversation
    assert "external_knowledge" not in conversation
    assert "list_knowledge" not in tool_names
    assert {"lookup_column_meaning", "search_knowledge"} <= tool_names
    assert tool_payloads[0] == {
        "found": True,
        "meaning": "Lifecycle state of an order.",
    }
    assert tool_payloads[1]["knowledge"][0]["name"] == "Order status lifecycle"
    assert "synthetic orders" not in model.calls[0][0]["content"]


def test_given_no_knowledge_catalog_when_agent_runs_then_preserves_original_tool_surface(
    database: DatabaseTools,
) -> None:
    # Arrange
    model = ScriptedModel([submit_answer_call("SELECT 1", "Ready.")])
    agent = SqlAgent(model, database)

    # Act
    agent.run("How many orders are there?")

    # Assert
    tool_names = {tool["function"]["name"] for tool in model.tool_definitions[0]}
    assert tool_names == {
        "list_tables",
        "inspect_schema",
        "inspect_table",
        "query",
        "submit_answer",
    }


def test_given_unsupported_knowledge_catalog_version_when_agent_created_then_rejects(
    database: DatabaseTools,
    mocker,
) -> None:
    # Arrange
    catalog = mocker.MagicMock(version="future-version")

    # Act
    with pytest.raises(AgentError, match="catalog version is unsupported"):
        SqlAgent(ScriptedModel([]), database, knowledge_catalog=catalog)


def test_given_empty_question_when_run_then_rejects_input(
    database: DatabaseTools,
) -> None:
    # Arrange
    agent = SqlAgent(ScriptedModel([]), database)

    # Act
    with pytest.raises(AgentError, match="non-empty question"):
        agent.run("  ")
