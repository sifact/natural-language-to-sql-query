"""Tests for scoring the agent on the fixed demo questions."""

import json

import pytest

from nl_sql.agent import AgentResult, ModelRateLimitError
from nl_sql.config import Settings
from nl_sql.data import DEMO_QUESTIONS, DemoQuestion, initialize_database
from nl_sql.database.sqlite_tools import DatabaseTools
from nl_sql.evaluation import (
    evaluate_demo_questions,
    summarize_demo_results,
    write_demo_report,
)

QUESTION = DemoQuestion(
    "q03",
    "What is the total value of shipped orders?",
    "SELECT SUM(oi.quantity * oi.unit_price) FROM order_items oi "
    "JOIN orders o ON o.order_id = oi.order_id WHERE o.status = 'shipped'",
    "business rule",
)


class FakeAgent:
    def __init__(self, sql: str | None = None, error: Exception | None = None) -> None:
        self.sql = sql
        self.error = error

    def run(self, question: str) -> AgentResult:
        if self.error is not None:
            raise self.error
        return AgentResult(
            answer="It is $350.",
            sql=self.sql,
            rows=[],
            clarification=None,
            incomplete=False,
            trace=[],
            input_tokens=100,
            output_tokens=20,
        )


@pytest.fixture
def database(tmp_path):
    return DatabaseTools(initialize_database(tmp_path / "demo.sqlite3"))


def evaluate(database, agent):
    return evaluate_demo_questions([QUESTION], lambda: agent, database, Settings())


def test_given_different_but_correct_sql_when_evaluated_then_passes(database) -> None:
    # Arrange
    agent = FakeAgent(
        "SELECT SUM(unit_price * quantity) AS total FROM order_items WHERE order_id IN "
        "(SELECT order_id FROM orders WHERE status = 'shipped')"
    )

    # Act
    [result] = evaluate(database, agent)

    # Assert
    assert (result.status, result.reason) == ("passed", None)


def test_given_sql_ignoring_business_rule_when_evaluated_then_fails(database) -> None:
    # Arrange
    agent = FakeAgent("SELECT SUM(quantity * unit_price) FROM order_items")

    # Act
    [result] = evaluate(database, agent)

    # Assert
    assert result.status == "failed"


def test_given_model_rate_limit_when_evaluated_then_records_error_not_failure(
    database,
) -> None:
    # Arrange
    agent = FakeAgent(error=ModelRateLimitError("quota"))

    # Act
    results = evaluate(database, agent)

    # Assert
    assert results[0].status == "error"
    assert summarize_demo_results(results)["accuracy"] is None


def test_given_results_when_report_written_then_contains_summary(database, tmp_path) -> None:
    # Arrange
    results = evaluate(database, FakeAgent(QUESTION.gold_sql))
    settings = Settings(model_url="https://model.example/v1", model_name="demo/model")

    # Act
    path = write_demo_report(results, settings, tmp_path / "report.json")

    # Assert
    report = json.loads(path.read_text())
    assert report["model"] == "demo/model"
    assert report["summary"]["passed"] == 1
    assert report["summary"]["accuracy"] == 1.0


def test_given_demo_questions_when_gold_sql_runs_then_every_query_returns_rows(
    database,
) -> None:
    # Act
    results = [database.query(question.gold_sql) for question in DEMO_QUESTIONS]

    # Assert
    assert all(result.rows for result in results)
    assert len({question.question_id for question in DEMO_QUESTIONS}) == len(DEMO_QUESTIONS)
