"""Score the agent on the fixed demo questions by comparing executed query results."""

from __future__ import annotations

import json
import statistics
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol

from ..agent.model_client import ModelError
from ..agent.sql_agent import AgentError, AgentResult
from ..config import Settings
from ..data.demo_questions import DemoQuestion
from ..database.provider import DatabaseToolError, QueryResult
from .matcher import match_results

DEMO_REPORT_DIRECTORY = Path("reports")
DEMO_REPORT_SCHEMA_VERSION = 1

DemoStatus = Literal["passed", "failed", "error"]


class QuestionAgent(Protocol):
    """The part of SqlAgent the evaluation needs; fakes stand in for it in tests."""

    def run(self, question: str) -> AgentResult: ...


class QueryRunner(Protocol):
    """Read-only query execution shared by gold and predicted SQL."""

    def query(self, sql: str) -> QueryResult: ...


@dataclass(frozen=True, slots=True)
class DemoQuestionResult:
    """Outcome for one question, kept small enough to publish in a report."""

    question_id: str
    question: str
    skill: str
    status: DemoStatus
    reason: str | None
    answer: str | None
    sql: str | None
    seconds: float
    input_tokens: int | None
    output_tokens: int | None


def evaluate_demo_questions(
    questions: Iterable[DemoQuestion],
    agent_factory: Callable[[], QuestionAgent],
    database: QueryRunner,
    settings: Settings,
    clock: Callable[[], float] = time.monotonic,
) -> list[DemoQuestionResult]:
    """Ask each question with a fresh agent and score its SQL by executed result.

    Scoring by result, not SQL text, accepts any correct query. Model or agent
    failures are recorded as ``error`` so infrastructure trouble is not reported
    as a wrong answer.
    """
    results = []
    for item in questions:
        started = clock()
        try:
            outcome = agent_factory().run(item.question)
        except (AgentError, ModelError) as error:
            results.append(_result(item, "error", type(error).__name__, None, clock() - started))
            continue
        status, reason = _score(item, outcome, database, settings)
        results.append(_result(item, status, reason, outcome, clock() - started))
    return results


def _score(
    item: DemoQuestion,
    outcome: AgentResult,
    database: QueryRunner,
    settings: Settings,
) -> tuple[DemoStatus, str | None]:
    if outcome.sql is None:
        return "failed", "asked_for_clarification" if outcome.clarification else "no_sql"
    try:
        predicted = database.query(outcome.sql)
    except DatabaseToolError:
        return "failed", "sql_error"
    match = match_results(database.query(item.gold_sql), predicted, settings)
    return ("passed", None) if match.status == "passed" else ("failed", match.reason_code)


def _result(
    item: DemoQuestion,
    status: DemoStatus,
    reason: str | None,
    outcome: AgentResult | None,
    seconds: float,
) -> DemoQuestionResult:
    return DemoQuestionResult(
        question_id=item.question_id,
        question=item.question,
        skill=item.skill,
        status=status,
        reason=reason,
        answer=outcome.answer if outcome else None,
        sql=outcome.sql if outcome else None,
        seconds=round(seconds, 1),
        input_tokens=outcome.input_tokens if outcome else None,
        output_tokens=outcome.output_tokens if outcome else None,
    )


def summarize_demo_results(results: list[DemoQuestionResult]) -> dict[str, object]:
    """Headline numbers; errors are excluded from accuracy and counted separately."""
    scored = [result for result in results if result.status != "error"]
    passed = sum(result.status == "passed" for result in scored)
    seconds = [result.seconds for result in scored]
    tokens = [
        (result.input_tokens or 0) + (result.output_tokens or 0)
        for result in scored
        if result.input_tokens is not None or result.output_tokens is not None
    ]
    return {
        "questions": len(results),
        "scored": len(scored),
        "passed": passed,
        "errors": len(results) - len(scored),
        "accuracy": round(passed / len(scored), 3) if scored else None,
        "median_seconds": round(statistics.median(seconds), 1) if seconds else None,
        "median_tokens": round(statistics.median(tokens)) if tokens else None,
    }


def write_demo_report(
    results: list[DemoQuestionResult],
    settings: Settings,
    path: Path | None = None,
) -> Path:
    """Write a publishable report; demo data is synthetic, so questions and SQL are kept."""
    target = path or DEMO_REPORT_DIRECTORY / f"demo-eval-{_slug(settings.model_name)}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": DEMO_REPORT_SCHEMA_VERSION,
        "model": settings.model_name,
        "run_timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "summary": summarize_demo_results(results),
        "results": [asdict(result) for result in results],
    }
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return target


def _slug(model_name: str) -> str:
    return "".join(character if character.isalnum() else "-" for character in model_name).strip("-")
