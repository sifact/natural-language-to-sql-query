"""Run selected tasks through isolated databases and score submitted SQL."""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from ..agent.model_client import ChatCompletionClient, ModelError
from ..agent.sql_agent import AgentError, AgentResult, SqlAgent
from ..agent.tracing import TaskTracer, TraceError
from ..config import Settings, SettingsError
from ..data.assets import (
    BenchmarkAssetError,
    BenchmarkTask,
    DatabaseInventory,
    ModelTaskInput,
    PreflightResult,
)
from ..data.knowledge import load_public_catalog
from ..data.selection import TaskSelection, validate_run_limits
from ..database.provider import DatabaseProvider, DatabaseToolError
from ..database.task_sandbox import PostgresTaskSandbox, TaskSandboxError
from ..database.templates import verify_live_template_inventories
from .matcher import MatchStatus, match_results
from .metrics import EvaluationMetrics, TaskStatus, calculate_metrics


class Agent(Protocol):
    """Minimal agent interface required by the task runner."""

    def run(self, question: str) -> AgentResult: ...


AgentFactory = Callable[[ModelTaskInput, DatabaseProvider, TaskTracer, Settings], Agent]
TemplateVerifier = Callable[[PreflightResult, Settings], None]
TracerFactory = Callable[[str], TaskTracer]


class TaskRunnerError(Exception):
    """A fixed-category run gate failure that prevents task execution."""

    def __init__(self, message: str, *, reason_code: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


@dataclass(frozen=True, slots=True)
class TaskResult:
    """Redacted, per-task status and execution metadata."""

    instance_id: str
    status: TaskStatus
    reason_code: str | None
    duration_ms: int
    input_tokens: int | None
    output_tokens: int | None
    trace_path: str | None


@dataclass(frozen=True, slots=True)
class TaskRunResult:
    """A bounded task run with aggregate metrics omitted after infrastructure abort."""

    run_id: str
    started_at_utc: str
    dataset_revision: str
    max_tasks: int
    selected_task_ids: tuple[str, ...]
    tasks: tuple[TaskResult, ...]
    duration_ms: int
    metrics: EvaluationMetrics | None
    aborted: bool
    abort_reason: str | None


class TaskSandbox(Protocol):
    """Isolated database lifecycle consumed by the runner."""

    def validate_configuration(self) -> None: ...

    def open_task(
        self,
        task: BenchmarkTask,
        inventory: DatabaseInventory,
        run_id: str,
    ) -> AbstractContextManager[DatabaseProvider]: ...


class TaskRunner:
    """Enforce run gates, execute each task in a clone, and score submitted SQL."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        sandbox: TaskSandbox | None = None,
        agent_factory: AgentFactory | None = None,
        tracer_factory: TracerFactory = TaskTracer,
        template_verifier: TemplateVerifier = verify_live_template_inventories,
    ) -> None:
        """Create a runner with injectable offline-test boundaries."""
        self.settings = settings or Settings.from_env()
        self.sandbox = (
            sandbox if sandbox is not None else PostgresTaskSandbox(self.settings)
        )
        self.agent_factory = (
            agent_factory if agent_factory is not None else _create_agent
        )
        self.tracer_factory = tracer_factory
        self.template_verifier = template_verifier

    def run(
        self,
        preflight: PreflightResult,
        selection: TaskSelection,
        *,
        max_tasks: int = 1,
        confirm_run: bool = False,
        allow_model_data_transfer: bool = False,
    ) -> TaskRunResult:
        """Run a bounded selector-produced selection using only verified assets."""
        started = time.monotonic()
        started_at_utc = (
            datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")
        )
        if not confirm_run:
            raise TaskRunnerError(
                "Explicit benchmark run confirmation is required.",
                reason_code="run_confirmation_required",
            )
        if not allow_model_data_transfer:
            raise TaskRunnerError(
                "Explicit model data-transfer authorization is required.",
                reason_code="model_transfer_confirmation_required",
            )
        try:
            validate_run_limits(max_tasks, self.settings)
        except BenchmarkAssetError as error:
            raise TaskRunnerError(
                "The task count is outside the configured run limit.",
                reason_code="invalid_run_limit",
            ) from error
        if preflight.run_blockers:
            raise TaskRunnerError(
                "Benchmark assets are blocked by incomplete authorization.",
                reason_code="asset_authorization_blocked",
            )
        if not _has_complete_asset_set(preflight):
            raise TaskRunnerError(
                "The benchmark asset set is incomplete.",
                reason_code="incomplete_assets",
            )
        selected = selection.selected_tasks[:max_tasks]
        if not selected:
            raise TaskRunnerError(
                "No eligible benchmark tasks were selected.",
                reason_code="empty_selection",
            )
        tasks_by_id = {task.instance_id: task for task in preflight.tasks}
        if any(tasks_by_id.get(task.instance_id) != task for task in selected):
            raise TaskRunnerError(
                "The selected tasks do not match the validated preflight.",
                reason_code="invalid_selection",
            )
        if len({task.instance_id for task in selected}) != len(selected):
            raise TaskRunnerError(
                "A selected task identifier is duplicated.",
                reason_code="invalid_selection",
            )
        inventories: dict[str, DatabaseInventory] = {}
        for task in selected:
            inventory = preflight.databases.get(task.selected_database)
            if inventory is None:
                raise TaskRunnerError(
                    "A selected task has no validated database inventory.",
                    reason_code="incomplete_assets",
                )
            inventories[task.instance_id] = inventory

        run_id = uuid.uuid4().hex
        unsupported_reasons = {
            task.instance_id: _unsupported_task_reason(task, self.settings)
            for task in selected
        }
        eligible = [
            task for task in selected if unsupported_reasons[task.instance_id] is None
        ]
        if eligible:
            if self.agent_factory is _create_agent:
                try:
                    self.settings.require_model()
                except SettingsError as error:
                    raise TaskRunnerError(
                        "Model configuration is required before task execution.",
                        reason_code="model_configuration_missing",
                    ) from error
            try:
                self.sandbox.validate_configuration()
                self.template_verifier(preflight, self.settings)
            except TaskSandboxError as error:
                raise TaskRunnerError(
                    "Local benchmark database preflight failed.",
                    reason_code=error.reason_code,
                ) from error
            except BenchmarkAssetError as error:
                raise TaskRunnerError(
                    "Local benchmark database preflight failed.",
                    reason_code=error.category,
                ) from error

        outcomes: list[TaskResult] = []
        aborted = False
        abort_reason: str | None = None
        for task_index, task in enumerate(selected):
            task_started = time.monotonic()
            reason = unsupported_reasons[task.instance_id]
            if reason is not None:
                outcomes.append(
                    _task_result(
                        task,
                        status="unsupported",
                        reason_code=reason,
                        duration_ms=_elapsed_ms(task_started),
                    )
                )
                continue
            inventory = inventories[task.instance_id]
            trace_id = uuid.uuid5(uuid.UUID(hex=run_id), task.instance_id).hex
            trace_path: str | None = None
            try:
                with self.sandbox.open_task(task, inventory, run_id) as database:
                    tracer = self.tracer_factory(trace_id)
                    trace_path = tracer.path.name
                    agent = self.agent_factory(
                        task.model_input,
                        database,
                        tracer,
                        self.settings,
                    )
                    agent_result = agent.run(task.model_input.query)
                    task_result = self._evaluate(
                        task,
                        database,
                        agent_result,
                        trace_path,
                        task_started,
                    )
            except TaskSandboxError as error:
                task_result = _task_result(
                    task,
                    status="incomplete",
                    reason_code=error.reason_code,
                    duration_ms=_elapsed_ms(task_started),
                    trace_path=trace_path,
                )
                aborted = True
                abort_reason = error.reason_code
            except (AgentError, ModelError, TraceError, DatabaseToolError, OSError):
                task_result = _task_result(
                    task,
                    status="incomplete",
                    reason_code="task_execution_failed",
                    duration_ms=_elapsed_ms(task_started),
                    trace_path=trace_path,
                )
            outcomes.append(task_result)
            if aborted:
                outcomes.extend(
                    _task_result(
                        remaining,
                        status="skipped",
                        reason_code="run_aborted",
                        duration_ms=0,
                    )
                    for remaining in selected[task_index + 1 :]
                )
                break

        metrics = (
            None if aborted else calculate_metrics(result.status for result in outcomes)
        )
        return TaskRunResult(
            run_id=run_id,
            started_at_utc=started_at_utc,
            dataset_revision=preflight.dataset_revision,
            max_tasks=max_tasks,
            selected_task_ids=tuple(task.instance_id for task in selected),
            tasks=tuple(outcomes),
            duration_ms=_elapsed_ms(started),
            metrics=metrics,
            aborted=aborted,
            abort_reason=abort_reason,
        )

    def _evaluate(
        self,
        task: BenchmarkTask,
        database: DatabaseProvider,
        result: AgentResult,
        trace_path: str,
        started: float,
    ) -> TaskResult:
        if result.incomplete:
            return _task_result(
                task,
                status="incomplete",
                reason_code="agent_round_limit",
                duration_ms=_elapsed_ms(started),
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                trace_path=trace_path,
            )
        if result.clarification is not None:
            return _task_result(
                task,
                status="incomplete",
                reason_code="clarification_required",
                duration_ms=_elapsed_ms(started),
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                trace_path=trace_path,
            )
        if (
            not isinstance(result.answer, str)
            or not result.answer.strip()
            or not isinstance(result.sql, str)
            or not result.sql.strip()
        ):
            return _task_result(
                task,
                status="incomplete",
                reason_code="submission_missing",
                duration_ms=_elapsed_ms(started),
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                trace_path=trace_path,
            )
        if (
            len(result.answer) > self.settings.max_tool_argument_length
            or len(result.sql) > self.settings.max_tool_argument_length
        ):
            return _task_result(
                task,
                status="incomplete",
                reason_code="invalid_submission",
                duration_ms=_elapsed_ms(started),
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                trace_path=trace_path,
            )
        status, reason_code = _score_submission(
            task,
            database,
            result.sql,
            self.settings,
        )
        return _task_result(
            task,
            status=status,
            reason_code=reason_code,
            duration_ms=_elapsed_ms(started),
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            trace_path=trace_path,
        )


def _create_agent(
    model_input: ModelTaskInput,
    database: DatabaseProvider,
    tracer: TaskTracer,
    settings: Settings,
) -> Agent:
    """Construct the normal single-agent implementation after run gates pass.

    The task database's public knowledge base stands in for the article's domain
    hints; without it, questions built on defined business terms cannot be answered.
    """
    catalog = (
        load_public_catalog(model_input.selected_database)
        if model_input.selected_database
        else None
    )
    return SqlAgent(
        ChatCompletionClient(settings),
        database,
        knowledge_catalog=catalog,
        settings=settings,
        tracer=tracer,
    )


def _has_complete_asset_set(preflight: PreflightResult) -> bool:
    kinds = {asset.kind for asset in preflight.assets}
    return {
        "public_tasks",
        "gold_tasks",
        "database",
        "database_inventory",
    }.issubset(kinds)


def _unsupported_task_reason(task: BenchmarkTask, settings: Settings) -> str | None:
    if (
        not task.model_input.query.strip()
        or len(task.model_input.query) > settings.max_tool_argument_length
    ):
        return "invalid_task_input"
    record = task.evaluator.record
    canonical_sql = record.get("sol_sql")
    if not isinstance(canonical_sql, list) or not canonical_sql:
        return "missing_gold_answer"
    if len(canonical_sql) > 100 or any(
        not isinstance(statement, str)
        or not statement.strip()
        or len(statement) > settings.max_tool_argument_length
        for statement in canonical_sql
    ):
        return "invalid_gold_answer"
    test_cases = record.get("test_cases")
    if not isinstance(test_cases, list):
        return "invalid_test_case_semantics"
    if test_cases:
        return "unknown_test_case_semantics"
    for field in ("preprocess_sql", "clean_up_sqls"):
        statements = record.get(field)
        if not isinstance(statements, list) or any(
            not isinstance(statement, str) or not statement.strip()
            for statement in statements
        ):
            return "invalid_setup_sql"
    return None


def _score_submission(
    task: BenchmarkTask,
    database: DatabaseProvider,
    submitted_sql: str,
    settings: Settings,
) -> tuple[TaskStatus, str | None]:
    try:
        submitted_result = database.query(submitted_sql)
    except DatabaseToolError:
        return "failed", "submitted_query_failed"
    if submitted_result.truncated:
        return "unsupported", "submitted_result_truncated"

    canonical_sqls = task.evaluator.record["sol_sql"]
    if not isinstance(canonical_sqls, list):
        return "unsupported", "invalid_gold_answer"
    unmatched: list[tuple[MatchStatus, str | None]] = []
    for canonical_sql in canonical_sqls:
        if not isinstance(canonical_sql, str):
            return "unsupported", "invalid_gold_answer"
        try:
            canonical_result = database.query(canonical_sql)
        except DatabaseToolError:
            unmatched.append(("unsupported", "canonical_query_failed"))
            continue
        if canonical_result.truncated:
            unmatched.append(("unsupported", "canonical_result_truncated"))
            continue
        match = match_results(canonical_result, submitted_result, settings)
        if match.status == "passed":
            return "passed", None
        unmatched.append((match.status, match.reason_code))
    for status, reason in unmatched:
        if status == "unsupported":
            return "unsupported", reason or "unsupported_result"
    return "failed", unmatched[0][1] if unmatched else "missing_gold_answer"


def _task_result(
    task: BenchmarkTask,
    *,
    status: TaskStatus,
    reason_code: str | None,
    duration_ms: int,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    trace_path: str | None = None,
) -> TaskResult:
    return TaskResult(
        instance_id=task.instance_id,
        status=status,
        reason_code=reason_code,
        duration_ms=duration_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        trace_path=trace_path,
    )


def _elapsed_ms(started: float) -> int:
    return max(0, round((time.monotonic() - started) * 1_000))
