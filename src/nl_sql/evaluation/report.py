"""Build deterministic reports that contain statuses but no task content."""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import TypeAlias

from ..config import Settings
from ..data.assets import (
    BenchmarkAssetError,
    PreflightResult,
    resolve_report_path,
)
from ..data.selection import TaskSelection
from .metrics import TaskStatus, calculate_metrics
from .task_runner import TaskResult, TaskRunResult

Report: TypeAlias = dict[str, object]

_RUN_ID = re.compile(r"[0-9a-f]{32}\Z")
_TASK_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_DATASET_REVISION = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_MODEL_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/+-]{0,127}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_REASON_CODES = frozenset(
    {
        "agent_round_limit",
        "canonical_query_failed",
        "canonical_result_truncated",
        "cleanup_failed",
        "clarification_required",
        "clone_drop_failed",
        "clone_ownership_mismatch",
        "database_inventory_mismatch",
        "database_setup_failed",
        "driver_unavailable",
        "incomplete_assets",
        "invalid_clone_name",
        "invalid_gold_answer",
        "invalid_run_id",
        "invalid_setup_sql",
        "invalid_submission",
        "invalid_task_input",
        "invalid_test_case_semantics",
        "non_finite_number",
        "non_string_json_key",
        "unsupported_value_type",
        "unknown_test_case_semantics",
        "row_count_mismatch",
        "predicted_column_count_too_small",
        "column_values_mismatch",
        "row_values_mismatch",
        "truncated_result",
        "invalid_column_labels",
        "inconsistent_result_rows",
        "duplicate_column_labels",
        "missing_row_value",
        "invalid_result_row",
        "result_row_width_mismatch",
        "invalid_database_configuration",
        "database_configuration_missing",
        "database_roles_not_separated",
        "setup_sql_limit_exceeded",
        "submitted_query_failed",
        "submitted_result_truncated",
        "submission_missing",
        "task_execution_failed",
        "task_sql_failed",
        "unsafe_database_target",
        "unsupported_result",
        "run_aborted",
    }
)


class ReportError(ValueError):
    """Raised when a run contains values outside the redacted report contract."""


def build_report(
    run: TaskRunResult,
    settings: Settings | None = None,
    *,
    preflight: PreflightResult,
    selection: TaskSelection,
) -> Report:
    """Build reproducibility metadata without benchmark contents or credentials."""
    runtime = settings or Settings()
    if _RUN_ID.fullmatch(run.run_id) is None:
        raise ReportError("The run identifier is invalid.")
    if _DATASET_REVISION.fullmatch(run.dataset_revision) is None:
        raise ReportError("The dataset revision is invalid.")
    if run.dataset_revision != preflight.dataset_revision:
        raise ReportError("Run and preflight dataset revisions do not agree.")
    _safe_timestamp(run.started_at_utc)
    if run.aborted != (run.metrics is None):
        raise ReportError("Aborted runs must omit aggregate metrics.")
    if run.aborted != (run.abort_reason is not None):
        raise ReportError("The run abort status and reason do not agree.")
    if (
        type(run.duration_ms) is not int
        or run.duration_ms < 0
        or type(run.max_tasks) is not int
        or run.max_tasks < 1
        or tuple(task.instance_id for task in run.tasks) != run.selected_task_ids
        or tuple(sorted(run.selected_task_ids)) != run.selected_task_ids
    ):
        raise ReportError("Run metadata does not match the report contract.")
    if run.selected_task_ids != selection.selected_ids[: run.max_tasks]:
        raise ReportError("Run task IDs do not match the deterministic selection.")
    model_name = runtime.model_name or "unconfigured"
    if _MODEL_NAME.fullmatch(model_name) is None:
        raise ReportError("The configured model name is outside the report contract.")
    input_hashes = _input_file_hashes(preflight)
    criteria = selection.criteria
    try:
        json.dumps(criteria, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ReportError("Selection criteria are not JSON-safe.") from error

    counts = calculate_metrics(task.status for task in run.tasks)
    metrics = run.metrics
    if metrics is not None and counts != metrics:
        raise ReportError("Run metrics do not match its task outcomes.")
    score = None
    if metrics is not None:
        score = {
            "numerator": metrics.passed_tasks,
            "denominator": metrics.evaluated_tasks,
            "accuracy": str(metrics.accuracy) if metrics.accuracy is not None else None,
        }
    return {
        "schema_version": 1,
        "run_id": run.run_id,
        "run_timestamp_utc": run.started_at_utc,
        "dataset_revision": run.dataset_revision,
        "input_file_hashes": input_hashes,
        "selection_criteria": criteria,
        "selected_task_ids": list(run.selected_task_ids),
        "model": {
            "provider": "openai-compatible-chat-completions",
            "name": model_name,
        },
        "authorization": {
            "run_confirmed": True,
            "model_data_transfer_allowed": True,
        },
        "duration_ms": run.duration_ms,
        "resource_limits": {
            "max_tasks": run.max_tasks,
            "max_rounds_per_task": runtime.max_rounds,
            "max_tool_argument_length": runtime.max_tool_argument_length,
            "agent_row_cap": runtime.row_cap,
            "postgres_row_cap": runtime.postgres_row_cap,
            "sqlite_query_timeout_seconds": str(runtime.sqlite_query_timeout_seconds),
            "postgres_statement_timeout_seconds": (
                runtime.postgres_statement_timeout_seconds
            ),
            "postgres_lock_timeout_seconds": runtime.postgres_lock_timeout_seconds,
            "model_timeout_seconds": str(runtime.model_timeout_seconds),
            "seed": runtime.seed,
        },
        "numeric_tolerances": {
            "absolute": str(runtime.numeric_absolute_tolerance),
            "relative": str(runtime.numeric_relative_tolerance),
        },
        "status_counts": {
            "passed": counts.passed_tasks,
            "failed": counts.failed_tasks,
            "incomplete": counts.incomplete_tasks,
            "unsupported": counts.unsupported_tasks,
            "skipped": counts.skipped_tasks,
        },
        "score": score,
        "aborted": run.aborted,
        "abort_reason": _safe_reason(run.abort_reason),
        "tasks": [_task_payload(task) for task in run.tasks],
    }


def serialize_report(
    run: TaskRunResult,
    settings: Settings | None = None,
    *,
    preflight: PreflightResult,
    selection: TaskSelection,
) -> str:
    """Serialize a redacted report with stable key ordering and compact JSON."""
    return json.dumps(
        build_report(run, settings, preflight=preflight, selection=selection),
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def validate_report_destination(
    workspace_root: Path,
    report_name: str | None = None,
) -> None:
    """Prepare and validate the ignored report destination before running tasks."""
    try:
        resolve_report_path(
            workspace_root,
            report_name,
            create_directory=True,
        )
    except BenchmarkAssetError as error:
        raise ReportError("The benchmark report destination is not safe.") from error


def write_report(
    run: TaskRunResult,
    settings: Settings | None = None,
    *,
    preflight: PreflightResult,
    selection: TaskSelection,
    workspace_root: Path | None = None,
    report_name: str | None = None,
) -> str:
    """Write a new deterministic report below the ignored benchmark reports root."""
    report_text = serialize_report(
        run,
        settings,
        preflight=preflight,
        selection=selection,
    )
    root = Path.cwd() if workspace_root is None else workspace_root
    name = f"run-{run.run_id}.json" if report_name is None else report_name
    try:
        destination = resolve_report_path(root, name, create_directory=True)
    except BenchmarkAssetError as error:
        raise ReportError("The benchmark report destination is not safe.") from error
    try:
        with destination.open("x", encoding="utf-8") as file:
            file.write(report_text)
            file.write("\n")
    except FileExistsError as error:
        raise ReportError("The benchmark report already exists.") from error
    except OSError as error:
        raise ReportError("The benchmark report could not be written.") from error
    return destination.relative_to(root.resolve(strict=True)).as_posix()


def _input_file_hashes(preflight: PreflightResult) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for asset in preflight.assets:
        if (
            asset.kind
            not in {"public_tasks", "gold_tasks", "database", "database_inventory"}
            or _SHA256.fullmatch(asset.sha256) is None
        ):
            raise ReportError("A benchmark asset hash entry is invalid.")
        entries.append({"kind": asset.kind, "sha256": asset.sha256})
    return sorted(entries, key=lambda item: (item["kind"], item["sha256"]))


def _safe_timestamp(value: str) -> None:
    if not value.endswith("Z"):
        raise ReportError("The run timestamp must be UTC ISO 8601.")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ReportError("The run timestamp must be UTC ISO 8601.") from error
    if parsed.utcoffset() != timedelta(0):
        raise ReportError("The run timestamp must be UTC ISO 8601.")


def _task_payload(task: TaskResult) -> dict[str, object]:
    if not _TASK_ID.fullmatch(task.instance_id):
        raise ReportError("A task identifier is invalid.")
    if task.status not in {"passed", "failed", "incomplete", "unsupported", "skipped"}:
        raise ReportError("A task status is outside the report contract.")
    if (
        type(task.duration_ms) is not int
        or task.duration_ms < 0
        or any(
            value is not None and (type(value) is not int or value < 0)
            for value in (task.input_tokens, task.output_tokens)
        )
    ):
        raise ReportError("Task duration or token counts are invalid.")
    trace_path = _safe_trace_path(task.trace_path)
    return {
        "instance_id": task.instance_id,
        "status": _status_value(task.status),
        "reason_code": _safe_reason(task.reason_code),
        "duration_ms": task.duration_ms,
        "input_tokens": task.input_tokens,
        "output_tokens": task.output_tokens,
        "trace_path": trace_path,
    }


def _status_value(status: TaskStatus) -> str:
    return status


def _safe_reason(reason: str | None) -> str | None:
    if reason is None:
        return None
    if reason not in _REASON_CODES:
        raise ReportError("A report reason code is outside the fixed contract.")
    return reason


def _safe_trace_path(path: str | None) -> str | None:
    if path is None:
        return None
    candidate = PurePosixPath(path)
    windows_candidate = PureWindowsPath(path)
    if (
        candidate.is_absolute()
        or windows_candidate.is_absolute()
        or len(candidate.parts) != 1
        or candidate.name != path
        or "\\" in path
    ):
        raise ReportError("Trace paths in reports must be relative file names.")
    return path
