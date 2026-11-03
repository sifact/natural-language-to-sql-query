"""Task evaluation, result matching, metrics, and reports."""

from .demo_eval import (
    DemoQuestionResult,
    evaluate_demo_questions,
    summarize_demo_results,
    write_demo_report,
)
from .matcher import MatchResult, MatchStatus, match_results
from .metrics import EvaluationMetrics, TaskStatus, calculate_metrics
from .normalize import (
    NormalizedRow,
    NormalizedValue,
    UnsupportedValueError,
    normalize_value,
    values_match,
)
from .report import (
    Report,
    ReportError,
    build_report,
    serialize_report,
    validate_report_destination,
    write_report,
)
from .task_runner import (
    Agent,
    TaskResult,
    TaskRunner,
    TaskRunnerError,
    TaskRunResult,
    TaskSandbox,
)

__all__ = [
    "Agent",
    "DemoQuestionResult",
    "EvaluationMetrics",
    "MatchResult",
    "MatchStatus",
    "NormalizedRow",
    "NormalizedValue",
    "Report",
    "ReportError",
    "TaskResult",
    "TaskRunResult",
    "TaskRunner",
    "TaskRunnerError",
    "TaskSandbox",
    "TaskStatus",
    "UnsupportedValueError",
    "build_report",
    "calculate_metrics",
    "evaluate_demo_questions",
    "match_results",
    "normalize_value",
    "serialize_report",
    "summarize_demo_results",
    "validate_report_destination",
    "values_match",
    "write_demo_report",
    "write_report",
]
