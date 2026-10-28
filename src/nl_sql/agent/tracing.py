"""Append ordered, credential-redacted events to one local JSONL file per task."""

from __future__ import annotations

import json
import re
import stat
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

from ..database.provider import serialize_database_value

_TASK_ID = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_DEFAULT_TRACE_DIRECTORY = Path("data/benchmark/traces")
_SENSITIVE_KEY_PARTS = (
    "api_key",
    "authorization",
    "connection_string",
    "credential",
    "dsn",
    "endpoint_key",
    "model_key",
    "password",
    "secret",
    "token",
)
_BEARER = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]+=*")
_URL_PASSWORD = re.compile(r"(?i)(postgres(?:ql)?://[^:/@\s]+:)[^@\s]+@")
_DSN_PASSWORD = re.compile(r"(?i)(\bpassword\s*=\s*)('[^']*'|\"[^\"]*\"|[^;\s]+)")


class TraceError(Exception):
    """Raised when a trace cannot be safely initialized or encoded."""


class TaskTracer:
    """Append structured conversation and tool events to a task-specific trace."""

    def __init__(
        self,
        task_id: str,
        *,
        trace_directory: Path = _DEFAULT_TRACE_DIRECTORY,
    ) -> None:
        """Create a new trace file without overwriting an earlier task trace.

        Args:
            task_id: Stable, filesystem-safe task identifier.
            trace_directory: Ignored local directory for JSONL trace files.

        Raises:
            TraceError: If the task identifier is invalid or its trace already exists.
            OSError: If the trace directory or file cannot be created.
        """
        if not _TASK_ID.fullmatch(task_id):
            raise TraceError("Task ID must be a filesystem-safe identifier.")
        self._workspace_root = Path.cwd().resolve(strict=True)
        self._confined_to_workspace = trace_directory == _DEFAULT_TRACE_DIRECTORY
        self.path = (
            self._workspace_root / trace_directory / f"{task_id}.jsonl"
            if self._confined_to_workspace
            else trace_directory / f"{task_id}.jsonl"
        )
        self._validate_trace_path(allow_missing=True)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._validate_trace_path(allow_missing=True)
        try:
            with self.path.open("x", encoding="utf-8"):
                pass
        except FileExistsError as error:
            raise TraceError("A trace already exists for this task.") from error
        self._sequence = 0

    def record_message(self, message: Mapping[str, object]) -> None:
        """Record one model or tool message."""
        self._append("message", {"message": message})

    def record_tool_call(
        self,
        name: str,
        arguments: Mapping[str, object],
        result: Mapping[str, object],
    ) -> None:
        """Record a tool invocation, its arguments, and its result."""
        self._append(
            "tool_call",
            {"name": name, "arguments": arguments, "result": result},
        )

    def _append(self, event_type: str, data: Mapping[str, object]) -> None:
        self._validate_trace_path(allow_missing=False)
        record = {
            "sequence": self._sequence,
            "timestamp": datetime.now(UTC).isoformat(),
            "event": event_type,
            "data": _redact(data),
        }
        try:
            line = json.dumps(
                record,
                default=serialize_database_value,
                separators=(",", ":"),
            )
        except (TypeError, ValueError) as error:
            raise TraceError("Trace event contains unsupported data.") from error
        with self.path.open("a", encoding="utf-8") as trace_file:
            trace_file.write(f"{line}\n")
        self._sequence += 1

    def _validate_trace_path(self, *, allow_missing: bool) -> None:
        if not self._confined_to_workspace:
            return
        try:
            relative = self.path.relative_to(self._workspace_root)
            resolved_root = self._workspace_root.resolve(strict=True)
        except (OSError, ValueError) as error:
            raise TraceError("Trace path must remain inside the workspace.") from error

        current = self._workspace_root
        for part in relative.parts:
            current /= part
            try:
                mode = current.lstat().st_mode
            except FileNotFoundError:
                if allow_missing:
                    break
                raise TraceError("Trace path is unavailable.")
            except OSError as error:
                raise TraceError("Trace path could not be inspected.") from error
            if stat.S_ISLNK(mode):
                raise TraceError("Symlinks are not allowed in the trace path.")

        try:
            self.path.resolve(strict=False).relative_to(resolved_root)
        except (OSError, ValueError) as error:
            raise TraceError("Trace path must remain inside the workspace.") from error


def _redact(value: object) -> object:
    if isinstance(value, Mapping):
        redacted: dict[str, object] = {}
        for key, item in value.items():
            safe_key = str(key)
            normalized_key = safe_key.casefold().replace("-", "_")
            redacted[safe_key] = (
                "[REDACTED]"
                if any(part in normalized_key for part in _SENSITIVE_KEY_PARTS)
                else _redact(item)
            )
        return redacted
    if isinstance(value, (list, tuple)):
        return [_redact(item) for item in value]
    if isinstance(value, str):
        return _DSN_PASSWORD.sub(
            r"\1[REDACTED]",
            _URL_PASSWORD.sub(
                r"\1[REDACTED]@", _BEARER.sub("Bearer [REDACTED]", value)
            ),
        )
    return value
