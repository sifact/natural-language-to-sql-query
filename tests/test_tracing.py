"""Tests for task-scoped, credential-redacted JSONL tracing."""

import json

import pytest

from nl_sql.agent import TaskTracer, TraceError


def test_given_task_events_when_recorded_then_appends_ordered_jsonl_records(
    tmp_path,
) -> None:
    tracer = TaskTracer("task-7", trace_directory=tmp_path)

    tracer.record_message({"role": "user", "content": "Count shipped orders."})
    tracer.record_tool_call(
        "query",
        {"sql": "SELECT COUNT(*) FROM orders"},
        {"rows": [{"count": 3}]},
    )

    records = [
        json.loads(line)
        for line in tracer.path.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["sequence"] for record in records] == [0, 1]
    assert [record["event"] for record in records] == ["message", "tool_call"]
    assert records[1]["data"]["arguments"]["sql"] == "SELECT COUNT(*) FROM orders"


def test_given_credentials_in_events_when_recorded_then_redacts_secret_values(
    tmp_path,
) -> None:
    tracer = TaskTracer("task-8", trace_directory=tmp_path)

    tracer.record_tool_call(
        "connect",
        {
            "api_key": "api-secret",
            "sql": "SELECT 'postgresql://user:db-secret@localhost/db'",
            "password": "field-secret",
        },
        {"authorization": "Bearer bearer-secret"},
    )

    serialized = tracer.path.read_text(encoding="utf-8")
    assert "api-secret" not in serialized
    assert "db-secret" not in serialized
    assert "field-secret" not in serialized
    assert "bearer-secret" not in serialized
    assert "[REDACTED]" in serialized


@pytest.mark.parametrize("task_id", ["", "../task", "contains/slash", "with spaces"])
def test_given_unsafe_task_id_when_tracer_created_then_rejects_it(
    tmp_path,
    task_id: str,
) -> None:
    with pytest.raises(TraceError, match="filesystem-safe"):
        TaskTracer(task_id, trace_directory=tmp_path)


def test_given_existing_task_trace_when_tracer_created_again_then_does_not_overwrite(
    tmp_path,
) -> None:
    first = TaskTracer("task-9", trace_directory=tmp_path)

    with pytest.raises(TraceError, match="already exists"):
        TaskTracer("task-9", trace_directory=tmp_path)

    assert first.path.read_text(encoding="utf-8") == ""


def test_given_unserializable_event_when_recorded_then_surfaces_trace_error(
    tmp_path,
) -> None:
    tracer = TaskTracer("task-10", trace_directory=tmp_path)

    with pytest.raises(TraceError, match="unsupported data"):
        tracer.record_message({"content": object()})


def test_given_default_trace_directory_symlink_when_created_then_rejects_escape(
    tmp_path,
    monkeypatch,
) -> None:
    benchmark_directory = tmp_path / "data" / "benchmark"
    benchmark_directory.mkdir(parents=True)
    outside_directory = tmp_path / "outside"
    outside_directory.mkdir()
    (benchmark_directory / "traces").symlink_to(
        outside_directory,
        target_is_directory=True,
    )
    monkeypatch.chdir(tmp_path)

    with pytest.raises(TraceError, match="Symlinks"):
        TaskTracer("task-11")

    assert not (outside_directory / "task-11.jsonl").exists()


def test_given_data_ancestor_symlink_when_created_then_rejects_escape(
    tmp_path,
    monkeypatch,
) -> None:
    outside_directory = tmp_path / "outside"
    outside_directory.mkdir()
    (tmp_path / "data").symlink_to(outside_directory, target_is_directory=True)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(TraceError, match="Symlinks"):
        TaskTracer("task-14")

    assert not (outside_directory / "benchmark" / "traces").exists()


def test_given_trace_directory_replaced_by_symlink_when_appending_then_rejects_escape(
    tmp_path,
    monkeypatch,
) -> None:
    benchmark_directory = tmp_path / "data" / "benchmark"
    trace_directory = benchmark_directory / "traces"
    trace_directory.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    tracer = TaskTracer("task-12")
    outside_directory = tmp_path / "outside"
    outside_directory.mkdir()
    trace_directory.rename(benchmark_directory / "traces-original")
    trace_directory.symlink_to(outside_directory, target_is_directory=True)

    with pytest.raises(TraceError, match="Symlinks"):
        tracer.record_message({"role": "user", "content": "Do not append outside."})

    assert not (outside_directory / "task-12.jsonl").exists()


def test_given_trace_file_replaced_by_symlink_when_appending_then_rejects_escape(
    tmp_path,
    monkeypatch,
) -> None:
    (tmp_path / "data" / "benchmark" / "traces").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    tracer = TaskTracer("task-13")
    outside_file = tmp_path / "outside.jsonl"
    outside_file.write_text("existing data\n", encoding="utf-8")
    tracer.path.unlink()
    tracer.path.symlink_to(outside_file)

    with pytest.raises(TraceError, match="Symlinks"):
        tracer.record_message({"role": "user", "content": "Do not append outside."})

    assert outside_file.read_text(encoding="utf-8") == "existing data\n"
