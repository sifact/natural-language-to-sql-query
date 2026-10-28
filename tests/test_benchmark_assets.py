"""Tests for strict local benchmark asset loading and preflight."""

from __future__ import annotations

import hashlib
import json
import subprocess
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Self

import pytest

from nl_sql.config import BENCHMARK_SETUP_DSN_ENV
from nl_sql.data.assets import (
    ASSET_DECLARATION_PATH,
    BenchmarkAssetError,
    _load_local_benchmark_assets_at,
    load_public_base_lite_tasks,
)
from nl_sql.data.selection import validate_run_limits
from nl_sql.database.templates import (
    verify_live_template_inventories,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _asset_record(kind: str, relative_path: str, path: Path) -> dict[str, object]:
    return {
        "kind": kind,
        "path": relative_path,
        "sha256": _sha256(path),
        "source_url": "https://example.test/dataset",
        "retrieved_on": "2026-10-27",
        "terms_id": "test-license",
        "terms_url": "https://example.test/terms",
        "terms_verified_by_user": True,
        "verified_on": "2026-10-27",
        "model_transmission_allowed": True,
    }


def _public_record(instance_id: str = "task-1") -> dict[str, object]:
    return {
        "instance_id": instance_id,
        "selected_database": "db-one",
        "query": "Find the matching record.",
        "category": "Query",
        "sol_sql": [],
        "preprocess_sql": [],
        "clean_up_sqls": [],
    }


def _gold_record(instance_id: str = "task-1") -> dict[str, object]:
    return {
        "instance_id": instance_id,
        "selected_database": "db-one",
        "query": "Find the matching record.",
        "sol_sql": ["SELECT id FROM items"],
        "test_cases": [],
        "external_knowledge": [],
        "preprocess_sql": [],
        "clean_up_sqls": [],
        "category": "Query",
        "high_level": "selection",
        "conditions": [],
        "difficulty_tier": "easy",
    }


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _workspace(tmp_path: Path) -> tuple[Path, dict[str, Any], dict[str, Path]]:
    root = tmp_path
    subprocess.run(["git", "init", "--quiet", str(root)], check=True)
    (root / ".gitignore").write_text("data/benchmark/\n", encoding="utf-8")
    asset_root = root / "data/benchmark"
    database_root = asset_root / "db-one"
    database_root.mkdir(parents=True)

    public_path = asset_root / "public.jsonl"
    gold_path = asset_root / "gold.jsonl"
    inventory_path = database_root / "inventory.json"
    database_asset_path = database_root / "schema.sql"
    _write_jsonl(public_path, [_public_record()])
    _write_jsonl(gold_path, [_gold_record()])
    inventory_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "selected_database": "db-one",
                "base_tables": ["public.items"],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    database_asset_path.write_text(
        "CREATE TABLE public.items (id integer);\n", encoding="utf-8"
    )

    declaration: dict[str, Any] = {
        "schema_version": 1,
        "dataset_revision": "fixture-revision",
        "assets": [
            _asset_record("public_tasks", "public.jsonl", public_path),
            _asset_record("gold_tasks", "gold.jsonl", gold_path),
            _asset_record("database", "db-one/schema.sql", database_asset_path),
            _asset_record(
                "database_inventory", "db-one/inventory.json", inventory_path
            ),
        ],
        "databases": {
            "db-one": {
                "template_database": "nl_sql_template_db_one",
                "asset_path": "db-one",
                "inventory_path": "db-one/inventory.json",
            }
        },
    }
    _write_declaration(root, declaration)
    return (
        root,
        declaration,
        {
            "public": public_path,
            "gold": gold_path,
            "inventory": inventory_path,
            "database": database_asset_path,
            "declaration": root / Path(*ASSET_DECLARATION_PATH.parts),
        },
    )


def _write_declaration(root: Path, declaration: dict[str, Any]) -> None:
    path = root / Path(*ASSET_DECLARATION_PATH.parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(declaration, sort_keys=True), encoding="utf-8")


def _refresh_asset_hashes(declaration: dict[str, Any], paths: dict[str, Path]) -> None:
    path_by_relative = {
        "public.jsonl": paths["public"],
        "gold.jsonl": paths["gold"],
        "db-one/inventory.json": paths["inventory"],
        "db-one/schema.sql": paths["database"],
    }
    for asset in declaration["assets"]:
        asset["sha256"] = _sha256(path_by_relative[asset["path"]])


def test_given_complete_local_assets_when_preflight_then_splits_model_and_evaluator_data(
    tmp_path: Path,
) -> None:
    # Arrange
    root, _, paths = _workspace(tmp_path)
    source_bytes = {
        name: path.read_bytes() for name, path in paths.items() if name != "declaration"
    }

    # Act
    result = _load_local_benchmark_assets_at(root)

    # Assert
    assert result.dataset_revision == "fixture-revision"
    assert len(result.assets) == 4
    assert result.observed_categories == ("Query",)
    assert result.run_blockers == ()
    assert result.databases["db-one"].base_tables == ("public.items",)
    assert result.tasks[0].model_input.query == "Find the matching record."
    assert set(result.tasks[0].model_input.__dataclass_fields__) == {
        "query",
        "selected_database",
    }
    assert result.tasks[0].model_input.selected_database == "db-one"
    assert result.tasks[0].evaluator.record["sol_sql"] == ["SELECT id FROM items"]
    assert result.tasks[0].evaluator.record["test_cases"] == []
    assert {
        name: path.read_bytes() for name, path in paths.items() if name != "declaration"
    } == source_bytes


def test_given_evaluator_only_gold_when_preflight_then_inherits_public_metadata(
    tmp_path: Path,
) -> None:
    root, declaration, paths = _workspace(tmp_path)
    gold = _gold_record()
    for field in (
        "selected_database",
        "query",
        "category",
        "high_level",
        "conditions",
        "difficulty_tier",
        "preprocess_sql",
        "clean_up_sqls",
    ):
        del gold[field]
    _write_jsonl(paths["gold"], [gold])
    _refresh_asset_hashes(declaration, paths)
    _write_declaration(root, declaration)

    result = _load_local_benchmark_assets_at(root)

    record = result.tasks[0].evaluator.record
    assert record["sol_sql"] == ["SELECT id FROM items"]
    assert record["test_cases"] == []
    assert record["query"] == "Find the matching record."
    assert record["category"] == "Query"
    assert record["preprocess_sql"] == []
    assert record["clean_up_sqls"] == []


def test_given_public_base_lite_file_when_loaded_then_parses_all_270_tasks() -> None:
    # Act
    records = load_public_base_lite_tasks()

    # Assert
    assert len(records) == 270
    assert len({record["instance_id"] for record in records}) == 270
    assert {
        category: sum(record["category"] == category for record in records)
        for category in ("Management", "Query")
    } == {"Management": 90, "Query": 180}
    for record in records:
        for field in ("sol_sql", "preprocess_sql", "clean_up_sqls"):
            value = record[field]
            assert isinstance(value, list)
            assert all(isinstance(statement, str) for statement in value)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("duplicate_public", "Duplicate instance_id"),
        ("missing_gold", "one-to-one instance_id coverage"),
        ("malformed_jsonl", "Malformed JSONL"),
        ("missing_field", "missing required fields"),
        ("mismatched_query", "query values differ"),
        ("no_query_category", "expected Query category"),
    ],
)
def test_given_invalid_task_records_when_preflight_then_fails_closed(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    # Arrange
    root, declaration, paths = _workspace(tmp_path)
    if mutation == "duplicate_public":
        _write_jsonl(paths["public"], [_public_record(), _public_record()])
    elif mutation == "missing_gold":
        _write_jsonl(paths["gold"], [])
    elif mutation == "malformed_jsonl":
        paths["public"].write_text('{"instance_id":\n', encoding="utf-8")
    elif mutation == "missing_field":
        _write_jsonl(paths["gold"], [{"instance_id": "task-1"}])
    elif mutation == "mismatched_query":
        gold = _gold_record()
        gold["query"] = "Different question"
        _write_jsonl(paths["gold"], [gold])
    else:
        gold = _gold_record()
        gold["category"] = "Management"
        _write_jsonl(paths["gold"], [gold])
    _refresh_asset_hashes(declaration, paths)
    _write_declaration(root, declaration)

    # Act
    with pytest.raises(BenchmarkAssetError, match=message) as error:
        _load_local_benchmark_assets_at(root)
    if mutation == "no_query_category":
        assert "Observed categories: 'Management'" in str(error.value)


def test_given_duplicate_json_keys_when_preflight_then_rejects_record(
    tmp_path: Path,
) -> None:
    # Arrange
    root, declaration, paths = _workspace(tmp_path)
    paths["public"].write_text(
        '{"instance_id":"task-1","instance_id":"task-2",'
        '"selected_database":"db-one","query":"Find it"}\n',
        encoding="utf-8",
    )
    _refresh_asset_hashes(declaration, paths)
    _write_declaration(root, declaration)

    # Act / Assert
    with pytest.raises(BenchmarkAssetError, match="Malformed JSONL"):
        _load_local_benchmark_assets_at(root)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("sol_sql", "SELECT id FROM items"),
        ("preprocess_sql", [None]),
        ("clean_up_sqls", "DROP TABLE items"),
    ],
)
def test_given_non_list_sql_field_when_preflight_then_rejects_record(
    tmp_path: Path,
    field: str,
    invalid_value: object,
) -> None:
    # Arrange
    root, declaration, paths = _workspace(tmp_path)
    public = _public_record()
    public[field] = invalid_value
    _write_jsonl(paths["public"], [public])
    _refresh_asset_hashes(declaration, paths)
    _write_declaration(root, declaration)

    # Act / Assert
    with pytest.raises(BenchmarkAssetError, match=f"invalid {field} field"):
        _load_local_benchmark_assets_at(root)


def test_given_hash_mismatch_when_preflight_then_rejects_asset(tmp_path: Path) -> None:
    # Arrange
    root, declaration, paths = _workspace(tmp_path)
    paths["database"].write_text("modified\n", encoding="utf-8")
    _write_declaration(root, declaration)

    # Act / Assert
    with pytest.raises(BenchmarkAssetError, match="SHA-256 digest does not match"):
        _load_local_benchmark_assets_at(root)


def test_given_traversal_asset_path_when_preflight_then_rejects_path(
    tmp_path: Path,
) -> None:
    # Arrange
    root, declaration, _paths = _workspace(tmp_path)
    declaration["assets"][0]["path"] = "../outside.jsonl"
    _write_declaration(root, declaration)

    # Act / Assert
    with pytest.raises(BenchmarkAssetError, match="unsafe path"):
        _load_local_benchmark_assets_at(root)


def test_given_symlink_asset_when_preflight_then_rejects_path(tmp_path: Path) -> None:
    # Arrange
    root, declaration, _paths = _workspace(tmp_path)
    target = root / "outside.jsonl"
    target.write_text(json.dumps(_public_record()), encoding="utf-8")
    link = root / "data/benchmark/public-link.jsonl"
    link.symlink_to(target)
    declaration["assets"][0]["path"] = "public-link.jsonl"
    declaration["assets"][0]["sha256"] = _sha256(target)
    _write_declaration(root, declaration)

    # Act / Assert
    with pytest.raises(BenchmarkAssetError, match="Symlinks are not allowed"):
        _load_local_benchmark_assets_at(root)


def test_given_unsupported_declaration_version_when_preflight_then_rejects_version(
    tmp_path: Path,
) -> None:
    # Arrange
    root, declaration, _ = _workspace(tmp_path)
    declaration["schema_version"] = 2
    _write_declaration(root, declaration)

    # Act / Assert
    with pytest.raises(BenchmarkAssetError, match="unsupported schema version"):
        _load_local_benchmark_assets_at(root)


def test_given_unverified_terms_when_preflight_then_blocks_run_without_losing_inventory(
    tmp_path: Path,
) -> None:
    # Arrange
    root, declaration, _ = _workspace(tmp_path)
    declaration["assets"][0]["terms_verified_by_user"] = False
    _write_declaration(root, declaration)

    # Act
    result = _load_local_benchmark_assets_at(root)

    # Assert
    assert result.run_blockers == ("asset_terms_not_verified",)
    assert len(result.tasks) == 1


def test_given_model_transfer_not_allowed_when_preflight_then_blocks_run(
    tmp_path: Path,
) -> None:
    # Arrange
    root, declaration, _ = _workspace(tmp_path)
    declaration["assets"][0]["model_transmission_allowed"] = False
    _write_declaration(root, declaration)

    # Act
    result = _load_local_benchmark_assets_at(root)

    # Assert
    assert result.run_blockers == ("model_transmission_not_authorized",)


def test_given_missing_local_postgres_configuration_when_run_preflight_then_blocks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    root, _, _ = _workspace(tmp_path)
    preflight = _load_local_benchmark_assets_at(root)
    monkeypatch.delenv(BENCHMARK_SETUP_DSN_ENV, raising=False)

    # Act / Assert
    with pytest.raises(BenchmarkAssetError, match=BENCHMARK_SETUP_DSN_ENV):
        verify_live_template_inventories(preflight)


def test_given_local_postgres_inventory_when_run_preflight_then_compares_static_tables(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    psycopg = pytest.importorskip("psycopg")

    root, _, _ = _workspace(tmp_path)
    preflight = _load_local_benchmark_assets_at(root)
    monkeypatch.setenv(
        BENCHMARK_SETUP_DSN_ENV,
        "host=localhost dbname=nl_sql_setup user=local-setup",
    )
    monkeypatch.setattr(
        psycopg,
        "connect",
        lambda *_args, **_kwargs: _FakePostgresConnection([("public", "items")]),
    )

    # Act / Assert
    verify_live_template_inventories(preflight)


def test_given_remote_postgres_target_when_run_preflight_then_rejects_before_connect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    psycopg = pytest.importorskip("psycopg")
    root, _, _ = _workspace(tmp_path)
    preflight = _load_local_benchmark_assets_at(root)
    monkeypatch.setenv(
        BENCHMARK_SETUP_DSN_ENV,
        "host=database.example.test dbname=nl_sql_setup user=local-setup",
    )
    connect_calls: list[object] = []
    monkeypatch.setattr(
        psycopg,
        "connect",
        lambda *_args, **_kwargs: connect_calls.append(object()),
    )

    # Act / Assert
    with pytest.raises(BenchmarkAssetError, match="restricted to a local"):
        verify_live_template_inventories(preflight)
    assert connect_calls == []


class _FakeCursor:
    def __init__(self, rows: list[tuple[object, ...]]) -> None:
        self.rows = rows

    def fetchone(self) -> tuple[object, ...] | None:
        return self.rows[0] if self.rows else None

    def fetchall(self) -> list[tuple[object, ...]]:
        return self.rows


class _FakePostgresConnection:
    def __init__(self, table_rows: list[tuple[str, str]]) -> None:
        self.table_rows = table_rows

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def transaction(self) -> nullcontext[None]:
        return nullcontext()

    def execute(
        self,
        statement: str,
        _parameters: tuple[list[str]] | None = None,
    ) -> _FakeCursor:
        if "FROM pg_roles AS role" in statement:
            return _FakeCursor([(False, False, False, False, False, False)])
        if "information_schema.tables" in statement:
            return _FakeCursor(self.table_rows)
        return _FakeCursor([])


def test_given_live_template_inventory_mismatch_when_run_preflight_then_refuses_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    psycopg = pytest.importorskip("psycopg")
    root, _, _ = _workspace(tmp_path)
    preflight = _load_local_benchmark_assets_at(root)
    monkeypatch.setenv(
        BENCHMARK_SETUP_DSN_ENV,
        "host=localhost dbname=nl_sql_setup user=local-setup",
    )
    monkeypatch.setattr(
        psycopg,
        "connect",
        lambda *_args, **_kwargs: _FakePostgresConnection([("public", "different")]),
    )

    # Act / Assert
    with pytest.raises(
        BenchmarkAssetError, match="does not match its hashed declaration"
    ):
        verify_live_template_inventories(preflight)


def test_given_inventory_with_unsorted_or_duplicate_tables_when_preflight_then_rejects_it(
    tmp_path: Path,
) -> None:
    # Arrange
    root, declaration, paths = _workspace(tmp_path)
    paths["inventory"].write_text(
        json.dumps(
            {
                "schema_version": 1,
                "selected_database": "db-one",
                "base_tables": ["public.items", "public.items"],
            }
        ),
        encoding="utf-8",
    )
    _refresh_asset_hashes(declaration, paths)
    _write_declaration(root, declaration)

    # Act / Assert
    with pytest.raises(BenchmarkAssetError, match="unique sorted base tables"):
        _load_local_benchmark_assets_at(root)


@pytest.mark.parametrize("max_tasks", [0, -1, 101, True, 1.5])
def test_given_invalid_task_limit_when_validated_then_rejects_it(
    max_tasks: object,
) -> None:
    with pytest.raises(BenchmarkAssetError, match="--max-tasks"):
        validate_run_limits(max_tasks)  # type: ignore[arg-type]


@pytest.mark.parametrize("max_tasks", [1, 100])
def test_given_bounded_task_limit_when_validated_then_accepts_it(
    max_tasks: int,
) -> None:
    validate_run_limits(max_tasks)
