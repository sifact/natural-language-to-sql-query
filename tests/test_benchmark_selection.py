"""Tests for deterministic article-shaped benchmark task selection."""

from __future__ import annotations

from types import MappingProxyType

import pytest

from nl_sql.data.assets import (
    BenchmarkTask,
    DatabaseInventory,
    EvaluatorTaskInput,
    ModelTaskInput,
    PreflightResult,
)
from nl_sql.data.selection import (
    select_article_shaped_tasks,
)

pytest.importorskip("sqlglot")


_INVENTORY_TABLES = tuple(f"public.table_{index}" for index in range(10))
_ELIGIBLE_SQL = (
    "SELECT a.id FROM public.table_0 AS a "
    "JOIN public.table_1 AS b ON a.id = b.id "
    "WHERE a.status = 'active' AND b.count > 2"
)


def _task(
    instance_id: str,
    sql: str | list[object],
    *,
    category: str = "Query",
) -> BenchmarkTask:
    canonical_sqls = [sql] if isinstance(sql, str) else sql
    return BenchmarkTask(
        instance_id=instance_id,
        selected_database="fixture-db",
        model_input=ModelTaskInput(query=f"Question for {instance_id}"),
        evaluator=EvaluatorTaskInput(
            record={"category": category, "sol_sql": canonical_sqls}
        ),
    )


def _preflight(
    *tasks: BenchmarkTask,
    base_tables: tuple[str, ...] = _INVENTORY_TABLES,
) -> PreflightResult:
    inventory = DatabaseInventory(
        selected_database="fixture-db",
        template_database="fixture-template",
        asset_path="fixture-db",
        inventory_path="fixture-db/inventory.json",
        base_tables=base_tables,
    )
    return PreflightResult(
        dataset_revision="authored-fixture",
        assets=(),
        databases=MappingProxyType({"fixture-db": inventory}),
        tasks=tuple(tasks),
        observed_categories=("Query",),
        run_blockers=(),
    )


@pytest.mark.parametrize(
    ("sql", "selected"),
    [
        (_ELIGIBLE_SQL, True),
        (
            (
                "SELECT a.id FROM public.table_0 a "
                "JOIN public.table_0 AS repeated ON a.id = repeated.id "
                "JOIN public.table_1 b ON a.id = b.id "
                "WHERE a.status = 'active' AND b.count > 2"
            ),
            True,
        ),
        (
            (
                "SELECT a.id FROM public.table_0 a "
                "JOIN public.table_1 b ON a.id = b.id "
                "WHERE a.status = 'active' OR b.count > 2"
            ),
            False,
        ),
        (
            (
                "SELECT a.id FROM public.table_0 a "
                "JOIN public.table_1 b ON a.id = b.id "
                "WHERE a.status = 'active' AND "
                "(b.count > 2 OR b.count IS NULL)"
            ),
            True,
        ),
        (
            "SELECT a.id FROM public.table_0 a JOIN public.table_1 b ON a.id = b.id",
            False,
        ),
        (
            (
                "SELECT a.id FROM public.table_0 a "
                "JOIN public.table_1 b ON a.id = b.id "
                "WHERE a.id IN (SELECT c.id FROM public.table_2 c "
                "WHERE c.x = 1 AND c.y = 2)"
            ),
            False,
        ),
        (
            (
                "WITH selected AS (SELECT id FROM public.table_0) "
                "SELECT s.id FROM selected s JOIN public.table_1 b ON s.id = b.id "
                "WHERE b.x = 1 AND b.y = 2"
            ),
            False,
        ),
        (
            (
                "SELECT derived.id FROM "
                "(SELECT id FROM public.table_0) AS derived "
                "JOIN public.table_1 b ON derived.id = b.id "
                "WHERE b.x = 1 AND b.y = 2"
            ),
            False,
        ),
        (
            (
                "SELECT a.id FROM public.table_0 a "
                "JOIN public.table_1 b ON a.id = b.id "
                "WHERE a.x = 1 AND b.y = 2 "
                "UNION SELECT c.id FROM public.table_2 c"
            ),
            False,
        ),
        (
            (
                "WITH removed AS (DELETE FROM public.table_0 "
                "RETURNING id) SELECT a.id FROM public.table_1 a "
                "JOIN public.table_2 b ON a.id = b.id "
                "WHERE a.x = 1 AND b.y = 2"
            ),
            False,
        ),
        ("DELETE FROM public.table_0 WHERE id = 1", False),
        ("SELECT ( FROM public.table_0", False),
        (_ELIGIBLE_SQL + "; SELECT 1", False),
    ],
)
def test_selector_applies_sql_and_outer_query_rules(sql: str, selected: bool) -> None:
    result = select_article_shaped_tasks(_preflight(_task("task-1", sql)))

    assert result.selected_ids == (("task-1",) if selected else ())


def test_selector_uses_distinct_outer_tables_and_database_inventory_size() -> None:
    too_few_database_tables = _INVENTORY_TABLES[:-1]
    query_with_three_outer_tables = (
        "SELECT a.id FROM public.table_0 a "
        "JOIN public.table_1 b ON a.id = b.id "
        "JOIN public.table_2 c ON b.id = c.id "
        "WHERE a.x = 1 AND c.y = 2"
    )
    query_with_four_outer_tables = (
        "SELECT a.id FROM public.table_0 a "
        "JOIN public.table_1 b ON a.id = b.id "
        "JOIN public.table_2 c ON b.id = c.id "
        "JOIN public.table_3 d ON c.id = d.id "
        "WHERE a.x = 1 AND d.y = 2"
    )

    assert (
        select_article_shaped_tasks(
            _preflight(
                _task("too-small-inventory", _ELIGIBLE_SQL),
                base_tables=too_few_database_tables,
            )
        ).selected_ids
        == ()
    )
    assert select_article_shaped_tasks(
        _preflight(_task("three-tables", query_with_three_outer_tables))
    ).selected_ids == ("three-tables",)
    assert (
        select_article_shaped_tasks(
            _preflight(_task("four-tables", query_with_four_outer_tables))
        ).selected_ids
        == ()
    )


@pytest.mark.parametrize(
    ("canonical_sqls", "selected", "unsupported_reason"),
    [
        ([], False, "canonical_sql_missing"),
        ([_ELIGIBLE_SQL, _ELIGIBLE_SQL], True, None),
        (
            [
                _ELIGIBLE_SQL,
                (
                    "SELECT a.id FROM public.table_0 a "
                    "JOIN public.table_1 b ON a.id = b.id"
                ),
            ],
            False,
            None,
        ),
        ([_ELIGIBLE_SQL, "SELECT ( FROM public.table_0"], False, "canonical_sql_parse_error"),
        ([_ELIGIBLE_SQL, 42], False, "canonical_sql_invalid"),
    ],
)
def test_selector_requires_every_canonical_sql_alternative_to_be_eligible(
    canonical_sqls: list[object],
    selected: bool,
    unsupported_reason: str | None,
) -> None:
    result = select_article_shaped_tasks(
        _preflight(_task("alternatives", canonical_sqls))
    )

    assert result.selected_ids == (("alternatives",) if selected else ())
    assert tuple(task.reason_code for task in result.unsupported_tasks) == (
        (unsupported_reason,) if unsupported_reason is not None else ()
    )


def test_selector_returns_stable_lexical_ids_and_records_criteria() -> None:
    tasks = (
        _task("task-10", _ELIGIBLE_SQL),
        _task("task-2", _ELIGIBLE_SQL),
        _task("not-query", _ELIGIBLE_SQL, category="Other"),
    )

    result = select_article_shaped_tasks(_preflight(*tasks))

    assert result.selected_ids == ("task-10", "task-2")
    assert result.criteria["category"] == "Query"
    assert result.criteria["selection_id"] == "article-shaped-query-v2"
    assert result.criteria["outer_base_table_count"] == {"minimum": 2, "maximum": 3}
    assert result.criteria["outer_where_predicates_minimum"] == 2
    assert result.unsupported_tasks == ()


def test_selector_reports_query_records_it_cannot_classify() -> None:
    result = select_article_shaped_tasks(
        _preflight(_task("bad-sql", "SELECT ( FROM public.table_0"))
    )

    assert result.selected_ids == ()
    assert [
        (task.instance_id, task.field, task.reason_code)
        for task in result.unsupported_tasks
    ] == [("bad-sql", "sol_sql", "canonical_sql_parse_error")]
