"""Deterministic selection of the article-shaped PostgreSQL task slice."""

from __future__ import annotations

from dataclasses import dataclass

from ..config import Settings
from .assets import (
    BenchmarkAssetError,
    BenchmarkTask,
    DatabaseInventory,
    PreflightResult,
)


@dataclass(frozen=True, slots=True)
class UnsupportedTask:
    """A Query-category task the selector could not classify safely."""

    instance_id: str
    field: str
    reason_code: str


@dataclass(frozen=True, slots=True)
class TaskSelection:
    """Deterministic selected tasks and unsupported SQL records."""

    selected_tasks: tuple[BenchmarkTask, ...]
    unsupported_tasks: tuple[UnsupportedTask, ...]

    @property
    def selected_ids(self) -> tuple[str, ...]:
        """Return selected task IDs in stable lexical order."""
        return tuple(task.instance_id for task in self.selected_tasks)

    @property
    def criteria(self) -> dict[str, object]:
        """Return the versioned, JSON-compatible local selection rules."""
        return {
            "schema_version": 1,
            "selection_id": "article-shaped-query-v2",
            "category": "Query",
            "canonical_sql": {
                "dialect": "postgres",
                "statement": "each listed answer is exactly one SELECT",
                "empty_list": "unsupported",
                "compound_queries": "unsupported",
                "eligibility": "every listed answer must meet the query-slice criteria",
            },
            "outer_base_table_count": {"minimum": 2, "maximum": 3},
            "outer_where_predicates_minimum": 2,
            "database_base_table_count_minimum": 10,
            "table_count_scope": (
                "outer FROM/JOIN base tables; aliases and repeated references count "
                "once; CTE, derived-table, and subquery references are excluded"
            ),
            "where_predicate_counting": (
                "flatten top-level AND expressions; each OR expression counts as one"
            ),
        }


class _UnsupportedClassification(Exception):
    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


def _sql_identifier_value(identifier: object) -> str:
    from sqlglot import exp

    if not isinstance(identifier, exp.Identifier):
        raise _UnsupportedClassification("unclassified_table_identifier")
    value = str(identifier.this)
    return value if identifier.args.get("quoted") else value.lower()


def _outer_cte_names(select: object) -> set[str]:
    with_clause = select.args.get("with_")
    if with_clause is None:
        return set()
    names: set[str] = set()
    for cte in with_clause.expressions:
        alias = cte.args.get("alias")
        identifier = getattr(alias, "this", None)
        names.add(_sql_identifier_value(identifier))
    return names


def _resolve_outer_tables(
    select: object,
    inventory: DatabaseInventory,
) -> set[str]:
    from sqlglot import exp

    cte_names = _outer_cte_names(select)
    inventory_tables = set(inventory.base_tables)
    tables_by_name: dict[str, list[str]] = {}
    for qualified_name in inventory.base_tables:
        schema, table_name = qualified_name.split(".", maxsplit=1)
        tables_by_name.setdefault(table_name, []).append(f"{schema}.{table_name}")

    sources: list[object] = []
    from_clause = select.args.get("from_")
    if from_clause is not None:
        if from_clause.this is not None:
            sources.append(from_clause.this)
        sources.extend(from_clause.expressions)
    sources.extend(join.this for join in select.args.get("joins") or [])

    physical_tables: set[str] = set()
    for source in sources:
        if isinstance(source, (exp.Subquery, exp.Values)):
            continue
        if isinstance(source, exp.Lateral):
            if isinstance(source.this, (exp.Subquery, exp.Values)):
                continue
            raise _UnsupportedClassification("unclassified_lateral_source")
        if not isinstance(source, exp.Table):
            raise _UnsupportedClassification("unclassified_outer_table_source")
        if not isinstance(source.this, exp.Identifier):
            raise _UnsupportedClassification("table_valued_function")

        name = _sql_identifier_value(source.this)
        schema_identifier = source.args.get("db")
        catalog_identifier = source.args.get("catalog")
        if catalog_identifier is not None:
            raise _UnsupportedClassification("catalog_qualified_table")
        if schema_identifier is None and name in cte_names:
            continue
        if schema_identifier is None:
            candidates = tables_by_name.get(name, [])
            if len(candidates) != 1:
                raise _UnsupportedClassification("unresolved_or_ambiguous_table")
            qualified_name = candidates[0]
        else:
            schema = _sql_identifier_value(schema_identifier)
            qualified_name = f"{schema}.{name}"
            if qualified_name not in inventory_tables:
                raise _UnsupportedClassification("table_not_in_inventory")
        physical_tables.add(qualified_name)
    return physical_tables


def _where_predicate_count(select: object) -> int:
    from sqlglot import exp

    where_clause = select.args.get("where")
    if where_clause is None:
        return 0

    def count(expression: object) -> int:
        if isinstance(expression, exp.Paren):
            return count(expression.this)
        if isinstance(expression, exp.And):
            return count(expression.this) + count(expression.expression)
        return 1

    return count(where_clause.this)


def _classify_canonical_sql(
    sql: str,
    inventory: DatabaseInventory,
) -> tuple[set[str], int]:
    import sqlglot
    from sqlglot import exp

    if not sql.strip() or len(sql) > 20_000:
        raise _UnsupportedClassification("canonical_sql_size_or_empty")
    try:
        statements = [
            statement
            for statement in sqlglot.parse(sql, read="postgres")
            if statement is not None
        ]
    except sqlglot.errors.ParseError as error:
        raise _UnsupportedClassification("canonical_sql_parse_error") from error
    if len(statements) != 1:
        raise _UnsupportedClassification("canonical_sql_statement_count")
    statement = statements[0]
    if isinstance(statement, (exp.Union, exp.Intersect, exp.Except)):
        raise _UnsupportedClassification("compound_query")
    if not isinstance(statement, exp.Select):
        raise _UnsupportedClassification("canonical_sql_not_select")
    if statement.find(exp.Into) is not None or statement.find(exp.Lock) is not None:
        raise _UnsupportedClassification("unsafe_select_clause")
    if statement.find(exp.DML) is not None:
        raise _UnsupportedClassification("data_modifying_cte")
    return (
        _resolve_outer_tables(statement, inventory),
        _where_predicate_count(statement),
    )


def select_article_shaped_tasks(preflight: PreflightResult) -> TaskSelection:
    """Apply the documented deterministic Base-Lite Query-slice approximation."""
    try:
        import sqlglot  # noqa: F401
    except ImportError as error:
        raise BenchmarkAssetError(
            "SQL selection is unavailable; install the benchmark extra.",
            category="parser_unavailable",
        ) from error

    selected: list[BenchmarkTask] = []
    unsupported: list[UnsupportedTask] = []
    for task in preflight.tasks:
        if task.evaluator.record.get("category") != "Query":
            continue
        inventory = preflight.databases.get(task.selected_database)
        if inventory is None:
            unsupported.append(
                UnsupportedTask(task.instance_id, "selected_database", "inventory_missing")
            )
            continue
        canonical_sqls = task.evaluator.record.get("sol_sql")
        if not isinstance(canonical_sqls, list) or not canonical_sqls:
            unsupported.append(
                UnsupportedTask(task.instance_id, "sol_sql", "canonical_sql_missing")
            )
            continue
        if any(not isinstance(sql, str) for sql in canonical_sqls):
            unsupported.append(
                UnsupportedTask(task.instance_id, "sol_sql", "canonical_sql_invalid")
            )
            continue

        meets_criteria = True
        classification_error: _UnsupportedClassification | None = None
        for canonical_sql in canonical_sqls:
            try:
                tables, predicate_count = _classify_canonical_sql(
                    canonical_sql,
                    inventory,
                )
            except _UnsupportedClassification as error:
                classification_error = error
                break
            meets_criteria = meets_criteria and (
                10 <= len(inventory.base_tables)
                and 2 <= len(tables) <= 3
                and predicate_count >= 2
            )
        if classification_error is not None:
            unsupported.append(
                UnsupportedTask(
                    task.instance_id,
                    "sol_sql",
                    classification_error.reason_code,
                )
            )
            continue
        if meets_criteria:
            selected.append(task)

    selected.sort(key=lambda task: task.instance_id)
    unsupported.sort(key=lambda task: task.instance_id)
    return TaskSelection(tuple(selected), tuple(unsupported))


def validate_run_limits(max_tasks: int, settings: Settings | None = None) -> None:
    """Reject task counts outside the configured finite benchmark bound."""
    configured = settings or Settings()
    if (
        not isinstance(max_tasks, int)
        or isinstance(max_tasks, bool)
        or not 1 <= max_tasks <= configured.max_benchmark_tasks
    ):
        raise BenchmarkAssetError(
            f"--max-tasks must be between 1 and {configured.max_benchmark_tasks}.",
            category="invalid_run_options",
        )
