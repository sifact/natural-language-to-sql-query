"""Tests for safe, bounded SQLite tools over synthetic fixtures."""

from pathlib import Path

import pytest

from nl_sql.config import Settings
from nl_sql.data.demo import initialize_database
from nl_sql.database import DatabaseToolError, DatabaseTools, PostgresDatabaseTools


@pytest.fixture()
def database_path(tmp_path: Path) -> Path:
    path = tmp_path / "demo.sqlite3"
    initialize_database(path)
    return path


@pytest.fixture()
def tools(database_path: Path) -> DatabaseTools:
    return DatabaseTools(database_path)


def test_given_demo_database_when_list_tables_then_returns_sorted_user_tables(
    tools: DatabaseTools,
) -> None:
    # Act
    tables = tools.list_tables()

    # Assert
    assert tables == ["customers", "order_items", "orders", "products"]


def test_given_demo_database_when_inspect_schema_then_returns_columns_and_relationships(
    tools: DatabaseTools,
) -> None:
    # Act
    schema = tools.inspect_schema()

    # Assert
    orders = next(table for table in schema if table["table_name"] == "orders")
    assert "metadata" in {column["name"] for column in orders["columns"]}
    assert orders["relationships"] == [
        {
            "column": "customer_id",
            "referenced_table": "customers",
            "referenced_column": "customer_id",
        }
    ]


def test_given_known_table_when_inspect_table_then_returns_its_schema(
    tools: DatabaseTools,
) -> None:
    # Act
    schema = tools.inspect_table("order_items")

    # Assert
    assert schema["table_name"] == "order_items"
    assert len(schema["columns"]) == 4
    assert len(schema["relationships"]) == 2


def test_given_valid_select_when_query_then_returns_structured_rows(
    tools: DatabaseTools,
) -> None:
    # Act
    result = tools.query(
        "SELECT order_id, status FROM orders WHERE status = 'shipped' ORDER BY order_id"
    )

    # Assert
    assert result.columns == ["order_id", "status"]
    assert result.rows == [
        {"order_id": "O101", "status": "shipped"},
        {"order_id": "O102", "status": "shipped"},
        {"order_id": "O104", "status": "shipped"},
    ]
    assert result.truncated is False


@pytest.mark.parametrize(
    "sql",
    [
        "UPDATE orders SET status = 'cancelled'",
        "CREATE TABLE unexpected (id INTEGER)",
        "ATTACH DATABASE ':memory:' AS other",
        "SELECT load_extension('missing')",
        "SELECT random()",
        "SELECT order_id FROM orders; DELETE FROM orders",
    ],
)
def test_given_disallowed_sql_when_query_then_rejects_without_changing_data(
    tools: DatabaseTools,
    sql: str,
) -> None:
    # Arrange
    before = tools.query("SELECT order_id, status FROM orders ORDER BY order_id")

    # Act
    with pytest.raises(DatabaseToolError):
        tools.query(sql)

    # Assert
    after = tools.query("SELECT order_id, status FROM orders ORDER BY order_id")
    assert after.rows == before.rows


def test_given_large_select_when_query_then_caps_rows_and_reports_truncation(
    tools: DatabaseTools,
) -> None:
    # Act
    result = tools.query(
        """
        SELECT a.order_id
        FROM orders AS a
        CROSS JOIN orders AS b
        CROSS JOIN orders AS c
        CROSS JOIN orders AS d
        """
    )

    # Assert
    assert len(result.rows) == 100
    assert result.truncated is True


def test_given_non_terminating_read_when_query_then_interrupts_at_deadline(
    tools: DatabaseTools,
) -> None:
    # Arrange
    sql = """
    WITH RECURSIVE counter(value) AS (
        VALUES (0)
        UNION ALL
        SELECT value + 1 FROM counter
    )
    SELECT COUNT(*) FROM counter
    """

    # Act
    with pytest.raises(DatabaseToolError, match="2-second execution deadline"):
        tools.query(sql)


def test_given_missing_database_when_query_then_fails_closed(tmp_path: Path) -> None:
    # Arrange
    tools = DatabaseTools(tmp_path / "missing.sqlite3")

    # Act
    with pytest.raises(DatabaseToolError, match="read-only mode"):
        tools.query("SELECT 1")


@pytest.fixture()
def postgres_tools() -> PostgresDatabaseTools:
    return PostgresDatabaseTools("host=localhost dbname=nl_sql_test_unit")


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 1; SELECT 2",
        "UPDATE orders SET status = 'cancelled'",
        "SELECT order_id INTO copied_orders FROM orders",
        "SELECT order_id FROM orders FOR UPDATE",
        "SELECT pg_sleep(1)",
        "SELECT pg_catalog.pg_sleep(1)",
        "SELECT nextval('orders_order_id_seq')",
        "SELECT dblink_connect('host=remote')",
        "SELECT public.lower('unsafe schema qualification')",
    ],
)
def test_given_disallowed_postgres_sql_when_parsed_then_rejects_safely(
    postgres_tools: PostgresDatabaseTools,
    sql: str,
) -> None:
    # Act
    with pytest.raises(DatabaseToolError) as error:
        postgres_tools._parse_query(sql)

    # Assert
    assert error.value.category == "query_rejected"
    assert sql not in str(error.value)


def test_given_single_postgres_select_when_parsed_then_accepts_query(
    postgres_tools: PostgresDatabaseTools,
) -> None:
    # Act
    query = postgres_tools._parse_query(
        "WITH shipped AS (SELECT order_id FROM orders WHERE status = 'shipped') "
        "SELECT COUNT(*) FROM shipped"
    )

    # Assert
    assert query is not None


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT COUNT(*) FROM orders",
        "SELECT date_part('day', DATE '2024-01-01')",
        "SELECT date_trunc('day', DATE '2024-01-01')",
        "SELECT nullif(1, 2)",
    ],
)
def test_given_approved_postgres_function_when_parsed_then_accepts_query(
    postgres_tools: PostgresDatabaseTools,
    sql: str,
) -> None:
    # Act
    query = postgres_tools._parse_query(sql)

    # Assert
    assert query is not None


def test_given_cte_name_matching_system_table_when_validated_then_checks_cte_scope(
    postgres_tools: PostgresDatabaseTools,
    mocker,
) -> None:
    # Arrange
    mocker.patch.object(
        postgres_tools, "_table_records", return_value=[("public", "orders")]
    )
    connection = mocker.MagicMock()
    query = postgres_tools._parse_query(
        "WITH pg_class AS (SELECT * FROM orders) SELECT * FROM pg_class"
    )

    # Act
    postgres_tools._validate_query_tables(connection, query)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM pg_catalog.pg_class",
        "SELECT * FROM another_schema.orders",
        "SELECT * FROM missing_table",
    ],
)
def test_given_disallowed_postgres_table_when_validated_then_rejects(
    postgres_tools: PostgresDatabaseTools,
    mocker,
    sql: str,
) -> None:
    # Arrange
    mocker.patch.object(
        postgres_tools, "_table_records", return_value=[("public", "orders")]
    )
    connection = mocker.MagicMock()

    # Act
    with pytest.raises(DatabaseToolError) as error:
        postgres_tools._validate_query_tables(
            connection, postgres_tools._parse_query(sql)
        )

    # Assert
    assert error.value.category == "query_rejected"


def test_given_unsafe_postgres_target_when_validated_then_rejects_without_connecting(
    postgres_tools: PostgresDatabaseTools,
) -> None:
    # Arrange
    unsafe = PostgresDatabaseTools("host=example.com dbname=nl_sql_test_remote")

    # Act
    with pytest.raises(DatabaseToolError) as error:
        unsafe._validate_target()

    # Assert
    assert error.value.category == "unsafe_target"
    assert "example.com" not in str(error.value)


def test_given_local_postgres_target_when_validated_then_pins_loopback_connection() -> (
    None
):
    # Arrange
    tools = PostgresDatabaseTools("host=localhost dbname=nl_sql_test_pinned")

    # Act
    target_options = tools._validate_target()

    # Assert
    assert target_options == {
        "host": "localhost",
        "hostaddr": "127.0.0.1",
        "dbname": "nl_sql_test_pinned",
    }


def test_given_default_postgres_provider_when_created_then_uses_finite_limits() -> None:
    # Act
    tools = PostgresDatabaseTools("host=localhost dbname=nl_sql_test_defaults")

    # Assert
    assert tools.statement_timeout_seconds == 30
    assert tools.lock_timeout_seconds == 5
    assert tools.max_result_rows == 1_000


def test_given_query_role_with_only_active_database_access_when_verified_then_accepts(
    mocker,
) -> None:
    # Arrange
    connection = mocker.MagicMock()
    role_cursor = mocker.MagicMock()
    database_cursor = mocker.MagicMock()
    role_cursor.fetchone.return_value = (
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    )
    database_cursor.fetchall.return_value = [("nl_sql_test_unit", True)]
    connection.execute.side_effect = [role_cursor, database_cursor]
    connection.info.dbname = "nl_sql_test_unit"

    # Act
    PostgresDatabaseTools._verify_role(connection)


@pytest.mark.parametrize(
    ("role", "databases"),
    [
        (
            (True, False, False, False, False, False, False),
            [("nl_sql_test_unit", True)],
        ),
        (
            (False, False, False, False, False, False, False),
            [("nl_sql_test_unit", True), ("other", True)],
        ),
        (
            (False, False, False, False, False, False, True),
            [("nl_sql_test_unit", True)],
        ),
    ],
)
def test_given_query_role_with_elevated_privileges_when_verified_then_rejects(
    mocker,
    role: tuple[bool, bool, bool, bool, bool, bool, bool],
    databases: list[tuple[str, bool]],
) -> None:
    # Arrange
    connection = mocker.MagicMock()
    role_cursor = mocker.MagicMock()
    database_cursor = mocker.MagicMock()
    role_cursor.fetchone.return_value = role
    database_cursor.fetchall.return_value = databases
    connection.execute.side_effect = [role_cursor, database_cursor]
    connection.info.dbname = "nl_sql_test_unit"

    # Act
    with pytest.raises(DatabaseToolError) as error:
        PostgresDatabaseTools._verify_role(connection)

    # Assert
    assert error.value.category == "unsafe_role"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("postgres_statement_timeout_seconds", 301),
        ("postgres_lock_timeout_seconds", 61),
        ("postgres_row_cap", 10_001),
    ],
)
def test_given_out_of_policy_runtime_setting_when_created_then_rejects(
    field: str,
    value: int,
) -> None:
    # Act
    with pytest.raises(ValueError):
        Settings(**{field: value})


@pytest.mark.parametrize(
    "allowed_schemas",
    [("public; DROP SCHEMA public",), ("pg_catalog",)],
)
def test_given_out_of_policy_postgres_schema_when_created_then_rejects(
    allowed_schemas: tuple[str, ...],
) -> None:
    # Act
    with pytest.raises(DatabaseToolError) as error:
        PostgresDatabaseTools(
            "host=localhost dbname=nl_sql_test_unit",
            allowed_schemas=allowed_schemas,
        )

    # Assert
    assert error.value.category == "invalid_configuration"


def test_given_local_postgres_test_dsn_when_integration_test_runs_then_uses_unique_schema() -> (
    None
):
    # Arrange
    import os
    import uuid

    dsn = os.environ.get("NL_SQL_TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("NL_SQL_TEST_POSTGRES_DSN is not configured.")

    import psycopg
    from psycopg import sql as psycopg_sql
    from psycopg.conninfo import conninfo_to_dict

    schema_name = f"nl_sql_test_{uuid.uuid4().hex}"
    provider = PostgresDatabaseTools(
        dsn, allowed_schemas=(schema_name,), settings=Settings(postgres_row_cap=1)
    )
    target_options = provider._validate_target()
    try:
        database_name = conninfo_to_dict(dsn).get("dbname", "")
    except (psycopg.Error, ValueError):
        pytest.fail(
            "NL_SQL_TEST_POSTGRES_DSN must be a valid local test connection string."
        )
    if not database_name.startswith("nl_sql_test_"):
        pytest.fail("NL_SQL_TEST_POSTGRES_DSN must target an nl_sql_test_ database.")
    schema_created = False

    try:
        with psycopg.connect(dsn, connect_timeout=5, **target_options) as connection:
            connection.execute(
                psycopg_sql.SQL("CREATE SCHEMA {}").format(
                    psycopg_sql.Identifier(schema_name)
                )
            )
            connection.execute(
                psycopg_sql.SQL(
                    "CREATE TABLE {}.items (id integer PRIMARY KEY)"
                ).format(psycopg_sql.Identifier(schema_name))
            )
            connection.execute(
                psycopg_sql.SQL("INSERT INTO {}.items VALUES (1), (2)").format(
                    psycopg_sql.Identifier(schema_name)
                )
            )
        schema_created = True

        result = provider.query(f'SELECT id FROM "{schema_name}".items ORDER BY id')

        assert result.columns == ["id"]
        assert result.rows == [{"id": 1}]
        assert result.truncated is True
    finally:
        if schema_created:
            with psycopg.connect(
                dsn, connect_timeout=5, **target_options
            ) as connection:
                connection.execute(
                    psycopg_sql.SQL("DROP SCHEMA {} CASCADE").format(
                        psycopg_sql.Identifier(schema_name)
                    )
                )
