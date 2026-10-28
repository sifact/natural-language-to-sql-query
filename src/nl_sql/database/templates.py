"""Verify local PostgreSQL templates against their declared inventories."""

from __future__ import annotations

from ..config import Settings
from ..data.assets import BenchmarkAssetError, PreflightResult

_SETUP_ROLE_CHECK_SQL = """
WITH RECURSIVE memberships(role_oid) AS (
    SELECT oid FROM pg_roles WHERE rolname = current_user
    UNION
    SELECT member.roleid
    FROM pg_auth_members AS member
    JOIN memberships AS inherited ON inherited.role_oid = member.member
)
SELECT role.rolsuper,
       role.rolcreaterole,
       role.rolreplication,
       role.rolbypassrls,
       EXISTS (
           SELECT 1
           FROM memberships
           JOIN pg_roles AS inherited ON inherited.oid = memberships.role_oid
           WHERE inherited.rolname IN (
               'pg_read_server_files',
               'pg_write_server_files',
               'pg_execute_server_program'
           )
       ) AS has_server_access_role,
       EXISTS (
           SELECT 1
           FROM memberships
           JOIN pg_roles AS inherited ON inherited.oid = memberships.role_oid
           WHERE inherited.rolsuper
              OR inherited.rolcreaterole
              OR inherited.rolreplication
              OR inherited.rolbypassrls
       ) AS has_unsafe_role_membership
FROM pg_roles AS role
WHERE role.rolname = current_user
"""
_BASE_TABLE_INVENTORY_SQL = """
SELECT table_schema, table_name
FROM information_schema.tables
WHERE table_type = 'BASE TABLE'
  AND table_schema = ANY(%s)
ORDER BY table_schema, table_name
"""


def verify_live_template_inventories(
    preflight: PreflightResult,
    settings: Settings | None = None,
) -> None:
    """Compare local template catalogs before any task SQL or model request."""
    runtime = settings or Settings.from_env()
    dsn = runtime.benchmark_setup_dsn
    if not dsn:
        raise BenchmarkAssetError(
            "NL_SQL_BENCHMARK_SETUP_DSN must name a local PostgreSQL setup connection.",
            category="database_configuration_missing",
        )
    try:
        import psycopg
        from psycopg.conninfo import conninfo_to_dict
    except ImportError as error:
        raise BenchmarkAssetError(
            "PostgreSQL support is unavailable; install the benchmark extra.",
            category="driver_unavailable",
        ) from error

    try:
        options = conninfo_to_dict(dsn)
    except (psycopg.Error, TypeError, ValueError) as error:
        raise BenchmarkAssetError(
            "The local PostgreSQL connection configuration is invalid.",
            category="invalid_database_configuration",
        ) from error
    host = options.get("host", "")
    host_address = options.get("hostaddr", "")
    if (
        options.get("service")
        or (
            host
            and host not in {"localhost", "127.0.0.1", "::1"}
            and not host.startswith("/")
        )
        or (host_address and host_address not in {"localhost", "127.0.0.1", "::1"})
    ):
        raise BenchmarkAssetError(
            "Benchmark database access is restricted to a local PostgreSQL server.",
            category="unsafe_database_target",
        )

    try:
        for inventory in preflight.databases.values():
            allowed_schemas = tuple(
                sorted(
                    {table.split(".", maxsplit=1)[0] for table in inventory.base_tables}
                )
            ) or ("public",)
            if host.startswith("/"):
                target_options = {
                    "host": host,
                    "hostaddr": "",
                    "dbname": inventory.template_database,
                }
            else:
                pinned_address = host_address or (
                    "::1" if host == "::1" else "127.0.0.1"
                )
                target_options = {
                    "host": host or "localhost",
                    "hostaddr": pinned_address,
                    "dbname": inventory.template_database,
                }
            with (
                psycopg.connect(dsn, connect_timeout=5, **target_options) as connection,
                connection.transaction(),
            ):
                connection.execute("SET TRANSACTION READ ONLY")
                connection.execute(
                    "SELECT set_config('statement_timeout', %s, true), "
                    "set_config('lock_timeout', %s, true)",
                    (
                        f"{runtime.postgres_statement_timeout_seconds * 1_000}ms",
                        f"{runtime.postgres_lock_timeout_seconds * 1_000}ms",
                    ),
                )
                role = connection.execute(_SETUP_ROLE_CHECK_SQL).fetchone()
                if role is None or any(role):
                    raise BenchmarkAssetError(
                        "The PostgreSQL setup role has unsupported elevated privileges.",
                        category="unsafe_database_role",
                    )
                rows = connection.execute(
                    _BASE_TABLE_INVENTORY_SQL,
                    (list(allowed_schemas),),
                ).fetchall()
            actual_tables = tuple(sorted(f"{schema}.{table}" for schema, table in rows))
            if actual_tables != inventory.base_tables:
                raise BenchmarkAssetError(
                    "A live PostgreSQL template inventory does not match its hashed "
                    "declaration.",
                    category="database_inventory_mismatch",
                )
    except BenchmarkAssetError:
        raise
    except (psycopg.Error, TypeError, ValueError) as error:
        raise BenchmarkAssetError(
            "A local PostgreSQL template could not be safely verified.",
            category="database_unavailable",
        ) from error
