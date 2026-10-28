"""Tests for centralized runtime settings and finite resource limits."""

from decimal import Decimal

import pytest

from nl_sql.config import Settings, SettingsError


def test_given_default_settings_when_created_then_uses_safe_pipeline_limits() -> None:
    settings = Settings()

    assert settings.max_rounds == 15
    assert settings.row_cap == 100
    assert settings.postgres_row_cap == 1_000
    assert settings.sqlite_query_timeout_seconds == 2.0
    assert settings.postgres_statement_timeout_seconds == 30
    assert settings.postgres_lock_timeout_seconds == 5
    assert settings.model_timeout_seconds == 30.0
    assert settings.numeric_absolute_tolerance == Decimal(0)
    assert settings.numeric_relative_tolerance == Decimal(0)


def test_given_environment_settings_when_loaded_then_parses_all_configured_values() -> None:
    settings = Settings.from_env(
        {
            "NL_SQL_BASE_URL": "https://model.example/v1/",
            "NL_SQL_MODEL": "demo-model",
            "NL_SQL_API_KEY": "test-key",
            "NL_SQL_MAX_ROUNDS": "12",
            "NL_SQL_ROW_CAP": "77",
            "NL_SQL_POSTGRES_ROW_CAP": "900",
            "NL_SQL_QUERY_TIMEOUT_SECONDS": "3.5",
            "NL_SQL_POSTGRES_STATEMENT_TIMEOUT_SECONDS": "45",
            "NL_SQL_POSTGRES_LOCK_TIMEOUT_SECONDS": "8",
            "NL_SQL_MODEL_TIMEOUT_SECONDS": "60",
            "NL_SQL_NUMERIC_ABS_TOLERANCE": "0.01",
            "NL_SQL_NUMERIC_REL_TOLERANCE": "0.001",
            "NL_SQL_SEED": "42",
            "NL_SQL_BENCHMARK_SETUP_DSN": "dbname=postgres user=setup",
            "NL_SQL_BENCHMARK_QUERY_DSN": "dbname=postgres user=query",
        }
    )

    assert settings.model_url == "https://model.example/v1"
    assert settings.completion_url == "https://model.example/v1/chat/completions"
    assert settings.model_name == "demo-model"
    assert settings.model_key == "test-key"
    assert settings.max_rounds == 12
    assert settings.row_cap == 77
    assert settings.postgres_row_cap == 900
    assert settings.sqlite_query_timeout_seconds == 3.5
    assert settings.postgres_statement_timeout_seconds == 45
    assert settings.postgres_lock_timeout_seconds == 8
    assert settings.model_timeout_seconds == 60
    assert settings.numeric_absolute_tolerance == Decimal("0.01")
    assert settings.numeric_relative_tolerance == Decimal("0.001")
    assert settings.seed == 42
    assert settings.benchmark_setup_dsn == "dbname=postgres user=setup"
    assert settings.benchmark_query_dsn == "dbname=postgres user=query"
    assert "test-key" not in repr(settings)
    assert "user=setup" not in repr(settings)
    assert "user=query" not in repr(settings)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_rounds", 16),
        ("row_cap", 0),
        ("postgres_row_cap", 10_001),
        ("sqlite_query_timeout_seconds", float("inf")),
        ("model_timeout_seconds", 0),
        ("postgres_statement_timeout_seconds", 301),
        ("postgres_lock_timeout_seconds", 61),
        ("numeric_absolute_tolerance", Decimal("-0.01")),
        ("numeric_relative_tolerance", Decimal("NaN")),
        ("seed", -1),
    ],
)
def test_given_invalid_runtime_setting_when_created_then_rejects(
    field: str,
    value: object,
) -> None:
    with pytest.raises(SettingsError):
        Settings(**{field: value})


def test_given_invalid_numeric_environment_value_when_loaded_then_reports_setting() -> None:
    with pytest.raises(SettingsError, match="NL_SQL_NUMERIC_ABS_TOLERANCE"):
        Settings.from_env({"NL_SQL_NUMERIC_ABS_TOLERANCE": "not-a-decimal"})
