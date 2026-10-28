"""Validated runtime configuration for the SQL prototype."""

from __future__ import annotations

import ipaddress
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from urllib.parse import urlsplit

BENCHMARK_SETUP_DSN_ENV = "NL_SQL_BENCHMARK_SETUP_DSN"
BENCHMARK_QUERY_DSN_ENV = "NL_SQL_BENCHMARK_QUERY_DSN"


class SettingsError(ValueError):
    """Raised when a runtime setting is missing or outside its safe range."""


@dataclass(frozen=True, slots=True)
class Settings:
    """Application settings shared by the CLI and pipeline components."""

    model_url: str = ""
    model_name: str = ""
    model_key: str = field(default="", repr=False)
    max_rounds: int = 15
    max_tool_argument_length: int = 20_000
    row_cap: int = 100
    postgres_row_cap: int = 1_000
    # Rows of each query result shown to the model; scoring re-runs the full query.
    model_preview_rows: int = 20
    sqlite_query_timeout_seconds: float = 2.0
    postgres_statement_timeout_seconds: int = 30
    postgres_lock_timeout_seconds: int = 5
    model_timeout_seconds: float = 30.0
    numeric_absolute_tolerance: Decimal = Decimal(0)
    numeric_relative_tolerance: Decimal = Decimal(0)
    seed: int = 0
    max_benchmark_tasks: int = 100
    benchmark_setup_dsn: str = field(default="", repr=False)
    benchmark_query_dsn: str = field(default="", repr=False)

    def __post_init__(self) -> None:
        self._validate_endpoint()
        self._validate_integer("max_rounds", self.max_rounds, 1, 15)
        self._validate_integer(
            "max_tool_argument_length", self.max_tool_argument_length, 1, 100_000
        )
        self._validate_integer("row_cap", self.row_cap, 1, 10_000)
        self._validate_integer("postgres_row_cap", self.postgres_row_cap, 1, 10_000)
        self._validate_integer(
            "model_preview_rows", self.model_preview_rows, 1, 10_000
        )
        self._validate_integer(
            "postgres_statement_timeout_seconds",
            self.postgres_statement_timeout_seconds,
            1,
            300,
        )
        self._validate_integer(
            "postgres_lock_timeout_seconds", self.postgres_lock_timeout_seconds, 1, 60
        )
        self._validate_integer("seed", self.seed, 0, 4_294_967_295)
        self._validate_integer("max_benchmark_tasks", self.max_benchmark_tasks, 1, 100)
        self._validate_timeout(
            "sqlite_query_timeout_seconds", self.sqlite_query_timeout_seconds, 300.0
        )
        self._validate_timeout("model_timeout_seconds", self.model_timeout_seconds, 300.0)
        self._validate_tolerance(
            "numeric_absolute_tolerance", self.numeric_absolute_tolerance
        )
        self._validate_tolerance(
            "numeric_relative_tolerance", self.numeric_relative_tolerance
        )

    @classmethod
    def from_env(
        cls,
        environment: Mapping[str, str] | None = None,
    ) -> Settings:
        """Create validated settings from ``NL_SQL_*`` environment variables."""
        values = os.environ if environment is None else environment
        return cls(
            model_url=values.get("NL_SQL_BASE_URL", "").strip().rstrip("/"),
            model_name=values.get("NL_SQL_MODEL", "").strip(),
            model_key=values.get("NL_SQL_API_KEY", "").strip(),
            max_rounds=_integer_setting(values, "NL_SQL_MAX_ROUNDS", 15),
            max_tool_argument_length=_integer_setting(
                values, "NL_SQL_MAX_TOOL_ARGUMENT_LENGTH", 20_000
            ),
            row_cap=_integer_setting(values, "NL_SQL_ROW_CAP", 100),
            postgres_row_cap=_integer_setting(values, "NL_SQL_POSTGRES_ROW_CAP", 1_000),
            model_preview_rows=_integer_setting(
                values, "NL_SQL_MODEL_PREVIEW_ROWS", 20
            ),
            sqlite_query_timeout_seconds=_float_setting(
                values, "NL_SQL_QUERY_TIMEOUT_SECONDS", 2.0
            ),
            postgres_statement_timeout_seconds=_integer_setting(
                values, "NL_SQL_POSTGRES_STATEMENT_TIMEOUT_SECONDS", 30
            ),
            postgres_lock_timeout_seconds=_integer_setting(
                values, "NL_SQL_POSTGRES_LOCK_TIMEOUT_SECONDS", 5
            ),
            model_timeout_seconds=_float_setting(
                values, "NL_SQL_MODEL_TIMEOUT_SECONDS", 30.0
            ),
            numeric_absolute_tolerance=_decimal_setting(
                values, "NL_SQL_NUMERIC_ABS_TOLERANCE", Decimal(0)
            ),
            numeric_relative_tolerance=_decimal_setting(
                values, "NL_SQL_NUMERIC_REL_TOLERANCE", Decimal(0)
            ),
            seed=_integer_setting(values, "NL_SQL_SEED", 0),
            max_benchmark_tasks=_integer_setting(
                values, "NL_SQL_MAX_BENCHMARK_TASKS", 100
            ),
            benchmark_setup_dsn=values.get(BENCHMARK_SETUP_DSN_ENV, "").strip(),
            benchmark_query_dsn=values.get(BENCHMARK_QUERY_DSN_ENV, "").strip(),
        )

    @property
    def completion_url(self) -> str:
        """Return the Chat Completions endpoint for the configured model URL."""
        return f"{self.model_url}/chat/completions"

    def require_model(self) -> None:
        """Require complete model connection settings before a request is attempted."""
        missing = [
            name
            for name, value in (
                ("NL_SQL_BASE_URL", self.model_url),
                ("NL_SQL_MODEL", self.model_name),
                ("NL_SQL_API_KEY", self.model_key),
            )
            if not value
        ]
        if missing:
            raise SettingsError(
                f"Missing required model configuration: {', '.join(missing)}."
            )

    def _validate_endpoint(self) -> None:
        if not self.model_url:
            return
        try:
            parsed = urlsplit(self.model_url)
            hostname = parsed.hostname
        except ValueError as error:
            raise SettingsError("NL_SQL_BASE_URL must be a valid HTTP(S) endpoint.") from error
        if (
            parsed.scheme not in {"https", "http"}
            or hostname is None
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise SettingsError(
                "NL_SQL_BASE_URL must be an HTTP(S) endpoint without credentials or query data."
            )
        if parsed.scheme == "http" and not _is_loopback(hostname):
            raise SettingsError("Remote model endpoints must use HTTPS.")

    @staticmethod
    def _validate_integer(name: str, value: int, minimum: int, maximum: int) -> None:
        if type(value) is not int or not minimum <= value <= maximum:
            raise SettingsError(
                f"{name} must be an integer between {minimum} and {maximum}."
            )

    @staticmethod
    def _validate_timeout(name: str, value: float, maximum: float) -> None:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 < value <= maximum
        ):
            raise SettingsError(f"{name} must be finite and between 0 and {maximum:g}.")

    @staticmethod
    def _validate_tolerance(name: str, value: Decimal) -> None:
        if not isinstance(value, Decimal) or not value.is_finite() or value < 0:
            raise SettingsError(f"{name} must be a finite, non-negative decimal.")


def _is_loopback(hostname: str) -> bool:
    if hostname.casefold() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _integer_setting(
    environment: Mapping[str, str],
    name: str,
    default: int,
) -> int:
    try:
        return int(environment.get(name, str(default)))
    except ValueError as error:
        raise SettingsError(f"{name} must be an integer.") from error


def _float_setting(
    environment: Mapping[str, str],
    name: str,
    default: float,
) -> float:
    try:
        return float(environment.get(name, str(default)))
    except ValueError as error:
        raise SettingsError(f"{name} must be a number.") from error


def _decimal_setting(
    environment: Mapping[str, str],
    name: str,
    default: Decimal,
) -> Decimal:
    try:
        return Decimal(environment.get(name, str(default)))
    except InvalidOperation as error:
        raise SettingsError(f"{name} must be a decimal.") from error
