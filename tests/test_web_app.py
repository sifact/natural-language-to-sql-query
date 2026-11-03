"""Tests for the WSGI web app: routing, limits, and how failures reach the browser."""

import io
import json

import pytest

from nl_sql.agent import ModelRateLimitError
from nl_sql.config import Settings, SettingsError
from nl_sql.data import initialize_database
from nl_sql.serving import DemoService, RateLimiter, WebApp

SETTINGS = Settings(
    model_url="https://model.example/v1",
    model_name="demo-model",
    model_key="test-secret",
    model_choices=("demo-model", "bigger-model"),
)


@pytest.fixture
def app(tmp_path):
    service = DemoService(SETTINGS, initialize_database(tmp_path / "demo.sqlite3"), tmp_path)
    return WebApp(service, RateLimiter([(2, 60.0)]))


def call(app, method="GET", path="/", body=None, forwarded_for="203.0.113.7"):
    raw = b"" if body is None else json.dumps(body).encode("utf-8")
    environ = {
        "REQUEST_METHOD": method,
        "PATH_INFO": path,
        "CONTENT_LENGTH": str(len(raw)),
        "wsgi.input": io.BytesIO(raw),
        "REMOTE_ADDR": "10.0.0.1",
        "HTTP_X_FORWARDED_FOR": forwarded_for,
    }
    captured = {}

    def start_response(status, headers):
        captured["status"] = int(status.split()[0])
        captured["headers"] = dict(headers)

    payload = b"".join(app(environ, start_response))
    return captured["status"], captured["headers"], payload


def test_given_index_when_requested_then_serves_page_with_security_headers(app) -> None:
    # Act
    status, headers, body = call(app)

    # Assert
    assert status == 200
    assert b"<html" in body
    assert "script-src 'self'" in headers["Content-Security-Policy"]
    assert headers["X-Content-Type-Options"] == "nosniff"


def test_given_info_request_when_served_then_hides_credentials(app) -> None:
    # Act
    status, _headers, body = call(app, path="/api/info")

    # Assert
    info = json.loads(body)
    assert status == 200
    assert info["models"] == ["demo-model", "bigger-model"]
    assert "test-secret" not in body.decode()


def test_given_model_rate_limit_when_ask_then_returns_429_with_retry_after(
    app, mocker
) -> None:
    # Arrange
    mocker.patch.object(
        DemoService,
        "ask",
        side_effect=ModelRateLimitError("Daily token quota is used up.", retry_after_seconds=247.1),
    )

    # Act
    status, headers, body = call(app, "POST", "/api/ask", {"question": "How many orders?"})

    # Assert
    assert status == 429
    assert headers["Retry-After"] == "248"
    assert json.loads(body) == {"error": "Daily token quota is used up."}


def test_given_visitor_over_limit_when_ask_then_rejects_without_model_call(
    app, mocker
) -> None:
    # Arrange
    ask = mocker.patch.object(DemoService, "ask", return_value={"answer": "4"})

    # Act
    statuses = [
        call(app, "POST", "/api/ask", {"question": "How many orders?"})[0] for _ in range(3)
    ]
    other_visitor = call(
        app, "POST", "/api/ask", {"question": "How many orders?"}, forwarded_for="198.51.100.2"
    )[0]

    # Assert
    assert statuses == [200, 200, 429]
    assert other_visitor == 200
    assert ask.call_count == 3


def test_given_oversized_question_when_ask_then_rejects_before_rate_limit(app) -> None:
    # Act
    status, _headers, body = call(app, "POST", "/api/ask", {"question": "x" * 600})

    # Assert
    assert status == 400
    assert "under 500 characters" in json.loads(body)["error"]


def test_given_unoffered_model_when_ask_then_rejects_choice(app) -> None:
    # Act
    status, _headers, body = call(
        app, "POST", "/api/ask", {"question": "How many orders?", "model": "other-model"}
    )

    # Assert
    assert status == 400
    assert "not offered" in json.loads(body)["error"]


def test_given_data_request_when_served_then_lists_demo_tables(app) -> None:
    # Act
    status, _headers, body = call(app, path="/api/data")

    # Assert
    names = [table["name"] for table in json.loads(body)["tables"]]
    assert status == 200
    assert "orders" in names


def test_given_offered_model_when_settings_resolved_then_switches_model(tmp_path) -> None:
    # Arrange
    service = DemoService(SETTINGS, tmp_path / "demo.sqlite3")

    # Act
    chosen = service._settings_for("bigger-model")

    # Assert
    assert chosen.model_name == "bigger-model"
    with pytest.raises(SettingsError):
        service._settings_for("unknown")


def test_given_limiter_when_window_passes_then_allows_again() -> None:
    # Arrange
    now = [0.0]
    limiter = RateLimiter([(1, 60.0)], clock=lambda: now[0])

    # Act
    first, second = limiter.check("a"), limiter.check("a")
    now[0] = 61.0
    third = limiter.check("a")

    # Assert
    assert first is None
    assert second == pytest.approx(60.0)
    assert third is None
