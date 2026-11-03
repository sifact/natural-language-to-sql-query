"""WSGI web app shared by the local server and serverless hosting."""

from __future__ import annotations

import json
import math
import sqlite3
import time
from collections.abc import Callable, Iterable
from http import HTTPStatus
from importlib.resources import files

from ..agent.model_client import ModelError, ModelRateLimitError
from ..agent.sql_agent import AgentError
from ..config import Settings, SettingsError
from ..database.provider import DatabaseToolError, serialize_database_value
from .demo_service import DemoService, resolve_demo_database
from .rate_limit import RateLimiter

MAX_REQUEST_BYTES = 4_096
MAX_QUESTION_LENGTH = 500
DAY_SECONDS = 86_400.0

STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/static/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/static/styles.css": ("styles.css", "text/css; charset=utf-8"),
    "/static/theme.js": ("theme.js", "text/javascript; charset=utf-8"),
}

# The page loads only its own script and styles and calls only its own API.
SECURITY_HEADERS = [
    (
        "Content-Security-Policy",
        (
            "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; "
            "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'"
        ),
    ),
    ("X-Content-Type-Options", "nosniff"),
    ("Referrer-Policy", "no-referrer"),
]

StartResponse = Callable[[str, list[tuple[str, str]]], object]
Response = tuple[HTTPStatus, bytes, str, list[tuple[str, str]]]


class WebApp:
    """Route requests to the demo service with size and per-visitor rate limits."""

    def __init__(self, service: DemoService, limiter: RateLimiter) -> None:
        self.service = service
        self.limiter = limiter

    def __call__(self, environ: dict[str, object], start_response: StartResponse) -> Iterable[bytes]:
        started = time.monotonic()
        method = str(environ.get("REQUEST_METHOD", "GET"))
        path = str(environ.get("PATH_INFO", "/")) or "/"
        status, body, content_type, headers = self._route(method, path, environ)
        start_response(
            f"{status.value} {status.phrase}",
            [
                ("Content-Type", content_type),
                ("Content-Length", str(len(body))),
                # The page and its script ship together, so never serve a stale half.
                ("Cache-Control", "no-store" if path.startswith("/api/") else "no-cache"),
                *SECURITY_HEADERS,
                *headers,
            ],
        )
        elapsed = (time.monotonic() - started) * 1000
        print(f"nl-sql: {method} {path} -> {status.value} ({elapsed:.0f} ms)", flush=True)
        return [body]

    def _route(self, method: str, path: str, environ: dict[str, object]) -> Response:
        if method == "GET" and path in STATIC_FILES:
            name, content_type = STATIC_FILES[path]
            return HTTPStatus.OK, _static(name), content_type, []
        if method == "GET" and path == "/api/info":
            return _json(HTTPStatus.OK, self.service.info())
        if method == "GET" and path == "/api/data":
            return self._guarded(lambda: {"tables": self.service.data()})
        if method == "GET" and path == "/api/results":
            return _json(HTTPStatus.OK, {"reports": self.service.results()})
        if path == "/api/ask":
            if method != "POST":
                return _json(HTTPStatus.METHOD_NOT_ALLOWED, {"error": "Use POST."})
            return self._ask(environ)
        return _json(HTTPStatus.NOT_FOUND, {"error": "Not found."})

    def _ask(self, environ: dict[str, object]) -> Response:
        try:
            length = int(str(environ.get("CONTENT_LENGTH") or 0))
        except ValueError:
            length = 0
        if not 0 < length <= MAX_REQUEST_BYTES:
            return _json(HTTPStatus.BAD_REQUEST, {"error": "Request size is invalid."})
        try:
            payload = json.loads(environ["wsgi.input"].read(length))  # type: ignore[union-attr]
            question = str(payload.get("question", "")).strip()
            model = payload.get("model")
        except (json.JSONDecodeError, AttributeError, UnicodeDecodeError, KeyError):
            return _json(HTTPStatus.BAD_REQUEST, {"error": "Send JSON with a question."})
        if not question:
            return _json(HTTPStatus.BAD_REQUEST, {"error": "Type a question first."})
        if len(question) > MAX_QUESTION_LENGTH:
            return _json(
                HTTPStatus.BAD_REQUEST,
                {"error": f"Keep questions under {MAX_QUESTION_LENGTH} characters."},
            )
        wait = self.limiter.check(_client_key(environ))
        if wait is not None:
            return _too_many(
                "You've reached this demo's question limit. Please try again later.", wait
            )
        return self._guarded(
            lambda: self.service.ask(question, model if isinstance(model, str) else None)
        )

    @staticmethod
    def _guarded(action: Callable[[], dict[str, object]]) -> Response:
        try:
            return _json(HTTPStatus.OK, action())
        except ModelRateLimitError as error:
            return _too_many(str(error), error.retry_after_seconds)
        except SettingsError as error:
            return _json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
        except (AgentError, DatabaseToolError, ModelError, OSError, sqlite3.Error) as error:
            return _json(HTTPStatus.BAD_GATEWAY, {"error": str(error)})


def create_app(settings: Settings | None = None) -> WebApp:
    """Build the app from environment settings and the demo database.

    Model settings are not required here: a host that is missing them still serves the
    page, data and results, and each question answers with what is missing.
    """
    runtime = settings or Settings.from_env()
    limiter = RateLimiter(
        [(runtime.web_requests_per_minute, 60.0), (runtime.web_requests_per_day, DAY_SECONDS)]
    )
    return WebApp(DemoService(runtime, resolve_demo_database()), limiter)


def _client_key(environ: dict[str, object]) -> str:
    """Identify a visitor; behind a proxy the first forwarded address is the client."""
    forwarded = str(environ.get("HTTP_X_FORWARDED_FOR", "")).split(",")[0].strip()
    return forwarded or str(environ.get("REMOTE_ADDR", "unknown"))


def _static(name: str) -> bytes:
    return files(__package__).joinpath("static", name).read_bytes()


def _json(
    status: HTTPStatus,
    payload: dict[str, object],
    headers: list[tuple[str, str]] | None = None,
) -> Response:
    body = json.dumps(payload, ensure_ascii=False, default=serialize_database_value)
    return status, body.encode("utf-8"), "application/json", headers or []


def _too_many(message: str, retry_after: float | None) -> Response:
    # 429 + Retry-After tells the browser to back off rather than resend.
    headers = [] if retry_after is None else [("Retry-After", str(math.ceil(retry_after)))]
    return _json(HTTPStatus.TOO_MANY_REQUESTS, {"error": message}, headers)
