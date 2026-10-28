"""Local web UI for asking the agent questions about the demo database."""

from __future__ import annotations

import json
import sqlite3
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from urllib.parse import urlsplit

from ..agent.model_client import ChatCompletionClient, ModelError
from ..agent.sql_agent import AgentError, SqlAgent
from ..config import Settings, SettingsError
from ..data.demo import DEFAULT_DATABASE_PATH
from ..database.provider import DatabaseToolError, serialize_database_value
from ..database.sqlite_tools import DatabaseTools

# Bind to loopback only: the server holds the model API key.
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
MAX_REQUEST_BYTES = 4_096
MAX_QUESTION_LENGTH = 1_000


def _index_html() -> bytes:
    return files(__package__).joinpath("static/index.html").read_bytes()


def _ask(question: str, settings: Settings) -> dict[str, object]:
    """Run one agent question and shape the result for the browser."""
    started = time.monotonic()
    result = SqlAgent(
        ChatCompletionClient(settings),
        DatabaseTools(DEFAULT_DATABASE_PATH, settings=settings),
        settings=settings,
    ).run(question)
    return {
        "answer": result.answer,
        "sql": result.sql,
        "rows": result.rows,
        "clarification": result.clarification,
        "incomplete": result.incomplete,
        "steps": [{"tool": step.name, "outcome": step.outcome} for step in result.trace],
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "seconds": round(time.monotonic() - started, 1),
    }


def _data(settings: Settings) -> list[dict[str, object]]:
    """Return every demo table with its columns and rows; the demo data is tiny."""
    tools = DatabaseTools(DEFAULT_DATABASE_PATH, settings=settings)
    tables = []
    for table in tools.inspect_schema():
        name = str(table["table_name"])
        result = tools.query(f'SELECT * FROM "{name}"')
        tables.append(
            {
                "name": name,
                "columns": table["columns"],
                "rows": result.rows,
                "truncated": result.truncated,
            }
        )
    return tables


def _info(settings: Settings) -> dict[str, object]:
    """Describe which model writes the SQL, without exposing credentials."""
    return {"model": settings.model_name, "endpoint": urlsplit(settings.model_url).hostname}


class _Handler(BaseHTTPRequestHandler):
    settings: Settings

    def do_GET(self) -> None:
        if self.path in {"/", "/index.html"}:
            self._send(HTTPStatus.OK, _index_html(), "text/html; charset=utf-8")
        elif self.path == "/api/info":
            self._send_json(HTTPStatus.OK, _info(self.settings))
        elif self.path == "/api/data":
            try:
                self._send_json(HTTPStatus.OK, {"tables": _data(self.settings)})
            except (DatabaseToolError, OSError, sqlite3.Error) as error:
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(error)})
        else:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found."})

    def do_POST(self) -> None:
        if self.path != "/api/ask":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found."})
            return
        length = int(self.headers.get("Content-Length") or 0)
        if not 0 < length <= MAX_REQUEST_BYTES:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Request size is invalid."})
            return
        try:
            question = str(json.loads(self.rfile.read(length)).get("question", "")).strip()
        except (json.JSONDecodeError, AttributeError, UnicodeDecodeError):
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Send JSON with a question."})
            return
        if not question or len(question) > MAX_QUESTION_LENGTH:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Ask a non-empty, short question."})
            return
        try:
            self._send_json(HTTPStatus.OK, _ask(question, self.settings))
        except (AgentError, DatabaseToolError, ModelError, OSError, sqlite3.Error) as error:
            self._send_json(HTTPStatus.BAD_GATEWAY, {"error": str(error)})

    def _send_json(self, status: HTTPStatus, payload: dict[str, object]) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=serialize_database_value)
        self._send(status, body.encode("utf-8"), "application/json")

    def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        print(f"nl-sql serve: {self.command} {self.path} -> {args[1] if len(args) > 1 else ''}")


def serve(port: int = DEFAULT_PORT) -> None:
    """Serve the UI on loopback until interrupted.

    Raises:
        SettingsError: If the model endpoint is not configured.
    """
    settings = Settings.from_env()
    settings.require_model()
    handler = type("Handler", (_Handler,), {"settings": settings})
    with ThreadingHTTPServer((DEFAULT_HOST, port), handler) as server:
        print(f"nl-sql UI running at http://{DEFAULT_HOST}:{port}  (Ctrl+C to stop)")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nnl-sql UI stopped.")


__all__ = ["DEFAULT_PORT", "SettingsError", "serve"]
