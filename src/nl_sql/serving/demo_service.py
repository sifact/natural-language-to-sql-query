"""What the web demo can do: answer a question, show the data, report measured results."""

from __future__ import annotations

import dataclasses
import json
import tempfile
import time
from pathlib import Path
from urllib.parse import urlsplit

from ..agent.model_client import ChatCompletionClient
from ..agent.sql_agent import SqlAgent
from ..config import Settings, SettingsError
from ..data.demo import DEFAULT_DATABASE_PATH, initialize_database
from ..data.demo_questions import DEMO_QUESTIONS
from ..database.sqlite_tools import DatabaseTools
from ..evaluation.demo_eval import DEMO_REPORT_DIRECTORY

# Examples that show the most in one click: rules, JSON fields, rankings, anti-joins.
EXAMPLE_QUESTION_IDS = frozenset({"q03", "q04", "q07", "q08", "q14", "q15"})

# Serverless file systems are read-only except the temp directory.
FALLBACK_DATABASE_PATH = Path(tempfile.gettempdir()) / "nl-sql-demo.sqlite3"


def resolve_demo_database(path: Path = DEFAULT_DATABASE_PATH) -> Path:
    """Use the local demo database, or build the deterministic one in temp storage."""
    if path.exists():
        return path
    if not FALLBACK_DATABASE_PATH.exists():
        initialize_database(FALLBACK_DATABASE_PATH, force=True)
    return FALLBACK_DATABASE_PATH


class DemoService:
    """Answer questions on the synthetic database for the web UI."""

    def __init__(
        self,
        settings: Settings,
        database_path: Path,
        report_directory: Path = DEMO_REPORT_DIRECTORY,
    ) -> None:
        self.settings = settings
        self.database_path = database_path
        self.report_directory = report_directory

    def info(self) -> dict[str, object]:
        """Describe the models and limits without exposing credentials or the endpoint URL."""
        return {
            "model": self.settings.model_name,
            "models": list(self.settings.available_models),
            "endpoint": urlsplit(self.settings.model_url).hostname,
            "examples": [
                question.question
                for question in DEMO_QUESTIONS
                if question.question_id in EXAMPLE_QUESTION_IDS
            ],
            "limits": {
                "per_minute": self.settings.web_requests_per_minute,
                "per_day": self.settings.web_requests_per_day,
            },
        }

    def ask(self, question: str, model: str | None = None) -> dict[str, object]:
        """Run one agent question and shape the result, including its working, for the UI."""
        settings = self._settings_for(model)
        started = time.monotonic()
        result = SqlAgent(
            ChatCompletionClient(settings),
            DatabaseTools(self.database_path, settings=settings),
            settings=settings,
        ).run(question)
        return {
            "answer": result.answer,
            "sql": result.sql,
            "rows": result.rows,
            "clarification": result.clarification,
            "incomplete": result.incomplete,
            "steps": [{"tool": step.name, "outcome": step.outcome} for step in result.trace],
            "model": settings.model_name,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "seconds": round(time.monotonic() - started, 1),
        }

    def data(self) -> list[dict[str, object]]:
        """Return every demo table with its columns and rows; the demo data is tiny."""
        tools = DatabaseTools(self.database_path, settings=self.settings)
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

    def results(self) -> list[dict[str, object]]:
        """Return every committed demo evaluation report, newest first."""
        reports = []
        for path in sorted(self.report_directory.glob("demo-eval-*.json")):
            try:
                reports.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
        return sorted(reports, key=lambda report: str(report.get("run_timestamp_utc")), reverse=True)

    def _settings_for(self, model: str | None) -> Settings:
        if not model or model == self.settings.model_name:
            return self.settings
        if model not in self.settings.available_models:
            raise SettingsError("That model is not offered in this demo.")
        return dataclasses.replace(self.settings, model_name=model)
