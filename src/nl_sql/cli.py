"""Command-line interface for the local SQL exploration prototype."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

from .agent.model_client import ChatCompletionClient, ModelError
from .agent.sql_agent import AgentError, SqlAgent
from .config import Settings, SettingsError
from .data.assets import BenchmarkAssetError, load_local_benchmark_assets
from .data.demo import DEFAULT_DATABASE_PATH, initialize_database
from .data.selection import select_article_shaped_tasks, validate_run_limits
from .database.provider import (
    DatabaseToolError,
    serialize_database_value,
)
from .database.sqlite_tools import DatabaseTools
from .evaluation import (
    ReportError,
    TaskRunner,
    TaskRunnerError,
    validate_report_destination,
    write_report,
)
from .serving import DEFAULT_PORT, serve

EXIT_SUCCESS = 0
EXIT_FAILURE = 1
EXIT_INCOMPLETE = 2


def create_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="nl-sql", description="Explore the synthetic SQL demo."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    init_parser = subparsers.add_parser(
        "init", help="Create the synthetic demo database."
    )
    init_parser.add_argument(
        "--force",
        action="store_true",
        help="Replace the existing demo database.",
    )
    ask_parser = subparsers.add_parser(
        "ask", help="Ask a question about the synthetic demo."
    )
    ask_parser.add_argument("question", nargs="+", help="Natural-language question.")
    serve_parser = subparsers.add_parser(
        "serve", help="Open a local browser UI for asking the synthetic demo."
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Local port to listen on (default: {DEFAULT_PORT}).",
    )
    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Validate or run an explicitly authorized local benchmark.",
    )
    benchmark_subparsers = benchmark_parser.add_subparsers(
        dest="benchmark_command",
        required=True,
    )
    benchmark_subparsers.add_parser(
        "preflight",
        help="Validate local assets without database or model access.",
    )
    run_parser = benchmark_subparsers.add_parser(
        "run",
        help="Run selected tasks with explicit safety opt-ins.",
    )
    run_parser.add_argument(
        "--confirm-run",
        action="store_true",
        required=True,
        help="Confirm that this command may execute benchmark tasks.",
    )
    run_parser.add_argument(
        "--allow-model-data-transfer",
        action="store_true",
        required=True,
        help="Allow benchmark questions and database context to reach the configured model.",
    )
    run_parser.add_argument(
        "--max-tasks",
        type=int,
        default=1,
        help="Maximum tasks to execute (1-100; default: 1).",
    )
    run_parser.add_argument(
        "--report",
        metavar="FILENAME",
        help=(
            "Optional JSON filename beneath data/benchmark/reports/ "
            "(default: generated from the run ID)."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the requested CLI command.

    Args:
        argv: Optional argument list, primarily for tests.

    Returns:
        The process exit code.
    """
    parser = create_parser()
    arguments = parser.parse_args(argv)
    if arguments.command == "benchmark":
        if arguments.benchmark_command == "preflight":
            try:
                result = load_local_benchmark_assets()
                selection = select_article_shaped_tasks(result)
            except BenchmarkAssetError as error:
                print(f"nl-sql: benchmark preflight failed: {error}", file=sys.stderr)
                return EXIT_FAILURE
            if not selection.selected_tasks:
                print(
                    "nl-sql: benchmark preflight failed: no tasks meet the "
                    "article-shaped query slice criteria.",
                    file=sys.stderr,
                )
                return EXIT_FAILURE
            print(
                "Validated "
                f"{len(result.tasks)} task records and {len(result.assets)} declared assets."
            )
            print(f"Observed categories: {', '.join(result.observed_categories)}")
            print(f"Selected task IDs: {', '.join(selection.selected_ids)}")
            print(f"Unsupported Query records: {len(selection.unsupported_tasks)}")
            if result.run_blockers:
                print(
                    "Benchmark run is blocked by incomplete authorization or transfer "
                    "attestations.",
                    file=sys.stderr,
                )
                return EXIT_FAILURE
            print("No database connection or model request was made.")
            return EXIT_SUCCESS

        try:
            settings = Settings.from_env()
            validate_run_limits(arguments.max_tasks, settings)
            result = load_local_benchmark_assets()
            selection = select_article_shaped_tasks(result)
        except (BenchmarkAssetError, SettingsError) as error:
            print(f"nl-sql: benchmark run blocked: {error}", file=sys.stderr)
            return EXIT_FAILURE
        if not selection.selected_tasks:
            print(
                "nl-sql: benchmark run blocked: no tasks meet the article-shaped "
                "query slice criteria.",
                file=sys.stderr,
            )
            return EXIT_FAILURE
        if result.run_blockers:
            print(
                "nl-sql: benchmark run blocked by incomplete authorization or transfer "
                "attestations.",
                file=sys.stderr,
            )
            return EXIT_FAILURE
        try:
            validate_report_destination(Path.cwd(), arguments.report)
        except ReportError:
            print(
                "nl-sql: benchmark run blocked by an unsafe report destination.",
                file=sys.stderr,
            )
            return EXIT_FAILURE
        try:
            run = TaskRunner(settings).run(
                result,
                selection,
                max_tasks=arguments.max_tasks,
                confirm_run=arguments.confirm_run,
                allow_model_data_transfer=arguments.allow_model_data_transfer,
            )
            report_path = write_report(
                run,
                settings,
                preflight=result,
                selection=selection,
                report_name=arguments.report,
            )
        except TaskRunnerError as error:
            print(
                f"nl-sql: benchmark run failed ({error.reason_code}).",
                file=sys.stderr,
            )
            return EXIT_FAILURE
        except ReportError:
            print("nl-sql: benchmark report could not be written.", file=sys.stderr)
            return EXIT_FAILURE
        counts = run.metrics
        if counts is None:
            print("Benchmark score unavailable because the run aborted.")
        else:
            print(
                f"Score: {counts.passed_tasks}/{counts.evaluated_tasks} "
                f"({counts.accuracy if counts.accuracy is not None else 'n/a'})"
            )
        print(f"Report: {report_path}")
        if run.aborted or any(
            task.status in {"incomplete", "unsupported", "skipped"}
            for task in run.tasks
        ):
            return EXIT_INCOMPLETE
        return EXIT_SUCCESS

    if arguments.command == "init":
        try:
            database_path = initialize_database(
                DEFAULT_DATABASE_PATH, force=arguments.force
            )
        except (FileExistsError, OSError, sqlite3.Error) as error:
            print(f"nl-sql: {error}", file=sys.stderr)
            return EXIT_FAILURE
        print(f"Initialized demo database: {database_path}")
        return EXIT_SUCCESS

    if arguments.command == "serve":
        if not DEFAULT_DATABASE_PATH.exists():
            print("nl-sql: run `nl-sql init` first to create the demo database.", file=sys.stderr)
            return EXIT_FAILURE
        try:
            serve(arguments.port)
        except (SettingsError, OSError) as error:
            print(f"nl-sql: {error}", file=sys.stderr)
            return EXIT_FAILURE
        return EXIT_SUCCESS

    if arguments.command == "ask":
        try:
            settings = Settings.from_env()
            result = SqlAgent(
                ChatCompletionClient(settings),
                DatabaseTools(DEFAULT_DATABASE_PATH, settings=settings),
                settings=settings,
            ).run(" ".join(arguments.question))
        except (
            AgentError,
            DatabaseToolError,
            ModelError,
            SettingsError,
            OSError,
            sqlite3.Error,
        ) as error:
            print(f"nl-sql: {error}", file=sys.stderr)
            return EXIT_FAILURE

        if result.clarification is not None:
            print(f"Clarification needed: {result.clarification}")
            return EXIT_INCOMPLETE
        if result.incomplete:
            print(
                "nl-sql: model/tool round limit reached before a final answer.",
                file=sys.stderr,
            )
            return EXIT_INCOMPLETE
        if result.answer is None:
            print("nl-sql: agent returned no answer.", file=sys.stderr)
            return EXIT_FAILURE
        if result.sql is None and any(call.outcome == "error" for call in result.trace):
            print(
                "nl-sql: a database tool call failed before a query succeeded.",
                file=sys.stderr,
            )
            return EXIT_FAILURE

        print(f"Answer: {result.answer}")
        if result.sql is not None:
            print(f"SQL: {result.sql}")
            print(
                "Rows: "
                f"{json.dumps(result.rows, ensure_ascii=False, default=serialize_database_value)}"
            )
        return EXIT_SUCCESS

    parser.error("Unsupported command.")
    return EXIT_FAILURE
