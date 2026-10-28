---
title: Natural-Language SQL Demo
description: Run a local, read-only SQL exploration prototype with a configurable model endpoint.
---

## Overview

This project is a small local prototype inspired by [Microsoft's
natural-language SQL query generation article](https://devblogs.microsoft.com/ise/llm-sql-query-generation/).
The default workflow uses a configurable OpenAI-compatible Chat Completions
endpoint to inspect a synthetic SQLite database, draft read-only SQL, and return
an answer with the final query and rows. An optional local benchmark pipeline
supports a user-authorized LiveSQLBench Base-Lite PostgreSQL slice.

The benchmark adapter is an approximation, not an exact reproduction of the
article's task IDs, model conditions, prompts, or evaluation environment. The
bundled database contains synthetic order data only; its tests do not establish
benchmark accuracy.

## Requirements

* Python 3.11 or later
* `uv`

Install the project and development tools:

```bash
uv sync
```

For the optional PostgreSQL benchmark components, install the `benchmark`
extra. It adds Psycopg and SQLGlot:

```bash
uv sync --extra benchmark
```

## Initialize and ask a question

Create the local demo database:

```bash
uv run nl-sql init
```

The command creates `data/demo.sqlite3`. It will not replace an existing
database unless you explicitly use `--force`:

```bash
uv run nl-sql init --force
```

Set the model configuration before running a question:

```bash
export NL_SQL_BASE_URL="https://your-model-endpoint.example/v1"
export NL_SQL_MODEL="your-model-name"
export NL_SQL_API_KEY="your-api-key"
```

Ask a natural-language question about the synthetic fixture:

```bash
uv run nl-sql ask "How many orders are there?"
```

The CLI and pipeline components read validated runtime configuration from
[`Settings`](src/nl_sql/config.py). These environment variables control the
model endpoint and runtime limits:

| Variable | Default | Purpose |
|----------|---------|---------|
| `NL_SQL_BASE_URL` | required for model calls | OpenAI-compatible API base URL |
| `NL_SQL_MODEL` | required for model calls | Model name sent to the endpoint |
| `NL_SQL_API_KEY` | required for model calls | Bearer credential; never included in reports |
| `NL_SQL_MAX_ROUNDS` | `15` | Maximum model/tool rounds per question |
| `NL_SQL_MAX_TOOL_ARGUMENT_LENGTH` | `20000` | Maximum characters in tool arguments and submissions |
| `NL_SQL_ROW_CAP` | `100` | Maximum SQLite rows returned to the agent |
| `NL_SQL_POSTGRES_ROW_CAP` | `1000` | Maximum PostgreSQL rows returned to the agent |
| `NL_SQL_QUERY_TIMEOUT_SECONDS` | `2` | SQLite query deadline |
| `NL_SQL_POSTGRES_STATEMENT_TIMEOUT_SECONDS` | `30` | PostgreSQL statement deadline |
| `NL_SQL_POSTGRES_LOCK_TIMEOUT_SECONDS` | `5` | PostgreSQL lock wait deadline |
| `NL_SQL_MODEL_TIMEOUT_SECONDS` | `30` | Model request deadline |
| `NL_SQL_NUMERIC_ABS_TOLERANCE` | `0` | Absolute tolerance for result comparison |
| `NL_SQL_NUMERIC_REL_TOLERANCE` | `0` | Relative tolerance for result comparison |
| `NL_SQL_SEED` | `0` | Reproducible pipeline seed |
| `NL_SQL_MAX_BENCHMARK_TASKS` | `100` | Maximum task count allowed in a run |
| `NL_SQL_BENCHMARK_SETUP_DSN` | unset | Local PostgreSQL setup role for template checks and clone lifecycle |
| `NL_SQL_BENCHMARK_QUERY_DSN` | unset | Distinct local read-only query role for task clones |

Remote model endpoints must use HTTPS. HTTP is accepted only for loopback
development endpoints such as `http://localhost:8080/v1`. The supported API
surface is the project's OpenAI-compatible Chat Completions subset.

## Local benchmark assets and preflight

The benchmark path is optional and uses only local files beneath
`data/benchmark/`. That entire directory is ignored by Git. The loader expects
the public task file at
`data/benchmark/public-base-lite/livesqlbench_data.jsonl`; it reads gold task
files and PostgreSQL database files only when each is listed and hash-verified
in the local `data/benchmark/asset-declaration.json`. The declaration also maps
each selected database to its local asset directory, inventory, and PostgreSQL
template name. Database files and inventory must be complete and declared.
Gold JSONL may contain only evaluator fields (`instance_id`, `sol_sql`,
`external_knowledge`, and `test_cases`); matching public metadata is inherited
by `instance_id` during preflight. The supplied
`livesqlbench_gt_kg_testcases_0528.jsonl` can therefore be placed at
`data/benchmark/gold-tasks/` and declared as the `gold_tasks` asset. Its
Python-shaped `test_cases` remain evaluator data and are not executed by this
project; tasks that require those semantics stay explicitly unsupported rather
than running arbitrary dataset code.

The declaration records provenance and terms for each asset, including whether
the user has verified applicable terms and whether model transmission is
permitted. Those are user-owned attestations: do not infer or set
`terms_verified_by_user` on the user's behalf. No benchmark declaration, gold
answers, database assets, traces, or reports are committed or downloaded by
these instructions.

Run static validation after the authorized local files and declaration are
present:

```bash
uv run nl-sql benchmark preflight
```

Preflight checks the declaration, file hashes, terms and transfer gates,
task/evaluator joins, database inventories, and the deterministic task slice.
It does not connect to PostgreSQL or call a model. Missing assets or incomplete
authorization are reported as blockers; preflight does not create missing gold
answers or attestations.

The chosen slice is a reproducible approximation: `Query` tasks whose canonical
SQL alternatives are SELECT-only, use two or three distinct outer base tables,
contain at least two outer `WHERE` predicates, and belong to databases with at
least ten base tables. The article does not publish its exact task IDs. The
local public-file regression test parses all 270 records, but the public file
does not contain gold answers or test cases, so it cannot produce a benchmark
score.

## Authorized benchmark execution and reports

A later, separately authorized run requires complete local assets, applicable
user-verified terms, a dedicated local PostgreSQL target, and model settings.
The setup and query DSNs must name different restricted roles. The setup role
creates and owns per-task clones; the non-elevated query role is read-only and
must not be able to connect to other databases on that target. Each task uses
a run-owned clone and the sandbox removes only that exact clone.

The run requires both opt-ins; the model-transfer opt-in is separate from the
asset terms declaration:

```bash
uv run nl-sql benchmark run \
  --confirm-run \
  --allow-model-data-transfer \
  --max-tasks 1
```

You can choose a simple JSON filename, which is still written only beneath
`data/benchmark/reports/`:

```bash
uv run nl-sql benchmark run \
  --confirm-run \
  --allow-model-data-transfer \
  --max-tasks 1 \
  --report local-run.json
```

The default filename is generated from the run ID. The CLI rejects path
traversal, absolute or outside paths, symlinked report directories, and
overwrites. Reports are verified as Git-ignored before execution and are not
committed.

The agent can send the task question and permitted schema, knowledge, and tool
results to the configured endpoint. Gold SQL, test cases, and evaluator-only
annotations stay on the evaluator side. The report records the dataset
revision, input hashes, selection criteria and sorted task IDs, model/provider
name, UTC start time, resource limits, numeric tolerances, fixed task
statuses/reasons, status counts, score numerator/denominator, token counts, and
relative trace filenames. It omits API keys, DSNs, endpoint URLs, absolute
paths, questions, SQL, result rows, gold answers, test-case contents, raw
traces, and raw database errors. Only passed and failed tasks enter the score
denominator; an infrastructure-aborted run has no aggregate score.

## Pipeline modules

The code is organized by pipeline stage, and each core module has a short
responsibility guide alongside it. Public APIs are re-exported from each
subpackage's `__init__.py`. For example:

```python
from nl_sql.config import Settings
from nl_sql.data import load_local_benchmark_assets, select_article_shaped_tasks
from nl_sql.evaluation import TaskRunner, build_report, write_report
```

* [`config.py`](src/nl_sql/config.py) centralizes validated runtime settings.
* [`data/`](src/nl_sql/data/) contains the [asset loader](src/nl_sql/data/guide/assets.md),
  [task selector](src/nl_sql/data/guide/selection.md), [knowledge catalog](src/nl_sql/data/guide/knowledge.md),
  and [synthetic demo](src/nl_sql/data/guide/demo.md).
* [`database/`](src/nl_sql/database/) contains the shared [provider contract](src/nl_sql/database/guide/provider.md),
  [SQLite tools](src/nl_sql/database/guide/sqlite_tools.md), [PostgreSQL tools](src/nl_sql/database/guide/postgres_tools.md),
  [template checks](src/nl_sql/database/guide/templates.md), and the [isolated task sandbox](src/nl_sql/database/guide/task_sandbox.md).
* [`agent/`](src/nl_sql/agent/) contains the [agent loop](src/nl_sql/agent/guide/sql_agent.md),
  [model client](src/nl_sql/agent/guide/model_client.md), [tool schemas](src/nl_sql/agent/guide/tools.md),
  and [task tracing](src/nl_sql/agent/guide/tracing.md).
* [`evaluation/`](src/nl_sql/evaluation/) contains the [value normalizer](src/nl_sql/evaluation/guide/normalize.md),
  [result matcher](src/nl_sql/evaluation/guide/matcher.md), [task runner](src/nl_sql/evaluation/guide/task_runner.md),
  [metrics](src/nl_sql/evaluation/guide/metrics.md), and [report](src/nl_sql/evaluation/guide/report.md).

## Data flow and safety

No model request occurs during installation, initialization, or testing. Each
`nl-sql ask` sends the question and selected schema/tool results to the endpoint
configured in `NL_SQL_BASE_URL`. An authorized benchmark run can likewise
transfer benchmark questions and permitted database context only when
`--allow-model-data-transfer` is supplied. The API key is sent as a bearer
credential and never belongs in benchmark files or reports. Use an endpoint you
trust; do not include sensitive information in questions or database values.

The CLI always uses the bundled database. It does not accept a database path.
SQLite opens that file in read-only mode, and an authorizer rejects mutations
and other disallowed operations. Query results are capped at 100 rows and
queries are interrupted after two seconds by default. The agent exposes four
database tools and stops after at most 15 model/tool rounds per question.

The data is synthetic, but model-generated SQL and answers are not guaranteed
to be semantically correct. Review the query and rows before relying on an
answer.

## Exit outcomes

| Exit code | Meaning |
|-----------|---------|
| `0`       | The command completed successfully |
| `1`       | Configuration, provider, or database tool failure |
| `2`       | The agent needs clarification or reached its round limit |

## Tests

Run the offline test suite and lint:

```bash
uv run pytest
uv run ruff check .
```

Tests use temporary SQLite databases and scripted model responses. They do not
require an API key or make network requests. Benchmark runner/report tests use
fake models, providers, and a temporary local Git workspace. The public-file
test reads the local 270-record dataset; tests do not use gold answers, connect
to PostgreSQL, or execute real benchmark tasks. The optional PostgreSQL
integration test is skipped unless an explicit disposable test DSN is
configured. Fixed SQLite examples cover four orders, a shipped-order total of
`350`, and two expedited orders.

## Limitations

This prototype demonstrates an interactive discovery-and-query flow and an
offline-testable local evaluation path; it is not a production service or a
guarantee of query correctness. Its fixed SQLite examples validate only the
synthetic fixture and scripted responses. The article does not disclose its
exact benchmark task IDs, model conditions, or numeric tolerances, so this
project's selected slice and local evaluation settings do not reproduce the
article's experiment or reported scores.

An optional live smoke test can be run by configuring a trusted endpoint and
asking a question about the bundled synthetic data. No live endpoint request
or benchmark task was run as part of this project implementation.