"""Strict, non-mutating loading and preflight for local benchmark assets."""

from __future__ import annotations

import hashlib
import json
import re
import stat
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import NoReturn
from urllib.parse import urlsplit

ASSET_DECLARATION_PATH = PurePosixPath("data/benchmark/asset-declaration.json")
BENCHMARK_ROOT = PurePosixPath("data/benchmark")
REPORTS_ROOT = PurePosixPath("data/benchmark/reports")
PUBLIC_BASE_LITE_PATH = PurePosixPath(
    "data/benchmark/public-base-lite/livesqlbench_data.jsonl"
)
ASSET_SCHEMA_VERSION = 1
INVENTORY_SCHEMA_VERSION = 1
_SHA256_PATTERN = re.compile(r"[0-9a-fA-F]{64}\Z")
_REPORT_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,123}\.json\Z")
_DATABASE_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_$]{0,62}\Z")
_BASE_TABLE_PATTERN = re.compile(
    r"(?P<schema>[A-Za-z_][A-Za-z0-9_$]{0,62})\."
    r"(?P<table>[A-Za-z_][A-Za-z0-9_$]{0,62})\Z"
)
_ASSET_FIELDS = frozenset(
    {
        "kind",
        "path",
        "sha256",
        "source_url",
        "retrieved_on",
        "terms_id",
        "terms_url",
        "terms_verified_by_user",
        "verified_on",
        "model_transmission_allowed",
    }
)
_DECLARATION_FIELDS = frozenset(
    {"schema_version", "dataset_revision", "assets", "databases"}
)
_DATABASE_MAPPING_FIELDS = frozenset(
    {"template_database", "asset_path", "inventory_path"}
)
_SQL_LIST_FIELDS = frozenset({"sol_sql", "preprocess_sql", "clean_up_sqls"})
_PUBLIC_REQUIRED_FIELDS = frozenset(
    {
        "instance_id",
        "selected_database",
        "query",
        "category",
        *_SQL_LIST_FIELDS,
    }
)
_GOLD_REQUIRED_FIELDS = frozenset(
    {
        "instance_id",
        "sol_sql",
        "test_cases",
        "external_knowledge",
    }
)
_STRING_FIELDS = frozenset(
    {
        "instance_id",
        "selected_database",
        "query",
        "category",
    }
)


class BenchmarkAssetError(ValueError):
    """A safe, fixed-category error in the local benchmark asset contract."""

    def __init__(self, message: str, *, category: str = "invalid_assets") -> None:
        super().__init__(message)
        self.category = category


@dataclass(frozen=True, slots=True)
class AssetProvenance:
    """Verified provenance for one file beneath the ignored benchmark root."""

    kind: str
    path: str
    sha256: str
    source_url: str
    retrieved_on: str
    terms_id: str
    terms_url: str
    terms_verified_by_user: bool
    verified_on: str
    model_transmission_allowed: bool


@dataclass(frozen=True, slots=True)
class DatabaseInventory:
    """Static, hashed inventory of one provisioned PostgreSQL template."""

    selected_database: str
    template_database: str
    asset_path: str
    inventory_path: str
    base_tables: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ModelTaskInput:
    """The only task data eligible to be passed to the model."""

    query: str
    # Names the public knowledge files to expose; never gold or evaluator data.
    selected_database: str = ""


@dataclass(frozen=True, slots=True)
class EvaluatorTaskInput:
    """Gold and evaluator annotations, kept separate from model input."""

    record: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class BenchmarkTask:
    """One validated public/evaluator task pair."""

    instance_id: str
    selected_database: str
    model_input: ModelTaskInput
    evaluator: EvaluatorTaskInput


@dataclass(frozen=True, slots=True)
class PreflightResult:
    """Validated local assets and task pairs, without opening a database."""

    dataset_revision: str
    assets: tuple[AssetProvenance, ...]
    databases: Mapping[str, DatabaseInventory]
    tasks: tuple[BenchmarkTask, ...]
    observed_categories: tuple[str, ...]
    run_blockers: tuple[str, ...]


def _fail(message: str, *, category: str = "invalid_assets") -> NoReturn:
    raise BenchmarkAssetError(message, category=category)


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate object key")
        result[key] = value
    return result


def _reject_json_constant(_value: str) -> object:
    raise ValueError("non-standard JSON constant")


def _load_json(path: Path, *, description: str) -> object:
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(
                file,
                object_pairs_hook=_strict_object,
                parse_constant=_reject_json_constant,
            )
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        ValueError,
        RecursionError,
    ) as error:
        raise BenchmarkAssetError(
            f"Could not read valid {description}.",
            category="malformed_json",
        ) from error


def _load_jsonl(path: Path, *, asset_path: str) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    try:
        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    _fail(
                        f"Blank JSONL record in {asset_path} at line {line_number}.",
                        category="malformed_jsonl",
                    )
                try:
                    value = json.loads(
                        line,
                        object_pairs_hook=_strict_object,
                        parse_constant=_reject_json_constant,
                    )
                except (json.JSONDecodeError, ValueError, RecursionError) as error:
                    raise BenchmarkAssetError(
                        f"Malformed JSONL record in {asset_path} at line {line_number}.",
                        category="malformed_jsonl",
                    ) from error
                if not isinstance(value, dict):
                    _fail(
                        f"JSONL record in {asset_path} at line {line_number} must be an object.",
                        category="malformed_jsonl",
                    )
                records.append(value)
    except (OSError, UnicodeError) as error:
        raise BenchmarkAssetError(
            f"Could not read JSONL asset {asset_path}.",
            category="asset_unavailable",
        ) from error
    return records


def _require_object(value: object, *, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        _fail(f"{context} must be a JSON object.")
    return value


def _require_string(
    value: object,
    *,
    field: str,
    context: str,
    allow_empty: bool = False,
) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        _fail(f"{context} has an invalid or missing {field} field.")
    return value


def _require_exact_fields(
    value: Mapping[str, object],
    required: frozenset[str],
    *,
    allowed: frozenset[str] | None = None,
    context: str,
) -> None:
    missing = required - value.keys()
    extra = value.keys() - (allowed if allowed is not None else required)
    if missing:
        _fail(f"{context} is missing required fields.")
    if extra:
        _fail(f"{context} contains unsupported fields.")


def _validate_date(value: object, *, field: str) -> str:
    text = _require_string(value, field=field, context="Asset declaration")
    try:
        parsed = date.fromisoformat(text)
    except ValueError as error:
        raise BenchmarkAssetError(
            f"Asset declaration has an invalid {field} date.",
            category="invalid_assets",
        ) from error
    if parsed.isoformat() != text:
        _fail(f"Asset declaration has an invalid {field} date.")
    return text


def _validate_url(value: object, *, field: str) -> str:
    text = _require_string(value, field=field, context="Asset declaration")
    try:
        parsed = urlsplit(text)
    except ValueError:
        _fail(f"Asset declaration has an invalid {field} URL.")
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
    ):
        _fail(f"Asset declaration has an invalid {field} URL.")
    return text


def _safe_relative_path(value: object, *, field: str) -> PurePosixPath:
    text = _require_string(value, field=field, context="Asset declaration")
    path = PurePosixPath(text)
    if (
        "\\" in text
        or "\x00" in text
        or path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in text.split("/"))
    ):
        _fail(f"Asset declaration has an unsafe {field} path.")
    return path


def _assert_no_symlink_components(
    path: Path,
    *,
    relative_to: Path,
    allow_missing: bool = False,
) -> None:
    try:
        relative = path.relative_to(relative_to)
    except ValueError as error:
        raise BenchmarkAssetError(
            "A benchmark path resolves outside data/benchmark.",
            category="unsafe_path",
        ) from error
    current = relative_to
    for part in relative.parts:
        current = current / part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            if allow_missing:
                return
            _fail(
                "A declared benchmark path does not exist.",
                category="asset_unavailable",
            )
        except OSError as error:
            raise BenchmarkAssetError(
                "A benchmark path could not be inspected.",
                category="asset_unavailable",
            ) from error
        if stat.S_ISLNK(mode):
            _fail(
                "Symlinks are not allowed in benchmark asset paths.",
                category="unsafe_path",
            )


def _resolve_existing_file(
    root: Path,
    relative: PurePosixPath,
    *,
    description: str,
) -> Path:
    benchmark_root = root / Path(*BENCHMARK_ROOT.parts)
    candidate = benchmark_root.joinpath(*relative.parts)
    _assert_no_symlink_components(candidate, relative_to=benchmark_root)
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(benchmark_root.resolve(strict=True))
    except (OSError, ValueError) as error:
        raise BenchmarkAssetError(
            f"{description} must resolve beneath data/benchmark.",
            category="unsafe_path",
        ) from error
    if not resolved.is_file():
        _fail(f"{description} must be a regular file.", category="unsafe_path")
    return resolved


def _resolve_existing_directory(
    root: Path,
    relative: PurePosixPath,
    *,
    description: str,
) -> Path:
    benchmark_root = root / Path(*BENCHMARK_ROOT.parts)
    candidate = benchmark_root.joinpath(*relative.parts)
    _assert_no_symlink_components(candidate, relative_to=benchmark_root)
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(benchmark_root.resolve(strict=True))
    except (OSError, ValueError) as error:
        raise BenchmarkAssetError(
            f"{description} must resolve beneath data/benchmark.",
            category="unsafe_path",
        ) from error
    if not resolved.is_dir():
        _fail(f"{description} must be a directory.", category="unsafe_path")
    return resolved


def _require_ignored(root: Path, relative_paths: tuple[PurePosixPath, ...]) -> None:
    for relative in relative_paths:
        try:
            completed = subprocess.run(
                [
                    "git",
                    "-C",
                    str(root),
                    "check-ignore",
                    "--no-index",
                    "-q",
                    "--",
                    relative.as_posix(),
                ],
                check=False,
                capture_output=True,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise BenchmarkAssetError(
                "Could not verify the benchmark ignore rule with git.",
                category="ignore_rule_unavailable",
            ) from error
        if completed.returncode != 0:
            _fail(
                "The benchmark declaration, assets, and reports must be ignored by git.",
                category="ignore_rule_missing",
            )


def resolve_report_path(
    workspace_root: Path,
    report_name: str | None = None,
    *,
    create_directory: bool = False,
) -> Path:
    """Resolve the ignored reports directory or a safe, non-overwriting JSON path."""
    if report_name is not None and not _REPORT_NAME_PATTERN.fullmatch(report_name):
        _fail(
            "The report name must be a simple JSON filename.",
            category="unsafe_report_path",
        )
    try:
        root = workspace_root.resolve(strict=True)
    except OSError as error:
        raise BenchmarkAssetError(
            "The workspace root is unavailable.",
            category="workspace_unavailable",
        ) from error
    benchmark_root = root / Path(*BENCHMARK_ROOT.parts)
    _assert_no_symlink_components(benchmark_root, relative_to=root)
    if not benchmark_root.is_dir():
        _fail(
            "Local benchmark assets are unavailable beneath data/benchmark.",
            category="asset_unavailable",
        )
    relative_report = REPORTS_ROOT / (report_name or ".report-write-check.json")
    _require_ignored(root, (relative_report,))

    reports_directory = root / Path(*REPORTS_ROOT.parts)
    _assert_no_symlink_components(
        reports_directory,
        relative_to=benchmark_root,
        allow_missing=True,
    )
    if reports_directory.exists() and not reports_directory.is_dir():
        _fail(
            "The benchmark reports path must be a directory.",
            category="unsafe_report_path",
        )
    if create_directory and not reports_directory.exists():
        try:
            reports_directory.mkdir()
        except OSError as error:
            raise BenchmarkAssetError(
                "The benchmark reports directory could not be created.",
                category="report_unavailable",
            ) from error
        _assert_no_symlink_components(
            reports_directory,
            relative_to=benchmark_root,
        )

    if report_name is None:
        try:
            resolved_directory = reports_directory.resolve(strict=False)
            resolved_directory.relative_to(benchmark_root.resolve(strict=True))
        except (OSError, ValueError) as error:
            raise BenchmarkAssetError(
                "The reports directory must resolve beneath data/benchmark.",
                category="unsafe_report_path",
            ) from error
        return resolved_directory

    candidate = reports_directory / report_name
    _assert_no_symlink_components(
        candidate,
        relative_to=benchmark_root,
        allow_missing=True,
    )
    if candidate.exists() or candidate.is_symlink():
        _fail(
            "The benchmark report already exists.",
            category="report_exists",
        )
    try:
        resolved = candidate.resolve(strict=False)
        resolved.relative_to(benchmark_root.resolve(strict=True))
    except (OSError, ValueError) as error:
        raise BenchmarkAssetError(
            "The report path must resolve beneath data/benchmark/reports.",
            category="unsafe_report_path",
        ) from error
    return resolved


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise BenchmarkAssetError(
            "A declared benchmark asset could not be read.",
            category="asset_unavailable",
        ) from error
    return digest.hexdigest()


def _validate_asset_record(
    value: object,
    *,
    root: Path,
    index: int,
) -> tuple[AssetProvenance, Path]:
    context = f"Asset record {index}"
    record = _require_object(value, context=context)
    _require_exact_fields(record, _ASSET_FIELDS, context=context)
    kind = _require_string(record["kind"], field="kind", context=context)
    if kind not in {"public_tasks", "gold_tasks", "database", "database_inventory"}:
        _fail(f"{context} has an unsupported kind.")
    relative = _safe_relative_path(record["path"], field="path")
    asset_path = _resolve_existing_file(root, relative, description="A declared asset")
    _require_ignored(root, (BENCHMARK_ROOT / relative,))

    expected_hash = _require_string(record["sha256"], field="sha256", context=context)
    if _SHA256_PATTERN.fullmatch(expected_hash) is None:
        _fail(f"{context} has an invalid SHA-256 digest.")
    actual_hash = _hash_file(asset_path)
    if actual_hash.casefold() != expected_hash.casefold():
        _fail(f"The SHA-256 digest does not match asset {relative.as_posix()}.")

    source_url = _validate_url(record["source_url"], field="source_url")
    terms_id = _require_string(record["terms_id"], field="terms_id", context=context)
    terms_url = _validate_url(record["terms_url"], field="terms_url")
    retrieved_on = _validate_date(record["retrieved_on"], field="retrieved_on")
    verified_on = _validate_date(record["verified_on"], field="verified_on")
    terms_verified = record["terms_verified_by_user"]
    transmission_allowed = record["model_transmission_allowed"]
    if not isinstance(terms_verified, bool) or not isinstance(
        transmission_allowed, bool
    ):
        _fail(f"{context} must explicitly attest both authorization booleans.")

    return (
        AssetProvenance(
            kind=kind,
            path=relative.as_posix(),
            sha256=actual_hash,
            source_url=source_url,
            retrieved_on=retrieved_on,
            terms_id=terms_id,
            terms_url=terms_url,
            terms_verified_by_user=terms_verified,
            verified_on=verified_on,
            model_transmission_allowed=transmission_allowed,
        ),
        asset_path,
    )


def _validate_inventory(
    path: Path,
    *,
    asset_path: str,
    selected_database: str,
) -> tuple[str, ...]:
    value = _require_object(
        _load_json(path, description="database inventory"),
        context=f"Database inventory {asset_path}",
    )
    _require_exact_fields(
        value,
        frozenset({"schema_version", "selected_database", "base_tables"}),
        context=f"Database inventory {asset_path}",
    )
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != INVENTORY_SCHEMA_VERSION
    ):
        _fail(f"Database inventory {asset_path} has an unsupported schema version.")
    inventory_database = _require_string(
        value["selected_database"],
        field="selected_database",
        context=f"Database inventory {asset_path}",
    )
    if inventory_database != selected_database:
        _fail(f"Database inventory {asset_path} does not match its database mapping.")
    base_tables = value["base_tables"]
    if not isinstance(base_tables, list) or any(
        not isinstance(table, str) for table in base_tables
    ):
        _fail(f"Database inventory {asset_path} has an invalid base_tables field.")
    if base_tables != sorted(set(base_tables)):
        _fail(f"Database inventory {asset_path} must list unique sorted base tables.")
    for table in base_tables:
        match = _BASE_TABLE_PATTERN.fullmatch(table)
        if (
            match is None
            or match["schema"].casefold()
            in {
                "information_schema",
                "pg_catalog",
            }
            or match["schema"].casefold().startswith("pg_")
        ):
            _fail(f"Database inventory {asset_path} contains an invalid base table.")
    return tuple(base_tables)


def _walk_regular_files(directory: Path) -> set[Path]:
    files: set[Path] = set()
    try:
        for child in directory.iterdir():
            mode = child.lstat().st_mode
            if stat.S_ISLNK(mode):
                _fail(
                    "Symlinks are not allowed beneath database asset roots.",
                    category="unsafe_path",
                )
            if stat.S_ISDIR(mode):
                files.update(_walk_regular_files(child))
            elif stat.S_ISREG(mode):
                files.add(child.resolve(strict=True))
            else:
                _fail("Only regular files are allowed beneath database asset roots.")
    except OSError as error:
        raise BenchmarkAssetError(
            "A database asset directory could not be inspected.",
            category="asset_unavailable",
        ) from error
    return files


def _validate_database_mappings(
    value: object,
    *,
    root: Path,
    assets: tuple[AssetProvenance, ...],
    resolved_assets: Mapping[str, Path],
) -> dict[str, DatabaseInventory]:
    mappings = _require_object(value, context="Database mappings")
    if not mappings:
        _fail("The asset declaration must map at least one database.")

    by_path = {asset.path: asset for asset in assets}
    inventories: dict[str, DatabaseInventory] = {}
    assigned_database_files: set[str] = set()
    assigned_roots: set[str] = set()
    for selected_database, raw_mapping in mappings.items():
        if not isinstance(selected_database, str) or not selected_database.strip():
            _fail("Database mappings must use non-empty database identifiers.")
        context = f"Database mapping for {selected_database}"
        mapping = _require_object(raw_mapping, context=context)
        _require_exact_fields(mapping, _DATABASE_MAPPING_FIELDS, context=context)
        template = _require_string(
            mapping["template_database"],
            field="template_database",
            context=context,
        )
        if _DATABASE_IDENTIFIER_PATTERN.fullmatch(
            template
        ) is None or not template.startswith("nl_sql_"):
            _fail(f"{context} must name a local nl_sql_ PostgreSQL template.")

        asset_root_relative = _safe_relative_path(
            mapping["asset_path"],
            field="asset_path",
        )
        asset_root = _resolve_existing_directory(
            root,
            asset_root_relative,
            description=f"{context} asset_path",
        )
        asset_root_name = asset_root_relative.as_posix()
        if asset_root_name in assigned_roots:
            _fail("Database mappings must not share an asset root.")
        assigned_roots.add(asset_root_name)

        inventory_relative = _safe_relative_path(
            mapping["inventory_path"],
            field="inventory_path",
        )
        inventory_asset_path = inventory_relative.as_posix()
        inventory_asset = by_path.get(inventory_asset_path)
        if inventory_asset is None or inventory_asset.kind != "database_inventory":
            _fail(f"{context} must reference a hashed database_inventory asset.")
        inventory = _validate_inventory(
            resolved_assets[inventory_asset_path],
            asset_path=inventory_asset_path,
            selected_database=selected_database,
        )

        root_path = PurePosixPath(asset_root_name)
        database_asset_names = {
            asset.path
            for asset in assets
            if asset.kind == "database"
            and PurePosixPath(asset.path).is_relative_to(root_path)
        }
        if not database_asset_names:
            _fail(f"{context} has no per-file database asset declarations.")
        for asset_name in database_asset_names:
            if asset_name in assigned_database_files:
                _fail(
                    "A database asset cannot be shared by multiple database mappings."
                )
            assigned_database_files.add(asset_name)

        actual_files = _walk_regular_files(asset_root)
        expected_paths = {resolved_assets[name] for name in database_asset_names}
        inventory_file = resolved_assets[inventory_asset_path]
        if inventory_file.is_relative_to(asset_root):
            expected_paths.add(inventory_file)
        if actual_files != expected_paths:
            _fail(f"{context} must declare and hash every file beneath its asset root.")

        inventories[selected_database] = DatabaseInventory(
            selected_database=selected_database,
            template_database=template,
            asset_path=asset_root_name,
            inventory_path=inventory_asset_path,
            base_tables=inventory,
        )

    declared_database_files = {
        asset.path for asset in assets if asset.kind == "database"
    }
    if assigned_database_files != declared_database_files:
        _fail("Every database asset must belong to exactly one database mapping.")
    return inventories


def _validate_task_records(
    public_records: list[dict[str, object]],
    gold_records: list[dict[str, object]],
    *,
    databases: Mapping[str, DatabaseInventory],
) -> tuple[BenchmarkTask, ...]:
    public_by_id: dict[str, dict[str, object]] = {}
    gold_by_id: dict[str, dict[str, object]] = {}

    for label, records, target, required in (
        ("public", public_records, public_by_id, _PUBLIC_REQUIRED_FIELDS),
        ("gold", gold_records, gold_by_id, _GOLD_REQUIRED_FIELDS),
    ):
        for record_number, record in enumerate(records, start=1):
            context = f"{label} task record {record_number}"
            missing = required - record.keys()
            if missing:
                _fail(f"{context} is missing required fields.")
            for field in _STRING_FIELDS & record.keys():
                _require_string(record[field], field=field, context=context)
            _validate_sql_list_fields(record, context=context)
            instance_id = record["instance_id"]
            if not isinstance(instance_id, str) or not instance_id.strip():
                _fail(f"{context} has an invalid instance_id field.")
            if instance_id in target:
                _fail(f"Duplicate instance_id in {label} task records.")
            target[instance_id] = record

    if public_by_id.keys() != gold_by_id.keys():
        _fail("Public and gold task files do not have one-to-one instance_id coverage.")
    if not public_by_id:
        _fail("Public and gold task files must each contain at least one record.")

    tasks: list[BenchmarkTask] = []
    for instance_id in sorted(public_by_id):
        public = public_by_id[instance_id]
        gold = gold_by_id[instance_id]
        context = f"Task record {instance_id}"
        selected_database = public["selected_database"]
        query = public["query"]
        if not isinstance(selected_database, str) or not isinstance(query, str):
            _fail(f"{context} has an invalid public task record.")
        if selected_database not in databases:
            _fail(f"{context} references an unresolved selected_database.")
        for field in ("selected_database", "query"):
            if field in gold and gold[field] != public[field]:
                _fail(f"Public and gold {field} values differ for task {instance_id}.")
        if "instance_id" in gold and gold["instance_id"] != instance_id:
            _fail(f"Public and gold instance_id values differ for task {instance_id}.")

        evaluator_record = dict(public)
        evaluator_record.update(gold)
        tasks.append(
            BenchmarkTask(
                instance_id=instance_id,
                selected_database=selected_database,
                model_input=ModelTaskInput(
                    query=query, selected_database=selected_database
                ),
                evaluator=EvaluatorTaskInput(
                    record=MappingProxyType(evaluator_record)
                ),
            )
        )
    return tuple(tasks)


def _validate_sql_list_fields(
    record: Mapping[str, object],
    *,
    context: str,
) -> None:
    for field in _SQL_LIST_FIELDS & record.keys():
        statements = record[field]
        if not isinstance(statements, list) or any(
            not isinstance(statement, str) for statement in statements
        ):
            _fail(f"{context} has an invalid {field} field.")


def load_local_benchmark_assets() -> PreflightResult:
    """Validate local declarations and task pairs beneath the current workspace."""
    return _load_local_benchmark_assets_at(Path.cwd())


def _load_local_benchmark_assets_at(workspace_root: Path) -> PreflightResult:
    """Load test fixtures from an explicit workspace root.

    The declaration path is fixed at ``data/benchmark/asset-declaration.json``.
    All assets must be regular, non-symlink files beneath the ignored benchmark root.
    """
    try:
        root = workspace_root.resolve(strict=True)
    except OSError as error:
        raise BenchmarkAssetError(
            "The workspace root is unavailable.",
            category="workspace_unavailable",
        ) from error
    if not root.is_dir():
        _fail(
            "The workspace root must be a directory.", category="workspace_unavailable"
        )

    _require_ignored(
        root,
        (ASSET_DECLARATION_PATH, REPORTS_ROOT / "preflight.json"),
    )
    benchmark_root = root / Path(*BENCHMARK_ROOT.parts)
    _assert_no_symlink_components(benchmark_root, relative_to=root)
    if not benchmark_root.is_dir() or benchmark_root.is_symlink():
        _fail(
            "Local benchmark assets are unavailable beneath data/benchmark.",
            category="asset_unavailable",
        )

    declaration_path = root / Path(*ASSET_DECLARATION_PATH.parts)
    _assert_no_symlink_components(declaration_path, relative_to=benchmark_root)
    declaration = _require_object(
        _load_json(declaration_path, description="asset declaration"),
        context="Asset declaration",
    )
    _require_exact_fields(declaration, _DECLARATION_FIELDS, context="Asset declaration")
    if (
        type(declaration["schema_version"]) is not int
        or declaration["schema_version"] != ASSET_SCHEMA_VERSION
    ):
        _fail("The asset declaration has an unsupported schema version.")
    dataset_revision = _require_string(
        declaration["dataset_revision"],
        field="dataset_revision",
        context="Asset declaration",
    )
    asset_values = declaration["assets"]
    if not isinstance(asset_values, list) or not asset_values:
        _fail("The asset declaration must contain per-file asset records.")

    provenance: list[AssetProvenance] = []
    resolved_assets: dict[str, Path] = {}
    for index, value in enumerate(asset_values, start=1):
        asset, resolved = _validate_asset_record(value, root=root, index=index)
        if asset.path in resolved_assets:
            _fail("The asset declaration contains a duplicate path.")
        provenance.append(asset)
        resolved_assets[asset.path] = resolved

    assets = tuple(provenance)
    kinds = {asset.kind for asset in assets}
    required_kinds = {"public_tasks", "gold_tasks", "database", "database_inventory"}
    if not required_kinds.issubset(kinds):
        _fail("The asset declaration is missing a required asset kind.")

    databases = _validate_database_mappings(
        declaration["databases"],
        root=root,
        assets=assets,
        resolved_assets=resolved_assets,
    )
    public_records: list[dict[str, object]] = []
    gold_records: list[dict[str, object]] = []
    for asset in assets:
        if asset.kind == "public_tasks":
            public_records.extend(
                _load_jsonl(resolved_assets[asset.path], asset_path=asset.path)
            )
        elif asset.kind == "gold_tasks":
            gold_records.extend(
                _load_jsonl(resolved_assets[asset.path], asset_path=asset.path)
            )
    tasks = _validate_task_records(public_records, gold_records, databases=databases)
    categories = tuple(
        sorted(
            {
                task.evaluator.record["category"]
                for task in tasks
                if isinstance(task.evaluator.record.get("category"), str)
            }
        )
    )
    if "Query" not in categories:
        observed = (
            ", ".join(repr(category[:64]) for category in categories[:20]) or "(none)"
        )
        if len(categories) > 20:
            observed += ", ..."
        _fail(
            "The local task records do not contain the expected Query category. "
            f"Observed categories: {observed}."
        )
    run_blockers = tuple(
        sorted(
            {
                "asset_terms_not_verified"
                for asset in assets
                if not asset.terms_verified_by_user
            }
            | {
                "model_transmission_not_authorized"
                for asset in assets
                if not asset.model_transmission_allowed
            }
        )
    )
    return PreflightResult(
        dataset_revision=dataset_revision,
        assets=assets,
        databases=MappingProxyType(databases),
        tasks=tasks,
        observed_categories=categories,
        run_blockers=run_blockers,
    )


def load_public_base_lite_tasks() -> tuple[Mapping[str, object], ...]:
    """Parse the local public Base-Lite task file without requiring gold assets."""
    try:
        root = Path.cwd().resolve(strict=True)
    except OSError as error:
        raise BenchmarkAssetError(
            "The workspace root is unavailable.",
            category="workspace_unavailable",
        ) from error
    if not root.is_dir():
        _fail(
            "The workspace root must be a directory.", category="workspace_unavailable"
        )

    _require_ignored(root, (PUBLIC_BASE_LITE_PATH,))
    path = _resolve_existing_file(
        root,
        PUBLIC_BASE_LITE_PATH.relative_to(BENCHMARK_ROOT),
        description="The public Base-Lite task file",
    )
    records = _load_jsonl(path, asset_path=PUBLIC_BASE_LITE_PATH.as_posix())
    validated: list[Mapping[str, object]] = []
    seen_ids: set[str] = set()
    for record_number, record in enumerate(records, start=1):
        context = f"Public Base-Lite task record {record_number}"
        missing = _PUBLIC_REQUIRED_FIELDS - record.keys()
        if missing:
            _fail(f"{context} is missing required fields.")
        for field in _STRING_FIELDS & record.keys():
            _require_string(record[field], field=field, context=context)
        _validate_sql_list_fields(record, context=context)
        instance_id = record["instance_id"]
        if not isinstance(instance_id, str) or not instance_id.strip():
            _fail(f"{context} has an invalid instance_id field.")
        if instance_id in seen_ids:
            _fail("Duplicate instance_id in public Base-Lite task records.")
        seen_ids.add(instance_id)
        validated.append(MappingProxyType(dict(record)))
    return tuple(validated)
