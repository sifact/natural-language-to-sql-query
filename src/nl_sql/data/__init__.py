"""Validated task data, selection, knowledge, and demo fixtures."""

from .assets import (
    AssetProvenance,
    BenchmarkAssetError,
    BenchmarkTask,
    DatabaseInventory,
    EvaluatorTaskInput,
    ModelTaskInput,
    PreflightResult,
    load_local_benchmark_assets,
    load_public_base_lite_tasks,
    resolve_report_path,
)
from .demo import DEFAULT_DATABASE_PATH, initialize_database
from .demo_questions import DEMO_QUESTIONS, DemoQuestion
from .knowledge import (
    KNOWLEDGE_CATALOG_VERSION,
    ColumnMeaning,
    InMemoryKnowledgeCatalog,
    KnowledgeCatalog,
    KnowledgeDefinition,
    load_public_catalog,
)
from .selection import (
    TaskSelection,
    UnsupportedTask,
    select_article_shaped_tasks,
    validate_run_limits,
)

__all__ = [
    "DEFAULT_DATABASE_PATH",
    "DEMO_QUESTIONS",
    "KNOWLEDGE_CATALOG_VERSION",
    "AssetProvenance",
    "BenchmarkAssetError",
    "BenchmarkTask",
    "ColumnMeaning",
    "DatabaseInventory",
    "DemoQuestion",
    "EvaluatorTaskInput",
    "InMemoryKnowledgeCatalog",
    "KnowledgeCatalog",
    "KnowledgeDefinition",
    "ModelTaskInput",
    "PreflightResult",
    "TaskSelection",
    "UnsupportedTask",
    "initialize_database",
    "load_local_benchmark_assets",
    "load_public_base_lite_tasks",
    "load_public_catalog",
    "resolve_report_path",
    "select_article_shaped_tasks",
    "validate_run_limits",
]
