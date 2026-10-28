"""Versioned, in-memory database knowledge discovery contracts."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

KNOWLEDGE_CATALOG_VERSION = "nl-sql-knowledge-v1"
MAX_KNOWLEDGE_RESULTS = 5
MAX_KNOWLEDGE_QUERY_LENGTH = 2_000
PUBLIC_METADATA_ROOT = Path("data/benchmark/public-base-lite")
_DATABASE_NAME_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,62}\Z")
_TOKEN_PATTERN = re.compile(r"[\w]+", re.UNICODE)
_STOP_WORDS = frozenset(
    {
        "about",
        "and",
        "are",
        "does",
        "for",
        "from",
        "how",
        "in",
        "is",
        "of",
        "on",
        "or",
        "the",
        "to",
        "into",
        "what",
        "when",
        "where",
        "which",
        "with",
    }
)


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in _TOKEN_PATTERN.findall(text.casefold().replace("_", " "))
        if token not in _STOP_WORDS
    }


@dataclass(frozen=True, slots=True)
class ColumnMeaning:
    """Public database meaning for one table column."""

    table_name: str
    column_name: str
    meaning: str


@dataclass(frozen=True, slots=True)
class KnowledgeDefinition:
    """Human-readable database knowledge without an internal or task-required ID."""

    name: str
    definition: str


class KnowledgeCatalog(Protocol):
    """Versioned, read-only lookup interface exposed to the single agent."""

    version: str

    def lookup_column_meaning(
        self,
        table_name: str,
        column_name: str,
    ) -> ColumnMeaning | None: ...

    def search_knowledge(
        self,
        query: str,
        limit: int,
    ) -> list[KnowledgeDefinition]: ...


class InMemoryKnowledgeCatalog:
    """Provide deterministic column lookups and bounded relevance-ranked search."""

    version = KNOWLEDGE_CATALOG_VERSION

    def __init__(
        self,
        column_meanings: Sequence[ColumnMeaning] = (),
        knowledge_definitions: Sequence[KnowledgeDefinition] = (),
    ) -> None:
        """Create a catalog from caller-validated, human-readable knowledge records."""
        self._column_meanings: dict[tuple[str, str], ColumnMeaning] = {}
        self._knowledge_definitions = tuple(knowledge_definitions)

        for item in column_meanings:
            table_name = item.table_name.strip()
            column_name = item.column_name.strip()
            meaning = item.meaning.strip()
            key = (table_name.casefold(), column_name.casefold())
            if (
                not table_name
                or not column_name
                or not meaning
                or key in self._column_meanings
            ):
                raise ValueError("Column meanings must be non-empty and unique.")
            self._column_meanings[key] = ColumnMeaning(table_name, column_name, meaning)

        seen_names: set[str] = set()
        for item in self._knowledge_definitions:
            name = item.name.strip()
            definition = item.definition.strip()
            normalized_name = name.casefold()
            if not name or not definition or normalized_name in seen_names:
                raise ValueError(
                    "Knowledge definitions must have non-empty unique names and text."
                )
            seen_names.add(normalized_name)
        self._knowledge_definitions = tuple(
            KnowledgeDefinition(item.name.strip(), item.definition.strip())
            for item in self._knowledge_definitions
        )

    def lookup_column_meaning(
        self,
        table_name: str,
        column_name: str,
    ) -> ColumnMeaning | None:
        """Return one matching meaning without exposing other catalog entries."""
        return self._column_meanings.get(
            (table_name.casefold(), column_name.casefold())
        )

    def search_knowledge(
        self,
        query: str,
        limit: int,
    ) -> list[KnowledgeDefinition]:
        """Return up to five stable, relevance-ranked human-readable definitions."""
        if not query.strip() or len(query) > MAX_KNOWLEDGE_QUERY_LENGTH:
            raise ValueError("Knowledge search requires a non-empty bounded query.")
        if (
            not isinstance(limit, int)
            or isinstance(limit, bool)
            or not 1 <= limit <= MAX_KNOWLEDGE_RESULTS
        ):
            raise ValueError("Knowledge search limit must be between one and five.")

        query_terms = _tokens(query)
        if not query_terms:
            return []

        documents = [
            (
                item,
                _tokens(f"{item.name} {item.definition}"),
            )
            for item in self._knowledge_definitions
        ]
        document_frequency = Counter(
            token for _, tokens in documents for token in tokens
        )
        total_documents = len(documents)
        ranked: list[tuple[float, KnowledgeDefinition]] = []
        for item, tokens in documents:
            matching_terms = query_terms & tokens
            if not matching_terms:
                continue
            score = sum(
                math.log((total_documents + 1) / (document_frequency[token] + 1)) + 1
                for token in matching_terms
            )
            ranked.append((score, item))

        ranked.sort(key=lambda result: (-result[0], result[1].name.casefold()))
        return [item for _, item in ranked[: min(limit, MAX_KNOWLEDGE_RESULTS)]]


def load_public_catalog(
    database: str,
    root: Path = PUBLIC_METADATA_ROOT,
) -> InMemoryKnowledgeCatalog | None:
    """Load one Base-Lite database's public knowledge base and column meanings.

    These public metadata files play the role of the article's domain hints. Knowledge
    IDs are dropped so the agent never sees the evaluator-only ``external_knowledge``
    labels. Returns ``None`` when the database has no local metadata.
    """
    if _DATABASE_NAME_PATTERN.fullmatch(database) is None:
        raise ValueError("Knowledge metadata requires a plain database identifier.")
    directory = root / database
    knowledge_path = directory / f"{database}_kb.jsonl"
    meanings_path = directory / f"{database}_column_meaning_base.json"
    if not knowledge_path.is_file() and not meanings_path.is_file():
        return None

    definitions: list[KnowledgeDefinition] = []
    seen_names: set[str] = set()
    if knowledge_path.is_file():
        for line in knowledge_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            name = str(record.get("knowledge") or "").strip()
            parts = (record.get("description"), record.get("definition"))
            text = " ".join(str(part).strip() for part in parts if part)
            if name and text and name.casefold() not in seen_names:
                seen_names.add(name.casefold())
                definitions.append(KnowledgeDefinition(name, text))

    meanings: list[ColumnMeaning] = []
    seen_columns: set[tuple[str, str]] = set()
    if meanings_path.is_file():
        raw_meanings = json.loads(meanings_path.read_text(encoding="utf-8"))
        for key, value in raw_meanings.items():
            # Keys look like "database|Table|Column"; JSON columns map to nested dicts.
            _, _, rest = key.partition("|")
            table_name, _, column_name = rest.partition("|")
            meaning = value if isinstance(value, str) else json.dumps(value)
            column_key = (table_name.casefold(), column_name.casefold())
            if table_name and column_name and column_key not in seen_columns:
                seen_columns.add(column_key)
                meanings.append(ColumnMeaning(table_name, column_name, meaning))

    return InMemoryKnowledgeCatalog(meanings, definitions)
