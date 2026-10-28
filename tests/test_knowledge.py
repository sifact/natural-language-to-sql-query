"""Tests for the versioned in-memory database knowledge catalog."""

import pytest

from nl_sql.data.knowledge import (
    KNOWLEDGE_CATALOG_VERSION,
    ColumnMeaning,
    InMemoryKnowledgeCatalog,
    KnowledgeDefinition,
)

MAX_LIMIT = 5


def test_given_column_meaning_when_looked_up_case_insensitively_then_returns_one_record() -> (
    None
):
    # Arrange
    meaning = ColumnMeaning("orders", "status", "Current lifecycle state.")
    catalog = InMemoryKnowledgeCatalog(column_meanings=[meaning])

    # Act
    result = catalog.lookup_column_meaning("ORDERS", "Status")

    # Assert
    assert result == meaning


def test_given_unknown_column_when_looked_up_then_returns_not_found() -> None:
    # Arrange
    catalog = InMemoryKnowledgeCatalog()

    # Act
    result = catalog.lookup_column_meaning("orders", "unknown")

    # Assert
    assert result is None


def test_given_matching_knowledge_when_searched_then_ranks_relevance_and_caps_results() -> (
    None
):
    # Arrange
    entries = [
        KnowledgeDefinition(
            "Invoice reconciliation",
            "Reconciliation matches invoices with received payments.",
        ),
        KnowledgeDefinition(
            "Invoice records",
            "Invoice details are stored with an issue date.",
        ),
        KnowledgeDefinition(
            "Invoice status",
            "The status shows whether an invoice is open or paid.",
        ),
        KnowledgeDefinition(
            "Invoice currency",
            "The currency identifies the invoice denomination.",
        ),
        KnowledgeDefinition(
            "Invoice totals",
            "The total is the amount due on an invoice.",
        ),
        KnowledgeDefinition(
            "Invoice exceptions",
            "Exceptions are recorded for an invoice that needs review.",
        ),
    ]
    catalog = InMemoryKnowledgeCatalog(knowledge_definitions=entries)

    # Act
    results = catalog.search_knowledge("invoice reconciliation", limit=MAX_LIMIT)

    # Assert
    assert len(results) == 5
    assert results[0].name == "Invoice reconciliation"
    assert all(not hasattr(item, "id") for item in results)


def test_given_invalid_search_limit_when_searched_then_rejects_request() -> None:
    # Arrange
    catalog = InMemoryKnowledgeCatalog()

    # Act
    with pytest.raises(ValueError, match="between one and five"):
        catalog.search_knowledge("order status", limit=6)


def test_given_catalog_when_created_then_exposes_supported_contract_version() -> None:
    # Arrange
    catalog = InMemoryKnowledgeCatalog()

    # Act
    version = catalog.version

    # Assert
    assert version == KNOWLEDGE_CATALOG_VERSION
