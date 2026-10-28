"""Model-visible read-only database and knowledge tool schemas."""

from __future__ import annotations

from ..data.knowledge import (
    MAX_KNOWLEDGE_QUERY_LENGTH,
    MAX_KNOWLEDGE_RESULTS,
)

DATABASE_TOOLS: list[dict[str, object]] = [
    {
        "type": "function",
        "function": {
            "name": "list_tables",
            "description": "List tables in the configured database.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_schema",
            "description": "Inspect columns and relationships for every table.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_table",
            "description": "Inspect columns and relationships for one table.",
            "parameters": {
                "type": "object",
                "properties": {"table_name": {"type": "string"}},
                "required": ["table_name"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query",
            "description": "Execute one read-only SELECT and return bounded rows.",
            "parameters": {
                "type": "object",
                "properties": {"sql": {"type": "string"}},
                "required": ["sql"],
                "additionalProperties": False,
            },
        },
    },
]

KNOWLEDGE_TOOLS: list[dict[str, object]] = [
    {
        "type": "function",
        "function": {
            "name": "lookup_column_meaning",
            "description": "Look up the documented meaning of one table column.",
            "parameters": {
                "type": "object",
                "properties": {
                    "table_name": {"type": "string"},
                    "column_name": {"type": "string"},
                },
                "required": ["table_name", "column_name"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "Search for up to five knowledge definitions relevant to a question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": MAX_KNOWLEDGE_QUERY_LENGTH,
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": MAX_KNOWLEDGE_RESULTS,
                    },
                },
                "required": ["query", "limit"],
                "additionalProperties": False,
            },
        },
    },
]

SUBMISSION_TOOLS: list[dict[str, object]] = [
    {
        "type": "function",
        "function": {
            "name": "submit_answer",
            "description": (
                "Submit the final SQL and concise answer to complete this task. "
                "Plain text alone does not complete the task."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string", "minLength": 1},
                    "answer": {"type": "string", "minLength": 1},
                },
                "required": ["sql", "answer"],
                "additionalProperties": False,
            },
        },
    }
]
