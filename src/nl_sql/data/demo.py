"""Create the deterministic, synthetic SQLite demo database."""

from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path

DEFAULT_DATABASE_PATH = Path("data/demo.sqlite3")

DEMO_DOMAIN_HINTS = """\
The demo contains synthetic customer orders.
An order is included in shipped totals only when orders.status is "shipped".
Order value is the sum of order_items.quantity multiplied by order_items.unit_price.
The orders.metadata column contains JSON text; metadata.expedited is a Boolean.
Order item prices are historical prices and need not equal the current product price.
"""

SCHEMA = (
    """
    CREATE TABLE customers (
        customer_id TEXT PRIMARY KEY,
        customer_name TEXT NOT NULL,
        region TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE products (
        product_id TEXT PRIMARY KEY,
        product_name TEXT NOT NULL,
        current_price REAL NOT NULL CHECK (current_price >= 0)
    )
    """,
    """
    CREATE TABLE orders (
        order_id TEXT PRIMARY KEY,
        customer_id TEXT NOT NULL REFERENCES customers(customer_id),
        status TEXT NOT NULL,
        order_date TEXT NOT NULL,
        metadata TEXT NOT NULL CHECK (json_valid(metadata))
    )
    """,
    """
    CREATE TABLE order_items (
        order_id TEXT NOT NULL REFERENCES orders(order_id),
        product_id TEXT NOT NULL REFERENCES products(product_id),
        quantity INTEGER NOT NULL CHECK (quantity > 0),
        unit_price REAL NOT NULL CHECK (unit_price >= 0),
        PRIMARY KEY (order_id, product_id)
    )
    """,
)

CUSTOMERS = (
    ("C001", "Northstar Labs", "West"),
    ("C002", "Contoso Retail", "East"),
    ("C003", "Fabrikam Works", "Central"),
)

PRODUCTS = (
    ("P001", "Widget", 80.0),
    ("P002", "Sensor", 55.0),
    ("P003", "Support Plan", 60.0),
)

ORDERS = (
    ("O101", "C001", "shipped", "2025-10-01", {"expedited": True, "channel": "direct"}),
    ("O102", "C002", "shipped", "2025-10-04", {"expedited": False, "channel": "partner"}),
    ("O103", "C001", "pending", "2025-10-08", {"expedited": True, "channel": "direct"}),
    ("O104", "C003", "shipped", "2025-10-10", {"expedited": False, "channel": "direct"}),
)

ORDER_ITEMS = (
    ("O101", "P001", 2, 75.0),
    ("O101", "P002", 1, 50.0),
    ("O102", "P002", 2, 50.0),
    ("O103", "P003", 1, 50.0),
    ("O104", "P001", 1, 50.0),
)


def initialize_database(path: Path = DEFAULT_DATABASE_PATH, *, force: bool = False) -> Path:
    """Create the synthetic demo database without replacing existing data by default.

    Args:
        path: File path where the SQLite database is created.
        force: Whether to atomically replace an existing database.

    Returns:
        The path of the initialized database.

    Raises:
        FileExistsError: If the target exists and force is false.
        sqlite3.Error: If SQLite cannot create the database.
    """
    target = path.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not force:
        raise FileExistsError(f"Database already exists: {target}. Use --force to replace it.")

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)

        with sqlite3.connect(temporary_path) as connection:
            connection.execute("PRAGMA foreign_keys = ON")
            for statement in SCHEMA:
                connection.execute(statement)
            connection.executemany(
                "INSERT INTO customers VALUES (?, ?, ?)",
                CUSTOMERS,
            )
            connection.executemany(
                "INSERT INTO products VALUES (?, ?, ?)",
                PRODUCTS,
            )
            connection.executemany(
                "INSERT INTO orders VALUES (?, ?, ?, ?, ?)",
                (
                    (*order[:4], json.dumps(order[4], sort_keys=True))
                    for order in ORDERS
                ),
            )
            connection.executemany(
                "INSERT INTO order_items VALUES (?, ?, ?, ?)",
                ORDER_ITEMS,
            )

        temporary_path.replace(target)
        return target
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
