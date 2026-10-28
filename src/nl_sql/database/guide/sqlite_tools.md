# sqlite_tools.py

Provides bounded, read-only schema discovery and SQL queries for SQLite.

* Opens the configured database in read-only mode.
* Allows only the demo's tables, functions, and read operations.
* Enforces row and execution-time limits from `Settings`.
* Returns truncation state when a query exceeds the row cap.
* Preserves sorted table listing and relationship metadata.
