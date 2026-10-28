# postgres_tools.py

Provides bounded read-only tools for local, run-owned PostgreSQL databases.

* Restricts targets to loopback servers or Unix sockets and `nl_sql_` databases.
* Requires a non-elevated role with access only to the active database.
* Executes within read-only transactions and approved schemas.
* Applies statement, lock, and result limits from `Settings`.
* Rejects unsafe SQL and returns fixed-category errors.
