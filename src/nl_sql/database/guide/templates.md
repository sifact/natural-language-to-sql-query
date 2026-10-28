# templates.py

Checks local PostgreSQL template catalogs against declared static inventories.

* Reads the setup DSN from centralized `Settings`.
* Restricts connections to loopback hosts or Unix sockets.
* Verifies role privileges and uses a read-only transaction.
* Applies configured statement and lock deadlines.
* Fails before task execution when the live inventory differs.
