# task_sandbox.py

Manages one local PostgreSQL clone for each selected evaluation task.

* Requires distinct named setup and query roles with loopback-only connection settings.
* Copies only the validated template database into a generated run-owned clone.
* Restricts clone connection access to the configured query role.
* Runs hash-verified task setup and cleanup SQL only inside that task's clone.
* Drops only a clone created by the current sandbox and surfaces cleanup failures.
