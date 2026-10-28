# tools.py

Declares the model-visible JSON schemas for database, knowledge, and submission tools.

* Lists only the read-only database discovery and query operations.
* Adds knowledge lookup schemas when a catalog is configured.
* Requires explicit SQL and answer submission to complete an agent run.
* Restricts schema properties and required arguments explicitly.
* Bounds knowledge and SQL inputs and leaves runtime validation in the agent pipeline.
