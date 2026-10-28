# knowledge.py

Defines bounded, read-only database meaning and knowledge lookup contracts.

* Validates column meanings and definition records at catalog construction.
* Looks up one column meaning without exposing unrelated entries.
* Ranks knowledge definitions by deterministic query relevance.
* Caps search results and query length.
* Keeps task-required hidden identifiers out of the agent-facing API.
