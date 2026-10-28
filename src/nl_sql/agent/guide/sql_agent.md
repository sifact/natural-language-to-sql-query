# sql_agent.py

Coordinates bounded model turns with read-only database and knowledge tools.

* Keeps evaluator-only task data out of model context and validates tool calls before dispatch.
* Completes only after a validated `submit_answer(sql, answer)` tool call.
* Returns the submitted SQL rather than substituting the latest executed query.
* Requests clarification for material ambiguity and surfaces incomplete runs at the round limit.
* Writes every conversation message and tool call when a task tracer is configured.
