# tracing.py

Writes structured, ordered conversation and tool events to one local JSONL file per task.

* Creates a new task trace under the ignored local benchmark trace directory.
* Appends messages and tool arguments/results in observed event order.
* Redacts credential fields and common credential-bearing strings before serialization.
* Surfaces file and serialization failures instead of returning trace-less success.
* Does not expose traces as report content or copy them into aggregate artifacts.
