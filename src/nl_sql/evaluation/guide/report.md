# report.py

Builds reproducible redacted manifests and score summaries from task-run records.

* Records dataset hashes, selection rules and IDs, model name, UTC start time, and resource limits.
* Excludes questions, SQL, answers, rows, credentials, raw errors, and absolute paths.
* Reports every task status and reason, with score numerator and denominator for evaluated tasks.
* Omits aggregate score when task execution aborts.
* Writes a new JSON file only beneath the verified ignored reports directory.
