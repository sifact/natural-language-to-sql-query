# task_runner.py

Executes selected tasks through isolated task databases and scores submitted SQL.

* Uses a selector-produced selection and requires explicit opt-ins plus verified assets.
* Keeps model-visible question data separate from evaluator-only answers and setup SQL.
* Validates the local template before opening one run-owned clone per eligible task.
* Executes canonical and submitted SQL through the clone's bounded read-only provider.
* Emits fixed task statuses and aborts without aggregate metrics after sandbox cleanup failure.
