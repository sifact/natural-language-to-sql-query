# demo_eval.py

Scores the agent on the fixed demo questions by comparing executed query results.

* Asks each question with a fresh agent so earlier answers cannot leak into later ones.
* Runs the agent's SQL and the gold SQL read-only and compares rows with `match_results`.
* Records model or agent failures as `error`, kept out of accuracy and counted separately.
* Summarizes accuracy, median latency and median tokens per question.
* Writes a publishable JSON report to `reports/demo-eval-<model>.json`, which the About page reads.
