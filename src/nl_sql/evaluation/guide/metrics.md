# metrics.py

Calculates benchmark accuracy from comparable task outcomes.

* Counts passed and failed tasks as evaluated results.
* Excludes incomplete, unsupported, and skipped tasks from the accuracy denominator.
* Returns an exact rational accuracy rather than a rounded floating-point score.
* Reports every task status count alongside the evaluated-task denominator.
* Returns no accuracy when no task has a comparable result.
