# normalize.py

Normalizes SQL result values for deterministic comparison.

* Preserves distinctions among NULL, booleans, numbers, strings, dates, and JSON.
* Trims and casefolds strings without converting arbitrary values to text.
* Compares numeric values as Decimal with configured absolute and relative tolerances.
* Normalizes date, time, and datetime values to ISO 8601 representations.
* Reports values with ambiguous or unsupported semantics explicitly.
