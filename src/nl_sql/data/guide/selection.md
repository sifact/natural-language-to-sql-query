# selection.py

Selects the deterministic, article-shaped PostgreSQL task slice.

* Parses canonical SQL with SQLGlot's PostgreSQL dialect.
* Requires every listed SQL alternative to meet the local filter.
* Counts distinct outer base tables and top-level `WHERE` predicates.
* Returns lexically sorted task IDs and versioned selection criteria.
* Reports unclassifiable SQL with fixed reason codes.
