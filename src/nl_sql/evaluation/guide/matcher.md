# matcher.py

Matches gold and predicted SQL results by their normalized values.

* Ignores output column names and permits additional predicted columns.
* Requires every gold column to map to a distinct value-equivalent predicted column.
* Matches rows without depending on their order.
* Uses one-to-one bipartite matching so duplicate rows retain their multiplicity.
* Refuses truncated results and reports mismatches with stable reason codes.
