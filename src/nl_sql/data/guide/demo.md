# demo.py

Creates the deterministic synthetic SQLite database used by the offline demo.

* Stores schema and example rows as source-controlled fixtures.
* Writes the database atomically to the configured demo path.
* Preserves existing data unless replacement is explicitly requested.
* Uses synthetic customer, product, order, and order-item records.
* Makes no model or network requests.
