# provider.py

Defines the shared read-only database result, error, and provider contracts.

* Standardizes schema discovery and query operations.
* Carries result columns, rows, and truncation state.
* Categorizes provider failures for safe routing.
* Serializes supported date, decimal, UUID, and interval values.
* Keeps provider-specific execution policy in database tool modules.
