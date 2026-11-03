# model_client.py

Sends validated OpenAI-compatible chat-completion requests through an injectable transport.

* Reads endpoint, model, key, and timeout from `Settings`.
* Sends the configured key as a bearer credential only to the configured endpoint.
* Waits out short rate limits but fails fast with `ModelRateLimitError` when the wait is too long (e.g. a spent daily quota), redacting provider bodies.
* Validates responses and tool calls, repairing calls the host rejects only for a leaked template token in the tool name.
* Captures available prompt and completion token usage without inventing missing counts.
