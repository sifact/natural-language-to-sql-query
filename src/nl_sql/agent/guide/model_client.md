# model_client.py

Sends validated OpenAI-compatible chat-completion requests through an injectable transport.

* Reads endpoint, model, key, and timeout from `Settings`.
* Sends the configured key as a bearer credential only to the configured endpoint.
* Redacts transport failures and rejects oversized or malformed responses.
* Validates tool-call structure and JSON arguments.
* Captures available prompt and completion token usage without inventing missing counts.
