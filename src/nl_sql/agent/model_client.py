"""Configurable OpenAI-compatible chat-completions client."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ..config import Settings, SettingsError

MAX_RESPONSE_BYTES = 1_000_000
# Free-tier endpoints cap tokens per minute; waiting out a 429 beats failing the task.
MAX_RATE_LIMIT_RETRIES = 8
DEFAULT_RATE_LIMIT_WAIT_SECONDS = 20.0
MAX_RATE_LIMIT_WAIT_SECONDS = 65.0
# Some hosts reject a malformed model tool call with HTTP 400; a fresh sample usually
# succeeds, so a few retries avoid failing the whole task on one bad generation.
MAX_MALFORMED_TOOL_CALL_RETRIES = 3

Transport = Callable[[Request, float], bytes]


class ModelError(Exception):
    """Raised when model configuration, transport, or response parsing fails."""


@dataclass(frozen=True)
class ToolCall:
    """A validated function call requested by the model."""

    call_id: str
    name: str
    arguments: dict[str, object]


@dataclass(frozen=True)
class ModelResponse:
    """A supported model message and its requested tool calls."""

    content: str | None
    tool_calls: list[ToolCall]
    input_tokens: int | None = None
    output_tokens: int | None = None


class ModelClient(Protocol):
    """Structural interface consumed by the agent and its scripted tests."""

    def complete(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]],
    ) -> ModelResponse:
        """Request one model turn."""


class ChatCompletionClient:
    """Send one chat-completions request through an injectable transport."""

    def __init__(
        self,
        settings: Settings,
        transport: Transport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Create a model client.

        Args:
            settings: Validated application settings.
            transport: Optional HTTP transport, injectable for offline tests.
            sleep: Wait function used between rate-limited retries.
        """
        try:
            settings.require_model()
        except SettingsError as error:
            raise ModelError(str(error)) from error
        self.settings = settings
        self.transport = self._default_transport if transport is None else transport
        self.sleep = sleep
        self.timeout = settings.model_timeout_seconds

    def complete(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]],
    ) -> ModelResponse:
        """Send a request and validate the supported response shape.

        Args:
            messages: Chat messages for this model turn.
            tools: Function tools available to the model.

        Returns:
            A parsed message and validated function calls.

        Raises:
            ModelError: If transport or response processing fails.
        """
        payload = {
            "model": self.settings.model_name,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
        }
        try:
            body = json.dumps(payload).encode("utf-8")
        except (TypeError, ValueError):
            raise ModelError("Model request contains unsupported data.") from None

        request = Request(
            self.settings.completion_url,
            data=body,
            headers={
                "Authorization": f"Bearer {self.settings.model_key}",
                "Content-Type": "application/json",
                # Some hosted endpoints reject urllib's default agent string.
                "User-Agent": "nl-sql/0.1",
            },
            method="POST",
        )
        response_body = self._send_with_rate_limit_retries(request)

        if len(response_body) > MAX_RESPONSE_BYTES:
            raise ModelError("Model endpoint response exceeded the size limit.")

        try:
            decoded = json.loads(response_body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise ModelError("Model endpoint returned invalid JSON.") from None
        return self._parse_response(decoded)

    def _send_with_rate_limit_retries(self, request: Request) -> bytes:
        malformed_retries = 0
        for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
            try:
                return self.transport(request, self.timeout)
            except HTTPError as error:
                if (
                    _is_malformed_tool_call(error)
                    and malformed_retries < MAX_MALFORMED_TOOL_CALL_RETRIES
                ):
                    malformed_retries += 1
                    continue
                if error.code != 429 or attempt == MAX_RATE_LIMIT_RETRIES:
                    raise ModelError(
                        f"Model endpoint returned HTTP {error.code}."
                    ) from None
                self.sleep(_retry_after_seconds(error))
            except (URLError, TimeoutError, OSError):
                raise ModelError(
                    "Model request failed; check endpoint availability and configuration."
                ) from None
        raise ModelError("Model endpoint kept rate-limiting the request.")

    @staticmethod
    def _default_transport(request: Request, timeout: float) -> bytes:
        with urlopen(request, timeout=timeout) as response:
            return response.read(MAX_RESPONSE_BYTES + 1)

    @staticmethod
    def _parse_response(payload: object) -> ModelResponse:
        if not isinstance(payload, dict):
            raise ModelError("Model endpoint returned an unsupported response shape.")
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise ModelError("Model endpoint returned an unsupported response shape.")
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise ModelError("Model endpoint returned an unsupported response shape.")

        content = message.get("content")
        if content is not None and not isinstance(content, str):
            raise ModelError("Model endpoint returned unsupported message content.")

        raw_calls = message.get("tool_calls", [])
        if not isinstance(raw_calls, list):
            raise ModelError("Model endpoint returned invalid tool calls.")
        tool_calls = [ChatCompletionClient._parse_tool_call(call) for call in raw_calls]
        input_tokens, output_tokens = ChatCompletionClient._parse_usage(
            payload.get("usage")
        )
        return ModelResponse(
            content=content,
            tool_calls=tool_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    @staticmethod
    def _parse_usage(usage: object) -> tuple[int | None, int | None]:
        if usage is None:
            return None, None
        if not isinstance(usage, dict):
            raise ModelError("Model endpoint returned invalid token usage.")
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        if prompt_tokens is None and completion_tokens is None:
            return None, None
        if (
            (
                prompt_tokens is not None
                and (
                    not isinstance(prompt_tokens, int)
                    or isinstance(prompt_tokens, bool)
                    or prompt_tokens < 0
                )
            )
            or (
                completion_tokens is not None
                and (
                    not isinstance(completion_tokens, int)
                    or isinstance(completion_tokens, bool)
                    or completion_tokens < 0
                )
            )
        ):
            raise ModelError("Model endpoint returned invalid token usage.")
        return prompt_tokens, completion_tokens

    @staticmethod
    def _parse_tool_call(raw_call: object) -> ToolCall:
        if not isinstance(raw_call, dict):
            raise ModelError("Model endpoint returned an invalid tool call.")
        call_id = raw_call.get("id")
        function = raw_call.get("function")
        if raw_call.get("type") != "function" or not isinstance(call_id, str) or not call_id:
            raise ModelError("Model endpoint returned an unsupported tool-call shape.")
        if not isinstance(function, dict):
            raise ModelError("Model endpoint returned an unsupported tool-call shape.")
        name = function.get("name")
        raw_arguments = function.get("arguments")
        if not isinstance(name, str) or not name or not isinstance(raw_arguments, str):
            raise ModelError("Model endpoint returned an unsupported tool-call shape.")
        try:
            arguments = json.loads(raw_arguments)
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise ModelError("Model endpoint returned invalid tool arguments.") from None
        if not isinstance(arguments, dict):
            raise ModelError("Model endpoint tool arguments must be a JSON object.")
        return ToolCall(call_id=call_id, name=name, arguments=arguments)


def _retry_after_seconds(error: HTTPError) -> float:
    """Honor the endpoint's Retry-After header, bounded to keep runs predictable."""
    header = error.headers.get("Retry-After") if error.headers else None
    try:
        wait = float(header) if header is not None else DEFAULT_RATE_LIMIT_WAIT_SECONDS
    except ValueError:
        wait = DEFAULT_RATE_LIMIT_WAIT_SECONDS
    return min(max(wait, 1.0), MAX_RATE_LIMIT_WAIT_SECONDS)


def _is_malformed_tool_call(error: HTTPError) -> bool:
    """Detect the host-side rejection of a malformed tool call from the model."""
    if error.code != 400:
        return False
    try:
        body = error.read(10_000)
    except OSError:
        return False
    return b"tool_use_failed" in body
