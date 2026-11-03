"""Configurable OpenAI-compatible chat-completions client."""

from __future__ import annotations

import json
import math
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ..config import Settings, SettingsError

MAX_RESPONSE_BYTES = 1_000_000
# Free-tier endpoints cap tokens per minute; waiting out a 429 beats failing the task.
# A longer requested wait (e.g. a spent daily quota) fails fast instead of hanging.
MAX_RATE_LIMIT_RETRIES = 8
DEFAULT_RATE_LIMIT_WAIT_SECONDS = 20.0
MAX_RATE_LIMIT_WAIT_SECONDS = 65.0
# Provider wording for per-day caps, e.g. Groq's "tokens per day (TPD)".
DAILY_QUOTA_MARKERS = (b"per day", b"(tpd)", b"(rpd)")
# Some hosts reject a malformed model tool call with HTTP 400; a fresh sample usually
# succeeds, so a few retries avoid failing the whole task on one bad generation.
MAX_MALFORMED_TOOL_CALL_RETRIES = 3

Transport = Callable[[Request, float], bytes]


class ModelError(Exception):
    """Raised when model configuration, transport, or response parsing fails."""


class ModelRateLimitError(ModelError):
    """Raised when the endpoint keeps rate-limiting or asks for a wait too long to honor."""

    def __init__(self, message: str, retry_after_seconds: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


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
        tool_names = {
            tool["function"]["name"]
            for tool in tools
            if isinstance(tool.get("function"), dict)
        }
        response_body = self._send_with_rate_limit_retries(request, tool_names)

        if len(response_body) > MAX_RESPONSE_BYTES:
            raise ModelError("Model endpoint response exceeded the size limit.")

        try:
            decoded = json.loads(response_body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise ModelError("Model endpoint returned invalid JSON.") from None
        return self._parse_response(decoded)

    def _send_with_rate_limit_retries(
        self, request: Request, tool_names: set[str]
    ) -> bytes:
        malformed_retries = 0
        for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
            try:
                return self.transport(request, self.timeout)
            except HTTPError as error:
                # Close each rejected response so retries don't leak sockets.
                with error:
                    failed_generation = _malformed_tool_call_generation(error)
                    if failed_generation is not None:
                        repaired = _repair_tool_call(failed_generation, tool_names)
                        if repaired is not None:
                            return repaired
                        if malformed_retries == MAX_MALFORMED_TOOL_CALL_RETRIES:
                            raise ModelError(
                                "The model kept producing an invalid tool call; "
                                "try again or use a different model."
                            ) from None
                        malformed_retries += 1
                        continue
                    if error.code != 429:
                        raise ModelError(
                            f"Model endpoint returned HTTP {error.code}."
                        ) from None
                    wait = _retry_after_seconds(error)
                    if wait > MAX_RATE_LIMIT_WAIT_SECONDS:
                        raise _long_rate_limit_error(error, wait) from None
                    if attempt == MAX_RATE_LIMIT_RETRIES:
                        break
                self.sleep(wait)
            except (URLError, TimeoutError, OSError):
                raise ModelError(
                    "Model request failed; check endpoint availability and configuration."
                ) from None
        raise ModelRateLimitError(
            "Model endpoint kept rate-limiting the request; try again shortly."
        )

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
    """Read the endpoint's requested wait; callers decide whether it is worth honoring."""
    header = error.headers.get("Retry-After") if error.headers else None
    try:
        wait = float(header) if header is not None else DEFAULT_RATE_LIMIT_WAIT_SECONDS
    except ValueError:
        wait = DEFAULT_RATE_LIMIT_WAIT_SECONDS
    return max(wait, 1.0)


def _long_rate_limit_error(error: HTTPError, wait: float) -> ModelRateLimitError:
    """Explain a wait too long to sit through, without echoing the provider's body."""
    try:
        body = error.read(10_000).lower()
    except OSError:
        body = b""
    minutes = math.ceil(wait / 60)
    unit = "minute" if minutes == 1 else "minutes"
    limit = (
        "daily token quota is used up"
        if any(marker in body for marker in DAILY_QUOTA_MARKERS)
        else "rate limit was reached"
    )
    return ModelRateLimitError(
        f"The model endpoint's {limit}; try again in about {minutes} {unit} "
        "or use a different model.",
        retry_after_seconds=wait,
    )


def _malformed_tool_call_generation(error: HTTPError) -> str | None:
    """Return the rejected generation when the host refused a malformed tool call.

    Returns an empty string when the rejection carries no generation to repair.
    """
    if error.code != 400:
        return None
    try:
        body = error.read(MAX_RESPONSE_BYTES)
    except OSError:
        return None
    if b"tool_use_failed" not in body:
        return None
    try:
        generation = json.loads(body)["error"]["failed_generation"]
    except (json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError):
        return ""
    return generation if isinstance(generation, str) else ""


def _repair_tool_call(generation: str, tool_names: set[str]) -> bytes | None:
    """Rebuild a rejected call whose only defect is a leaked chat-template token.

    gpt-oss models on some hosts append tokens like ``<|channel|>commentary`` to the
    tool name; the arguments are intact, so a repair is cheaper than a resample.
    Returns a chat-completions body that goes through normal validation, or None.
    """
    try:
        call = json.loads(generation)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(call, dict) or not isinstance(call.get("name"), str):
        return None
    name = call["name"].split("<|", 1)[0].strip()
    arguments = call.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    if name not in tool_names or not isinstance(arguments, dict):
        return None
    tool_call = {
        "id": f"repaired_{uuid.uuid4().hex[:12]}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }
    message = {"role": "assistant", "content": None, "tool_calls": [tool_call]}
    return json.dumps({"choices": [{"message": message}]}).encode("utf-8")
