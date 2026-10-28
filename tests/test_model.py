"""Tests for provider configuration and the offline model client boundary."""

import json
from io import BytesIO
from urllib.error import HTTPError, URLError

import pytest

from nl_sql.agent.model_client import ChatCompletionClient, ModelError
from nl_sql.config import Settings, SettingsError

VALID_ENV = {
    "NL_SQL_BASE_URL": "https://model.example/v1",
    "NL_SQL_MODEL": "demo-model",
    "NL_SQL_API_KEY": "test-secret",
}


def response_bytes(
    message: dict[str, object],
    usage: dict[str, object] | None = None,
) -> bytes:
    payload: dict[str, object] = {"choices": [{"message": message}]}
    if usage is not None:
        payload["usage"] = usage
    return json.dumps(payload).encode("utf-8")


def test_given_missing_model_settings_when_load_config_then_names_missing_variables() -> None:
    # Act
    with pytest.raises(ModelError, match="NL_SQL_BASE_URL.*NL_SQL_MODEL.*NL_SQL_API_KEY"):
        ChatCompletionClient(Settings.from_env({}))


def test_given_remote_http_endpoint_when_load_config_then_rejects_before_request() -> None:
    # Arrange
    environment = {**VALID_ENV, "NL_SQL_BASE_URL": "http://model.example/v1"}

    # Act
    with pytest.raises(SettingsError, match="Remote model endpoints must use HTTPS"):
        Settings.from_env(environment)


@pytest.mark.parametrize(
    "base_url",
    ["http://localhost:8080/v1", "http://127.0.0.1:8080/v1", "http://[::1]:8080/v1"],
)
def test_given_loopback_http_endpoint_when_load_config_then_accepts_local_development(
    base_url: str,
) -> None:
    # Arrange
    environment = {**VALID_ENV, "NL_SQL_BASE_URL": base_url}

    # Act
    config = Settings.from_env(environment)

    # Assert
    assert config.completion_url == f"{base_url}/chat/completions"


def test_given_tool_call_response_when_complete_then_parses_supported_contract(mocker) -> None:
    # Arrange
    config = Settings.from_env(VALID_ENV)
    response_body = response_bytes(
        {
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "list_tables", "arguments": "{}"},
                }
            ],
        }
    )
    transport = mocker.Mock(return_value=response_body)
    client = ChatCompletionClient(config, transport=transport)

    # Act
    result = client.complete([{"role": "user", "content": "List tables"}], [])

    # Assert
    assert result.tool_calls[0].arguments == {}
    request = transport.call_args.args[0]
    assert request.get_header("Authorization") == "Bearer test-secret"
    assert request.full_url == config.completion_url


def test_given_token_usage_when_complete_then_returns_available_counts() -> None:
    # Arrange
    body = response_bytes(
        {"content": "Done.", "tool_calls": []},
        {"prompt_tokens": 19, "completion_tokens": 7},
    )
    client = ChatCompletionClient(
        Settings.from_env(VALID_ENV),
        transport=lambda _request, _timeout: body,
    )

    # Act
    result = client.complete([], [])

    # Assert
    assert result.input_tokens == 19
    assert result.output_tokens == 7


def test_given_partial_token_usage_when_complete_then_preserves_missing_count() -> None:
    # Arrange
    body = response_bytes(
        {"content": "Done.", "tool_calls": []},
        {"prompt_tokens": 19},
    )
    client = ChatCompletionClient(
        Settings.from_env(VALID_ENV),
        transport=lambda _request, _timeout: body,
    )

    # Act
    result = client.complete([], [])

    # Assert
    assert result.input_tokens == 19
    assert result.output_tokens is None


def test_given_invalid_token_usage_when_complete_then_rejects_usage() -> None:
    body = response_bytes(
        {"content": "Done.", "tool_calls": []},
        {"prompt_tokens": -1, "completion_tokens": 7},
    )
    client = ChatCompletionClient(
        Settings.from_env(VALID_ENV),
        transport=lambda _request, _timeout: body,
    )

    with pytest.raises(ModelError, match="invalid token usage"):
        client.complete([], [])


def test_given_malformed_response_when_complete_then_reports_provider_shape_error() -> None:
    # Arrange
    config = Settings.from_env(VALID_ENV)
    client = ChatCompletionClient(config, transport=lambda _request, _timeout: b"{}")

    # Act
    with pytest.raises(ModelError, match="unsupported response shape"):
        client.complete([], [])


def test_given_invalid_tool_arguments_when_complete_then_rejects_arguments() -> None:
    # Arrange
    config = Settings.from_env(VALID_ENV)
    body = response_bytes(
        {
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "query", "arguments": "not-json"},
                }
            ]
        }
    )
    client = ChatCompletionClient(config, transport=lambda _request, _timeout: body)

    # Act
    with pytest.raises(ModelError, match="invalid tool arguments"):
        client.complete([], [])


def test_given_http_error_body_with_sensitive_content_when_complete_then_redacts_body(
    mocker,
) -> None:
    # Arrange
    config = Settings.from_env(VALID_ENV)
    error = HTTPError(
        config.completion_url,
        403,
        "forbidden",
        hdrs=None,
        fp=BytesIO(b"test-secret request content"),
    )
    mocker.patch("nl_sql.agent.model_client.urlopen", side_effect=error)
    client = ChatCompletionClient(config)

    # Act
    with pytest.raises(ModelError) as captured:
        client.complete([], [])

    # Assert
    assert str(captured.value) == "Model endpoint returned HTTP 403."
    assert "test-secret" not in str(captured.value)
    assert "request content" not in str(captured.value)


def test_given_transport_error_with_sensitive_reason_when_complete_then_redacts_reason(
    mocker,
) -> None:
    # Arrange
    config = Settings.from_env(VALID_ENV)
    mocker.patch("nl_sql.agent.model_client.urlopen", side_effect=URLError("test-secret"))
    client = ChatCompletionClient(config)

    # Act
    with pytest.raises(ModelError) as captured:
        client.complete([], [])

    # Assert
    assert "test-secret" not in str(captured.value)
