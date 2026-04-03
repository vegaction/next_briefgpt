from __future__ import annotations

import json
from unittest.mock import Mock, patch

import pytest
import requests

from briefgpt_arxiv.config import LLMEndpointSettings
from briefgpt_arxiv.llm_client import (
    GeminiAPIClient,
    OpenAICompatibleClient,
    TransientLLMResponseError,
    create_llm_client,
)


@patch("briefgpt_arxiv.llm_client.time.sleep", return_value=None)
@patch("briefgpt_arxiv.llm_client.requests.post")
def test_openai_compatible_client_retries_transient_errors(mock_post, _mock_sleep) -> None:
    success_response = Mock()
    success_response.raise_for_status.return_value = None
    success_response.json.return_value = {"choices": [{"message": {"content": '{"ok": true}'}}]}
    mock_post.side_effect = [
        requests.ConnectionError("temporary ssl eof"),
        success_response,
    ]

    client = OpenAICompatibleClient(
        endpoint=LLMEndpointSettings(provider="openai_compatible", model_name="test-model"),
        api_key="test-key",
    )
    payload = client.generate_json(system_instruction="Return JSON.", user_text="{}")

    assert payload == {"ok": True}
    assert mock_post.call_count == 2


@patch("briefgpt_arxiv.llm_client.time.sleep", return_value=None)
@patch("briefgpt_arxiv.llm_client.requests.post")
def test_openai_compatible_client_retries_chunked_encoding_errors(mock_post, _mock_sleep) -> None:
    success_response = Mock()
    success_response.raise_for_status.return_value = None
    success_response.json.return_value = {"choices": [{"message": {"content": '{"ok": true}'}}]}
    mock_post.side_effect = [
        requests.exceptions.ChunkedEncodingError("Response ended prematurely"),
        success_response,
    ]

    client = OpenAICompatibleClient(
        endpoint=LLMEndpointSettings(provider="openai_compatible", model_name="test-model"),
        api_key="test-key",
    )
    payload = client.generate_json(system_instruction="Return JSON.", user_text="{}")

    assert payload == {"ok": True}
    assert mock_post.call_count == 2


@patch("briefgpt_arxiv.llm_client.requests.post")
def test_openai_compatible_client_sets_explicit_timeout(mock_post) -> None:
    success_response = Mock()
    success_response.raise_for_status.return_value = None
    success_response.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
    mock_post.return_value = success_response

    client = OpenAICompatibleClient(
        endpoint=LLMEndpointSettings(provider="openai_compatible", model_name="test-model"),
        api_key="test-key",
        timeout_seconds=17.5,
    )
    client.generate_text(system_instruction="hi", user_text="there")

    _args, kwargs = mock_post.call_args
    assert ("https://openrouter.ai/api/v1/chat/completions" == kwargs["url"] if "url" in kwargs else _args[0])
    assert kwargs["timeout"] == 17.5
    assert isinstance(kwargs["data"], str)


@patch("briefgpt_arxiv.llm_client.time.sleep", return_value=None)
@patch("briefgpt_arxiv.llm_client.requests.post")
def test_openai_compatible_client_does_not_retry_non_retryable_status_errors(mock_post, _mock_sleep) -> None:
    response = Mock(status_code=400, text="bad request")
    mock_post.side_effect = requests.HTTPError("bad request", response=response)

    client = OpenAICompatibleClient(
        endpoint=LLMEndpointSettings(provider="openai_compatible", model_name="test-model"),
        api_key="test-key",
    )

    with pytest.raises(requests.HTTPError):
        client.generate_text(system_instruction="hi", user_text="there")

    assert mock_post.call_count == 1


@patch("briefgpt_arxiv.llm_client.requests.post")
def test_openai_compatible_client_sends_structured_message_parts_without_response_format(mock_post) -> None:
    success_response = Mock()
    success_response.raise_for_status.return_value = None
    success_response.json.return_value = {"choices": [{"message": {"content": '{"ok": true}'}}]}
    mock_post.return_value = success_response

    client = OpenAICompatibleClient(
        endpoint=LLMEndpointSettings(provider="openai_compatible", model_name="test-model"),
        api_key="test-key",
    )
    client.generate_json(system_instruction="Return JSON.", user_text="Base prompt")

    payload = json.loads(mock_post.call_args.kwargs["data"])
    assert "response_format" not in payload
    assert payload["messages"][0]["content"] == [{"type": "text", "text": "Return JSON."}]
    assert payload["messages"][1]["content"] == [{"type": "text", "text": "Base prompt"}]
    assert payload["reasoning"] == {"enabled": True}


@patch("briefgpt_arxiv.llm_client.time.sleep", return_value=None)
@patch("briefgpt_arxiv.llm_client.requests.post")
def test_openai_compatible_client_retries_provider_body_errors(mock_post, _mock_sleep) -> None:
    error_response = Mock()
    error_response.raise_for_status.return_value = None
    error_response.json.return_value = {
        "choices": [
            {
                "error": {
                    "code": 502,
                    "message": "Network connection lost.",
                    "metadata": {"error_type": "provider_unavailable"},
                },
                "message": {"content": None},
            }
        ]
    }
    success_response = Mock()
    success_response.raise_for_status.return_value = None
    success_response.json.return_value = {"choices": [{"message": {"content": '{"ok": true}'}}]}
    mock_post.side_effect = [error_response, success_response]

    client = OpenAICompatibleClient(
        endpoint=LLMEndpointSettings(provider="openai_compatible", model_name="test-model"),
        api_key="test-key",
    )

    payload = client.generate_json(system_instruction="Return JSON.", user_text="{}")

    assert payload == {"ok": True}
    assert mock_post.call_count == 2


@patch("briefgpt_arxiv.llm_client.requests.post")
def test_openai_compatible_client_raises_non_retryable_provider_body_errors(mock_post) -> None:
    error_response = Mock()
    error_response.raise_for_status.return_value = None
    error_response.json.return_value = {
        "choices": [
            {
                "error": {
                    "code": 400,
                    "message": "Unsupported parameter.",
                    "metadata": {"error_type": "invalid_request_error"},
                },
                "message": {"content": None},
            }
        ]
    }
    mock_post.return_value = error_response

    client = OpenAICompatibleClient(
        endpoint=LLMEndpointSettings(provider="openai_compatible", model_name="test-model"),
        api_key="test-key",
    )

    with pytest.raises(RuntimeError, match="returned an error in the response body"):
        client.generate_text(system_instruction="hi", user_text="there")


def test_openai_compatible_extract_text_raises_transient_error_for_retryable_body_error() -> None:
    response = {
        "choices": [
            {
                "error": {
                    "code": 502,
                    "message": "Network connection lost.",
                    "metadata": {"error_type": "provider_unavailable"},
                },
                "message": {"content": None},
            }
        ]
    }

    with pytest.raises(TransientLLMResponseError):
        OpenAICompatibleClient._extract_text(response)


@patch("briefgpt_arxiv.llm_client.time.sleep", return_value=None)
@patch("briefgpt_arxiv.llm_client.requests.post")
def test_gemini_api_client_retries_transient_http_errors(mock_post, _mock_sleep) -> None:
    success_response = Mock()
    success_response.raise_for_status.return_value = None
    success_response.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": '{"ok": true}'}]}}]
    }
    mock_post.side_effect = [
        requests.ConnectionError("temporary ssl eof"),
        success_response,
    ]

    client = GeminiAPIClient(
        endpoint=LLMEndpointSettings(provider="gemini", model_name="gemini-test-model"),
        api_key="test-key",
    )
    payload = client.generate_json(system_instruction="Return JSON.", user_text="{}")

    assert payload == {"ok": True}
    assert mock_post.call_count == 2


@patch("briefgpt_arxiv.llm_client.requests.post")
def test_gemini_api_client_uses_json_string_body_without_response_schema(mock_post) -> None:
    success_response = Mock()
    success_response.raise_for_status.return_value = None
    success_response.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": '{"ok": true}'}]}}]
    }
    mock_post.return_value = success_response

    client = GeminiAPIClient(
        endpoint=LLMEndpointSettings(provider="gemini", model_name="gemini-test-model"),
        api_key="test-key",
    )
    client.generate_json(system_instruction="Return JSON.", user_text="Prompt body")

    _args, kwargs = mock_post.call_args
    payload = json.loads(kwargs["data"])
    assert payload["contents"][0]["parts"][0]["text"] == "Prompt body"
    assert "generationConfig" not in payload


def test_factory_selects_openai_compatible_provider() -> None:
    with patch("briefgpt_arxiv.llm_client.OpenAICompatibleClient") as mock_client:
        endpoint = LLMEndpointSettings(provider="openai_compatible", model_name="test-model")
        create_llm_client(endpoint)

    mock_client.assert_called_once_with(endpoint)


@patch("briefgpt_arxiv.llm_client.requests.post")
def test_openai_compatible_client_honors_endpoint_reasoning_override(mock_post) -> None:
    success_response = Mock()
    success_response.raise_for_status.return_value = None
    success_response.json.return_value = {"choices": [{"message": {"content": '{"ok": true}'}}]}
    mock_post.return_value = success_response

    client = OpenAICompatibleClient(
        endpoint=LLMEndpointSettings(
            provider="openai_compatible",
            model_name="test-model",
            reasoning_enabled=False,
        ),
        api_key="test-key",
    )
    client.generate_json(system_instruction="Return JSON.", user_text="Base prompt")

    payload = json.loads(mock_post.call_args.kwargs["data"])
    assert "reasoning" not in payload


def test_factory_selects_gemini_provider() -> None:
    with patch("briefgpt_arxiv.llm_client.GeminiAPIClient") as mock_client:
        endpoint = LLMEndpointSettings(provider="gemini", model_name="gemini-test-model")
        create_llm_client(endpoint)

    mock_client.assert_called_once_with(endpoint)
