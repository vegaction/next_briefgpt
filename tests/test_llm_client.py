from __future__ import annotations

import json
from unittest import TestCase
from unittest.mock import Mock, patch

import requests

from briefgpt_arxiv.config import LLMEndpointSettings
from briefgpt_arxiv.llm_client import (
    GeminiAPIClient,
    OpenAICompatibleClient,
    TransientLLMResponseError,
    create_llm_client,
)


class LLMClientTests(TestCase):
    @patch("briefgpt_arxiv.llm_client.time.sleep", return_value=None)
    @patch("briefgpt_arxiv.llm_client.requests.post")
    def test_openai_compatible_client_retries_transient_errors(self, mock_post: Mock, _mock_sleep: Mock) -> None:
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

        self.assertEqual({"ok": True}, payload)
        self.assertEqual(2, mock_post.call_count)

    @patch("briefgpt_arxiv.llm_client.requests.post")
    def test_openai_compatible_client_sets_explicit_timeout(self, mock_post: Mock) -> None:
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
        self.assertEqual("https://openrouter.ai/api/v1/chat/completions", kwargs["url"] if "url" in kwargs else _args[0])
        self.assertEqual(17.5, kwargs["timeout"])
        self.assertIsInstance(kwargs["data"], str)

    @patch("briefgpt_arxiv.llm_client.time.sleep", return_value=None)
    @patch("briefgpt_arxiv.llm_client.requests.post")
    def test_openai_compatible_client_does_not_retry_non_retryable_status_errors(
        self,
        mock_post: Mock,
        _mock_sleep: Mock,
    ) -> None:
        response = Mock(status_code=400, text="bad request")
        mock_post.side_effect = requests.HTTPError("bad request", response=response)

        client = OpenAICompatibleClient(
            endpoint=LLMEndpointSettings(provider="openai_compatible", model_name="test-model"),
            api_key="test-key",
        )

        with self.assertRaises(requests.HTTPError):
            client.generate_text(system_instruction="hi", user_text="there")

        self.assertEqual(1, mock_post.call_count)

    @patch("briefgpt_arxiv.llm_client.requests.post")
    def test_openai_compatible_client_sends_structured_message_parts_without_response_format(self, mock_post: Mock) -> None:
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
        self.assertNotIn("response_format", payload)
        self.assertEqual(
            [{"type": "text", "text": "Return JSON."}],
            payload["messages"][0]["content"],
        )
        self.assertEqual(
            [{"type": "text", "text": "Base prompt"}],
            payload["messages"][1]["content"],
        )
        self.assertEqual({"enabled": True}, payload["reasoning"])

    @patch("briefgpt_arxiv.llm_client.time.sleep", return_value=None)
    @patch("briefgpt_arxiv.llm_client.requests.post")
    def test_openai_compatible_client_retries_provider_body_errors(self, mock_post: Mock, _mock_sleep: Mock) -> None:
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

        self.assertEqual({"ok": True}, payload)
        self.assertEqual(2, mock_post.call_count)

    @patch("briefgpt_arxiv.llm_client.requests.post")
    def test_openai_compatible_client_raises_non_retryable_provider_body_errors(self, mock_post: Mock) -> None:
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

        with self.assertRaisesRegex(RuntimeError, "returned an error in the response body"):
            client.generate_text(system_instruction="hi", user_text="there")

    def test_openai_compatible_extract_text_raises_transient_error_for_retryable_body_error(self) -> None:
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

        with self.assertRaises(TransientLLMResponseError):
            OpenAICompatibleClient._extract_text(response)

    @patch("briefgpt_arxiv.llm_client.time.sleep", return_value=None)
    @patch("briefgpt_arxiv.llm_client.requests.post")
    def test_gemini_api_client_retries_transient_http_errors(self, mock_post: Mock, _mock_sleep: Mock) -> None:
        success_response = Mock()
        success_response.raise_for_status.return_value = None
        success_response.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": '{"ok": true}',
                            }
                        ]
                    }
                }
            ]
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

        self.assertEqual({"ok": True}, payload)
        self.assertEqual(2, mock_post.call_count)

    @patch("briefgpt_arxiv.llm_client.requests.post")
    def test_gemini_api_client_uses_json_string_body_without_response_schema(self, mock_post: Mock) -> None:
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
        self.assertEqual("Prompt body", payload["contents"][0]["parts"][0]["text"])
        self.assertNotIn("generationConfig", payload)

    def test_factory_selects_openai_compatible_provider(self) -> None:
        with patch("briefgpt_arxiv.llm_client.OpenAICompatibleClient") as mock_client:
            endpoint = LLMEndpointSettings(provider="openai_compatible", model_name="test-model")
            create_llm_client(endpoint)

        mock_client.assert_called_once_with(endpoint)

    @patch("briefgpt_arxiv.llm_client.requests.post")
    def test_openai_compatible_client_honors_endpoint_reasoning_override(self, mock_post: Mock) -> None:
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
        self.assertNotIn("reasoning", payload)

    def test_factory_selects_gemini_provider(self) -> None:
        with patch("briefgpt_arxiv.llm_client.GeminiAPIClient") as mock_client:
            endpoint = LLMEndpointSettings(provider="gemini", model_name="gemini-test-model")
            create_llm_client(endpoint)

        mock_client.assert_called_once_with(endpoint)
