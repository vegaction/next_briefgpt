from __future__ import annotations

from unittest import TestCase
from unittest.mock import Mock, patch

import httpx

from briefgpt_arxiv.config import LLMEndpointSettings
from briefgpt_arxiv.llm_client import GeminiAPIClient, OpenAICompatibleClient, create_llm_client


class LLMClientTests(TestCase):
    @patch("briefgpt_arxiv.llm_client.time.sleep", return_value=None)
    @patch("briefgpt_arxiv.llm_client.OpenAI")
    def test_openai_compatible_client_retries_transient_errors(self, mock_openai: Mock, _mock_sleep: Mock) -> None:
        success_response = Mock()
        success_response.choices = [Mock(message=Mock(content='{"ok": true}'))]
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = [
            RuntimeError("temporary ssl eof"),
            success_response,
        ]
        mock_openai.return_value = mock_client

        client = OpenAICompatibleClient(
            endpoint=LLMEndpointSettings(provider="openai_compatible", model_name="test-model"),
            api_key="test-key",
        )
        payload = client.generate_json(
            system_instruction="Return JSON.",
            user_text="{}",
            response_json_schema={"type": "object"},
        )

        self.assertEqual({"ok": True}, payload)
        self.assertEqual(2, mock_client.chat.completions.create.call_count)

    @patch("briefgpt_arxiv.llm_client.time.sleep", return_value=None)
    @patch("briefgpt_arxiv.llm_client.httpx.post")
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
            httpx.ConnectError("temporary ssl eof"),
            success_response,
        ]

        client = GeminiAPIClient(
            endpoint=LLMEndpointSettings(provider="gemini", model_name="gemini-test-model"),
            api_key="test-key",
        )
        payload = client.generate_json(
            system_instruction="Return JSON.",
            user_text="{}",
            response_json_schema={"type": "object"},
        )

        self.assertEqual({"ok": True}, payload)
        self.assertEqual(2, mock_post.call_count)

    def test_factory_selects_openai_compatible_provider(self) -> None:
        with patch("briefgpt_arxiv.llm_client.OpenAICompatibleClient") as mock_client:
            endpoint = LLMEndpointSettings(provider="openai_compatible", model_name="test-model")
            create_llm_client(endpoint)

        mock_client.assert_called_once_with(endpoint)

    def test_factory_selects_gemini_provider(self) -> None:
        with patch("briefgpt_arxiv.llm_client.GeminiAPIClient") as mock_client:
            endpoint = LLMEndpointSettings(provider="gemini", model_name="gemini-test-model")
            create_llm_client(endpoint)

        mock_client.assert_called_once_with(endpoint)
