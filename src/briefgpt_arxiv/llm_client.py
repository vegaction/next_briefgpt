from __future__ import annotations

import json
import time

import httpx
from openai import OpenAI

from briefgpt_arxiv.config import LLMEndpointSettings, settings


class LLMConfigurationError(RuntimeError):
    """Raised when the configured LLM provider is missing required configuration."""


class BaseLLMClient:
    def __init__(self, endpoint: LLMEndpointSettings):
        self.endpoint = endpoint
        self.model_name = endpoint.model_name

    def generate_json(self, system_instruction: str, user_text: str, response_json_schema: dict) -> dict:
        raise NotImplementedError

    def generate_text(self, system_instruction: str, user_text: str) -> str:
        raise NotImplementedError


class GeminiAPIClient(BaseLLMClient):
    def __init__(self, endpoint: LLMEndpointSettings, api_key: str | None = None):
        super().__init__(endpoint)
        self.api_key = api_key or settings.gemini_api_key
        if not self.api_key:
            raise LLMConfigurationError("GEMINI_API_KEY is required for the Gemini provider.")

    def generate_json(self, system_instruction: str, user_text: str, response_json_schema: dict) -> dict:
        payload = {
            "system_instruction": {"parts": [{"text": system_instruction}]},
            "contents": [{"parts": [{"text": user_text}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseJsonSchema": response_json_schema,
            },
        }
        response = self._post(payload)
        text_response = self._extract_text(response)
        return json.loads(text_response)

    def generate_text(self, system_instruction: str, user_text: str) -> str:
        payload = {
            "system_instruction": {"parts": [{"text": system_instruction}]},
            "contents": [{"parts": [{"text": user_text}]}],
        }
        response = self._post(payload)
        return self._extract_text(response)

    def _post(self, payload: dict) -> dict:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = httpx.post(
                    url,
                    headers={
                        "x-goog-api-key": self.api_key,
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=60.0,
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as exc:
                last_error = exc
                if attempt == 2:
                    break
                time.sleep(1.0 * (attempt + 1))
        assert last_error is not None
        raise last_error

    @staticmethod
    def _extract_text(response_json: dict) -> str:
        candidates = response_json.get("candidates") or []
        if not candidates:
            raise RuntimeError(f"Gemini returned no candidates: {response_json}")
        parts = candidates[0].get("content", {}).get("parts", [])
        text_parts = [part.get("text", "") for part in parts if part.get("text")]
        if not text_parts:
            raise RuntimeError(f"Gemini response did not contain text: {response_json}")
        return "".join(text_parts)


class OpenAICompatibleClient(BaseLLMClient):
    def __init__(
        self,
        endpoint: LLMEndpointSettings,
        api_key: str | None = None,
        base_url: str | None = None,
        reasoning_enabled: bool | None = None,
    ):
        super().__init__(endpoint)
        self.api_key = api_key or settings.openrouter_api_key
        self.base_url = base_url or settings.openrouter_base_url
        self.reasoning_enabled = (
            settings.openrouter_reasoning_enabled if reasoning_enabled is None else reasoning_enabled
        )
        if not self.api_key:
            raise LLMConfigurationError("OPEN_ROUTER_API_KEY is required for the OpenAI-compatible provider.")
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )

    def generate_json(self, system_instruction: str, user_text: str, response_json_schema: dict) -> dict:
        response = self._create_completion(
            system_instruction=system_instruction,
            user_text=user_text,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_response",
                    "strict": True,
                    "schema": response_json_schema,
                },
            },
        )
        text_response = self._extract_text(response)
        return json.loads(text_response)

    def generate_text(self, system_instruction: str, user_text: str) -> str:
        response = self._create_completion(
            system_instruction=system_instruction,
            user_text=user_text,
        )
        return self._extract_text(response)

    def _create_completion(
        self,
        *,
        system_instruction: str,
        user_text: str,
        response_format: dict | None = None,
    ):
        request_kwargs = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_text},
            ],
        }
        if response_format is not None:
            request_kwargs["response_format"] = response_format
        if self.reasoning_enabled:
            request_kwargs["extra_body"] = {"reasoning": {"enabled": True}}

        last_error: Exception | None = None
        for attempt in range(3):
            try:
                return self.client.chat.completions.create(**request_kwargs)
            except Exception as exc:
                last_error = exc
                if attempt == 2:
                    break
                time.sleep(1.0 * (attempt + 1))
        assert last_error is not None
        raise last_error

    @staticmethod
    def _extract_text(response) -> str:
        choices = getattr(response, "choices", None) or []
        if not choices:
            raise RuntimeError(f"OpenAI-compatible provider returned no choices: {response}")
        message = getattr(choices[0], "message", None)
        if message is None:
            raise RuntimeError(f"OpenAI-compatible response did not contain a message: {response}")
        content = getattr(message, "content", None)
        if isinstance(content, str) and content:
            return content
        if isinstance(content, list):
            text_parts = []
            for item in content:
                text = getattr(item, "text", None)
                if text:
                    text_parts.append(text)
                    continue
                if isinstance(item, dict) and item.get("text"):
                    text_parts.append(item["text"])
            if text_parts:
                return "".join(text_parts)
        raise RuntimeError(f"OpenAI-compatible response did not contain text content: {response}")


def create_llm_client(endpoint: LLMEndpointSettings) -> BaseLLMClient:
    if endpoint.provider == "gemini":
        return GeminiAPIClient(endpoint)
    if endpoint.provider == "openai_compatible":
        return OpenAICompatibleClient(endpoint)
    raise LLMConfigurationError(f"Unsupported LLM provider {endpoint.provider!r}.")
