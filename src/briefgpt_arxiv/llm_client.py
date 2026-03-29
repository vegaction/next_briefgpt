from __future__ import annotations

import json
import time

import requests

from briefgpt_arxiv.config import LLMEndpointSettings, settings


class LLMConfigurationError(RuntimeError):
    """Raised when the configured LLM provider is missing required configuration."""


class TransientLLMResponseError(RuntimeError):
    """Raised when the provider returned a retryable non-HTTP response error."""


def parse_json_text(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 3:
            cleaned = "\n".join(lines[1:-1]).strip()
    start_positions = [pos for pos in [cleaned.find("{"), cleaned.find("[")] if pos != -1]
    end_positions = [pos for pos in [cleaned.rfind("}"), cleaned.rfind("]")] if pos != -1]
    if start_positions and end_positions:
        start = min(start_positions)
        end = max(end_positions)
        cleaned = cleaned[start : end + 1]
    return json.loads(cleaned)


def post_json_with_retries(
    *,
    url: str,
    headers: dict[str, str],
    payload: dict,
    timeout_seconds: float,
    should_retry_http_error,
) -> dict:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = requests.post(
                url,
                headers=headers,
                data=json.dumps(payload),
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as exc:
            if not should_retry_http_error(exc):
                raise
            last_error = exc
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_error = exc
        if attempt == 2:
            break
        time.sleep(1.0 * (attempt + 1))
    assert last_error is not None
    raise last_error


class BaseLLMClient:
    def __init__(self, endpoint: LLMEndpointSettings):
        self.model_name = endpoint.model_name

    def generate_json(self, system_instruction: str, user_text: str) -> dict:
        raise NotImplementedError

    def generate_text(self, system_instruction: str, user_text: str) -> str:
        raise NotImplementedError


class GeminiAPIClient(BaseLLMClient):
    def __init__(self, endpoint: LLMEndpointSettings, api_key: str | None = None):
        super().__init__(endpoint)
        self.api_key = api_key or settings.gemini_api_key
        if not self.api_key:
            raise LLMConfigurationError("GEMINI_API_KEY is required for the Gemini provider.")

    def generate_json(self, system_instruction: str, user_text: str) -> dict:
        return parse_json_text(
            self._generate_text_response(system_instruction=system_instruction, user_text=user_text)
        )

    def generate_text(self, system_instruction: str, user_text: str) -> str:
        return self._generate_text_response(system_instruction=system_instruction, user_text=user_text)

    @staticmethod
    def _build_payload(*, system_instruction: str, user_text: str) -> dict:
        return {
            "system_instruction": {"parts": [{"text": system_instruction}]},
            "contents": [{"parts": [{"text": user_text}]}],
        }

    def _post(self, payload: dict) -> dict:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"
        return post_json_with_retries(
            url=url,
            headers={
                "x-goog-api-key": self.api_key,
                "Content-Type": "application/json",
            },
            payload=payload,
            timeout_seconds=60.0,
            should_retry_http_error=lambda _exc: True,
        )

    def _generate_text_response(self, *, system_instruction: str, user_text: str) -> str:
        response = self._post(self._build_payload(system_instruction=system_instruction, user_text=user_text))
        return self._extract_text(response)

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
        timeout_seconds: float | None = None,
    ):
        super().__init__(endpoint)
        self.api_key = api_key or settings.openrouter_api_key
        self.base_url = base_url or settings.openrouter_base_url
        self.reasoning_enabled = (
            endpoint.reasoning_enabled
            if endpoint.reasoning_enabled is not None
            else (settings.openrouter_reasoning_enabled if reasoning_enabled is None else reasoning_enabled)
        )
        self.timeout_seconds = settings.openrouter_timeout_seconds if timeout_seconds is None else timeout_seconds
        if not self.api_key:
            raise LLMConfigurationError("OPEN_ROUTER_API_KEY is required for the OpenAI-compatible provider.")

    def generate_json(self, system_instruction: str, user_text: str) -> dict:
        return self._generate_with_retries(
            system_instruction=system_instruction,
            user_text=user_text,
            transform=parse_json_text,
            retryable_exceptions=(TransientLLMResponseError, json.JSONDecodeError),
        )

    def generate_text(self, system_instruction: str, user_text: str) -> str:
        return self._generate_with_retries(
            system_instruction=system_instruction,
            user_text=user_text,
            transform=lambda text: text,
            retryable_exceptions=(TransientLLMResponseError,),
        )

    def _create_completion(
        self,
        *,
        system_instruction: str,
        user_text: str,
    ):
        request_body = {
            "model": self.model_name,
            "messages": self._build_messages(
                system_instruction=system_instruction,
                user_text=user_text,
            ),
        }
        if self.reasoning_enabled:
            request_body["reasoning"] = {"enabled": True}

        return post_json_with_retries(
            url=f"{self.base_url.rstrip('/')}/chat/completions",
            headers=self._build_headers(),
            payload=request_body,
            timeout_seconds=self.timeout_seconds,
            should_retry_http_error=self._should_retry_http_error,
        )

    def _generate_with_retries(
        self,
        *,
        system_instruction: str,
        user_text: str,
        transform,
        retryable_exceptions: tuple[type[Exception], ...],
    ):
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = self._create_completion(
                    system_instruction=system_instruction,
                    user_text=user_text,
                )
                return transform(self._extract_text(response))
            except Exception as exc:
                if not isinstance(exc, retryable_exceptions):
                    raise
                last_error = exc
                if attempt == 2:
                    break
                time.sleep(1.0 * (attempt + 1))
        assert last_error is not None
        raise last_error

    @staticmethod
    def _build_messages(*, system_instruction: str, user_text: str) -> list[dict]:
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_instruction}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": user_text}],
            },
        ]

    def _build_headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if settings.openrouter_site_url:
            headers["HTTP-Referer"] = settings.openrouter_site_url
        if settings.openrouter_site_name:
            headers["X-OpenRouter-Title"] = settings.openrouter_site_name
        return headers

    @staticmethod
    def _should_retry_http_error(exc: requests.HTTPError) -> bool:
        status_code = exc.response.status_code if exc.response is not None else None
        return status_code is None or status_code == 429 or status_code >= 500

    @staticmethod
    def _extract_text(response) -> str:
        choices = response.get("choices") or []
        if not choices:
            raise RuntimeError(f"OpenAI-compatible provider returned no choices: {response}")
        choice_error = choices[0].get("error")
        if choice_error:
            message = choice_error.get("message") or "unknown provider error"
            code = choice_error.get("code")
            metadata = choice_error.get("metadata") or {}
            error_type = metadata.get("error_type")
            exc_message = (
                f"OpenAI-compatible provider returned an error in the response body: "
                f"code={code!r} error_type={error_type!r} message={message!r}"
            )
            if (
                code is None
                or (isinstance(code, int) and code >= 500)
                or error_type in {"provider_unavailable", "server_error", "rate_limited"}
            ):
                raise TransientLLMResponseError(exc_message)
            raise RuntimeError(exc_message)
        message = choices[0].get("message")
        if message is None:
            raise RuntimeError(f"OpenAI-compatible response did not contain a message: {response}")
        content = message.get("content")
        if isinstance(content, str) and content:
            return content
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("text"):
                    text_parts.append(item["text"])
                    continue
                text = getattr(item, "text", None)
                if text:
                    text_parts.append(text)
            if text_parts:
                return "".join(text_parts)
        raise RuntimeError(f"OpenAI-compatible response did not contain text content: {response}")


def create_llm_client(endpoint: LLMEndpointSettings) -> BaseLLMClient:
    if endpoint.provider == "gemini":
        return GeminiAPIClient(endpoint)
    if endpoint.provider == "openai_compatible":
        return OpenAICompatibleClient(endpoint)
    raise LLMConfigurationError(f"Unsupported LLM provider {endpoint.provider!r}.")
