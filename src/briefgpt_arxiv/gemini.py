from __future__ import annotations

import json

import httpx

from briefgpt_arxiv.config import settings


class GeminiClient:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or settings.gemini_api_key
        self.model = model or settings.gemini_model
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is required for GeminiClient.")

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
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
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
