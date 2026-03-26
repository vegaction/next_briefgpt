from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


def normalize_env_value(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped or stripped in {"***", "<your-key>", "changeme"}:
        return None
    return stripped


def parse_bool_env(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def load_dotenv(dotenv_path: str = ".env") -> None:
    path = Path(dotenv_path)
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        os.environ.setdefault(key, value)


def load_yaml_config(config_path: str = "config.yaml") -> dict:
    path = Path(config_path)
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise RuntimeError(f"{config_path} must contain a top-level mapping.")
    return payload


def get_yaml_mapping(config: dict, key: str) -> dict:
    value = config.get(key) or {}
    if not isinstance(value, dict):
        raise RuntimeError(f"config.yaml field '{key}' must be a mapping.")
    return value


def get_endpoint_config(config: dict, endpoint_name: str) -> dict:
    value = config.get(endpoint_name) or {}
    if not isinstance(value, dict):
        raise RuntimeError(f"config.yaml llm.{endpoint_name} must be a mapping.")
    return value


load_dotenv()
yaml_config = load_yaml_config()
llm_yaml_config = get_yaml_mapping(yaml_config, "llm")


@dataclass(slots=True)
class LLMEndpointSettings:
    provider: str
    model_name: str


def default_llm_provider() -> str:
    if normalize_env_value(os.getenv("OPEN_ROUTER_API_KEY")):
        return "openai_compatible"
    if normalize_env_value(os.getenv("GEMINI_API_KEY")):
        return "gemini"
    return "openai_compatible"


def default_model_name_for_provider(provider: str) -> str:
    if provider == "gemini":
        return os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
    return os.getenv("OPEN_ROUTER_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")


def build_llm_endpoint_settings(endpoint_name: str) -> LLMEndpointSettings:
    endpoint_config = get_endpoint_config(llm_yaml_config, endpoint_name)
    provider = str(endpoint_config.get("provider") or default_llm_provider())
    model_name = str(endpoint_config.get("model_name") or default_model_name_for_provider(provider))
    return LLMEndpointSettings(provider=provider, model_name=model_name)


@dataclass(slots=True)
class Settings:
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./briefgpt.db")
    artifact_root: Path = Path(os.getenv("ARTIFACT_ROOT", "./artifacts"))
    openrouter_api_key: str | None = normalize_env_value(os.getenv("OPEN_ROUTER_API_KEY"))
    gemini_api_key: str | None = normalize_env_value(os.getenv("GEMINI_API_KEY"))
    openrouter_base_url: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    openrouter_reasoning_enabled: bool = parse_bool_env(os.getenv("OPENROUTER_REASONING_ENABLED"), True)
    summary_debug_log_path: Path = Path(os.getenv("SUMMARY_DEBUG_LOG_PATH", "./logs/summary_debug.jsonl"))
    parser_llm: LLMEndpointSettings = field(default_factory=lambda: build_llm_endpoint_settings("parser"))
    extractor_llm: LLMEndpointSettings = field(default_factory=lambda: build_llm_endpoint_settings("extractor"))

    def has_llm_api_key(self, provider: str) -> bool:
        if provider == "gemini":
            return bool(self.gemini_api_key)
        if provider == "openai_compatible":
            return bool(self.openrouter_api_key)
        raise RuntimeError(f"Unsupported LLM provider {provider!r}")


settings = Settings()
