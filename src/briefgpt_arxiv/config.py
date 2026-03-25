from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def normalize_env_value(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped or stripped in {"***", "<your-key>", "changeme"}:
        return None
    return stripped


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


load_dotenv()


@dataclass(slots=True)
class Settings:
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./briefgpt.db")
    artifact_root: Path = Path(os.getenv("ARTIFACT_ROOT", "./artifacts"))
    gemini_api_key: str | None = normalize_env_value(os.getenv("GEMINI_API_KEY"))
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
    summary_debug_log_path: Path = Path(os.getenv("SUMMARY_DEBUG_LOG_PATH", "./logs/summary_debug.jsonl"))


settings = Settings()
