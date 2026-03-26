from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path


ARXIV_VERSION_PATTERN = re.compile(r"^(?P<base>.+?)(?P<version>v\d+)$")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str) -> list[str]:
    normalized = normalize_whitespace(text)
    if not normalized:
        return []
    parts = re.split(r"(?<=[.!?])\s+", normalized)
    return [part for part in parts if part]


def split_arxiv_id(identifier: str) -> tuple[str, str | None]:
    normalized = identifier.strip()
    match = ARXIV_VERSION_PATTERN.match(normalized)
    if match is None:
        return normalized, None
    return match.group("base"), match.group("version")


def format_arxiv_id(arxiv_id: str, version: str | None = None) -> str:
    return f"{arxiv_id}{version or ''}"


def arxiv_version_number(version: str | None) -> int:
    if not version:
        return 0
    try:
        return int(version.removeprefix("v"))
    except ValueError:
        return 0


def utcnow_naive() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)
