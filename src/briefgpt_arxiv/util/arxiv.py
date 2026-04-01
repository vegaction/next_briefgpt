from __future__ import annotations

import re


ARXIV_VERSION_PATTERN = re.compile(r"^(?P<base>.+?)(?P<version>v\d+)$")


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
