from __future__ import annotations

from datetime import UTC, datetime

from briefgpt_arxiv.util.arxiv import (  # noqa: F401
    ARXIV_VERSION_PATTERN,
    arxiv_version_number,
    format_arxiv_id,
    split_arxiv_id,
)
from briefgpt_arxiv.util.fileutil import ensure_parent, load_json, sha256sum  # noqa: F401
from briefgpt_arxiv.util.text import normalize_whitespace, split_sentences  # noqa: F401


def utcnow_naive() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)
