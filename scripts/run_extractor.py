from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sqlalchemy import select

from briefgpt_arxiv.config import settings
from briefgpt_arxiv.db import SessionLocal, init_db
from briefgpt_arxiv.models import Paper
from briefgpt_arxiv.services.extractor import ExtractionConfigurationError, ExtractorService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run extractor only for an existing parsed paper.")
    parser.add_argument("paper_ref", help="Paper id or arXiv id, for example `1` or `2603.15726v1`.")
    parser.add_argument(
        "--skip-if-ready",
        action="store_true",
        help="Reuse existing extraction outputs when citation mentions already exist.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON instead of a text summary.")
    return parser


def resolve_paper(session, paper_ref: str) -> Paper | None:
    if paper_ref.isdigit():
        return session.get(Paper, int(paper_ref))
    return session.scalar(select(Paper).where(Paper.arxiv_id == paper_ref))


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args()
    init_db()

    with SessionLocal() as session:
        paper = resolve_paper(session, args.paper_ref)
        if paper is None:
            print(f"Paper not found: {args.paper_ref}", file=sys.stderr)
            return 1
        try:
            result = ExtractorService(session).extract_for_paper_result(
                paper.id,
                rerun=not args.skip_if_ready,
            )
        except ExtractionConfigurationError as exc:
            print(str(exc), file=sys.stderr)
            return 2

    payload = {
        "paper_id": result.paper_id,
        "arxiv_id": result.arxiv_id,
        "status": result.status,
        "model_name": result.model_name,
        "mentions_created": result.mentions_created,
        "extractions_created": result.extractions_created,
        "cleanup_performed": result.cleanup_performed,
        "summary_debug_log_path": str(settings.summary_debug_log_path.resolve()),
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    print(f"{payload['arxiv_id']} -> paper_id={payload['paper_id']}")
    print(
        "  "
        f"extract={payload['status']} model={payload['model_name']} "
        f"mentions={payload['mentions_created']} cleanup={payload['cleanup_performed']}"
    )
    print(f"  summary_debug_log={payload['summary_debug_log_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
