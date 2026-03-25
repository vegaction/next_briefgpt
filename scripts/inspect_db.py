from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from sqlalchemy import func, select, text
from sqlalchemy.orm import joinedload

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from briefgpt_arxiv.db import SessionLocal  # noqa: E402
from briefgpt_arxiv.models import (  # noqa: E402
    CitationBlock,
    CitationMention,
    IngestionJob,
    Paper,
    PaperReference,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect the briefgpt-arxiv database.")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("overview", help="Show high-level table counts and papers.")

    paper = subparsers.add_parser("paper", help="Show one paper with references and extracted mentions.")
    paper.add_argument("arxiv_id", help="arXiv id, e.g. 2603.15726v1")

    mentions = subparsers.add_parser("mentions", help="List citation mentions.")
    mentions.add_argument("--arxiv-id", dest="arxiv_id", help="Filter by arXiv id")
    mentions.add_argument("--limit", type=int, default=20, help="Max rows to print")

    extractions = subparsers.add_parser("extractions", help="List citation extractions.")
    extractions.add_argument("--arxiv-id", dest="arxiv_id", help="Filter by arXiv id")
    extractions.add_argument("--limit", type=int, default=20, help="Max rows to print")

    blocks = subparsers.add_parser("blocks", help="List citation blocks.")
    blocks.add_argument("--arxiv-id", dest="arxiv_id", help="Filter by arXiv id")
    blocks.add_argument("--limit", type=int, default=20, help="Max rows to print")

    jobs = subparsers.add_parser("jobs", help="List ingestion jobs.")
    jobs.add_argument("--job-type", dest="job_type", help="Filter by job type")
    jobs.add_argument("--target-id", dest="target_id", type=int, help="Filter by target paper id")
    jobs.add_argument("--limit", type=int, default=20, help="Max rows to print")

    dump = subparsers.add_parser("dump", help="Print all table schemas and the first N rows from each table.")
    dump.add_argument("--limit", type=int, default=20, help="Rows per table")
    dump.add_argument(
        "--citation-mentions-csv",
        default="citation_mentions_first_rows.csv",
        help="CSV path for the first N citation_mentions rows.",
    )

    sql = subparsers.add_parser("sql", help="Run a raw SQL query.")
    sql.add_argument("query", help="SQL query to run")

    return parser


def print_rows(rows: list[dict]) -> None:
    if not rows:
        print("No rows.")
        return
    headers = list(rows[0].keys())
    widths = {
        header: max(len(str(header)), *(len(str(row.get(header, ""))) for row in rows))
        for header in headers
    }
    print(" | ".join(str(header).ljust(widths[header]) for header in headers))
    print("-+-".join("-" * widths[header] for header in headers))
    for row in rows:
        print(" | ".join(str(row.get(header, "")).ljust(widths[header]) for header in headers))


def run_overview() -> None:
    with SessionLocal() as session:
        counts = {
            "papers": session.scalar(select(func.count(Paper.id))),
            "paper_references": session.scalar(select(func.count(PaperReference.id))),
            "citation_blocks": session.scalar(select(func.count(CitationBlock.id))),
            "citation_mentions": session.scalar(select(func.count(CitationMention.id))),
            "extracted_mentions": session.scalar(
                select(func.count(CitationMention.id)).where(CitationMention.intent_label.is_not(None))
            ),
        }
        print("Counts")
        print_rows([counts])
        print()
        papers = list(session.scalars(select(Paper).order_by(Paper.id)))
        print("Papers")
        print_rows(
            [
                {
                    "id": paper.id,
                    "arxiv_id": paper.arxiv_id,
                    "title": paper.title,
                    "current_version": paper.current_version,
                    "parse_status": paper.parse_status,
                    "ingest_status": paper.ingest_status,
                }
                for paper in papers
            ]
        )


def run_paper(arxiv_id: str) -> None:
    with SessionLocal() as session:
        paper = session.scalar(select(Paper).where(Paper.arxiv_id == arxiv_id))
        if paper is None:
            print(f"No paper found for {arxiv_id}")
            return
        print("Paper")
        print_rows(
            [
                {
                    "id": paper.id,
                    "arxiv_id": paper.arxiv_id,
                    "title": paper.title,
                    "current_version": paper.current_version,
                    "parse_status": paper.parse_status,
                    "ingest_status": paper.ingest_status,
                }
            ]
        )
        print()
        references = list(
            session.execute(
                select(PaperReference)
                .where(PaperReference.paper_id == paper.id)
                .options(
                    joinedload(PaperReference.citation_mentions)
                    .joinedload(CitationMention.citation_block)
                )
            )
            .unique()
            .scalars()
        )
        print("References")
        print_rows(
            [
                {
                    "id": ref.id,
                    "local_ref_id": ref.local_ref_id,
                    "title": ref.title,
                    "year": ref.year,
                    "mentions": len(ref.citation_mentions),
                }
                for ref in references
            ]
        )
        print()
        print("Mention Details")
        mention_rows = []
        for ref in references:
            for mention in sorted(ref.citation_mentions, key=lambda item: item.mention_order):
                mention_rows.append(
                    {
                        "mention_id": mention.id,
                        "local_ref_id": ref.local_ref_id,
                        "section_title": mention.citation_block.section_title,
                        "intent": mention.intent_label or "",
                        "summary": mention.summary or "",
                    }
                )
        print_rows(mention_rows)


def run_mentions(arxiv_id: str | None, limit: int) -> None:
    with SessionLocal() as session:
        stmt = (
            select(CitationMention)
            .join(CitationMention.paper_reference)
            .join(CitationMention.citation_block)
            .join(PaperReference.paper)
            .options(
                joinedload(CitationMention.paper_reference),
                joinedload(CitationMention.citation_block),
            )
            .order_by(CitationMention.id)
            .limit(limit)
        )
        if arxiv_id:
            stmt = stmt.where(Paper.arxiv_id == arxiv_id)
        mentions = list(session.scalars(stmt))
        print_rows(
            [
                {
                    "id": mention.id,
                    "paper": mention.paper_reference.paper.arxiv_id,
                    "local_ref_id": mention.paper_reference.local_ref_id,
                    "section_title": mention.citation_block.section_title,
                    "citation_mention": mention.citation_mention,
                    "intent": mention.intent_label or "",
                }
                for mention in mentions
            ]
        )


def run_extractions(arxiv_id: str | None, limit: int) -> None:
    with SessionLocal() as session:
        stmt = (
            select(CitationMention)
            .join(CitationMention.paper_reference)
            .join(CitationMention.citation_block)
            .join(PaperReference.paper)
            .where(CitationMention.intent_label.is_not(None))
            .options(
                joinedload(CitationMention.paper_reference),
                joinedload(CitationMention.citation_block),
            )
            .order_by(CitationMention.id)
            .limit(limit)
        )
        if arxiv_id:
            stmt = stmt.where(Paper.arxiv_id == arxiv_id)
        mentions = list(session.scalars(stmt))
        print_rows(
            [
                {
                    "mention_id": mention.id,
                    "paper": mention.paper_reference.paper.arxiv_id,
                    "local_ref_id": mention.paper_reference.local_ref_id,
                    "section_title": mention.citation_block.section_title,
                    "intent": mention.intent_label,
                    "summary": mention.summary,
                    "status": mention.status,
                }
                for mention in mentions
            ]
        )


def run_blocks(arxiv_id: str | None, limit: int) -> None:
    with SessionLocal() as session:
        stmt = (
            select(CitationBlock)
            .join(CitationBlock.paper)
            .options(joinedload(CitationBlock.paper))
            .order_by(CitationBlock.id)
            .limit(limit)
        )
        if arxiv_id:
            stmt = stmt.where(Paper.arxiv_id == arxiv_id)
        blocks = list(session.scalars(stmt))
        print_rows(
            [
                {
                    "id": block.id,
                    "paper": block.paper.arxiv_id,
                    "section_title": block.section_title,
                    "raw_citation_keys": block.raw_citation_keys,
                    "has_citations": block.has_citations,
                    "raw_text": block.raw_text[:100],
                }
                for block in blocks
            ]
        )


def run_jobs(job_type: str | None, target_id: int | None, limit: int) -> None:
    with SessionLocal() as session:
        stmt = select(IngestionJob).order_by(IngestionJob.id.desc()).limit(limit)
        if job_type:
            stmt = stmt.where(IngestionJob.job_type == job_type)
        if target_id is not None:
            stmt = stmt.where(IngestionJob.target_id == target_id)
        jobs = list(session.scalars(stmt))
        paper_ids = [job.target_id for job in jobs if job.target_id]
        paper_map = {
            paper.id: paper.arxiv_id
            for paper in session.scalars(select(Paper).where(Paper.id.in_(paper_ids)))
        }
        print_rows(
            [
                {
                    "id": job.id,
                    "job_type": job.job_type,
                    "target_id": job.target_id,
                    "paper": paper_map.get(job.target_id, ""),
                    "status": job.status,
                    "attempt_count": job.attempt_count,
                    "started_at": job.started_at,
                    "finished_at": job.finished_at,
                    "error_message": (job.error_message or "")[:120],
                }
                for job in jobs
            ]
        )


def run_sql(query: str) -> None:
    with SessionLocal() as session:
        result = session.execute(text(query))
        rows = [dict(row._mapping) for row in result]
        print_rows(rows)


def write_csv_rows(path: Path, rows: list[Row]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({header: row[header] for header in headers})


def run_dump(limit: int, citation_mentions_csv: str) -> None:
    from sqlite3 import Row, connect

    database_url = str(SessionLocal.kw["bind"].url)
    if not database_url.startswith("sqlite:///"):
        raise RuntimeError("dump currently supports sqlite databases only.")
    db_path = database_url.replace("sqlite:///", "", 1)

    conn = connect(db_path)
    conn.row_factory = Row
    cur = conn.cursor()
    csv_path = Path(citation_mentions_csv)
    try:
        tables = [
            row[0]
            for row in cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        for table in tables:
            print(f"=== SCHEMA: {table} ===")
            schema = cur.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
                (table,),
            ).fetchone()[0]
            print(schema)
            print()
            print(f"=== FIRST {limit} ROWS: {table} ===")
            rows = cur.execute(f'SELECT * FROM "{table}" LIMIT {limit}').fetchall()
            if not rows:
                print("(no rows)")
            else:
                headers = rows[0].keys()
                print(" | ".join(headers))
                for row in rows:
                    print(" | ".join(str(row[col]) for col in headers))
                if table == "citation_mentions":
                    write_csv_rows(csv_path, rows)
                    print()
                    print(f"(wrote citation_mentions CSV to {csv_path})")
            print()
    finally:
        conn.close()


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    command = args.command or "overview"

    if command == "overview":
        run_overview()
    elif command == "paper":
        run_paper(args.arxiv_id)
    elif command == "mentions":
        run_mentions(args.arxiv_id, args.limit)
    elif command == "extractions":
        run_extractions(args.arxiv_id, args.limit)
    elif command == "blocks":
        run_blocks(args.arxiv_id, args.limit)
    elif command == "jobs":
        run_jobs(args.job_type, args.target_id, args.limit)
    elif command == "dump":
        run_dump(args.limit, args.citation_mentions_csv)
    elif command == "sql":
        run_sql(args.query)
    else:
        parser.error(f"Unknown command: {command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
