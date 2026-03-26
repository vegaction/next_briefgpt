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

from briefgpt_arxiv.config import settings
from briefgpt_arxiv.db import SessionLocal, init_db
from briefgpt_arxiv.models import Artifact, Paper
from briefgpt_arxiv.services.contracts import PipelineRunResult
from briefgpt_arxiv.services.extractor import ExtractorService
from briefgpt_arxiv.services.jobs import JobTracker
from briefgpt_arxiv.services.orchestrator import OrchestratorService
from briefgpt_arxiv.services.parser import ParserService
from briefgpt_arxiv.utils import format_arxiv_id, sha256sum, split_arxiv_id


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the crawl -> parse -> extract pipeline for arXiv ids.")
    parser.add_argument("arxiv_ids", nargs="+", help="arXiv ids to process")
    parser.add_argument(
        "--mode",
        choices=["crawl", "local-artifacts"],
        default="crawl",
        help="Whether to fetch from arXiv or restore inputs from local artifacts.",
    )
    parser.add_argument(
        "--skip-parse-if-parsed",
        action="store_true",
        help="Reuse existing parse outputs when the paper is already parsed.",
    )
    parser.add_argument(
        "--skip-extract-if-ready",
        action="store_true",
        help="Reuse existing extraction outputs when citation mentions already exist.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON instead of a text summary.")
    return parser


def resolve_local_artifact_dir(arxiv_id: str) -> tuple[Path, str]:
    base_arxiv_id, version = split_arxiv_id(arxiv_id)
    if version is None:
        raise ValueError(
            f"Local artifact mode requires a versioned arXiv id like `2603.15726v1`; got {arxiv_id!r}."
        )
    artifact_dir = settings.artifact_root / base_arxiv_id / version
    if not artifact_dir.exists():
        raise ValueError(f"Local artifact directory not found: {artifact_dir}")
    return artifact_dir, version


def upsert_local_artifact_paper(session, arxiv_id: str) -> Paper:
    base_arxiv_id, input_version = split_arxiv_id(arxiv_id)
    artifact_dir, version = resolve_local_artifact_dir(arxiv_id)
    paper = session.query(Paper).filter_by(arxiv_id=base_arxiv_id, version=input_version or version).one_or_none()
    if paper is None:
        paper = Paper(
            arxiv_id=base_arxiv_id,
            version=version,
            title=format_arxiv_id(base_arxiv_id, version),
            abstract="",
            ingest_status="fetched",
            parse_status="pending",
        )
        session.add(paper)
        session.flush()
    else:
        paper.version = version
        if paper.title is None or not paper.title.strip():
            paper.title = format_arxiv_id(base_arxiv_id, version)
        if paper.ingest_status == "discovered":
            paper.ingest_status = "fetched"

    artifact_specs = [
        ("pdf", artifact_dir / "pdf.pdf"),
        ("source", artifact_dir / "source.tar"),
        ("pdf_text", artifact_dir / "pdf.txt"),
    ]
    restored_artifacts: list[str] = []
    for artifact_type, path in artifact_specs:
        if not path.exists():
            continue
        artifact = (
            session.query(Artifact)
            .filter_by(paper_id=paper.id, artifact_type=artifact_type)
            .one_or_none()
        )
        if artifact is None:
            artifact = Artifact(paper_id=paper.id, artifact_type=artifact_type, uri=str(path))
            session.add(artifact)
        artifact.uri = str(path)
        artifact.checksum = sha256sum(path)
        artifact.size_bytes = path.stat().st_size
        restored_artifacts.append(artifact_type)
    if not restored_artifacts:
        raise ValueError(f"No local artifacts found under {artifact_dir}")

    job_tracker = JobTracker(session)
    job = job_tracker.start(job_type="crawl", target_id=paper.id)
    job_tracker.finish(job, error_message=f"Restored from local artifacts: {', '.join(restored_artifacts)}.")
    session.commit()
    return paper


def run_local_artifact_pipeline(
    session,
    arxiv_ids: list[str],
    *,
    rerun_parse: bool,
    rerun_extract: bool,
) -> list[PipelineRunResult]:
    parser = ParserService(session)
    extractor = ExtractorService(session)
    results: list[PipelineRunResult] = []
    for arxiv_id in arxiv_ids:
        paper = upsert_local_artifact_paper(session, arxiv_id)
        parse_result = parser.parse_paper(paper.id, rerun=rerun_parse)
        extract_result = extractor.extract_for_paper_result(paper.id, rerun=rerun_extract)
        results.append(
            PipelineRunResult(
                paper_id=paper.id,
                arxiv_id=paper.arxiv_id,
                version=paper.version,
                crawl_status="local-artifacts",
                parse=parse_result,
                extract=extract_result,
            )
        )
    return results


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args()
    init_db()
    with SessionLocal() as session:
        if args.mode == "crawl":
            results = OrchestratorService(session).run_pipeline_for_arxiv_ids(
                args.arxiv_ids,
                rerun_parse=not args.skip_parse_if_parsed,
                rerun_extract=not args.skip_extract_if_ready,
            )
        else:
            results = run_local_artifact_pipeline(
                session,
                args.arxiv_ids,
                rerun_parse=not args.skip_parse_if_parsed,
                rerun_extract=not args.skip_extract_if_ready,
            )

    payload = [
        {
            "paper_id": result.paper_id,
            "arxiv_id": result.arxiv_id,
            "version": result.version,
            "crawl_status": result.crawl_status,
            "parse": {
                "status": result.parse.status,
                "version": result.parse.version,
                "source_artifact_type": result.parse.source_artifact_type,
                "sections_created": result.parse.sections_created,
                "references_created": result.parse.references_created,
                "citation_blocks_created": result.parse.citation_blocks_created,
                "cleanup_performed": result.parse.cleanup_performed,
            },
            "extract": {
                "status": result.extract.status,
                "version": result.extract.version,
                "model_name": result.extract.model_name,
                "mentions_created": result.extract.mentions_created,
                "extractions_created": result.extract.extractions_created,
                "cleanup_performed": result.extract.cleanup_performed,
            },
        }
        for result in results
    ]
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    for item in payload:
        print(
            f"{format_arxiv_id(item['arxiv_id'], item['version'])} "
            f"-> paper_id={item['paper_id']} crawl={item['crawl_status']}"
        )
        print(
            "  "
            f"parse={item['parse']['status']} source={item['parse']['source_artifact_type']} "
            f"sections={item['parse']['sections_created']} refs={item['parse']['references_created']} "
            f"blocks={item['parse']['citation_blocks_created']} cleanup={item['parse']['cleanup_performed']}"
        )
        print(
            "  "
            f"extract={item['extract']['status']} model={item['extract']['model_name']} "
            f"mentions={item['extract']['mentions_created']} cleanup={item['extract']['cleanup_performed']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
