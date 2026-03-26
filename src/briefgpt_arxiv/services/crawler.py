from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from xml.etree import ElementTree

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from briefgpt_arxiv.config import settings
from briefgpt_arxiv.models import Artifact, Paper
from briefgpt_arxiv.services.jobs import JobTracker
from briefgpt_arxiv.utils import ensure_parent, format_arxiv_id, sha256sum, split_arxiv_id


ATOM_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


@dataclass(slots=True)
class ArxivPaperRecord:
    arxiv_id: str
    version: str
    title: str
    abstract: str
    primary_category: str | None
    published_at: datetime | None
    updated_at: datetime | None
    pdf_url: str
    source_url: str


class ArxivClient:
    api_url = "https://export.arxiv.org/api/query"

    def fetch_record(self, arxiv_id: str) -> ArxivPaperRecord:
        response = httpx.get(
            self.api_url,
            params={"search_query": f"id:{arxiv_id}", "start": 0, "max_results": 1},
            timeout=30.0,
        )
        response.raise_for_status()
        return self.parse_record(response.text)

    def download(self, url: str, destination: Path) -> tuple[str, int]:
        ensure_parent(destination)
        with httpx.stream("GET", url, timeout=60.0, follow_redirects=True) as response:
            response.raise_for_status()
            with destination.open("wb") as handle:
                for chunk in response.iter_bytes():
                    handle.write(chunk)
        return sha256sum(destination), destination.stat().st_size

    def parse_record(self, xml_text: str) -> ArxivPaperRecord:
        root = ElementTree.fromstring(xml_text)
        entry = root.find("atom:entry", ATOM_NS)
        if entry is None:
            raise ValueError("No arXiv entry found in response.")

        entry_id = entry.findtext("atom:id", default="", namespaces=ATOM_NS).rsplit("/", 1)[-1]
        title = (entry.findtext("atom:title", default="", namespaces=ATOM_NS) or "").strip()
        abstract = (entry.findtext("atom:summary", default="", namespaces=ATOM_NS) or "").strip()
        published_at = self._parse_datetime(entry.findtext("atom:published", default="", namespaces=ATOM_NS))
        updated_at = self._parse_datetime(entry.findtext("atom:updated", default="", namespaces=ATOM_NS))
        primary_category = None
        category_element = entry.find("arxiv:primary_category", ATOM_NS)
        if category_element is not None:
            primary_category = category_element.attrib.get("term")
        arxiv_id, version = split_arxiv_id(entry_id)
        versioned_id = format_arxiv_id(arxiv_id, version or "v1")
        pdf_url = f"https://arxiv.org/pdf/{versioned_id}.pdf"
        source_url = f"https://arxiv.org/e-print/{versioned_id}"

        return ArxivPaperRecord(
            arxiv_id=arxiv_id,
            version=version or "v1",
            title=title,
            abstract=abstract,
            primary_category=primary_category,
            published_at=published_at,
            updated_at=updated_at,
            pdf_url=pdf_url,
            source_url=source_url,
        )

    @staticmethod
    def _parse_datetime(raw: str) -> datetime | None:
        if not raw:
            return None
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(UTC).replace(tzinfo=None)


class CrawlerService:
    def __init__(self, session: Session, client: ArxivClient | None = None, artifact_root: Path | None = None):
        self.session = session
        self.client = client or ArxivClient()
        self.artifact_root = artifact_root or settings.artifact_root
        self.job_tracker = JobTracker(session)

    def crawl_arxiv_ids(self, arxiv_ids: list[str]) -> list[Paper]:
        papers: list[Paper] = []
        for arxiv_id in arxiv_ids:
            job = self.job_tracker.start(job_type="crawl", target_id=0)
            try:
                record = self.client.fetch_record(arxiv_id)
                paper = self._upsert_paper(record)
                job.target_id = paper.id
                self._persist_artifact(paper, "pdf", record.pdf_url, suffix=".pdf")
                self._persist_artifact(paper, "source", record.source_url, suffix=".tar")
                paper.ingest_status = "fetched"
                self.job_tracker.finish(job)
                papers.append(paper)
                self.session.commit()
            except Exception as exc:
                self.session.rollback()
                self.job_tracker.record_failure(job_type="crawl", target_id=0, error_message=str(exc))
                self.session.commit()
                raise
        return papers

    def _upsert_paper(self, record: ArxivPaperRecord) -> Paper:
        paper = self.session.scalar(
            select(Paper).where(
                Paper.arxiv_id == record.arxiv_id,
                Paper.version == record.version,
            )
        )
        if paper is None:
            paper = Paper(
                arxiv_id=record.arxiv_id,
                version=record.version,
                title=record.title,
                abstract=record.abstract,
            )
            self.session.add(paper)
            self.session.flush()
        paper.title = record.title
        paper.abstract = record.abstract
        paper.primary_category = record.primary_category
        paper.published_at = record.published_at
        paper.updated_at_source = record.updated_at
        if not paper.parse_status:
            paper.parse_status = "pending"
        return paper

    def _persist_artifact(self, paper: Paper, artifact_type: str, source_url: str, suffix: str) -> None:
        destination = self.artifact_root / paper.arxiv_id / paper.version / f"{artifact_type}{suffix}"
        checksum, size_bytes = self.client.download(source_url, destination)
        artifact = self.session.scalar(
            select(Artifact).where(
                Artifact.paper_id == paper.id,
                Artifact.artifact_type == artifact_type,
            )
        )
        if artifact is None:
            artifact = Artifact(paper_id=paper.id, artifact_type=artifact_type, uri=str(destination))
            self.session.add(artifact)
        artifact.uri = str(destination)
        artifact.checksum = checksum
        artifact.size_bytes = size_bytes
