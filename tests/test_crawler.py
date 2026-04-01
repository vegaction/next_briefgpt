from __future__ import annotations

from pathlib import Path

from briefgpt_arxiv.models import Artifact, IngestionJob, Paper
from briefgpt_arxiv.services.crawler import ArxivClient, ArxivPaperRecord, CrawlerService
from tests.conftest import FIXTURES


class FakeArxivClient(ArxivClient):
    def __init__(self) -> None:
        self.downloads: list[Path] = []

    def fetch_record(self, arxiv_id: str) -> ArxivPaperRecord:
        return self.parse_record((FIXTURES / "arxiv_feed.xml").read_text())

    def download(self, url: str, destination: Path) -> tuple[str, int]:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if "pdf" in url:
            destination.write_text("fake pdf content")
        else:
            destination.write_text("fake source content")
        self.downloads.append(destination)
        return "checksum", destination.stat().st_size


def test_crawl_creates_paper_and_artifacts(db_session, tempdir) -> None:
    service = CrawlerService(
        db_session,
        client=FakeArxivClient(),
        artifact_root=Path(tempdir.name),
    )

    papers = service.crawl_arxiv_ids(["2603.15726v1"])

    assert len(papers) == 1
    assert db_session.query(Paper).count() == 1
    assert db_session.query(Artifact).count() == 2
    assert papers[0].ingest_status == "fetched"
    job = db_session.query(IngestionJob).one()
    assert job.job_type == "crawl"
    assert job.status == "completed"
    assert job.attempt_count == 1
    assert job.finished_at is not None


def test_crawl_is_idempotent_for_same_version(db_session, tempdir) -> None:
    service = CrawlerService(
        db_session,
        client=FakeArxivClient(),
        artifact_root=Path(tempdir.name),
    )

    service.crawl_arxiv_ids(["2603.15726v1"])
    service.crawl_arxiv_ids(["2603.15726v1"])

    assert db_session.query(Paper).count() == 1
    assert db_session.query(Artifact).count() == 2
