from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from briefgpt_arxiv.models import Artifact, IngestionJob, Paper
from briefgpt_arxiv.services.crawler import ArxivClient, ArxivPaperRecord, CrawlerService
from tests.helpers import FIXTURES, get_session, reset_database


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


class CrawlerServiceTests(TestCase):
    def setUp(self) -> None:
        reset_database()
        self.session = get_session()
        self.tempdir = TemporaryDirectory()

    def tearDown(self) -> None:
        self.session.close()
        self.tempdir.cleanup()

    def test_crawl_creates_paper_and_artifacts(self) -> None:
        service = CrawlerService(
            self.session,
            client=FakeArxivClient(),
            artifact_root=Path(self.tempdir.name),
        )

        papers = service.crawl_arxiv_ids(["2603.15726v1"])

        self.assertEqual(1, len(papers))
        self.assertEqual(1, self.session.query(Paper).count())
        self.assertEqual(2, self.session.query(Artifact).count())
        self.assertEqual("fetched", papers[0].ingest_status)
        job = self.session.query(IngestionJob).one()
        self.assertEqual("crawl", job.job_type)
        self.assertEqual("completed", job.status)
        self.assertEqual(1, job.attempt_count)
        self.assertIsNotNone(job.finished_at)

    def test_crawl_is_idempotent_for_same_version(self) -> None:
        service = CrawlerService(
            self.session,
            client=FakeArxivClient(),
            artifact_root=Path(self.tempdir.name),
        )

        service.crawl_arxiv_ids(["2603.15726v1"])
        service.crawl_arxiv_ids(["2603.15726v1"])

        self.assertEqual(1, self.session.query(Paper).count())
        self.assertEqual(2, self.session.query(Artifact).count())
