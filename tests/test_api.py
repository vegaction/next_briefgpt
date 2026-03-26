from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from fastapi.testclient import TestClient

from briefgpt_arxiv.config import settings
from briefgpt_arxiv.main import app
from briefgpt_arxiv.models import Artifact, CitationMention, Paper
from briefgpt_arxiv.services.extractor import BaseExtractionClient, ExtractedCitation, ExtractorService
from tests.helpers import FIXTURES, get_session, reset_database


class FakeExtractionClient(BaseExtractionClient):
    model_name = "fake-llm"

    def annotate_candidates(self, candidates, raw_text, section_title, references, debug_context=None):
        return [
            ExtractedCitation(
                raw_citation_key=candidates[0].raw_citation_key,
                citation_mention=candidates[0].citation_mention,
                sentence_text=candidates[0].sentence_text,
                section_title=candidates[0].section_title,
                mention_order=candidates[0].mention_order,
                intent_label="method_use",
                summary="ReAct is used as a planning baseline and method reference.",
            )
        ]


class ApiTests(TestCase):
    def setUp(self) -> None:
        reset_database()
        self.client = TestClient(app)
        self.session = get_session()
        self.tempdir = TemporaryDirectory()

    def tearDown(self) -> None:
        self.session.close()
        self.tempdir.cleanup()

    def test_parse_extract_and_query_endpoints(self) -> None:
        paper = Paper(arxiv_id="2603.15726", version="v1", title="Sample", abstract="A", ingest_status="fetched")
        self.session.add(paper)
        self.session.flush()
        paper.parse_status = "pending"
        structured_path = Path(self.tempdir.name) / "structured.json"
        structured_path.write_text((FIXTURES / "doc2json_sample.json").read_text())
        self.session.add(Artifact(paper_id=paper.id, artifact_type="structured_parse", uri=str(structured_path)))
        self.session.commit()

        parse_response = self.client.post(f"/parse/{paper.id}")
        self.assertEqual(200, parse_response.status_code)
        self.assertEqual(2, parse_response.json()["citation_blocks_created"])

        ExtractorService(self.session, client=FakeExtractionClient()).extract_for_paper(paper.id)

        paper_response = self.client.get("/papers/2603.15726v1")
        refs_response = self.client.get("/papers/2603.15726v1/references")
        search_response = self.client.get("/citations/search", params={"intent": "method_use"})

        self.assertEqual(200, paper_response.status_code)
        self.assertEqual(200, refs_response.status_code)
        self.assertEqual(200, search_response.status_code)
        self.assertEqual("2603.15726", paper_response.json()["arxiv_id"])
        self.assertEqual("v1", paper_response.json()["version"])
        self.assertEqual("parsed", paper_response.json()["parse_status"])
        self.assertGreaterEqual(len(refs_response.json()), 3)
        self.assertGreaterEqual(self.session.query(CitationMention).count(), 1)
        first_mention = refs_response.json()[0]["mentions"][0]
        self.assertIn("section_title", first_mention)
        self.assertIn("cited_arxiv_id", refs_response.json()[0])
        extraction = first_mention["extraction"]
        if extraction is not None:
            self.assertNotIn("evidence_text", extraction)
            self.assertNotIn("confidence", extraction)
        first_search = search_response.json()[0]
        self.assertEqual("2603.15726", first_search["paper_arxiv_id"])
        self.assertEqual("v1", first_search["paper_version"])

    def test_extract_endpoint_returns_503_without_llm_configuration(self) -> None:
        original_api_key = settings.gemini_api_key
        settings.gemini_api_key = None
        try:
            paper = Paper(arxiv_id="2603.15726", version="v1", title="Sample", abstract="A", ingest_status="parsed")
            self.session.add(paper)
            self.session.commit()

            response = self.client.post(f"/extract/{paper.id}")

            self.assertEqual(503, response.status_code)
            self.assertIn("GEMINI_API_KEY", response.json()["detail"])
        finally:
            settings.gemini_api_key = original_api_key
