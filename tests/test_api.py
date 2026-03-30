from __future__ import annotations

from pathlib import Path
import re
from tempfile import TemporaryDirectory
from unittest import TestCase

from fastapi.testclient import TestClient

from briefgpt_arxiv.config import settings
from briefgpt_arxiv.main import app
from briefgpt_arxiv.models import Artifact, CitationMention, Paper
from briefgpt_arxiv.services.extractor import ExtractorService
from tests.helpers import FIXTURES, get_session, reset_database


class ApiTests(TestCase):
    def setUp(self) -> None:
        reset_database()
        self.original_openrouter_api_key = settings.openrouter_api_key
        self.original_gemini_api_key = settings.gemini_api_key
        settings.openrouter_api_key = None
        settings.gemini_api_key = None
        self.client = TestClient(app)
        self.session = get_session()
        self.tempdir = TemporaryDirectory()

    def tearDown(self) -> None:
        settings.openrouter_api_key = self.original_openrouter_api_key
        settings.gemini_api_key = self.original_gemini_api_key
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

        llm_client = TestClientLLM()
        ExtractorService(self.session, llm_client=llm_client).extract_for_paper(paper.id)

        paper_response = self.client.get("/papers/2603.15726v1")
        refs_response = self.client.get("/papers/2603.15726v1/references")
        search_response = self.client.get("/citations/search", params={"intent": "method_basis"})

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
        original_openrouter_api_key = settings.openrouter_api_key
        original_gemini_api_key = settings.gemini_api_key
        settings.openrouter_api_key = None
        settings.gemini_api_key = None
        try:
            paper = Paper(arxiv_id="2603.15726", version="v1", title="Sample", abstract="A", ingest_status="parsed")
            self.session.add(paper)
            self.session.commit()

            response = self.client.post(f"/extract/{paper.id}")

            self.assertEqual(503, response.status_code)
            self.assertIn("provider", response.json()["detail"])
        finally:
            settings.openrouter_api_key = original_openrouter_api_key
            settings.gemini_api_key = original_gemini_api_key


class TestClientLLM:
    model_name = "fake-llm"

    def generate_json(self, system_instruction: str, user_text: str) -> dict:
        candidates_block = re.search(r"### Candidates\s+```json\s+(.*?)\s+```", user_text, re.DOTALL)
        candidates_text = candidates_block.group(1) if candidates_block else user_text
        mention_orders = sorted({int(value) for value in re.findall(r'"mention_order":\s*(\d+)', candidates_text)})
        return {
            "items": [
                {
                    "mention_order": mention_order,
                    "intent_label": "method_basis" if mention_order == 0 else "comparison",
                    "summary": (
                        "ReAct is used as a planning baseline and method reference."
                        if mention_order == 0
                        else "TravelPlanner is used as a comparison baseline."
                    ),
                }
                for mention_order in mention_orders
            ]
        }
