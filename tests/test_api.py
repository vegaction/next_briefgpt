from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from briefgpt_arxiv.models import Artifact, Base, CitationMention, Paper
from briefgpt_arxiv.services.extractor import ExtractorService
from tests.conftest import FIXTURES, override_settings


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


@pytest.fixture()
def api_env(tmp_path, no_llm_keys):
    """Provide a TestClient + session backed by a file-based SQLite (thread-safe for TestClient)."""
    db_path = tmp_path / "test_api.db"
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    TestSession = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)

    from briefgpt_arxiv.db import get_db
    from briefgpt_arxiv.main import app

    def _override_get_db():
        session = TestSession()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = _override_get_db
    client = TestClient(app)
    session = TestSession()
    yield client, session
    session.close()
    app.dependency_overrides.clear()
    engine.dispose()


def test_parse_extract_and_query_endpoints(api_env, tmp_path) -> None:
    client, session = api_env
    paper = Paper(arxiv_id="2603.15726", version="v1", title="Sample", abstract="A", ingest_status="fetched")
    session.add(paper)
    session.flush()
    paper.parse_status = "pending"
    structured_path = tmp_path / "structured.json"
    structured_path.write_text((FIXTURES / "doc2json_sample.json").read_text())
    session.add(Artifact(paper_id=paper.id, artifact_type="structured_parse", uri=str(structured_path)))
    session.commit()

    parse_response = client.post(f"/parse/{paper.id}")
    assert parse_response.status_code == 200
    assert parse_response.json()["citation_blocks_created"] == 2

    llm_client = TestClientLLM()
    ExtractorService(session, llm_client=llm_client).extract_for_paper(paper.id)

    paper_response = client.get("/papers/2603.15726v1")
    refs_response = client.get("/papers/2603.15726v1/references")
    search_response = client.get("/citations/search", params={"intent": "method_basis"})

    assert paper_response.status_code == 200
    assert refs_response.status_code == 200
    assert search_response.status_code == 200
    assert paper_response.json()["arxiv_id"] == "2603.15726"
    assert paper_response.json()["version"] == "v1"
    assert paper_response.json()["parse_status"] == "parsed"
    assert len(refs_response.json()) >= 3
    assert session.query(CitationMention).count() >= 1
    first_mention = refs_response.json()[0]["mentions"][0]
    assert "section_title" in first_mention
    assert "cited_arxiv_id" in refs_response.json()[0]
    extraction = first_mention["extraction"]
    if extraction is not None:
        assert "evidence_text" not in extraction
        assert "confidence" not in extraction
    first_search = search_response.json()[0]
    assert first_search["paper_arxiv_id"] == "2603.15726"
    assert first_search["paper_version"] == "v1"


def test_extract_endpoint_returns_503_without_llm_configuration(api_env) -> None:
    client, session = api_env
    with override_settings(openrouter_api_key=None, gemini_api_key=None):
        paper = Paper(arxiv_id="2603.15726", version="v1", title="Sample", abstract="A", ingest_status="parsed")
        session.add(paper)
        session.commit()

        response = client.post(f"/extract/{paper.id}")

        assert response.status_code == 503
        assert "provider" in response.json()["detail"]
