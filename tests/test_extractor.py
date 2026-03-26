from __future__ import annotations

from unittest import TestCase

from briefgpt_arxiv.config import settings
from briefgpt_arxiv.models import (
    CitationBlock,
    CitationMention,
    IngestionJob,
    Paper,
    PaperReference,
)
from briefgpt_arxiv.services.extractor import (
    BaseExtractionClient,
    CitationCandidate,
    ExtractedCitation,
    ExtractionConfigurationError,
    ExtractorService,
    build_citation_candidates,
    normalize_extracted_summary,
)
from tests.helpers import get_session, reset_database


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
                    intent_label="support",
                    summary="Uses ReAct as evidence for strong planning behavior.",
                ),
                ExtractedCitation(
                    raw_citation_key=candidates[1].raw_citation_key,
                    citation_mention=candidates[1].citation_mention,
                    sentence_text=candidates[1].sentence_text,
                    section_title=candidates[1].section_title,
                    mention_order=candidates[1].mention_order,
                    intent_label="comparison",
                    summary="Uses TravelPlanner as a comparison baseline.",
                ),
            ]


class ExtractorServiceTests(TestCase):
    def setUp(self) -> None:
        reset_database()
        self.session = get_session()
        paper = Paper(arxiv_id="2603.15726", version="v1", title="Sample", abstract="A", ingest_status="parsed")
        self.session.add(paper)
        self.session.flush()
        paper.parse_status = "parsed"
        self.session.flush()
        block_text = "Recent works such as ReAct BIBREF0 and TravelPlanner BIBREF1 demonstrate strong planning behavior."
        self.session.add_all(
            [
                PaperReference(
                    paper_id=paper.id,
                    local_ref_id="BIBREF0",
                    raw_text="ReAct raw text",
                    title="ReAct",
                ),
                PaperReference(
                    paper_id=paper.id,
                    local_ref_id="BIBREF1",
                    raw_text="TravelPlanner raw text",
                    title="TravelPlanner",
                ),
            ]
        )
        self.session.flush()
        block = CitationBlock(
            paper_id=paper.id,
            section_title="Introduction",
            section_path="Introduction",
            chunk_index=0,
            raw_text=block_text,
            raw_citation_keys=["BIBREF0", "BIBREF1"],
            has_citations=True,
            repair_used=False,
        )
        self.session.add(block)
        self.session.commit()
        self.paper_id = paper.id

    def tearDown(self) -> None:
        self.session.close()

    def test_extract_creates_mentions_and_extractions(self) -> None:
        service = ExtractorService(self.session, client=FakeExtractionClient())
        mentions, extractions = service.extract_for_paper(self.paper_id)

        self.assertEqual(2, mentions)
        self.assertEqual(2, extractions)
        self.assertEqual(2, self.session.query(CitationMention).count())
        mention = self.session.query(CitationMention).filter_by(intent_label="comparison").one()
        self.assertEqual("fake-llm", mention.model)
        job = self.session.query(IngestionJob).one()
        self.assertEqual("completed", job.status)
        self.assertEqual(1, job.attempt_count)
        self.assertIsNotNone(job.finished_at)

    def test_requires_llm_client_when_no_explicit_client_is_provided(self) -> None:
        original_api_key = settings.gemini_api_key
        settings.gemini_api_key = None
        try:
            with self.assertRaises(ExtractionConfigurationError):
                ExtractorService(self.session)
        finally:
            settings.gemini_api_key = original_api_key

    def test_extract_skip_reuses_existing_outputs_and_records_skipped_job(self) -> None:
        service = ExtractorService(self.session, client=FakeExtractionClient())
        service.extract_for_paper(self.paper_id)

        result = service.extract_for_paper_result(self.paper_id, rerun=False)

        self.assertEqual((2, 2), result.as_tuple())
        self.assertEqual("skipped", result.status)
        self.assertFalse(result.cleanup_performed)
        jobs = self.session.query(IngestionJob).order_by(IngestionJob.id).all()
        self.assertEqual(["completed", "skipped"], [job.status for job in jobs])
        self.assertEqual([1, 2], [job.attempt_count for job in jobs])

    def test_build_citation_candidates_derives_existing_fields_deterministically(self) -> None:
        candidates = build_citation_candidates(
            raw_text=(
                "Recent works such as ReAct BIBREF0 demonstrate strong planning behavior. "
                "TravelPlanner BIBREF1 is used for comparison."
            ),
            section_title="Introduction",
            raw_citation_keys=["BIBREF0", "BIBREF1"],
            references={
                "BIBREF0": {"title": "ReAct"},
                "BIBREF1": {"title": "TravelPlanner"},
            },
        )

        self.assertEqual(
            [
                CitationCandidate(
                    raw_citation_key="BIBREF0",
                    citation_mention="ReAct",
                    sentence_text="Recent works such as ReAct BIBREF0 demonstrate strong planning behavior.",
                    section_title="Introduction",
                    mention_order=0,
                ),
                CitationCandidate(
                    raw_citation_key="BIBREF1",
                    citation_mention="TravelPlanner",
                    sentence_text="TravelPlanner BIBREF1 is used for comparison.",
                    section_title="Introduction",
                    mention_order=1,
                ),
            ],
            candidates,
        )

    def test_normalize_extracted_summary_removes_author_year_lead_in(self) -> None:
        summary = (
            "Yao et al. (2024) introduce Tau-bench, a benchmark focusing on "
            "tool-agent-user interaction within real-world domains."
        )

        normalized = normalize_extracted_summary(
            summary=summary,
            citation_mention="Tau-bench",
        )

        self.assertEqual(
            "Tau-bench, a benchmark focusing on tool-agent-user interaction within real-world domains.",
            normalized,
        )

    def test_normalize_extracted_summary_keeps_non_bibliographic_summary(self) -> None:
        summary = "Tau-bench is used as a benchmark for real-world tool-agent-user interaction."

        normalized = normalize_extracted_summary(
            summary=summary,
            citation_mention="Tau-bench",
        )

        self.assertEqual(summary, normalized)
