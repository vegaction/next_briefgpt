from __future__ import annotations

import json
from pathlib import Path
from unittest import TestCase
from unittest.mock import Mock

from briefgpt_arxiv.config import settings
from briefgpt_arxiv.models import (
    Artifact,
    CitationBlock,
    CitationMention,
    IngestionJob,
    Paper,
    PaperReference,
)
from briefgpt_arxiv.services.extractor import (
    CitationCandidate,
    ExtractionConfigurationError,
    ExtractorService,
    build_citation_candidates,
    postprocess_extracted_summary,
    should_skip_extraction_block,
)
from tests.helpers import get_session, reset_database


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
                    title="ReAct",
                ),
                PaperReference(
                    paper_id=paper.id,
                    local_ref_id="BIBREF1",
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

    def _build_mock_llm_client(self) -> Mock:
        llm_client = Mock()
        llm_client.model_name = "fake-llm"
        llm_client.generate_json.return_value = {
            "items": [
                {
                    "mention_order": 0,
                    "intent_label": "supporting_evidence",
                    "summary": "Uses ReAct as evidence for strong planning behavior.",
                },
                {
                    "mention_order": 1,
                    "intent_label": "comparison",
                    "summary": "Uses TravelPlanner as a comparison baseline.",
                },
            ]
        }
        return llm_client

    def test_extract_creates_mentions_and_extractions(self) -> None:
        service = ExtractorService(self.session, llm_client=self._build_mock_llm_client())
        mentions, extractions = service.extract_for_paper(self.paper_id)

        self.assertEqual(2, mentions)
        self.assertEqual(2, extractions)
        self.assertEqual(2, self.session.query(CitationMention).count())
        mention = self.session.query(CitationMention).filter_by(intent_label="comparison").one()
        self.assertEqual("fake-llm", mention.model)
        report_artifact = self.session.query(Artifact).filter_by(
            paper_id=self.paper_id,
            artifact_type="extract_report",
        ).one()
        report = json.loads(Path(report_artifact.uri).read_text())
        self.assertEqual("ready", report["status"])
        self.assertEqual("fake-llm", report["model_name"])
        self.assertEqual(1, report["blocks_seen"])
        self.assertEqual(0, report["blocks_skipped_non_narrative"])
        self.assertEqual(2, report["candidates_total"])
        self.assertEqual(1, report["llm_call_count"])
        self.assertEqual(2, report["mentions_created"])
        job = self.session.query(IngestionJob).one()
        self.assertEqual("completed", job.status)
        self.assertEqual(1, job.attempt_count)
        self.assertIsNotNone(job.finished_at)

    def test_requires_llm_client_when_no_explicit_client_is_provided(self) -> None:
        original_openrouter_api_key = settings.openrouter_api_key
        original_gemini_api_key = settings.gemini_api_key
        settings.openrouter_api_key = None
        settings.gemini_api_key = None
        try:
            with self.assertRaises(ExtractionConfigurationError):
                ExtractorService(self.session)
        finally:
            settings.openrouter_api_key = original_openrouter_api_key
            settings.gemini_api_key = original_gemini_api_key

    def test_extract_skip_reuses_existing_outputs_and_records_skipped_job(self) -> None:
        service = ExtractorService(self.session, llm_client=self._build_mock_llm_client())
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

    def test_should_skip_extraction_block_for_tabular_benchmark_rows(self) -> None:
        raw_text = (
            "Qwen3.5-397B~ qwenteam2026qwen35 & 78.6 & 70.3 & 48.3 & -- & -- & 46.9 & -- \\\\ "
            "Tongyi-DeepResearch-30B~ team2025tongyi & 43.4 & 46.7 & 32.9 & 70.9 & 55.0 & -- & --"
        )

        self.assertTrue(should_skip_extraction_block(raw_text))

    def test_should_not_skip_extraction_block_for_narrative_benchmark_discussion(self) -> None:
        raw_text = (
            "On GAIA mialon2023gaia , MiroThinker-H1 reaches 88.5 and surpasses OpenAI-GPT-5, "
            "showing strong multi-step reasoning performance in a narrative comparison."
        )

        self.assertFalse(should_skip_extraction_block(raw_text))

    def test_extract_skips_non_narrative_tabular_blocks(self) -> None:
        block = self.session.query(CitationBlock).one()
        block.section_title = "Implementation Details"
        block.raw_text = (
            "Qwen3.5-397B~ BIBREF0 & 78.6 & 70.3 & 48.3 & -- & -- & 46.9 & -- \\\\ "
            "TravelPlanner~ BIBREF1 & 43.4 & 46.7 & 32.9 & 70.9 & 55.0 & -- & --"
        )
        self.session.commit()

        service = ExtractorService(self.session, llm_client=self._build_mock_llm_client())
        mentions, extractions = service.extract_for_paper(self.paper_id)

        self.assertEqual(0, mentions)
        self.assertEqual(0, extractions)
        self.assertEqual(0, self.session.query(CitationMention).count())

    def test_postprocess_extracted_summary_removes_author_year_lead_in(self) -> None:
        summary = (
            "Yao et al. (2024) introduce Tau-bench, a benchmark focusing on "
            "tool-agent-user interaction within real-world domains."
        )

        normalized = postprocess_extracted_summary(
            summary=summary,
            citation_mention="Tau-bench",
            sentence_text="",
            intent_label="benchmark_or_dataset",
        )

        self.assertEqual(
            "Tau-bench, a benchmark focusing on tool-agent-user interaction within real-world domains.",
            normalized,
        )

    def test_postprocess_extracted_summary_keeps_non_bibliographic_summary(self) -> None:
        summary = "Tau-bench is used as a benchmark for real-world tool-agent-user interaction."

        normalized = postprocess_extracted_summary(
            summary=summary,
            citation_mention="Tau-bench",
            sentence_text="",
            intent_label="benchmark_or_dataset",
        )

        self.assertEqual(summary, normalized)

    def test_postprocess_extracted_summary_removes_cited_wording(self) -> None:
        processed = postprocess_extracted_summary(
            summary=(
                "Introducing GPT-5.4 is cited as an example of a contemporary conversational LLM "
                "whose capabilities are insufficient for complex tasks."
            ),
            citation_mention="Introducing GPT-5.4",
            sentence_text="Recent advances require more than conversational ability.",
            intent_label="comparison",
        )

        self.assertNotIn("cited", processed.lower())
        self.assertIn("Introducing GPT-5.4 is an example", processed)

    def test_postprocess_extracted_summary_replaces_raw_citation_key_and_here(self) -> None:
        processed = postprocess_extracted_summary(
            summary=(
                "liu2024rpo demonstrates using an SFT loss on preferred trajectories "
                "as an implicit adversarial regularizer, a technique adapted here for training stability."
            ),
            citation_mention="Provably mitigating overoptimization in rlhf: Your sft loss is implicitly an adversarial regularizer",
            sentence_text="We optimize the model using DPO combined with an auxiliary SFT loss.",
            intent_label="method_basis",
        )

        self.assertNotIn("liu2024rpo", processed)
        self.assertNotIn("here", processed.lower())
        self.assertIn("Provably mitigating overoptimization in rlhf", processed)
        self.assertIn("describes an SFT loss", processed)

    def test_postprocess_extracted_summary_drops_score_only_benchmark_summary(self) -> None:
        processed = postprocess_extracted_summary(
            summary=(
                "DeepSearchQA bridges the comprehensiveness gap for deep research agents; "
                "MiroThinker-H1 scores 80.6 on this benchmark."
            ),
            citation_mention="DeepSearchQA: Bridging the Comprehensiveness Gap for Deep Research Agents",
            sentence_text=(
                "MiroThinker-H1 scores 80.6 on DeepSearchQA and sets strong benchmark results."
            ),
            intent_label="benchmark_or_dataset",
        )

        self.assertEqual("", processed)

    def test_postprocess_extracted_summary_drops_benchmark_sota_score_note(self) -> None:
        processed = postprocess_extracted_summary(
            summary=(
                "SEAL-0 raises the bar for reasoning in search-augmented language models; "
                "MiroThinker-H1 achieves 61.3, setting a new best result among all evaluated models."
            ),
            citation_mention="SealQA: Raising the Bar for Reasoning in Search-Augmented Language Models",
            sentence_text=(
                "MiroThinker-H1 achieves 61.3 on SEAL-0, setting a new best result among all evaluated models."
            ),
            intent_label="benchmark_or_dataset",
        )

        self.assertEqual("", processed)

    def test_llm_extraction_path_omits_reference_raw_text_from_prompt(self) -> None:
        block = self.session.query(CitationBlock).one()
        block.raw_text = "ReAct BIBREF0 is discussed here."
        block.raw_citation_keys = ["BIBREF0"]
        ref = self.session.query(PaperReference).filter_by(local_ref_id="BIBREF0").one()
        ref.year = 2023
        self.session.commit()

        llm_client = Mock()
        llm_client.model_name = "mock-llm"
        llm_client.generate_json.return_value = {
            "items": [
                {
                    "mention_order": 0,
                    "intent_label": "benchmark_or_dataset",
                    "summary": "ReAct is a planning benchmark.",
                }
            ]
        }
        service = ExtractorService(self.session, llm_client=llm_client)
        service.extract_for_paper(self.paper_id)

        kwargs = llm_client.generate_json.call_args.kwargs
        self.assertIn('"title": "ReAct"', kwargs["user_text"])
        self.assertIn('"year": 2023', kwargs["user_text"])
        self.assertNotIn("This should not be sent", kwargs["user_text"])
        self.assertIn("## Output Format", kwargs["user_text"])
        self.assertIn("Do not return any extra keys beyond the requested fields.", kwargs["user_text"])
        self.assertIn("The JSON object must contain:", kwargs["user_text"])
        self.assertIn("`items`", kwargs["user_text"])
        self.assertIn("`mention_order`", kwargs["user_text"])
        self.assertIn("`intent_label`", kwargs["user_text"])
        self.assertIn("`summary`", kwargs["user_text"])
        self.assertNotIn("The JSON must match this schema exactly", kwargs["user_text"])

    def test_llm_extraction_path_rejects_non_items_payload_shape(self) -> None:
        block = self.session.query(CitationBlock).one()
        block.raw_text = "DPO BIBREF0 is used here."
        block.raw_citation_keys = ["BIBREF0"]
        ref = self.session.query(PaperReference).filter_by(local_ref_id="BIBREF0").one()
        ref.title = "DPO"
        ref.year = 2023
        self.session.commit()

        llm_client = Mock()
        llm_client.model_name = "mock-llm"
        llm_client.generate_json.return_value = {
            "mention_order": 0,
            "intent_label": "method_basis",
            "summary": "DPO directly optimizes from preferences without a separate reward model.",
        }
        service = ExtractorService(self.session, llm_client=llm_client)

        with self.assertRaisesRegex(RuntimeError, "unexpected top-level keys"):
            service.extract_for_paper(self.paper_id)

    def test_llm_extraction_path_rejects_schema_wrapped_items_list(self) -> None:
        block = self.session.query(CitationBlock).one()
        block.raw_text = "DPO BIBREF0 is used here."
        block.raw_citation_keys = ["BIBREF0"]
        ref = self.session.query(PaperReference).filter_by(local_ref_id="BIBREF0").one()
        ref.title = "DPO"
        ref.year = 2023
        self.session.commit()

        llm_client = Mock()
        llm_client.model_name = "mock-llm"
        llm_client.generate_json.return_value = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": [
                        {
                            "mention_order": 0,
                            "intent_label": "method_basis",
                            "summary": "DPO directly optimizes from preferences without a separate reward model.",
                        }
                    ],
                }
            },
            "required": ["items"],
            "additionalProperties": False,
        }
        service = ExtractorService(self.session, llm_client=llm_client)

        with self.assertRaisesRegex(RuntimeError, "unexpected top-level keys"):
            service.extract_for_paper(self.paper_id)

    def test_llm_extraction_path_accepts_items_list_payload(self) -> None:
        block = self.session.query(CitationBlock).one()
        block.raw_text = "GPT-5.4 BIBREF0 is discussed here."
        block.raw_citation_keys = ["BIBREF0"]
        ref = self.session.query(PaperReference).filter_by(local_ref_id="BIBREF0").one()
        ref.title = "GPT-5.4"
        ref.year = 2026
        self.session.commit()

        llm_client = Mock()
        llm_client.model_name = "mock-llm"
        llm_client.generate_json.return_value = {
            "items": [
                {
                    "mention_order": 0,
                    "intent_label": "supporting_evidence",
                    "summary": "GPT-5.4 is cited as evidence of recent LLM progress.",
                }
            ]
        }
        service = ExtractorService(self.session, llm_client=llm_client)

        extracted = service.extract_for_paper(self.paper_id)

        self.assertEqual((1, 1), extracted)
        mention = self.session.query(CitationMention).one()
        self.assertEqual("supporting_evidence", mention.intent_label)

    def test_llm_extraction_path_skips_llm_call_when_candidates_are_empty(self) -> None:
        block = self.session.query(CitationBlock).one()
        block.raw_text = ""
        block.raw_citation_keys = []
        self.session.commit()

        llm_client = Mock()
        llm_client.model_name = "mock-llm"
        service = ExtractorService(self.session, llm_client=llm_client)

        extracted = service.extract_for_paper(self.paper_id)

        self.assertEqual((0, 0), extracted)
        llm_client.generate_json.assert_not_called()
