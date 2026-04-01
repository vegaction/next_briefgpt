from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

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
from tests.conftest import override_settings


@pytest.fixture()
def extractor_paper(db_session):
    paper = Paper(arxiv_id="2603.15726", version="v1", title="Sample", abstract="A", ingest_status="parsed")
    db_session.add(paper)
    db_session.flush()
    paper.parse_status = "parsed"
    db_session.flush()
    block_text = "Recent works such as ReAct BIBREF0 and TravelPlanner BIBREF1 demonstrate strong planning behavior."
    db_session.add_all(
        [
            PaperReference(paper_id=paper.id, local_ref_id="BIBREF0", title="ReAct"),
            PaperReference(paper_id=paper.id, local_ref_id="BIBREF1", title="TravelPlanner"),
        ]
    )
    db_session.flush()
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
    db_session.add(block)
    db_session.commit()
    return paper


def _build_mock_llm_client() -> Mock:
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


def test_extract_creates_mentions_and_extractions(db_session, extractor_paper) -> None:
    service = ExtractorService(db_session, llm_client=_build_mock_llm_client())
    mentions, extractions = service.extract_for_paper(extractor_paper.id)

    assert mentions == 2
    assert extractions == 2
    assert db_session.query(CitationMention).count() == 2
    mention = db_session.query(CitationMention).filter_by(intent_label="comparison").one()
    assert mention.model == "fake-llm"
    report_artifact = db_session.query(Artifact).filter_by(
        paper_id=extractor_paper.id, artifact_type="extract_report",
    ).one()
    report = json.loads(Path(report_artifact.uri).read_text())
    assert report["status"] == "ready"
    assert report["model_name"] == "fake-llm"
    assert report["blocks_seen"] == 1
    assert report["blocks_skipped_non_narrative"] == 0
    assert report["candidates_total"] == 2
    assert report["llm_call_count"] == 1
    assert report["mentions_created"] == 2
    job = db_session.query(IngestionJob).one()
    assert job.status == "completed"
    assert job.attempt_count == 1
    assert job.finished_at is not None


def test_requires_llm_client_when_no_explicit_client_is_provided(db_session, extractor_paper) -> None:
    with override_settings(openrouter_api_key=None, gemini_api_key=None):
        with pytest.raises(ExtractionConfigurationError):
            ExtractorService(db_session)


def test_extract_skip_reuses_existing_outputs_and_records_skipped_job(db_session, extractor_paper) -> None:
    service = ExtractorService(db_session, llm_client=_build_mock_llm_client())
    service.extract_for_paper(extractor_paper.id)

    result = service.extract_for_paper_result(extractor_paper.id, rerun=False)

    assert result.as_tuple() == (2, 2)
    assert result.status == "skipped"
    assert result.cleanup_performed is False
    jobs = db_session.query(IngestionJob).order_by(IngestionJob.id).all()
    assert [job.status for job in jobs] == ["completed", "skipped"]
    assert [job.attempt_count for job in jobs] == [1, 2]


def test_build_citation_candidates_derives_existing_fields_deterministically() -> None:
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

    assert candidates == [
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
    ]


def test_should_skip_extraction_block_for_tabular_benchmark_rows() -> None:
    raw_text = (
        "Qwen3.5-397B~ qwenteam2026qwen35 & 78.6 & 70.3 & 48.3 & -- & -- & 46.9 & -- \\\\ "
        "Tongyi-DeepResearch-30B~ team2025tongyi & 43.4 & 46.7 & 32.9 & 70.9 & 55.0 & -- & --"
    )
    assert should_skip_extraction_block(raw_text) is True


def test_should_not_skip_extraction_block_for_narrative_benchmark_discussion() -> None:
    raw_text = (
        "On GAIA mialon2023gaia , MiroThinker-H1 reaches 88.5 and surpasses OpenAI-GPT-5, "
        "showing strong multi-step reasoning performance in a narrative comparison."
    )
    assert should_skip_extraction_block(raw_text) is False


def test_extract_skips_non_narrative_tabular_blocks(db_session, extractor_paper) -> None:
    block = db_session.query(CitationBlock).one()
    block.section_title = "Implementation Details"
    block.raw_text = (
        "Qwen3.5-397B~ BIBREF0 & 78.6 & 70.3 & 48.3 & -- & -- & 46.9 & -- \\\\ "
        "TravelPlanner~ BIBREF1 & 43.4 & 46.7 & 32.9 & 70.9 & 55.0 & -- & --"
    )
    db_session.commit()

    service = ExtractorService(db_session, llm_client=_build_mock_llm_client())
    mentions, extractions = service.extract_for_paper(extractor_paper.id)

    assert mentions == 0
    assert extractions == 0
    assert db_session.query(CitationMention).count() == 0


def test_postprocess_extracted_summary_removes_author_year_lead_in() -> None:
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
    assert normalized == (
        "Tau-bench, a benchmark focusing on tool-agent-user interaction within real-world domains."
    )


def test_postprocess_extracted_summary_keeps_non_bibliographic_summary() -> None:
    summary = "Tau-bench is used as a benchmark for real-world tool-agent-user interaction."
    normalized = postprocess_extracted_summary(
        summary=summary,
        citation_mention="Tau-bench",
        sentence_text="",
        intent_label="benchmark_or_dataset",
    )
    assert normalized == summary


def test_postprocess_extracted_summary_removes_cited_wording() -> None:
    processed = postprocess_extracted_summary(
        summary=(
            "Introducing GPT-5.4 is cited as an example of a contemporary conversational LLM "
            "whose capabilities are insufficient for complex tasks."
        ),
        citation_mention="Introducing GPT-5.4",
        sentence_text="Recent advances require more than conversational ability.",
        intent_label="comparison",
    )
    assert "cited" not in processed.lower()
    assert "Introducing GPT-5.4 is an example" in processed


def test_postprocess_extracted_summary_replaces_raw_citation_key_and_here() -> None:
    processed = postprocess_extracted_summary(
        summary=(
            "liu2024rpo demonstrates using an SFT loss on preferred trajectories "
            "as an implicit adversarial regularizer, a technique adapted here for training stability."
        ),
        citation_mention="Provably mitigating overoptimization in rlhf: Your sft loss is implicitly an adversarial regularizer",
        sentence_text="We optimize the model using DPO combined with an auxiliary SFT loss.",
        intent_label="method_basis",
    )
    assert "liu2024rpo" not in processed
    assert "here" not in processed.lower()
    assert "Provably mitigating overoptimization in rlhf" in processed
    assert "describes an SFT loss" in processed


def test_postprocess_extracted_summary_drops_score_only_benchmark_summary() -> None:
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
    assert processed == ""


def test_postprocess_extracted_summary_drops_benchmark_sota_score_note() -> None:
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
    assert processed == ""


def test_llm_extraction_path_omits_reference_raw_text_from_prompt(db_session, extractor_paper) -> None:
    block = db_session.query(CitationBlock).one()
    block.raw_text = "ReAct BIBREF0 is discussed here."
    block.raw_citation_keys = ["BIBREF0"]
    ref = db_session.query(PaperReference).filter_by(local_ref_id="BIBREF0").one()
    ref.year = 2023
    db_session.commit()

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
    service = ExtractorService(db_session, llm_client=llm_client)
    service.extract_for_paper(extractor_paper.id)

    kwargs = llm_client.generate_json.call_args.kwargs
    assert '"title": "ReAct"' in kwargs["user_text"]
    assert '"year": 2023' in kwargs["user_text"]
    assert "This should not be sent" not in kwargs["user_text"]
    assert "## Output Format" in kwargs["user_text"]
    assert "Do not return any extra keys beyond the requested fields." in kwargs["user_text"]
    assert "The JSON object must contain:" in kwargs["user_text"]
    assert "`items`" in kwargs["user_text"]
    assert "`mention_order`" in kwargs["user_text"]
    assert "`intent_label`" in kwargs["user_text"]
    assert "`summary`" in kwargs["user_text"]
    assert "The JSON must match this schema exactly" not in kwargs["user_text"]


def test_llm_extraction_path_rejects_non_items_payload_shape(db_session, extractor_paper) -> None:
    block = db_session.query(CitationBlock).one()
    block.raw_text = "DPO BIBREF0 is used here."
    block.raw_citation_keys = ["BIBREF0"]
    ref = db_session.query(PaperReference).filter_by(local_ref_id="BIBREF0").one()
    ref.title = "DPO"
    ref.year = 2023
    db_session.commit()

    llm_client = Mock()
    llm_client.model_name = "mock-llm"
    llm_client.generate_json.return_value = {
        "mention_order": 0,
        "intent_label": "method_basis",
        "summary": "DPO directly optimizes from preferences without a separate reward model.",
    }
    service = ExtractorService(db_session, llm_client=llm_client)

    with pytest.raises(RuntimeError, match="unexpected top-level keys"):
        service.extract_for_paper(extractor_paper.id)


def test_llm_extraction_path_rejects_schema_wrapped_items_list(db_session, extractor_paper) -> None:
    block = db_session.query(CitationBlock).one()
    block.raw_text = "DPO BIBREF0 is used here."
    block.raw_citation_keys = ["BIBREF0"]
    ref = db_session.query(PaperReference).filter_by(local_ref_id="BIBREF0").one()
    ref.title = "DPO"
    ref.year = 2023
    db_session.commit()

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
    service = ExtractorService(db_session, llm_client=llm_client)

    with pytest.raises(RuntimeError, match="unexpected top-level keys"):
        service.extract_for_paper(extractor_paper.id)


def test_llm_extraction_path_accepts_items_list_payload(db_session, extractor_paper) -> None:
    block = db_session.query(CitationBlock).one()
    block.raw_text = "GPT-5.4 BIBREF0 is discussed here."
    block.raw_citation_keys = ["BIBREF0"]
    ref = db_session.query(PaperReference).filter_by(local_ref_id="BIBREF0").one()
    ref.title = "GPT-5.4"
    ref.year = 2026
    db_session.commit()

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
    service = ExtractorService(db_session, llm_client=llm_client)

    extracted = service.extract_for_paper(extractor_paper.id)

    assert extracted == (1, 1)
    mention = db_session.query(CitationMention).one()
    assert mention.intent_label == "supporting_evidence"


def test_llm_extraction_path_skips_llm_call_when_candidates_are_empty(db_session, extractor_paper) -> None:
    block = db_session.query(CitationBlock).one()
    block.raw_text = ""
    block.raw_citation_keys = []
    db_session.commit()

    llm_client = Mock()
    llm_client.model_name = "mock-llm"
    service = ExtractorService(db_session, llm_client=llm_client)

    extracted = service.extract_for_paper(extractor_paper.id)

    assert extracted == (0, 0)
    llm_client.generate_json.assert_not_called()
