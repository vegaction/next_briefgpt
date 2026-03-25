from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import logging

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from briefgpt_arxiv.config import settings
from briefgpt_arxiv.gemini import GeminiClient
from briefgpt_arxiv.models import (
    CitationBlock,
    CitationMention,
    Paper,
    PaperReference,
)
from briefgpt_arxiv.prompts import (
    EXTRACTION_JSON_SCHEMA,
    EXTRACTOR_PROMPT_VERSION,
    EXTRACTION_SYSTEM_TEMPLATE,
    EXTRACTION_USER_TEMPLATE,
    PROMPT_TEMPLATE_ENV,
)
from briefgpt_arxiv.services.contracts import ExtractionRunResult
from briefgpt_arxiv.services.jobs import JobTracker
from briefgpt_arxiv.utils import normalize_whitespace, split_sentences

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CitationCandidate:
    raw_citation_key: str
    citation_mention: str
    sentence_text: str
    section_title: str | None
    mention_order: int


@dataclass(slots=True)
class ExtractedCitation(CitationCandidate):
    intent_label: str
    summary: str


@dataclass(slots=True)
class CitationAnnotation:
    mention_order: int
    intent_label: str
    summary: str


class ExtractionConfigurationError(RuntimeError):
    """Raised when extraction is requested without a configured LLM client."""


class BaseExtractionClient:
    model_name = "extraction-client"

    def annotate_candidates(
        self,
        candidates: list[CitationCandidate],
        raw_text: str,
        section_title: str | None,
        references: dict[str, dict],
        debug_context: dict | None = None,
    ) -> list[ExtractedCitation]:
        raise NotImplementedError


class GeminiExtractionClient(BaseExtractionClient):
    def __init__(self) -> None:
        self.client = GeminiClient()
        self.model_name = settings.gemini_model

    def annotate_candidates(
        self,
        candidates: list[CitationCandidate],
        raw_text: str,
        section_title: str | None,
        references: dict[str, dict],
        debug_context: dict | None = None,
    ) -> list[ExtractedCitation]:
        prompt_candidates = [
            {
                "mention_order": candidate.mention_order,
                "raw_citation_key": candidate.raw_citation_key,
                "citation_mention": candidate.citation_mention,
                "sentence_text": candidate.sentence_text,
                "reference": {
                    "title": references.get(candidate.raw_citation_key, {}).get("title"),
                    "year": references.get(candidate.raw_citation_key, {}).get("year"),
                    "raw_text": references.get(candidate.raw_citation_key, {}).get("raw_text"),
                },
            }
            for candidate in candidates
        ]
        system_instruction = PROMPT_TEMPLATE_ENV.from_string(EXTRACTION_SYSTEM_TEMPLATE).render()
        user_text = PROMPT_TEMPLATE_ENV.from_string(EXTRACTION_USER_TEMPLATE).render(
            raw_text=raw_text,
            section_title=section_title,
            candidates=prompt_candidates,
        )
        self._append_summary_debug_log(
            event="summary_input",
            debug_context=debug_context,
            payload={
                "model_name": self.model_name,
                "prompt_version": EXTRACTOR_PROMPT_VERSION,
                "system_instruction": system_instruction,
                "user_text": user_text,
                "response_json_schema": EXTRACTION_JSON_SCHEMA,
                "prompt_candidates": prompt_candidates,
            },
        )
        try:
            payload = self.client.generate_json(
                system_instruction=system_instruction,
                user_text=user_text,
                response_json_schema=EXTRACTION_JSON_SCHEMA,
            )
        except Exception as exc:
            self._append_summary_debug_log(
                event="summary_error",
                debug_context=debug_context,
                payload={
                    "model_name": self.model_name,
                    "prompt_version": EXTRACTOR_PROMPT_VERSION,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            )
            raise
        self._append_summary_debug_log(
            event="summary_output",
            debug_context=debug_context,
            payload={
                "model_name": self.model_name,
                "prompt_version": EXTRACTOR_PROMPT_VERSION,
                "response_items": payload.get("items", []),
                "response_payload": payload,
            },
        )
        items = payload.get("items", [])
        annotations = [CitationAnnotation(**item) for item in items]
        self._validate_annotations(candidates, annotations)
        annotations_by_order = {item.mention_order: item for item in annotations}
        return [
            ExtractedCitation(
                raw_citation_key=candidate.raw_citation_key,
                citation_mention=candidate.citation_mention,
                sentence_text=candidate.sentence_text,
                section_title=candidate.section_title,
                mention_order=candidate.mention_order,
                intent_label=annotations_by_order[candidate.mention_order].intent_label,
                summary=annotations_by_order[candidate.mention_order].summary,
            )
            for candidate in candidates
        ]

    @staticmethod
    def _validate_annotations(
        candidates: list[CitationCandidate],
        annotations: list[CitationAnnotation],
    ) -> None:
        expected_orders = [candidate.mention_order for candidate in candidates]
        seen_orders = [item.mention_order for item in annotations]
        duplicate_orders = {order for order in seen_orders if seen_orders.count(order) > 1}
        missing_orders = sorted(set(expected_orders) - set(seen_orders))
        unexpected_orders = sorted(set(seen_orders) - set(expected_orders))
        if duplicate_orders or missing_orders or unexpected_orders:
            raise RuntimeError(
                "Extractor returned mismatched annotations: "
                f"duplicate_orders={sorted(duplicate_orders)} "
                f"missing_orders={missing_orders} "
                f"unexpected_orders={unexpected_orders}"
            )

    @staticmethod
    def _append_summary_debug_log(*, event: str, debug_context: dict | None, payload: dict) -> None:
        path = settings.summary_debug_log_path
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **(debug_context or {}),
            **payload,
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


class ExtractorService:
    def __init__(self, session: Session, client: BaseExtractionClient | None = None):
        if client is None:
            if not settings.gemini_api_key:
                raise ExtractionConfigurationError(
                    "Extraction requires GEMINI_API_KEY; heuristic summary fallback has been disabled."
                )
            client = GeminiExtractionClient()
        self.session = session
        self.client = client
        self.job_tracker = JobTracker(session)

    def extract_for_paper(self, paper_id: int) -> tuple[int, int]:
        return self.extract_for_paper_result(paper_id).as_tuple()

    def extract_for_paper_result(self, paper_id: int, *, rerun: bool = True) -> ExtractionRunResult:
        paper = self.session.get(Paper, paper_id)
        if paper is None:
            raise ValueError(f"Unknown paper id {paper_id}")
        existing_mentions = self._count_mentions(paper_id)
        if not rerun and existing_mentions:
            result = self._build_extraction_result(
                paper=paper,
                mentions_created=existing_mentions,
                extractions_created=existing_mentions,
                cleanup_performed=False,
                status="skipped",
            )
            job = self.job_tracker.start(job_type="extract", target_id=paper_id)
            self.job_tracker.finish(job, status="skipped", error_message="Reused existing extraction outputs.")
            self.session.commit()
            return result

        job = self.job_tracker.start(job_type="extract", target_id=paper_id)
        try:
            cleanup_performed = bool(existing_mentions)
            if cleanup_performed:
                self.clear_extractions(paper_id)
            blocks = list(
                self.session.scalars(
                    select(CitationBlock)
                    .where(CitationBlock.paper_id == paper_id, CitationBlock.has_citations.is_(True))
                )
            )
            reference_rows = list(
                self.session.scalars(select(PaperReference).where(PaperReference.paper_id == paper_id))
            )
            reference_map = {
                row.local_ref_id: {
                    "paper_reference_id": row.id,
                    "title": row.title,
                    "raw_text": row.raw_text,
                    "year": row.year,
                }
                for row in reference_rows
            }
            mentions_created = 0
            extractions_created = 0
            for block in blocks:
                candidates = build_citation_candidates(
                    raw_text=block.raw_text,
                    section_title=block.section_title,
                    raw_citation_keys=block.raw_citation_keys,
                    references=reference_map,
                )
                extracted = self.client.annotate_candidates(
                    candidates=candidates,
                    raw_text=block.raw_text,
                    section_title=block.section_title,
                    references=reference_map,
                    debug_context={
                        "paper_id": paper_id,
                        "block_id": block.id,
                        "section_title": block.section_title,
                        "raw_citation_keys": block.raw_citation_keys,
                        "raw_text": block.raw_text,
                        "candidates": [asdict(candidate) for candidate in candidates],
                    },
                )
                logger.info(
                    "Extracted %s candidate summaries for paper_id=%s block_id=%s section=%r model=%s",
                    len(extracted),
                    paper_id,
                    block.id,
                    block.section_title,
                    self.client.model_name,
                )
                for item in extracted:
                    if item.raw_citation_key not in reference_map:
                        logger.warning(
                            "Skipping extracted summary for unknown citation key paper_id=%s block_id=%s key=%s summary=%r",
                            paper_id,
                            block.id,
                            item.raw_citation_key,
                            item.summary,
                        )
                        continue
                    logger.info(
                        "Generated summary paper_id=%s block_id=%s key=%s section=%r summary=%r",
                        paper_id,
                        block.id,
                        item.raw_citation_key,
                        item.section_title,
                        item.summary,
                    )
                    mention = CitationMention(
                        citation_block_id=block.id,
                        paper_reference_id=reference_map[item.raw_citation_key]["paper_reference_id"],
                        citation_mention=item.citation_mention,
                        sentence_text=item.sentence_text,
                        mention_order=item.mention_order,
                        model=self.client.model_name,
                        prompt_version=EXTRACTOR_PROMPT_VERSION,
                        intent_label=item.intent_label,
                        summary=item.summary,
                        json_result={
                            "raw_citation_key": item.raw_citation_key,
                            "citation_mention": item.citation_mention,
                            "sentence_text": item.sentence_text,
                            "section_title": item.section_title,
                            "mention_order": item.mention_order,
                            "intent_label": item.intent_label,
                            "summary": item.summary,
                        },
                        status="completed",
                    )
                    self.session.add(mention)
                    mentions_created += 1
                    extractions_created += 1
            paper.ingest_status = "ready"
            self.job_tracker.finish(job)
            self.session.commit()
            return self._build_extraction_result(
                paper=paper,
                mentions_created=mentions_created,
                extractions_created=extractions_created,
                cleanup_performed=cleanup_performed,
                status="ready",
            )
        except Exception as exc:
            self.session.rollback()
            self.job_tracker.record_failure(job_type="extract", target_id=paper_id, error_message=str(exc))
            self.session.commit()
            raise

    def clear_extractions(self, paper_id: int) -> None:
        paper = self.session.get(Paper, paper_id)
        if paper is None:
            raise ValueError(f"Unknown paper id {paper_id}")
        self._clear_existing_extractions(paper_id)
        if paper.ingest_status == "ready":
            paper.ingest_status = "parsed"

    def _clear_existing_extractions(self, paper_id: int) -> None:
        block_ids = list(self.session.scalars(select(CitationBlock.id).where(CitationBlock.paper_id == paper_id)))
        if not block_ids:
            return
        mention_ids = list(
            self.session.scalars(select(CitationMention.id).where(CitationMention.citation_block_id.in_(block_ids)))
        )
        if mention_ids:
            self.session.execute(delete(CitationMention).where(CitationMention.id.in_(mention_ids)))

    def _count_mentions(self, paper_id: int) -> int:
        block_ids = list(self.session.scalars(select(CitationBlock.id).where(CitationBlock.paper_id == paper_id)))
        if not block_ids:
            return 0
        return len(
            list(self.session.scalars(select(CitationMention.id).where(CitationMention.citation_block_id.in_(block_ids))))
        )

    def _build_extraction_result(
        self,
        *,
        paper: Paper,
        mentions_created: int,
        extractions_created: int,
        cleanup_performed: bool,
        status: str,
    ) -> ExtractionRunResult:
        return ExtractionRunResult(
            paper_id=paper.id,
            arxiv_id=paper.arxiv_id,
            mentions_created=mentions_created,
            extractions_created=extractions_created,
            cleanup_performed=cleanup_performed,
            status=status,
            model_name=self.client.model_name,
        )


def build_citation_candidates(
    *,
    raw_text: str,
    section_title: str | None,
    raw_citation_keys: list[str],
    references: dict[str, dict],
) -> list[CitationCandidate]:
    normalized_text = normalize_whitespace(raw_text)
    if not normalized_text:
        return []
    sentences = split_sentences(raw_text)
    if not sentences:
        sentences = [normalized_text]
    candidates: list[CitationCandidate] = []
    for order, key in enumerate(raw_citation_keys):
        reference_title = (references.get(key) or {}).get("title") or key
        sentence_index = _find_sentence_index(sentences, key=key, reference_title=reference_title)
        candidates.append(
            CitationCandidate(
                raw_citation_key=key,
                citation_mention=reference_title,
                sentence_text=sentences[sentence_index],
                section_title=section_title,
                mention_order=order,
            )
        )
    return candidates


def _find_sentence_index(sentences: list[str], *, key: str, reference_title: str) -> int:
    for index, sentence in enumerate(sentences):
        if key in sentence:
            return index
    lowered_title = reference_title.lower()
    if lowered_title:
        for index, sentence in enumerate(sentences):
            if lowered_title in sentence.lower():
                return index
    return 0
