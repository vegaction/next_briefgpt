from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import logging
import re

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from briefgpt_arxiv.config import settings
from briefgpt_arxiv.llm_client import BaseLLMClient, create_llm_client
from briefgpt_arxiv.models import CitationBlock, CitationMention, Paper, PaperReference
from briefgpt_arxiv.prompts import (
    EXTRACTOR_PROMPT_VERSION,
    EXTRACTION_SYSTEM_TEMPLATE,
    EXTRACTION_USER_TEMPLATE,
    PROMPT_TEMPLATE_ENV,
)
from briefgpt_arxiv.services.contracts import ExtractionRunResult
from briefgpt_arxiv.services.jobs import JobTracker
from briefgpt_arxiv.utils import normalize_whitespace, split_sentences

logger = logging.getLogger(__name__)


TABULAR_SCORE_TOKEN_PATTERN = re.compile(r"(?<![A-Za-z])(?:\d{1,3}(?:\.\d+)?|--)(?![A-Za-z])")
RAW_CITATION_KEY_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]*20\d{2}[A-Za-z0-9_]*\b")
SUMMARY_SELF_REFERENCE_PATTERN = re.compile(
    r"\b(cited|mentioned here|used here|adapted here|applied here|extended here|this paper|our system|we|here)\b",
    re.IGNORECASE,
)
BENCHMARK_PERFORMANCE_PATTERN = re.compile(
    r"\b(score(?:s|d)?|achiev(?:e|es|ed)|reach(?:es|ed)?|outperform(?:s|ed)?|"
    r"surpass(?:es|ed)?|state-of-the-art|sota|best result|narrowing the gap)\b",
    re.IGNORECASE,
)


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


class ExtractionConfigurationError(RuntimeError):
    """Raised when extraction is requested without a configured LLM client."""


class ExtractorService:
    def __init__(
        self,
        session: Session,
        llm_client: BaseLLMClient | None = None,
    ) -> None:
        if llm_client is None:
            if not settings.has_llm_api_key(settings.extractor_llm.provider):
                raise ExtractionConfigurationError(
                    f"Extraction requires credentials for provider {settings.extractor_llm.provider!r}."
                )
            llm_client = create_llm_client(settings.extractor_llm)
        self.session = session
        self.llm_client = llm_client
        self.model_name = llm_client.model_name
        self.job_tracker = JobTracker(session)

    def extract_for_paper(self, paper_id: int) -> tuple[int, int]:
        return self.extract_for_paper_result(paper_id).as_tuple()

    def extract_for_paper_result(self, paper_id: int, *, rerun: bool = True) -> ExtractionRunResult:
        paper = self.session.get(Paper, paper_id)
        if paper is None:
            raise ValueError(f"Unknown paper id {paper_id}")

        existing_mentions = self._count_mentions(paper_id)
        if not rerun and existing_mentions:
            job = self.job_tracker.start(job_type="extract", target_id=paper_id)
            self.job_tracker.finish(job, status="skipped", error_message="Reused existing extraction outputs.")
            self.session.commit()
            return ExtractionRunResult(
                paper_id=paper.id,
                arxiv_id=paper.arxiv_id,
                version=paper.version,
                mentions_created=existing_mentions,
                extractions_created=existing_mentions,
                cleanup_performed=False,
                status="skipped",
                model_name=self.model_name,
            )

        job = self.job_tracker.start(job_type="extract", target_id=paper_id)
        try:
            if existing_mentions:
                self.clear_extractions(paper_id)

            blocks = list(
                self.session.scalars(
                    select(CitationBlock).where(
                        CitationBlock.paper_id == paper_id,
                        CitationBlock.has_citations.is_(True),
                    )
                )
            )
            references = {
                row.local_ref_id: {
                    "paper_reference_id": row.id,
                    "title": row.title,
                    "year": row.year,
                }
                for row in self.session.scalars(select(PaperReference).where(PaperReference.paper_id == paper_id))
            }

            mentions_created = 0
            for block in blocks:
                missing_reference_keys = [key for key in block.raw_citation_keys if key not in references]
                if missing_reference_keys:
                    logger.warning(
                        "Skipping unknown citation keys before extraction paper_id=%s block_id=%s keys=%s",
                        paper_id,
                        block.id,
                        missing_reference_keys,
                    )

                candidates = build_citation_candidates(
                    raw_text=block.raw_text,
                    section_title=block.section_title,
                    raw_citation_keys=[key for key in block.raw_citation_keys if key in references],
                    references=references,
                )
                if not candidates:
                    continue
                if should_skip_extraction_block(block.raw_text):
                    logger.info(
                        "Skipping non-narrative citation block before extraction paper_id=%s block_id=%s section=%r",
                        paper_id,
                        block.id,
                        block.section_title,
                    )
                    continue

                debug_context = {
                    "paper_id": paper_id,
                    "block_id": block.id,
                    "section_title": block.section_title,
                    "raw_citation_keys": block.raw_citation_keys,
                    "raw_text": block.raw_text,
                    "candidates": [asdict(candidate) for candidate in candidates],
                }
                prompt_candidates = [
                    {
                        "mention_order": candidate.mention_order,
                        "raw_citation_key": candidate.raw_citation_key,
                        "citation_mention": candidate.citation_mention,
                        "sentence_text": candidate.sentence_text,
                        "reference": {
                            "title": references.get(candidate.raw_citation_key, {}).get("title"),
                            "year": references.get(candidate.raw_citation_key, {}).get("year"),
                        },
                    }
                    for candidate in candidates
                ]
                system_instruction = PROMPT_TEMPLATE_ENV.from_string(EXTRACTION_SYSTEM_TEMPLATE).render()
                user_text = PROMPT_TEMPLATE_ENV.from_string(EXTRACTION_USER_TEMPLATE).render(
                    raw_text=block.raw_text,
                    section_title=block.section_title,
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
                        "prompt_candidates": prompt_candidates,
                    },
                )
                payload: dict | None = None
                try:
                    payload = self.llm_client.generate_json(
                        system_instruction=system_instruction,
                        user_text=user_text,
                    )
                    annotations = self._parse_annotations(payload)
                    self._validate_annotations(candidates, annotations)
                except Exception as exc:
                    self._append_summary_debug_log(
                        event="summary_error",
                        debug_context=debug_context,
                        payload={
                            "model_name": self.model_name,
                            "prompt_version": EXTRACTOR_PROMPT_VERSION,
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                            "response_payload": payload,
                        },
                    )
                    raise
                self._append_summary_debug_log(
                    event="summary_output",
                    debug_context=debug_context,
                    payload={
                        "model_name": self.model_name,
                        "prompt_version": EXTRACTOR_PROMPT_VERSION,
                        "response_items": annotations,
                        "response_payload": payload,
                    },
                )
                annotations_by_order = {item["mention_order"]: item for item in annotations}
                extracted = [
                    ExtractedCitation(
                        raw_citation_key=candidate.raw_citation_key,
                        citation_mention=candidate.citation_mention,
                        sentence_text=candidate.sentence_text,
                        section_title=candidate.section_title,
                        mention_order=candidate.mention_order,
                        intent_label=annotations_by_order[candidate.mention_order]["intent_label"],
                        summary=annotations_by_order[candidate.mention_order]["summary"],
                    )
                    for candidate in candidates
                ]
                logger.info(
                    "Extracted %s candidate summaries for paper_id=%s block_id=%s section=%r model=%s",
                    len(extracted),
                    paper_id,
                    block.id,
                    block.section_title,
                    self.model_name,
                )

                for item in extracted:
                    logger.info(
                        "Generated summary paper_id=%s block_id=%s key=%s section=%r summary=%r",
                        paper_id,
                        block.id,
                        item.raw_citation_key,
                        item.section_title,
                        item.summary,
                    )
                    summary = postprocess_extracted_summary(
                        summary=item.summary,
                        citation_mention=item.citation_mention,
                        sentence_text=item.sentence_text,
                        intent_label=item.intent_label,
                    )
                    self.session.add(
                        CitationMention(
                            citation_block_id=block.id,
                            paper_reference_id=references[item.raw_citation_key]["paper_reference_id"],
                            citation_mention=item.citation_mention,
                            sentence_text=item.sentence_text,
                            mention_order=item.mention_order,
                            model=self.model_name,
                            prompt_version=EXTRACTOR_PROMPT_VERSION,
                            intent_label=item.intent_label,
                            summary=summary,
                            json_result={
                                "raw_citation_key": item.raw_citation_key,
                                "citation_mention": item.citation_mention,
                                "sentence_text": item.sentence_text,
                                "section_title": item.section_title,
                                "mention_order": item.mention_order,
                                "intent_label": item.intent_label,
                                "summary": summary,
                            },
                            status="completed",
                        )
                    )
                    mentions_created += 1

            paper.ingest_status = "ready"
            self.job_tracker.finish(job)
            self.session.commit()
            return ExtractionRunResult(
                paper_id=paper.id,
                arxiv_id=paper.arxiv_id,
                version=paper.version,
                mentions_created=mentions_created,
                extractions_created=mentions_created,
                cleanup_performed=bool(existing_mentions),
                status="ready",
                model_name=self.model_name,
            )
        except Exception as exc:
            self.session.rollback()
            self.job_tracker.record_failure(job_type="extract", target_id=paper_id, error_message=str(exc))
            self.session.commit()
            raise

    @staticmethod
    def _parse_annotations(payload: dict) -> list[dict]:
        if not isinstance(payload, dict):
            raise RuntimeError(f"Extractor returned a non-object JSON payload: {payload!r}")
        if set(payload.keys()) != {"items"}:
            raise RuntimeError(f"Extractor returned unexpected top-level keys: {sorted(payload.keys())!r}")
        items = payload.get("items")
        if not isinstance(items, list):
            raise RuntimeError(f"Extractor returned a non-list `items` payload: {payload!r}")
        for item in items:
            if not isinstance(item, dict):
                raise RuntimeError(f"Extractor returned a non-object item payload: {payload!r}")
            if set(item.keys()) != {"mention_order", "intent_label", "summary"}:
                raise RuntimeError(f"Extractor returned unexpected item keys: {sorted(item.keys())!r}")
        return items

    @staticmethod
    def _validate_annotations(candidates: list[CitationCandidate], annotations: list[dict]) -> None:
        expected_orders = [candidate.mention_order for candidate in candidates]
        seen_orders = [int(item["mention_order"]) for item in annotations]
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

    def clear_extractions(self, paper_id: int) -> None:
        paper = self.session.get(Paper, paper_id)
        if paper is None:
            raise ValueError(f"Unknown paper id {paper_id}")
        self.session.execute(
            delete(CitationMention).where(
                CitationMention.citation_block_id.in_(
                    select(CitationBlock.id).where(CitationBlock.paper_id == paper_id)
                )
            )
        )
        if paper.ingest_status == "ready":
            paper.ingest_status = "parsed"

    def _count_mentions(self, paper_id: int) -> int:
        return int(
            self.session.scalar(
                select(func.count(CitationMention.id))
                .join(CitationMention.citation_block)
                .where(CitationBlock.paper_id == paper_id)
            )
            or 0
        )


def postprocess_extracted_summary(
    *,
    summary: str,
    citation_mention: str,
    sentence_text: str | None,
    intent_label: str | None,
) -> str:
    normalized_summary = normalize_whitespace(summary)
    normalized_mention = normalize_whitespace(citation_mention)
    if not normalized_summary or not normalized_mention:
        return normalized_summary

    normalized_summary = _replace_raw_key_lead_in(normalized_summary, normalized_mention)
    normalized_summary = _strip_document_internal_language(normalized_summary)
    normalized_summary = _smooth_title_as_subject(normalized_summary, normalized_mention)
    if _should_drop_benchmark_summary(
        summary=normalized_summary,
        citation_mention=normalized_mention,
        sentence_text=sentence_text or "",
        intent_label=intent_label or "",
    ):
        return ""

    lowered_summary = normalized_summary.lower()
    lowered_mention = normalized_mention.lower()
    title_index = lowered_summary.find(lowered_mention)
    if title_index <= 0:
        return normalized_summary

    lead_in = normalized_summary[:title_index]
    if not re.search(r"\(\d{4}\)", lead_in):
        return normalized_summary
    if not re.search(
        r"\b(introduce|introduces|introduced|present|presents|presented|propose|proposes|proposed|"
        r"describe|describes|described|define|defines|defined|develop|develops|developed|"
        r"provide|provides|provided|study|studies|studied)\b",
        lead_in.lower(),
    ):
        return normalized_summary
    return normalized_summary[title_index:]


def _replace_raw_key_lead_in(summary: str, citation_mention: str) -> str:
    if not RAW_CITATION_KEY_PATTERN.match(summary):
        return summary
    replacement = citation_mention.split(":", 1)[0].strip()
    if not replacement:
        return summary
    return RAW_CITATION_KEY_PATTERN.sub(replacement, summary, count=1)


def _strip_document_internal_language(summary: str) -> str:
    cleaned = summary
    for pattern, replacement in [
        (r"\bis cited as\b", "is"),
        (r"\bis cited\b", "is"),
        (r"\bused here to\b", "used to"),
        (r"\bused here\b", "used"),
        (r"\badapted here\b", "adapted"),
        (r"\bapplied here\b", "applied"),
        (r"\bextended here\b", "extended"),
        (r"\bthis benchmark\b", "the benchmark"),
    ]:
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(SUMMARY_SELF_REFERENCE_PATTERN, "", cleaned)
    return normalize_whitespace(cleaned)


def _smooth_title_as_subject(summary: str, citation_mention: str) -> str:
    subject = citation_mention.split(":", 1)[0].strip()
    if not subject:
        return summary
    return re.sub(
        rf"^{re.escape(subject)}\s+(demonstrates using|uses|use|employs|adopts)\b",
        f"{subject} describes",
        summary,
        count=1,
        flags=re.IGNORECASE,
    )


def _should_drop_benchmark_summary(
    *,
    summary: str,
    citation_mention: str,
    sentence_text: str,
    intent_label: str,
) -> bool:
    if intent_label != "benchmark_or_dataset":
        return False
    combined_text = f"{summary} {sentence_text}".lower()
    if not BENCHMARK_PERFORMANCE_PATTERN.search(combined_text):
        return False
    if not re.search(r"\b\d{1,3}(?:\.\d+)?\b", combined_text):
        return False
    if re.search(r"\b(mirothinker|our system|previous leading model|all evaluated models)\b", combined_text):
        return True
    return "the benchmark" in summary.lower()


def should_skip_extraction_block(raw_text: str) -> bool:
    normalized_text = normalize_whitespace(raw_text)
    if not normalized_text:
        return True

    ampersand_count = raw_text.count("&")
    row_break_count = raw_text.count("\\\\")
    score_token_count = len(TABULAR_SCORE_TOKEN_PATTERN.findall(raw_text))
    lowered = normalized_text.lower()

    if lowered.startswith("tab:") and (ampersand_count >= 2 or row_break_count >= 1):
        return True
    if ampersand_count >= 4 and score_token_count >= 6:
        return True
    if row_break_count >= 2 and score_token_count >= 6:
        return True
    return False


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
    sentences = split_sentences(raw_text) or [normalized_text]
    return [
        CitationCandidate(
            raw_citation_key=key,
            citation_mention=(references.get(key) or {}).get("title") or key,
            sentence_text=sentences[_find_sentence_index(
                sentences,
                key=key,
                reference_title=(references.get(key) or {}).get("title") or key,
            )],
            section_title=section_title,
            mention_order=order,
        )
        for order, key in enumerate(raw_citation_keys)
    ]


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
