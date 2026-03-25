from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class CrawlRequest(BaseModel):
    arxiv_ids: list[str] = Field(default_factory=list)


class CrawlResponseItem(BaseModel):
    paper_id: int
    arxiv_id: str
    version: str
    status: str


class ParseResponse(BaseModel):
    paper_id: int
    sections_created: int
    references_created: int
    citation_blocks_created: int
    status: str


class ExtractResponse(BaseModel):
    paper_id: int
    mentions_created: int
    extractions_created: int
    status: str


class CitationExtractionView(BaseModel):
    id: int
    intent_label: str
    summary: str
    model: str
    prompt_version: str
    status: str


class CitationMentionView(BaseModel):
    id: int
    paper_reference_id: int
    citation_mention: str
    sentence_text: str
    section_title: str | None
    mention_order: int
    extraction: CitationExtractionView | None = None


class PaperReferenceView(BaseModel):
    id: int
    local_ref_id: str
    raw_text: str
    title: str | None
    year: int | None
    venue: str | None
    mentions: list[CitationMentionView] = Field(default_factory=list)


class PaperView(BaseModel):
    id: int
    arxiv_id: str
    title: str
    abstract: str
    primary_category: str | None
    published_at: datetime | None
    updated_at_source: datetime | None
    current_version: str | None
    parse_status: str
    parsed_at: datetime | None
    ingest_status: str


class CitationSearchItem(BaseModel):
    mention_id: int
    paper_arxiv_id: str
    paper_title: str
    local_ref_id: str
    section_title: str | None
    intent_label: str | None
    summary: str | None
