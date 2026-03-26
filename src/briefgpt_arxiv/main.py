from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException, Query
from sqlalchemy import or_, select
from sqlalchemy.orm import Session, joinedload

from briefgpt_arxiv.db import get_db, init_db
from briefgpt_arxiv.models import CitationBlock, CitationMention, Paper, PaperReference
from briefgpt_arxiv.schemas import (
    CitationExtractionView,
    CitationMentionView,
    CitationSearchItem,
    CrawlRequest,
    CrawlResponseItem,
    ExtractResponse,
    PaperReferenceView,
    PaperView,
    ParseResponse,
)
from briefgpt_arxiv.services.crawler import CrawlerService
from briefgpt_arxiv.services.extractor import ExtractionConfigurationError, ExtractorService
from briefgpt_arxiv.services.parser import ParserService
from briefgpt_arxiv.utils import arxiv_version_number, split_arxiv_id

app = FastAPI(title="briefgpt arXiv citations")
init_db()


def get_mention_section_title(mention: CitationMention) -> str | None:
    return mention.citation_block.section_title


def build_extraction_view(mention: CitationMention) -> CitationExtractionView | None:
    if mention.intent_label is None:
        return None
    return CitationExtractionView(
        id=mention.id,
        intent_label=mention.intent_label,
        summary=mention.summary or "",
        model=mention.model or "",
        prompt_version=mention.prompt_version or "",
        status=mention.status or "completed",
    )


def resolve_paper(session: Session, identifier: str) -> Paper | None:
    arxiv_id, version = split_arxiv_id(identifier)
    if version is not None:
        return session.scalar(
            select(Paper).where(
                Paper.arxiv_id == arxiv_id,
                Paper.version == version,
            )
        )

    candidates = list(session.scalars(select(Paper).where(Paper.arxiv_id == arxiv_id)))
    if not candidates:
        return None
    return max(candidates, key=lambda paper: arxiv_version_number(paper.version))


@app.post("/crawl/arxiv", response_model=list[CrawlResponseItem])
def crawl_arxiv(request: CrawlRequest, session: Session = Depends(get_db)) -> list[CrawlResponseItem]:
    if not request.arxiv_ids:
        raise HTTPException(status_code=400, detail="arxiv_ids is required")
    papers = CrawlerService(session).crawl_arxiv_ids(request.arxiv_ids)
    return [
        CrawlResponseItem(
            paper_id=paper.id,
            arxiv_id=paper.arxiv_id,
            version=paper.version,
            status=paper.ingest_status,
        )
        for paper in papers
    ]


@app.post("/parse/{paper_id}", response_model=ParseResponse)
def parse_paper(paper_id: int, session: Session = Depends(get_db)) -> ParseResponse:
    try:
        result = ParserService(session).parse_paper(paper_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ParseResponse(
        paper_id=paper_id,
        sections_created=result.sections_created,
        references_created=result.references_created,
        citation_blocks_created=result.citation_blocks_created,
        status=result.status,
    )


@app.post("/extract/{paper_id}", response_model=ExtractResponse)
def extract_paper(paper_id: int, session: Session = Depends(get_db)) -> ExtractResponse:
    try:
        mentions_created, extractions_created = ExtractorService(session).extract_for_paper(paper_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ExtractionConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return ExtractResponse(
        paper_id=paper_id,
        mentions_created=mentions_created,
        extractions_created=extractions_created,
        status="ready",
    )


@app.get("/papers/{arxiv_id}", response_model=PaperView)
def get_paper(arxiv_id: str, session: Session = Depends(get_db)) -> PaperView:
    paper = resolve_paper(session, arxiv_id)
    if paper is None:
        raise HTTPException(status_code=404, detail=f"Unknown paper {arxiv_id}")
    return PaperView.model_validate(paper, from_attributes=True)


@app.get("/papers/{arxiv_id}/references", response_model=list[PaperReferenceView])
def get_paper_references(arxiv_id: str, session: Session = Depends(get_db)) -> list[PaperReferenceView]:
    paper = resolve_paper(session, arxiv_id)
    if paper is None:
        raise HTTPException(status_code=404, detail=f"Unknown paper {arxiv_id}")
    references = list(
        session.execute(
            select(PaperReference)
            .where(PaperReference.paper_id == paper.id)
            .options(
                joinedload(PaperReference.citation_mentions)
                .joinedload(CitationMention.citation_block)
            )
        )
        .unique()
        .scalars()
    )
    payload = []
    for reference in references:
        mentions = [
            CitationMentionView(
                id=mention.id,
                paper_reference_id=mention.paper_reference_id,
                citation_mention=mention.citation_mention,
                sentence_text=mention.sentence_text,
                section_title=get_mention_section_title(mention),
                mention_order=mention.mention_order,
                extraction=build_extraction_view(mention),
            )
            for mention in sorted(reference.citation_mentions, key=lambda item: item.mention_order)
        ]
        payload.append(
            PaperReferenceView(
                id=reference.id,
                local_ref_id=reference.local_ref_id,
                raw_text=reference.raw_text,
                title=reference.title,
                year=reference.year,
                venue=reference.venue,
                cited_arxiv_id=reference.cited_arxiv_id,
                cited_version=reference.cited_version,
                mentions=mentions,
            )
        )
    return payload


@app.get("/citations/search", response_model=list[CitationSearchItem])
def search_citations(
    intent: str | None = Query(default=None),
    keyword: str | None = Query(default=None),
    session: Session = Depends(get_db),
) -> list[CitationSearchItem]:
    statement = (
        select(CitationMention)
        .join(CitationMention.paper_reference)
        .join(PaperReference.paper)
        .join(CitationMention.citation_block)
        .options(
            joinedload(CitationMention.paper_reference),
            joinedload(CitationMention.citation_block),
        )
    )
    if intent:
        statement = statement.where(CitationMention.intent_label == intent)
    if keyword:
        needle = f"%{keyword.lower()}%"
        statement = statement.where(
            or_(
                CitationBlock.raw_text.ilike(needle),
                CitationMention.sentence_text.ilike(needle),
                CitationMention.summary.ilike(needle),
            )
        )
    mentions = list(session.scalars(statement))
    return [
        CitationSearchItem(
            mention_id=mention.id,
            paper_arxiv_id=mention.paper_reference.paper.arxiv_id,
            paper_version=mention.paper_reference.paper.version,
            paper_title=mention.paper_reference.paper.title,
            local_ref_id=mention.paper_reference.local_ref_id,
            section_title=get_mention_section_title(mention),
            intent_label=mention.intent_label,
            summary=mention.summary,
        )
        for mention in mentions
    ]
