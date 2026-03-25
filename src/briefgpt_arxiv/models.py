from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from briefgpt_arxiv.db import Base
from briefgpt_arxiv.utils import utcnow_naive


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow_naive, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=utcnow_naive,
        onupdate=utcnow_naive,
        nullable=False,
    )


class Paper(TimestampMixin, Base):
    __tablename__ = "papers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    arxiv_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    title: Mapped[str] = mapped_column(Text)
    abstract: Mapped[str] = mapped_column(Text, default="")
    primary_category: Mapped[str | None] = mapped_column(String(64), nullable=True)
    published_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    updated_at_source: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    current_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    ingest_status: Mapped[str] = mapped_column(String(32), default="discovered", index=True)
    parse_status: Mapped[str] = mapped_column(String(32), default="pending", index=True)
    parsed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    artifacts: Mapped[list["Artifact"]] = relationship(back_populates="paper", cascade="all, delete-orphan")
    references: Mapped[list["PaperReference"]] = relationship(back_populates="paper", cascade="all, delete-orphan")
    citation_blocks: Mapped[list["CitationBlock"]] = relationship(back_populates="paper", cascade="all, delete-orphan")


class Artifact(TimestampMixin, Base):
    __tablename__ = "artifacts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    paper_id: Mapped[int] = mapped_column(ForeignKey("papers.id"), index=True)
    artifact_type: Mapped[str] = mapped_column(String(32), index=True)
    uri: Mapped[str] = mapped_column(Text)
    checksum: Mapped[str | None] = mapped_column(String(128), nullable=True)
    size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)

    paper: Mapped["Paper"] = relationship(back_populates="artifacts")


class PaperReference(TimestampMixin, Base):
    __tablename__ = "paper_references"
    __table_args__ = (UniqueConstraint("paper_id", "local_ref_id", name="uq_paper_reference_local"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    paper_id: Mapped[int] = mapped_column(ForeignKey("papers.id"), index=True)
    local_ref_id: Mapped[str] = mapped_column(String(128))
    raw_text: Mapped[str] = mapped_column(Text)
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    authors_json: Mapped[list[dict] | None] = mapped_column(JSON, nullable=True)
    year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    venue: Mapped[str | None] = mapped_column(Text, nullable=True)

    paper: Mapped["Paper"] = relationship(back_populates="references")
    citation_mentions: Mapped[list["CitationMention"]] = relationship(back_populates="paper_reference")


class CitationBlock(TimestampMixin, Base):
    __tablename__ = "citation_blocks"
    __table_args__ = (UniqueConstraint("paper_id", "chunk_index", name="uq_citation_block_chunk"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    paper_id: Mapped[int] = mapped_column(ForeignKey("papers.id"), index=True)
    section_title: Mapped[str | None] = mapped_column(String(512), nullable=True)
    section_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    chunk_index: Mapped[int] = mapped_column(Integer)
    raw_text: Mapped[str] = mapped_column(Text)
    raw_citation_keys: Mapped[list[str]] = mapped_column(JSON)
    has_citations: Mapped[bool] = mapped_column(Boolean, default=False)
    repair_used: Mapped[bool] = mapped_column(Boolean, default=False)

    paper: Mapped["Paper"] = relationship(back_populates="citation_blocks")
    citation_mentions: Mapped[list["CitationMention"]] = relationship(back_populates="citation_block", cascade="all, delete-orphan")


class CitationMention(TimestampMixin, Base):
    __tablename__ = "citation_mentions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    citation_block_id: Mapped[int] = mapped_column(ForeignKey("citation_blocks.id"), index=True)
    paper_reference_id: Mapped[int] = mapped_column(ForeignKey("paper_references.id"), index=True)
    citation_mention: Mapped[str] = mapped_column(Text)
    sentence_text: Mapped[str] = mapped_column(Text)
    mention_order: Mapped[int] = mapped_column(Integer, default=0)
    model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    prompt_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    intent_label: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    json_result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    status: Mapped[str | None] = mapped_column(String(32), default="completed", index=True, nullable=True)

    citation_block: Mapped["CitationBlock"] = relationship(back_populates="citation_mentions")
    paper_reference: Mapped["PaperReference"] = relationship(back_populates="citation_mentions")


class IngestionJob(TimestampMixin, Base):
    __tablename__ = "ingestion_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    job_type: Mapped[str] = mapped_column(String(32), index=True)
    target_id: Mapped[int] = mapped_column(Integer, index=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    attempt_count: Mapped[int] = mapped_column(Integer, default=1)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow_naive)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
