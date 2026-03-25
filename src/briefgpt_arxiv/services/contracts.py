from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ParseInputSelection:
    artifact_type: str
    artifact_uri: str


@dataclass(slots=True)
class ParseRunResult:
    paper_id: int
    arxiv_id: str
    source_artifact_type: str
    source_artifact_uri: str
    sections_created: int
    references_created: int
    citation_blocks_created: int
    cleanup_performed: bool
    status: str

    def as_tuple(self) -> tuple[int, int, int]:
        return self.sections_created, self.references_created, self.citation_blocks_created


@dataclass(slots=True)
class ExtractionRunResult:
    paper_id: int
    arxiv_id: str
    mentions_created: int
    extractions_created: int
    cleanup_performed: bool
    status: str
    model_name: str

    def as_tuple(self) -> tuple[int, int]:
        return self.mentions_created, self.extractions_created


@dataclass(slots=True)
class PipelineRunResult:
    paper_id: int
    arxiv_id: str
    crawl_status: str
    parse: ParseRunResult
    extract: ExtractionRunResult
