from __future__ import annotations

from sqlalchemy.orm import Session

from briefgpt_arxiv.services.contracts import PipelineRunResult
from briefgpt_arxiv.services.crawler import CrawlerService
from briefgpt_arxiv.services.extractor import BaseExtractionClient, ExtractorService
from briefgpt_arxiv.services.parser import ParserRepairClient, ParserService


class OrchestratorService:
    def __init__(
        self,
        session: Session,
        repair_client: ParserRepairClient | None = None,
        extraction_client: BaseExtractionClient | None = None,
    ) -> None:
        self.session = session
        self.crawler = CrawlerService(session)
        self.parser = ParserService(session, repair_client=repair_client)
        self.extractor = ExtractorService(session, client=extraction_client)

    def run_for_arxiv_ids(self, arxiv_ids: list[str]) -> list[int]:
        return [result.paper_id for result in self.run_pipeline_for_arxiv_ids(arxiv_ids)]

    def run_pipeline_for_arxiv_ids(
        self,
        arxiv_ids: list[str],
        *,
        rerun_parse: bool = True,
        rerun_extract: bool = True,
    ) -> list[PipelineRunResult]:
        papers = self.crawler.crawl_arxiv_ids(arxiv_ids)
        results: list[PipelineRunResult] = []
        for paper in papers:
            parse_result = self.parser.parse_paper(paper.id, rerun=rerun_parse)
            extract_result = self.extractor.extract_for_paper_result(paper.id, rerun=rerun_extract)
            results.append(
                PipelineRunResult(
                    paper_id=paper.id,
                    arxiv_id=paper.arxiv_id,
                    crawl_status=paper.ingest_status,
                    parse=parse_result,
                    extract=extract_result,
                )
            )
        return results
