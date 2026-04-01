from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path
from unittest.mock import Mock, patch

from briefgpt_arxiv.models import Artifact, CitationBlock, IngestionJob, Paper, PaperReference
from briefgpt_arxiv.services.parser import (
    LLMParserRepairClient,
    ParseRepairResult,
    ParserRepairClient,
    ParserService,
)
from tests.conftest import FIXTURES, override_settings


class FakeRepairClient(ParserRepairClient):
    def repair(self, raw_text: str, candidate_keys: list[str]) -> ParseRepairResult:
        repaired_keys = list(candidate_keys)
        if "\\mycite{BIBREF2}" in raw_text and "BIBREF2" not in repaired_keys:
            repaired_keys.append("BIBREF2")
        return ParseRepairResult(
            raw_citation_keys=repaired_keys,
            cleaned_text=raw_text.replace("\\mycite{BIBREF2}", "BIBREF2"),
            used_repair="\\mycite{BIBREF2}" in raw_text,
        )


def _create_paper(db_session):
    paper = Paper(arxiv_id="2603.15726", version="v1", title="Sample", abstract="A")
    db_session.add(paper)
    db_session.flush()
    paper.parse_status = "pending"
    db_session.commit()
    return paper


def test_parse_structured_doc_creates_sections_references_and_blocks(db_session, tempdir, no_llm_keys) -> None:
    paper = _create_paper(db_session)
    structured_path = Path(tempdir.name) / "structured.json"
    structured_path.write_text((FIXTURES / "doc2json_sample.json").read_text())
    artifact = Artifact(paper_id=paper.id, artifact_type="structured_parse", uri=str(structured_path))
    db_session.add(artifact)
    db_session.commit()

    service = ParserService(db_session)
    result = service.parse_paper(paper.id)

    assert result.as_tuple() == (3, 3, 2)
    assert db_session.query(PaperReference).count() == 3
    assert db_session.query(CitationBlock).count() == 3
    assert db_session.query(CitationBlock).filter(CitationBlock.has_citations.is_(True)).count() == 2
    report_artifact = db_session.query(Artifact).filter_by(paper_id=paper.id, artifact_type="parse_report").one()
    report = json.loads(Path(report_artifact.uri).read_text())
    assert report["status"] == "parsed"
    assert report["source_artifact_type"] == "structured_parse"
    assert report["section_count"] == 3
    assert report["reference_count"] == 3
    assert report["citation_block_count"] == 2
    assert report["blocks_with_unresolved_keys"] == 0
    assert report_artifact.size_bytes == Path(report_artifact.uri).stat().st_size
    job = db_session.query(IngestionJob).one()
    assert job.status == "completed"
    assert job.attempt_count == 1
    assert job.finished_at is not None


def test_parse_result_includes_parse_contract_details(db_session, tempdir, no_llm_keys) -> None:
    paper = _create_paper(db_session)
    structured_path = Path(tempdir.name) / "structured.json"
    structured_path.write_text((FIXTURES / "doc2json_sample.json").read_text())
    artifact = Artifact(paper_id=paper.id, artifact_type="structured_parse", uri=str(structured_path))
    db_session.add(artifact)
    db_session.commit()

    result = ParserService(db_session).parse_paper(paper.id)

    assert result.paper_id == paper.id
    assert result.arxiv_id == "2603.15726"
    assert result.version == "v1"
    assert result.source_artifact_type == "structured_parse"
    assert result.source_artifact_uri == str(structured_path)
    assert result.cleanup_performed is False
    assert result.status == "parsed"


def test_parse_structured_doc_skips_repair_client(db_session, tempdir, no_llm_keys) -> None:
    paper = _create_paper(db_session)
    structured_path = Path(tempdir.name) / "structured.json"
    structured_path.write_text((FIXTURES / "doc2json_sample.json").read_text())
    artifact = Artifact(paper_id=paper.id, artifact_type="structured_parse", uri=str(structured_path))
    db_session.add(artifact)
    db_session.commit()

    repair_client = Mock(spec=ParserRepairClient)
    service = ParserService(db_session, repair_client=repair_client)
    service.parse_paper(paper.id)

    repair_client.repair.assert_not_called()


def test_parse_skip_reuses_existing_outputs_and_records_skipped_job(db_session, tempdir, no_llm_keys) -> None:
    paper = _create_paper(db_session)
    structured_path = Path(tempdir.name) / "structured.json"
    structured_path.write_text((FIXTURES / "doc2json_sample.json").read_text())
    artifact = Artifact(paper_id=paper.id, artifact_type="structured_parse", uri=str(structured_path))
    db_session.add(artifact)
    db_session.commit()

    service = ParserService(db_session)
    service.parse_paper(paper.id)
    skipped = service.parse_paper(paper.id, rerun=False)

    assert skipped.as_tuple() == (3, 3, 2)
    assert skipped.status == "skipped"
    assert skipped.cleanup_performed is False
    jobs = db_session.query(IngestionJob).order_by(IngestionJob.id).all()
    assert [job.status for job in jobs] == ["completed", "skipped"]
    assert [job.attempt_count for job in jobs] == [1, 2]


def test_parse_structured_doc_requires_explicit_latex_parse_object(db_session, tempdir, no_llm_keys) -> None:
    paper = _create_paper(db_session)
    structured_path = Path(tempdir.name) / "structured.json"
    structured_path.write_text('{"body_text": [], "bib_entries": {}}')
    artifact = Artifact(paper_id=paper.id, artifact_type="structured_parse", uri=str(structured_path))
    db_session.add(artifact)
    db_session.commit()

    import pytest
    with pytest.raises(ValueError, match="latex_parse"):
        ParserService(db_session).parse_paper(paper.id)


def test_parse_source_uses_repair_client_for_nonstandard_macros(db_session, tempdir, no_llm_keys) -> None:
    paper = _create_paper(db_session)
    source_path = Path(tempdir.name) / "source.tex"
    source_path.write_text((FIXTURES / "latex_sample.tex").read_text())
    artifact = Artifact(paper_id=paper.id, artifact_type="source", uri=str(source_path))
    db_session.add(artifact)
    db_session.commit()

    service = ParserService(db_session, repair_client=FakeRepairClient())
    _, refs, blocks = service.parse_paper(paper.id).as_tuple()

    assert refs == 3
    assert blocks == 2
    repaired_block = (
        db_session.query(CitationBlock)
        .filter(CitationBlock.repair_used.is_(True))
        .one()
    )
    assert "BIBREF2" in repaired_block.raw_citation_keys


def test_parse_source_skips_repair_for_standard_cite_macros(db_session, tempdir, no_llm_keys) -> None:
    paper = _create_paper(db_session)
    source_path = Path(tempdir.name) / "source.tex"
    source_path.write_text(
        r"""
\documentclass{article}
\begin{document}
\section{Intro}
This paragraph has no references at all.

We compare against prior work \cite{alpha2024}.

\begin{thebibliography}{9}
\bibitem{alpha2024} Alpha Systems for Planning. 2024.
\end{thebibliography}
\end{document}
"""
    )
    artifact = Artifact(paper_id=paper.id, artifact_type="source", uri=str(source_path))
    db_session.add(artifact)
    db_session.commit()

    repair_client = Mock(spec=ParserRepairClient)
    repair_client.repair.return_value = ParseRepairResult(
        raw_citation_keys=["alpha2024"],
        cleaned_text="We compare against prior work alpha2024.",
        used_repair=True,
    )

    service = ParserService(db_session, repair_client=repair_client)
    _, refs, blocks = service.parse_paper(paper.id).as_tuple()

    assert refs == 1
    assert blocks == 1
    assert repair_client.repair.call_count == 0


def test_parse_source_ignores_commented_citations(db_session, tempdir, no_llm_keys) -> None:
    paper = _create_paper(db_session)
    source_path = Path(tempdir.name) / "source.tex"
    source_path.write_text(
        r"""
\documentclass{article}
\begin{document}
\section{Intro}
% Legacy note \citep{old2025key}.
We compare against MiroFlow \citep{miromind2025miroflow}.

\begin{thebibliography}{9}
\bibitem{miromind2025miroflow} MiroFlow. 2025.
\end{thebibliography}
\end{document}
"""
    )
    artifact = Artifact(paper_id=paper.id, artifact_type="source", uri=str(source_path))
    db_session.add(artifact)
    db_session.commit()

    service = ParserService(db_session)
    _, refs, blocks = service.parse_paper(paper.id).as_tuple()

    assert refs == 1
    assert blocks == 1
    block = db_session.query(CitationBlock).filter(CitationBlock.has_citations.is_(True)).one()
    assert block.raw_citation_keys == ["miromind2025miroflow"]


def test_llm_repair_client_derives_used_repair_without_model_flag(no_llm_keys) -> None:
    with override_settings(openrouter_api_key="test-key"):
        client = LLMParserRepairClient()

        class StubLLMClient:
            def generate_json(self, **_kwargs) -> dict:
                return {
                    "raw_citation_keys": ["BIBREF0", "BIBREF2"],
                    "cleaned_text": "Text with BIBREF2",
                }

        client.client = StubLLMClient()
        repaired = client.repair("Text with \\mycite{BIBREF2}", ["BIBREF0"])

        assert repaired.raw_citation_keys == ["BIBREF0", "BIBREF2"]
        assert repaired.cleaned_text == "Text with BIBREF2"
        assert repaired.used_repair is True


def test_llm_repair_client_prompt_omits_schema_text(no_llm_keys) -> None:
    with override_settings(openrouter_api_key="test-key"):
        client = LLMParserRepairClient()
        client.client = Mock()
        client.client.generate_json.return_value = {
            "raw_citation_keys": ["BIBREF0"],
            "cleaned_text": "Text with BIBREF0",
        }

        client.repair("Text with BIBREF0", ["BIBREF0"])

        kwargs = client.client.generate_json.call_args.kwargs
        assert "## Output Format" in kwargs["user_text"]
        assert "Do not return any explanatory text before or after the JSON object." in kwargs["user_text"]
        assert "The JSON object must contain:" in kwargs["user_text"]
        assert "`raw_citation_keys`" in kwargs["user_text"]
        assert "`cleaned_text`" in kwargs["user_text"]
        assert "`used_repair`" in kwargs["user_text"]
        assert "The JSON must match this schema exactly" not in kwargs["user_text"]


def test_llm_repair_client_propagates_exception(no_llm_keys) -> None:
    with override_settings(openrouter_api_key="test-key"):
        client = LLMParserRepairClient()

        class FailingLLMClient:
            def generate_json(self, **_kwargs) -> dict:
                raise TimeoutError("stuck response")

        client.client = FailingLLMClient()
        import pytest
        with pytest.raises(TimeoutError):
            client.repair("Text with \\mycite{BIBREF2}", ["BIBREF0"])


def test_parse_source_tar_extracts_references_from_external_bib(db_session, tempdir, no_llm_keys) -> None:
    paper = _create_paper(db_session)
    source_path = Path(tempdir.name) / "source.tar"
    tex_text = r"""
\documentclass{article}
\begin{document}
\section{Intro}
We follow prior work \cite{alpha2024,beta2023}.
\bibliography{main}
\end{document}
"""
    bib_text = r"""
@article{alpha2024,
  title={Alpha Systems for Planning},
  author={Ada Lovelace and Grace Hopper},
  journal={Journal of Agents},
  year={2024}
}

@inproceedings{beta2023,
  title={Beta Benchmarks for Search},
  author={Alan Turing},
  booktitle={Proceedings of SearchConf},
  year={2023}
}
"""
    with tarfile.open(source_path, "w") as archive:
        for name, payload in {"main.tex": tex_text, "main.bib": bib_text}.items():
            encoded = payload.encode()
            info = tarfile.TarInfo(name=name)
            info.size = len(encoded)
            archive.addfile(info, io.BytesIO(encoded))
    artifact = Artifact(paper_id=paper.id, artifact_type="source", uri=str(source_path))
    db_session.add(artifact)
    db_session.commit()

    service = ParserService(db_session, repair_client=FakeRepairClient())
    _, refs, blocks = service.parse_paper(paper.id).as_tuple()

    assert refs == 2
    assert blocks == 1
    reference_rows = db_session.query(PaperReference).order_by(PaperReference.local_ref_id).all()
    assert reference_rows[0].local_ref_id == "alpha2024"
    assert reference_rows[0].title == "Alpha Systems for Planning"
    assert reference_rows[0].year == 2024
    assert reference_rows[0].venue == "Journal of Agents"
    assert reference_rows[1].local_ref_id == "beta2023"
    assert reference_rows[1].title == "Beta Benchmarks for Search"
    assert reference_rows[1].year == 2023


def test_pdf_text_fallback_parses_reference_blocks(db_session, tempdir, no_llm_keys) -> None:
    paper = _create_paper(db_session)
    pdf_text_path = Path(tempdir.name) / "paper.txt"
    pdf_text_path.write_text((FIXTURES / "pdf_fallback.txt").read_text())
    artifact = Artifact(paper_id=paper.id, artifact_type="pdf_text", uri=str(pdf_text_path))
    db_session.add(artifact)
    db_session.commit()

    service = ParserService(db_session)
    _, refs, blocks = service.parse_paper(paper.id).as_tuple()

    assert refs == 2
    assert blocks == 2


def test_pdf_text_fallback_reconstructs_paragraphs_and_citation_ranges(db_session, tempdir, no_llm_keys) -> None:
    paper = _create_paper(db_session)
    pdf_text_path = Path(tempdir.name) / "paper.txt"
    pdf_text_path.write_text((FIXTURES / "pdf_complex_sample.txt").read_text())
    artifact = Artifact(paper_id=paper.id, artifact_type="pdf_text", uri=str(pdf_text_path))
    db_session.add(artifact)
    db_session.commit()

    service = ParserService(db_session)
    _, refs, blocks = service.parse_paper(paper.id).as_tuple()

    assert refs == 5
    assert blocks == 2
    block_rows = db_session.query(CitationBlock).order_by(CitationBlock.id).all()
    citation_rows = [row for row in block_rows if row.has_citations]
    assert citation_rows[0].raw_citation_keys == ["REF1", "REF2", "REF3"]
    assert citation_rows[1].raw_citation_keys == ["REF4", "REF5"]
    assert "robust planning behavior in agents." in citation_rows[0].raw_text
    assert "API Match and Correctness." in citation_rows[1].raw_text
    section_titles = [block.section_title for block in block_rows]
    assert "1. Introduction" in section_titles
    assert "2. Metrics" in section_titles

    reference_rows = db_session.query(PaperReference).order_by(PaperReference.id).all()
    assert reference_rows[0].title == "Alpha systems for planning"
    assert reference_rows[-1].title == "Epsilon study of correctness"


def test_pdf_artifact_is_directly_parsed_and_persisted_as_pdf_text(db_session, tempdir, no_llm_keys) -> None:
    paper = _create_paper(db_session)
    pdf_path = Path(tempdir.name) / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")
    artifact = Artifact(paper_id=paper.id, artifact_type="pdf", uri=str(pdf_path))
    db_session.add(artifact)
    db_session.commit()

    class FakePage:
        def __init__(self, text: str) -> None:
            self._text = text

        def extract_text(self) -> str:
            return self._text

    class FakePdfReader:
        def __init__(self, _path: str) -> None:
            self.pages = [FakePage((FIXTURES / "pdf_fallback.txt").read_text())]

    with patch("pypdf.PdfReader", FakePdfReader):
        service = ParserService(db_session)
        _, refs, blocks = service.parse_paper(paper.id).as_tuple()

    assert refs == 2
    assert blocks == 2
    pdf_text_artifact = (
        db_session.query(Artifact)
        .filter(Artifact.paper_id == paper.id, Artifact.artifact_type == "pdf_text")
        .one()
    )
    assert Path(pdf_text_artifact.uri).exists()
